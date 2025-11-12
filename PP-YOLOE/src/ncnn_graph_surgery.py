#!/usr/bin/env python3
"""
NCNN/Vulkan Graph Surgery for PP-YOLOE ONNX

This pass prepares the customized PP-YOLOE detection head for NCNN+pnnx by:
- Forcing static shapes so Vulkan shaders avoid dynamic branches
- Rewriting costly ops into NCNN-friendly forms (Div→Mul, Pow rewrites, Slice→Gather, etc.)
- Folding shape computation chains and clearing redundant nodes
- Reporting ops that commonly block NCNN pipelines

Expected workflow:
1. export_to_onnx.sh → ppyoloe_crn_s_36e_pphuman.onnx (base model)
2. onnx_customize.py → ppyoloe_crn_s_36e_pphuman_cust.onnx (customized model)
3. ncnn_graph_surgery.py → ppyoloe_crn_s_36e_pphuman_cust_ncnn.onnx (NCNN-friendly model)
4. export_to_ncnn.sh → final .param/.bin via pnnx + ncnnoptimize
"""
import argparse
import os
import shutil
from typing import Tuple, List, Dict, Any, Optional
import sys

import onnx
import numpy as np
from onnx import helper, numpy_helper

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Reuse functions from profile_onnx
from onnx_profile import (
    load_model_info,
    parse_shape,
    run_benchmark,
)


def _build_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for nm in node.input:
            if not nm:
                continue
            consumers.setdefault(nm, []).append(node)
    return consumers


def fold_conv_batchnorm(model_path: str, out_path: str) -> str:
    """Fold Conv->BatchNormalization into a single Conv by absorbing BN params.

    Safety constraints:
    - Only when the BN input comes directly from a Conv output
    - That Conv output has a single consumer (the BN)
    - Conv weights and BN params are initializers (constants)
    - Supports Conv with or without bias
    """
    m = onnx.load(model_path)
    g = m.graph

    # Map initializer name -> array
    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    name_to_init: Dict[str, onnx.TensorProto] = {init.name: init for init in g.initializer}

    # Quick lookup for node by its first output
    out_to_node: Dict[str, onnx.NodeProto] = {}
    for node in g.node:
        if node.output:
            out_to_node[node.output[0]] = node

    consumers = _build_consumers(g)

    kept: List[onnx.NodeProto] = []
    removed: List[onnx.NodeProto] = []
    changed = 0

    for node in g.node:
        if node.op_type != "BatchNormalization":
            kept.append(node)
            continue
        if len(node.input) < 5:
            kept.append(node)
            continue

        x, gamma_n, beta_n, mean_n, var_n = node.input[:5]
        conv = out_to_node.get(x)
        if conv is None or conv.op_type != "Conv":
            kept.append(node)
            continue

        # ensure single consumer of conv output
        if len(consumers.get(conv.output[0], [])) != 1:
            kept.append(node)
            continue

        # Fetch conv weights and (optional) bias
        if len(conv.input) < 2:
            kept.append(node)
            continue
        w_name = conv.input[1]
        if w_name not in init_map:
            kept.append(node)
            continue
        W = init_map[w_name].astype(np.float32)
        b = None
        if len(conv.input) >= 3 and conv.input[2] in init_map:
            b = init_map[conv.input[2]].astype(np.float32)

        # Fetch BN params
        if not all(nm in init_map for nm in (gamma_n, beta_n, mean_n, var_n)):
            kept.append(node)
            continue
        gamma = init_map[gamma_n].astype(np.float32).reshape(-1)
        beta = init_map[beta_n].astype(np.float32).reshape(-1)
        mean = init_map[mean_n].astype(np.float32).reshape(-1)
        var = init_map[var_n].astype(np.float32).reshape(-1)

        eps = 1e-5
        for a in node.attribute:
            if a.name == "epsilon":
                try:
                    eps = float(a.f)
                except Exception:
                    eps = eps

        oc = W.shape[0]
        if gamma.shape[0] != oc or beta.shape[0] != oc or mean.shape[0] != oc or var.shape[0] != oc:
            kept.append(node)
            continue

        # Compute scale and new parameters
        std = np.sqrt(var + eps)
        scale = (gamma / std).reshape(oc, 1, 1, 1)
        W_new = W * scale
        if b is None:
            b0 = np.zeros((oc,), dtype=np.float32)
        else:
            b0 = b
        b_new = beta + (b0 - mean) * (gamma / std)

        # Update initializers in-place
        # weights
        if w_name in name_to_init:
            del name_to_init[w_name].raw_data
            name_to_init[w_name].CopyFrom(numpy_helper.from_array(W_new.astype(np.float32), name=w_name))
        else:
            # replace map and graph initializer list
            for i, init in enumerate(list(g.initializer)):
                if init.name == w_name:
                    g.initializer[i] = numpy_helper.from_array(W_new.astype(np.float32), name=w_name)
                    break

        # bias
        if len(conv.input) >= 3 and conv.input[2] in name_to_init:
            b_name = conv.input[2]
            del name_to_init[b_name].raw_data
            name_to_init[b_name].CopyFrom(numpy_helper.from_array(b_new.astype(np.float32), name=b_name))
        else:
            # create new bias initializer and hook into conv
            b_name = conv.name + "_bnfold_bias"
            # ensure unique name
            exist_names = {init.name for init in g.initializer}
            if b_name in exist_names:
                idx = 0
                while f"{b_name}_{idx}" in exist_names:
                    idx += 1
                b_name = f"{b_name}_{idx}"
            g.initializer.extend([numpy_helper.from_array(b_new.astype(np.float32), name=b_name)])
            if len(conv.input) >= 3:
                conv.input[2] = b_name
            else:
                conv.input.extend([b_name])

        # Rewire outputs: Conv now produces BN's output
        if node.output and conv.output:
            conv.output[0] = node.output[0]

        # Mark BN for removal (do not append to kept)
        removed.append(node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    # Rebuild node list without removed BNs
    new_nodes = [n for n in g.node if n not in removed]
    del g.node[:]
    g.node.extend(new_nodes)

    onnx.save(m, out_path)
    print(f"Folded Conv+BatchNorm: {changed} chain(s)")
    return out_path


def _extract_channel_vector(arr: np.ndarray, C: int) -> Optional[np.ndarray]:
    """Try to reduce arr to a (C,) vector by squeezing dims of size 1.
    Returns None if not possible or mismatched length.
    """
    a = np.array(arr)
    # Try common shapes first
    if a.ndim == 1 and a.shape[0] == C:
        return a.astype(np.float32).reshape(C)
    # Squeeze all ones
    squeezed = np.squeeze(a)
    if squeezed.ndim == 1 and squeezed.shape[0] == C:
        return squeezed.astype(np.float32).reshape(C)
    # Handle (1,C,1,1) explicitly
    if a.ndim == 4 and a.shape[1] == C and a.shape[0] in (1, ) and a.shape[2] in (1,) and a.shape[3] in (1,):
        return a.reshape(C).astype(np.float32)
    return None


def fold_conv_mul_scale(model_path: str, out_path: str) -> str:
    """Fold per-channel constant Mul after Conv into Conv weights/bias.

    Pattern: y = Mul(Conv(x, W, b), s) where s is constant broadcastable to (N,C,H,W) per-channel.
    Requires Conv output to be consumed only by this Mul.
    """
    m = onnx.load(model_path)
    g = m.graph

    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    name_to_init: Dict[str, onnx.TensorProto] = {init.name: init for init in g.initializer}
    consumers = _build_consumers(g)
    out_to_node: Dict[str, onnx.NodeProto] = {n.output[0]: n for n in g.node if n.output}

    removed: List[onnx.NodeProto] = []
    changed = 0

    for node in g.node:
        if node.op_type != "Mul" or len(node.input) != 2 or not node.output:
            continue
        a, b = node.input
        # identify conv output and scale constant
        conv_out = None
        scale_name = None
        if a in out_to_node and out_to_node[a].op_type == "Conv" and b in init_map:
            conv_out, scale_name = a, b
        elif b in out_to_node and out_to_node[b].op_type == "Conv" and a in init_map:
            conv_out, scale_name = b, a
        else:
            continue

        conv = out_to_node[conv_out]
        if len(consumers.get(conv.output[0], [])) != 1:
            continue

        # Fetch conv params
        if len(conv.input) < 2:
            continue
        w_name = conv.input[1]
        if w_name not in init_map:
            continue
        W = init_map[w_name].astype(np.float32)
        oc = W.shape[0]

        bias = None
        if len(conv.input) >= 3 and conv.input[2] in init_map:
            bias = init_map[conv.input[2]].astype(np.float32)

        s = _extract_channel_vector(init_map[scale_name], oc)
        if s is None:
            continue

        # Apply folding: W' = W * s[:,1,1,1]; b' = b * s
        scale4 = s.reshape(oc, 1, 1, 1)
        W_new = W * scale4
        if bias is not None:
            b_new = (bias * s).astype(np.float32)
        else:
            b_new = None

        # Update initializers
        if w_name in name_to_init:
            del name_to_init[w_name].raw_data
            name_to_init[w_name].CopyFrom(numpy_helper.from_array(W_new.astype(np.float32), name=w_name))
        else:
            for i, init in enumerate(list(g.initializer)):
                if init.name == w_name:
                    g.initializer[i] = numpy_helper.from_array(W_new.astype(np.float32), name=w_name)
                    break
        if b_new is not None:
            if len(conv.input) >= 3 and conv.input[2] in name_to_init:
                b_name = conv.input[2]
                del name_to_init[b_name].raw_data
                name_to_init[b_name].CopyFrom(numpy_helper.from_array(b_new.astype(np.float32), name=b_name))
            elif len(conv.input) >= 3:
                # replace existing by name
                for i, init in enumerate(list(g.initializer)):
                    if init.name == conv.input[2]:
                        g.initializer[i] = numpy_helper.from_array(b_new.astype(np.float32), name=conv.input[2])
                        break
            else:
                # create new bias
                b_name = conv.name + "_mulfold_bias"
                exist_names = {init.name for init in g.initializer}
                if b_name in exist_names:
                    idx = 0
                    while f"{b_name}_{idx}" in exist_names:
                        idx += 1
                    b_name = f"{b_name}_{idx}"
                g.initializer.extend([numpy_helper.from_array(b_new.astype(np.float32), name=b_name)])
                conv.input.extend([b_name])

        # Rewire conv output to mul output, drop Mul
        conv.output[0] = node.output[0]
        removed.append(node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    new_nodes = [n for n in g.node if n not in removed]
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, out_path)
    print(f"Folded Conv+Mul(scale): {changed} node(s)")
    return out_path


def fold_conv_add_bias(model_path: str, out_path: str) -> str:
    """Fold per-channel constant Add after Conv into Conv bias.

    Pattern: y = Add(Conv(x, W, b), c) where c is constant broadcastable to per-channel.
    Requires Conv output to be consumed only by this Add.
    """
    m = onnx.load(model_path)
    g = m.graph

    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    name_to_init: Dict[str, onnx.TensorProto] = {init.name: init for init in g.initializer}
    consumers = _build_consumers(g)
    out_to_node: Dict[str, onnx.NodeProto] = {n.output[0]: n for n in g.node if n.output}

    removed: List[onnx.NodeProto] = []
    changed = 0

    for node in g.node:
        if node.op_type != "Add" or len(node.input) != 2 or not node.output:
            continue
        a, b = node.input
        conv_out = None
        bias_name = None
        if a in out_to_node and out_to_node[a].op_type == "Conv" and b in init_map:
            conv_out, bias_name = a, b
        elif b in out_to_node and out_to_node[b].op_type == "Conv" and a in init_map:
            conv_out, bias_name = b, a
        else:
            continue

        conv = out_to_node[conv_out]
        if len(consumers.get(conv.output[0], [])) != 1:
            continue

        # Fetch conv params
        if len(conv.input) < 2:
            continue
        w_name = conv.input[1]
        if w_name not in init_map:
            continue
        W = init_map[w_name]
        oc = W.shape[0]

        add_vec = _extract_channel_vector(init_map[bias_name], oc)
        if add_vec is None:
            continue

        if len(conv.input) >= 3 and conv.input[2] in init_map:
            b_old = init_map[conv.input[2]].astype(np.float32)
            b_new = (b_old + add_vec).astype(np.float32)
            b_name = conv.input[2]
            if b_name in name_to_init:
                del name_to_init[b_name].raw_data
                name_to_init[b_name].CopyFrom(numpy_helper.from_array(b_new, name=b_name))
            else:
                for i, init in enumerate(list(g.initializer)):
                    if init.name == b_name:
                        g.initializer[i] = numpy_helper.from_array(b_new, name=b_name)
                        break
        else:
            b_name = conv.name + "_addfold_bias"
            exist_names = {init.name for init in g.initializer}
            if b_name in exist_names:
                idx = 0
                while f"{b_name}_{idx}" in exist_names:
                    idx += 1
                b_name = f"{b_name}_{idx}"
            g.initializer.extend([numpy_helper.from_array(add_vec.astype(np.float32), name=b_name)])
            if len(conv.input) >= 3:
                conv.input[2] = b_name
            else:
                conv.input.extend([b_name])

        # Rewire conv output to add output, drop Add
        conv.output[0] = node.output[0]
        removed.append(node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    new_nodes = [n for n in g.node if n not in removed]
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, out_path)
    print(f"Folded Conv+Add(bias): {changed} node(s)")
    return out_path


def fold_pad_into_conv(model_path: str, out_path: str) -> str:
    """Fold zero-constant Pad on NCHW into Conv pads attribute.

    Pattern: y = Conv(Pad(x, pads, mode='constant', value=0), W, b) -> Conv(x, W, b, pads+=...)
    Constraints:
    - Only H/W pads are nonzero (N/C pads must be 0)
    - Pad mode is 'constant' and value is 0 (or absent)
    - Conv's auto_pad must be not set
    - Conv must consume the Pad output as its X and Pad's output has single consumer
    """
    m = onnx.load(model_path)
    g = m.graph

    init_map: Dict[str, np.ndarray] = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    consumers = _build_consumers(g)
    out_to_node: Dict[str, onnx.NodeProto] = {n.output[0]: n for n in g.node if n.output}

    def get_attr_str(node: onnx.NodeProto, name: str, default: Optional[str] = None) -> Optional[str]:
        for a in node.attribute:
            if a.name == name and a.type == onnx.AttributeProto.STRING:
                try:
                    return a.s.decode() if isinstance(a.s, (bytes, bytearray)) else str(a.s)
                except Exception:
                    return default
        return default

    def has_attr(node: onnx.NodeProto, name: str) -> bool:
        return any(a.name == name for a in node.attribute)

    changed = 0
    removed: List[onnx.NodeProto] = []

    for node in g.node:
        if node.op_type != "Pad" or not node.output:
            continue
        y = node.output[0]
        # Single consumer and must be Conv
        if len(consumers.get(y, [])) != 1:
            continue
        conv = consumers[y][0]
        if conv.op_type != "Conv" or not conv.input:
            continue
        if conv.input[0] != y:
            continue

        # Get pads const from input[1] for Pad or attribute? (ONNX Pad v11+ uses input)
        if len(node.input) < 2 or node.input[1] not in init_map:
            continue
        pads = np.array(init_map[node.input[1]]).astype(np.int64).reshape(-1)
        # Optional constant value input
        pad_value = 0.0
        if len(node.input) >= 3 and node.input[2] in init_map:
            val = np.array(init_map[node.input[2]]).astype(np.float32)
            if val.size != 1:
                continue
            pad_value = float(val.reshape(-1)[0])
        mode = get_attr_str(node, "mode", "constant")
        if mode != "constant" or abs(pad_value) > 1e-8:
            continue

        # Expect rank 4: [N,C,H,W]; pads length 8
        if pads.size != 8:
            continue
        n_b, c_b, h_b, w_b, n_e, c_e, h_e, w_e = [int(v) for v in pads.tolist()]
        if any(v != 0 for v in (n_b, c_b, n_e, c_e)):
            continue
        if min(h_b, h_e, w_b, w_e) < 0:
            continue

        # Skip if Conv has auto_pad
        if has_attr(conv, "auto_pad"):
            continue

        # Read existing conv pads attribute (4 ints) if any
        curr_pads = [0, 0, 0, 0]
        pad_attr_idx = None
        for idx, a in enumerate(conv.attribute):
            if a.name == "pads" and a.type == onnx.AttributeProto.INTS:
                pa = list(a.ints)
                if len(pa) == 4:
                    curr_pads = [int(pa[0]), int(pa[1]), int(pa[2]), int(pa[3])]
                pad_attr_idx = idx
                break
        new_pads = [curr_pads[0] + h_b, curr_pads[1] + w_b, curr_pads[2] + h_e, curr_pads[3] + w_e]

        # Update conv attribute
        pads_attr = onnx.helper.make_attribute("pads", new_pads)
        if pad_attr_idx is not None:
            conv.attribute[pad_attr_idx].CopyFrom(pads_attr)
        else:
            conv.attribute.extend([pads_attr])

        # Bypass Pad: feed Conv with Pad's input
        conv.input[0] = node.input[0]
        removed.append(node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    new_nodes = [n for n in g.node if n not in removed]
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, out_path)
    print(f"Folded Pad into Conv: {changed} node(s)")
    return out_path


def simplify_transpose(model_path: str, out_path: str) -> str:
    """Remove identity Transpose and merge consecutive Transpose ops.

    - Transpose with perm=[0,1,2,3] (or rank identity) -> Identity
    - Transpose(Transpose(x, p1), p2) -> Transpose(x, compose(p2, p1)); if compose==identity, remove both
    Conservatively handles only when the first transpose output has a single consumer.
    """
    m = onnx.load(model_path)
    g = m.graph

    def get_perm(node: onnx.NodeProto) -> Optional[List[int]]:
        if node.op_type != "Transpose":
            return None
        for a in node.attribute:
            if a.name == "perm" and a.type == onnx.AttributeProto.INTS:
                return [int(v) for v in a.ints]
        return None

    def make_transpose(node: onnx.NodeProto, inp: str, out: str, perm: List[int]) -> onnx.NodeProto:
        n = helper.make_node("Transpose", [inp], [out], name=(node.name + "_m") if node.name else "TransposeMerged")
        n.attribute.extend([onnx.helper.make_attribute("perm", perm)])
        return n

    consumers = _build_consumers(g)

    kept: List[onnx.NodeProto] = []
    changed = 0

    # First pass: remove identity transpose
    for node in g.node:
        if node.op_type != "Transpose":
            kept.append(node)
            continue
        perm = get_perm(node)
        rank = None
        if perm is not None:
            rank = len(perm)
        if perm is not None and perm == list(range(rank)):
            # identity
            idn = helper.make_node("Identity", [node.input[0]], list(node.output), name=node.name + "_IdT")
            kept.append(idn)
            changed += 1
        else:
            kept.append(node)

    if changed:
        del g.node[:]
        g.node.extend(kept)
        kept = []

    # Recompute consumers after first pass
    consumers = _build_consumers(g)

    # Second pass: merge consecutive transpose
    for node in g.node:
        if node.op_type != "Transpose":
            kept.append(node)
            continue
        inp = node.input[0]
        prev = None
        for cand in g.node:
            if cand.output and cand.output[0] == inp and cand.op_type == "Transpose":
                prev = cand
                break
        if prev is None:
            kept.append(node)
            continue
        # Only safe to merge if prev output has single consumer
        if len(consumers.get(prev.output[0], [])) != 1:
            kept.append(node)
            continue
        p1 = get_perm(prev)
        p2 = get_perm(node)
        if p1 is None or p2 is None or len(p1) != len(p2):
            kept.append(node)
            continue
        # Compose p = p2 ∘ p1 (apply p1 then p2)
        composed = [p1[i] for i in p2]
        if composed == list(range(len(p1))):
            # identity -> bypass both
            idn = helper.make_node("Identity", [prev.input[0]], list(node.output), name=node.name + "_IdTT")
            # Do not keep prev nor node
            for n in (prev, node):
                if n in kept:
                    kept.remove(n)
            kept.append(idn)
            changed += 1
        else:
            # Merge into single transpose from prev input to node output
            merged = make_transpose(node, prev.input[0], node.output[0], composed)
            # Remove prev if present in kept
            for n in (prev, node):
                if n in kept:
                    kept.remove(n)
            kept.append(merged)
            changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Simplified Transpose: {changed} change(s)")
    return out_path

def shape_infer_model(model_path: str, out_path: str) -> str:
    try:
        inferred = onnx.shape_inference.infer_shapes_path(model_path)
        if isinstance(inferred, str) and os.path.exists(inferred):
            return inferred
    except Exception:
        pass
    m = onnx.load(model_path)
    m2 = onnx.shape_inference.infer_shapes(m)
    onnx.save(m2, out_path)
    return out_path


def fix_input_shapes(model_path: str, out_path: str, nchw: Tuple[int, int, int, int], extra_2d: int = 2) -> str:
    m = onnx.load(model_path)
    n, c, h, w = nchw
    for vi in m.graph.input:
        tt = vi.type.tensor_type
        rank = len(tt.shape.dim)
        if rank == 4 and ("image" in vi.name.lower() or "input" in vi.name.lower()):
            dims = [n, c, h, w]
            for i, d in enumerate(tt.shape.dim):
                d.dim_param = ""
                d.dim_value = int(dims[i])
        elif rank == 2 and ("scale" in vi.name.lower() or "shape" in vi.name.lower()):
            dims = [n, extra_2d]
            for i, d in enumerate(tt.shape.dim):
                d.dim_param = ""
                d.dim_value = int(dims[i])
        elif rank == 1:
            tt.shape.dim[0].dim_param = ""
            tt.shape.dim[0].dim_value = int(n)
    onnx.save(m, out_path)
    return out_path


def cast_graph_to_fp16(model_path: str, out_path: str) -> str:
    """Best-effort FP16 casting while preserving I/O dtypes."""
    try:
        import importlib
        mod = importlib.import_module('onnxmltools.utils.float16_converter')
        convert_float_to_float16 = getattr(mod, 'convert_float_to_float16')
    except Exception:
        shutil.copyfile(model_path, out_path)
        return out_path
    m = onnx.load(model_path)
    keep_io_types = {vi.name for vi in list(m.graph.input) + list(m.graph.output)}
    m_fp16 = convert_float_to_float16(m, keep_io_types=keep_io_types)
    onnx.save(m_fp16, out_path)
    return out_path


def split_large_concats(model_path: str, out_path: str, max_inputs: int = 8) -> str:
    """Split Concat nodes with too many inputs into a tree of smaller Concat nodes."""
    assert max_inputs >= 2
    m = onnx.load(model_path)
    g = m.graph
    total_splits = 0

    def unique_name(base: str) -> str:
        idx = 0
        existing = {n.name for n in g.node}
        existing.update({vi.name for vi in list(g.input) + list(g.output)})
        existing.update({init.name for init in g.initializer})
        name = f"{base}__{idx}"
        while name in existing:
            idx += 1
            name = f"{base}__{idx}"
        return name

    new_nodes: List[onnx.NodeProto] = []
    nodes_to_remove: List[onnx.NodeProto] = []

    for node in g.node:
        if node.op_type != "Concat":
            continue
        inputs = list(node.input)
        if len(inputs) <= max_inputs:
            continue
        axis = None
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
                break
        if axis is None:
            axis = 1

        current = inputs
        while len(current) > 1:
            next_level: List[str] = []
            for i in range(0, len(current), max_inputs):
                chunk = current[i : i + max_inputs]
                if len(chunk) == 1:
                    next_level.append(chunk[0])
                else:
                    out_name = unique_name(node.name + "_splitcat")
                    cnode = helper.make_node(
                        "Concat",
                        inputs=chunk,
                        outputs=[out_name],
                        name=unique_name(node.name + "_Concat"),
                        axis=axis,
                    )
                    new_nodes.append(cnode)
                    next_level.append(out_name)
            current = next_level

        final_out = current[0]
        if final_out != node.output[0]:
            id_node = helper.make_node(
                "Identity", inputs=[final_out], outputs=list(node.output), name=unique_name(node.name + "_Id")
            )
            new_nodes.append(id_node)
        nodes_to_remove.append(node)
        total_splits += 1

    if total_splits == 0:
        onnx.save(m, out_path)
        return out_path

    rebuilt: List[onnx.NodeProto] = []
    for node in g.node:
        if node in nodes_to_remove:
            continue
        rebuilt.append(node)
    rebuilt.extend(new_nodes)
    del g.node[:]
    g.node.extend(rebuilt)

    onnx.save(m, out_path)
    print(f"Split large Concat nodes: {total_splits} node(s) processed (max_inputs={max_inputs})")
    return out_path


def fold_static_shape_chains(model_path: str, out_path: str, iterations: int = 4) -> str:
    """Constant-fold common shape computation chains with configurable iterations."""
    m = onnx.load(model_path)
    g = m.graph

    value_shapes = {}
    def record_shape(vi):
        try:
            shp = []
            tt = vi.type.tensor_type
            for d in tt.shape.dim:
                if d.dim_value:
                    shp.append(int(d.dim_value))
                else:
                    shp.append(None)
            value_shapes[vi.name] = shp
        except Exception:
            pass
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        record_shape(vi)

    const_vals = {}
    for init in g.initializer:
        const_vals[init.name] = numpy_helper.to_array(init)

    def get_const(name: str):
        return const_vals.get(name)

    def set_const(target_name: str, arr: np.ndarray):
        for i, init in enumerate(list(g.initializer)):
            if init.name == target_name:
                del g.initializer[i]
                break
        g.initializer.extend([numpy_helper.from_array(arr, name=target_name)])
        const_vals[target_name] = arr

    nodes_to_remove: List[onnx.NodeProto] = []

    def try_fold(node: onnx.NodeProto) -> bool:
        op = node.op_type
        def get_attr(name, default=None):
            for a in node.attribute:
                if a.name == name:
                    if a.type == onnx.AttributeProto.INT:
                        return a.i
                    if a.type == onnx.AttributeProto.INTS:
                        return list(a.ints)
                    if a.type == onnx.AttributeProto.FLOAT:
                        return a.f
                    if a.type == onnx.AttributeProto.FLOATS:
                        return list(a.floats)
                    if a.type == onnx.AttributeProto.STRING:
                        return a.s
            return default

        if op == "Shape" and len(node.input) == 1:
            x = node.input[0]
            out = node.output[0]
            shp = value_shapes.get(x)
            if shp and all(d is not None for d in shp):
                arr = np.asarray(shp, dtype=np.int64)
                set_const(out, arr)
                nodes_to_remove.append(node)
                return True
            return False

        if op == "Gather" and len(node.input) >= 2:
            data = get_const(node.input[0])
            indices = get_const(node.input[1])
            if data is None or indices is None:
                return False
            axis = get_attr("axis", 0)
            try:
                out_arr = np.take(data, indices.astype(np.int64), axis=axis)
            except Exception:
                return False
            set_const(node.output[0], out_arr.astype(np.int64))
            nodes_to_remove.append(node)
            return True

        if op == "Unsqueeze" and len(node.input) == 1:
            x = get_const(node.input[0])
            if x is None:
                return False
            axes = get_attr("axes")
            if axes is None:
                return False
            arr = x
            for ax in sorted([int(a) for a in axes]):
                arr = np.expand_dims(arr, axis=ax)
            set_const(node.output[0], arr.astype(np.int64))
            nodes_to_remove.append(node)
            return True

        if op == "Concat" and len(node.input) >= 2:
            axis = get_attr("axis", 0)
            vals = [get_const(nm) for nm in node.input]
            if any(v is None for v in vals):
                return False
            try:
                out_arr = np.concatenate(vals, axis=axis)
            except Exception:
                return False
            set_const(node.output[0], out_arr.astype(vals[0].dtype))
            nodes_to_remove.append(node)
            return True

        if op == "Cast" and len(node.input) == 1:
            x = get_const(node.input[0])
            to = get_attr("to", None)
            if x is None or to is None:
                return False
            type_map = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.FLOAT16: np.float16,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.INT64: np.int64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.INT16: np.int16,
                onnx.TensorProto.INT8: np.int8,
                onnx.TensorProto.UINT8: np.uint8,
                onnx.TensorProto.BOOL: np.bool_,
            }
            dtype = type_map.get(int(to))
            if dtype is None:
                return False
            set_const(node.output[0], x.astype(dtype))
            nodes_to_remove.append(node)
            return True

        return False

    changed = True
    any_change = False
    for _ in range(iterations):
        if not changed:
            break
        changed = False
        for node in list(g.node):
            if node in nodes_to_remove:
                continue
            if try_fold(node):
                changed = True
                any_change = True

    if not any_change:
        onnx.save(m, out_path)
        return out_path

    kept = [n for n in g.node if n not in nodes_to_remove]
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Folded static shape chains: {len(nodes_to_remove)} node(s) replaced by constants ({iterations} iterations)")
    return out_path


def rewrite_div_by_const(model_path: str, out_path: str) -> str:
    """Replace Div(x, c) where c is constant with Mul(x, 1/c)."""
    m = onnx.load(model_path)
    g = m.graph
    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    new_inits = []
    new_nodes = []
    changed = 0

    def unique_name(base: str) -> str:
        idx = 0
        existing = {n.name for n in g.node}
        existing.update({vi.name for vi in list(g.input) + list(g.output)})
        existing.update({init.name for init in g.initializer})
        name = f"{base}__{idx}"
        while name in existing:
            idx += 1
            name = f"{base}__{idx}"
        return name

    kept: List[onnx.NodeProto] = []
    for node in g.node:
        if node.op_type != "Div" or len(node.input) != 2:
            kept.append(node)
            continue
        x, y = node.input
        y_val = consts.get(y)
        if y_val is None:
            kept.append(node)
            continue
        if np.any(y_val == 0):
            kept.append(node)
            continue
        recip = (1.0 / y_val.astype(np.float32)).astype(np.float32)
        recip_name = unique_name(node.name + "_recip")
        new_inits.append(numpy_helper.from_array(recip, name=recip_name))
        mul_node = helper.make_node("Mul", [x, recip_name], list(node.output), name=unique_name(node.name + "_Mul"))
        new_nodes.append(mul_node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    g.initializer.extend(new_inits)
    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote Div by constant: {changed} node(s)")
    return out_path


def rewrite_pow_patterns(model_path: str, out_path: str) -> str:
    """Rewrite Pow(x, c) where c is constant: 2->Mul(x,x), 0.5->Sqrt(x), -1->Reciprocal(x), 1->Identity."""
    m = onnx.load(model_path)
    g = m.graph
    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    const_node_vals = {}
    for node in g.node:
        if node.op_type != "Constant" or not node.output:
            continue
        out_name = node.output[0]
        arr = None
        for a in node.attribute:
            if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                try:
                    arr = numpy_helper.to_array(a.t)
                except Exception:
                    arr = None
                break
            if a.name == "value_float" and a.type == onnx.AttributeProto.FLOAT:
                arr = np.array(a.f, dtype=np.float32)
                break
            if a.name == "value_floats" and a.type == onnx.AttributeProto.FLOATS:
                arr = np.array(list(a.floats), dtype=np.float32)
                break
            if a.name == "value_int" and a.type == onnx.AttributeProto.INT:
                arr = np.array(a.i, dtype=np.int64)
                break
            if a.name == "value_ints" and a.type == onnx.AttributeProto.INTS:
                arr = np.array(list(a.ints), dtype=np.int64)
                break
        if arr is not None:
            const_node_vals[out_name] = arr

    def get_scalar(v):
        arr = consts.get(v)
        if arr is None:
            arr = const_node_vals.get(v)
        if arr is None:
            return None
        try:
            flat = np.array(arr).reshape(-1)
            if flat.size == 0:
                return None
            if flat.size > 1 and not np.allclose(flat, flat[0]):
                return None
            return float(flat[0])
        except Exception:
            return None

    def unique_name(base: str) -> str:
        idx = 0
        existing = {n.name for n in g.node}
        existing.update({vi.name for vi in list(g.input) + list(g.output)})
        existing.update({init.name for init in g.initializer})
        name = f"{base}__{idx}"
        while name in existing:
            idx += 1
            name = f"{base}__{idx}"
        return name

    kept: List[onnx.NodeProto] = []
    new_nodes: List[onnx.NodeProto] = []
    changed = 0
    for node in g.node:
        if node.op_type != "Pow" or len(node.input) != 2:
            kept.append(node)
            continue
        x, y = node.input
        c = get_scalar(y)
        if c is None:
            kept.append(node)
            continue
        out = list(node.output)
        if abs(c - 2.0) < 1e-6:
            new_nodes.append(helper.make_node("Mul", [x, x], out, name=unique_name(node.name + "_MulPow2")))
            changed += 1
            continue
        if abs(c - 3.0) < 1e-6:
            t1 = unique_name(node.name + "_sq")
            new_nodes.append(helper.make_node("Mul", [x, x], [t1], name=unique_name(node.name + "_MulSq")))
            new_nodes.append(helper.make_node("Mul", [t1, x], out, name=unique_name(node.name + "_MulPow3")))
            changed += 1
            continue
        if abs(c - 4.0) < 1e-6:
            t1 = unique_name(node.name + "_sq")
            t2 = unique_name(node.name + "_p4")
            new_nodes.append(helper.make_node("Mul", [x, x], [t1], name=unique_name(node.name + "_MulSq")))
            new_nodes.append(helper.make_node("Mul", [t1, t1], [t2], name=unique_name(node.name + "_MulSq2")))
            new_nodes.append(helper.make_node("Identity", [t2], out, name=unique_name(node.name + "_IdPow4")))
            changed += 1
            continue
        if abs(c - 0.5) < 1e-6:
            new_nodes.append(helper.make_node("Sqrt", [x], out, name=unique_name(node.name + "_Sqrt")))
            changed += 1
            continue
        if abs(c + 0.5) < 1e-6:
            s_out = unique_name(node.name + "_SqrtTmp")
            new_nodes.append(helper.make_node("Sqrt", [x], [s_out], name=unique_name(node.name + "_Sqrt")))
            new_nodes.append(helper.make_node("Reciprocal", [s_out], out, name=unique_name(node.name + "_RecipSqrt")))
            changed += 1
            continue
        if abs(c + 1.0) < 1e-6:
            new_nodes.append(helper.make_node("Reciprocal", [x], out, name=unique_name(node.name + "_Recip")))
            changed += 1
            continue
        if abs(c + 2.0) < 1e-6:
            t1 = unique_name(node.name + "_sq")
            new_nodes.append(helper.make_node("Mul", [x, x], [t1], name=unique_name(node.name + "_MulSq")))
            new_nodes.append(helper.make_node("Reciprocal", [t1], out, name=unique_name(node.name + "_RecipPow2")))
            changed += 1
            continue
        if abs(c + 3.0) < 1e-6:
            t1 = unique_name(node.name + "_sq")
            t2 = unique_name(node.name + "_p3")
            new_nodes.append(helper.make_node("Mul", [x, x], [t1], name=unique_name(node.name + "_MulSq")))
            new_nodes.append(helper.make_node("Mul", [t1, x], [t2], name=unique_name(node.name + "_MulPow3")))
            new_nodes.append(helper.make_node("Reciprocal", [t2], out, name=unique_name(node.name + "_RecipPow3")))
            changed += 1
            continue
        if abs(c - 1.0) < 1e-6:
            new_nodes.append(helper.make_node("Identity", [x], out, name=unique_name(node.name + "_Id")))
            changed += 1
            continue

        # Generic small integer exponents via exponentiation by squaring (abs(n) in [2..8])
        try:
            n = int(round(c))
        except Exception:
            n = None
        if n is not None and abs(c - n) < 1e-6 and abs(n) >= 2 and abs(n) <= 8:
            pos = abs(n)
            # Build x^pos
            nodes_chain: List[onnx.NodeProto] = []
            result_name = x
            base_name = x
            cur_pow = 1
            # Precompute powers of two using squaring
            pow_name = base_name
            bit = 1
            target = pos
            accum_name = None
            while (1 << (bit - 1)) <= target:
                if bit == 1:
                    pow_name = base_name  # x^(1)
                else:
                    # square previous pow_name: x^(2^(bit-1)) -> x^(2^bit)
                    next_pow = unique_name(node.name + f"_p2^{bit}")
                    nodes_chain.append(helper.make_node("Mul", [pow_name, pow_name], [next_pow], name=unique_name(node.name + f"_MulSq_{bit}")))
                    pow_name = next_pow
                # If this bit is set in target, multiply into accumulator
                if (target >> (bit - 1)) & 1:
                    if accum_name is None:
                        accum_name = pow_name
                    else:
                        new_accum = unique_name(node.name + f"_acc_{bit}")
                        nodes_chain.append(helper.make_node("Mul", [accum_name, pow_name], [new_accum], name=unique_name(node.name + f"_MulAcc_{bit}")))
                        accum_name = new_accum
                bit += 1
            if accum_name is None:
                # Should not happen for pos>=2, but guard
                kept.append(node)
                continue
            final_name = accum_name
            if n < 0:
                # Reciprocal for negative powers
                recip_out = unique_name(node.name + "_RecipPow")
                nodes_chain.append(helper.make_node("Reciprocal", [final_name], [recip_out], name=unique_name(node.name + "_RecipPow")))
                final_name = recip_out
            # Connect to original outputs
            nodes_chain.append(helper.make_node("Identity", [final_name], out, name=unique_name(node.name + "_PowExpand")))
            new_nodes.extend(nodes_chain)
            changed += 1
            continue
        kept.append(node)

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote Pow patterns: {changed} node(s)")
    return out_path



def rewrite_slice_to_gather(model_path: str, out_path: str) -> str:
    """Rewrite simple Slice with fixed single-axis indices into Gather for better CoreML support."""
    m = onnx.load(model_path)
    g = m.graph
    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    new_nodes: List[onnx.NodeProto] = []
    nodes_to_remove: List[onnx.NodeProto] = []
    changed = 0
    for node in g.node:
        if node.op_type != "Slice":
            continue
        if len(node.input) < 3:
            continue
        data, starts, ends = node.input[:3]
        axes = None
        steps = None
        if len(node.input) >= 4:
            axes = consts.get(node.input[3])
        if len(node.input) >= 5:
            steps = consts.get(node.input[4])
        s = consts.get(starts)
        e = consts.get(ends)
        if s is None or e is None:
            continue
        try:
            s = np.array(s).astype(np.int64).reshape(-1)
            e = np.array(e).astype(np.int64).reshape(-1)
            ax = np.array(axes).astype(np.int64).reshape(-1) if axes is not None else np.array([0], dtype=np.int64)
            st = np.array(steps).astype(np.int64).reshape(-1) if steps is not None else np.array([1], dtype=np.int64)
        except Exception:
            continue
        if not (len(s) == len(e) == len(ax) == len(st) == 1):
            continue
        if int(st[0]) != 1:
            continue
        if int(e[0]) - int(s[0]) != 1:
            continue
        idx = int(s[0])
        axis = int(ax[0])
        idx_name = node.name + "_gather_idx"
        g.initializer.extend([numpy_helper.from_array(np.array([idx], dtype=np.int64), name=idx_name)])
        gather = helper.make_node("Gather", [data, idx_name], list(node.output), name=node.name + "_toGather", axis=axis)
        new_nodes.append(gather)
        nodes_to_remove.append(node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    kept = [n for n in g.node if n not in nodes_to_remove]
    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote Slice -> Gather: {changed} node(s)")
    return out_path


# NEW OPTIMIZATIONS FOR ANE


def rewrite_slice_range_to_gather(model_path: str, out_path: str) -> str:
    """Rewrite Slice with static scalar range (start/end) and step=1 into a Gather with constant indices.

    Notes:
    - Handles only single-axis slices with scalar (or size-1) starts/ends/axis/step.
    - Skips negative or decreasing ranges and non-unit steps.
    - Prefer to run this BEFORE the single-element rewrite to catch ranges >= 2.
    """
    m = onnx.load(model_path)
    g = m.graph
    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    def unique_name(base: str) -> str:
        idx = 0
        existing = {n.name for n in g.node}
        existing.update({vi.name for vi in list(g.input) + list(g.output)})
        existing.update({init.name for init in g.initializer})
        name = f"{base}__{idx}"
        while name in existing:
            idx += 1
            name = f"{base}__{idx}"
        return name

    def as_scalar_int(arr):
        if arr is None:
            return None
        try:
            a = np.array(arr).astype(np.int64)
            if a.size != 1:
                return None
            return int(a.reshape(-1)[0])
        except Exception:
            return None

    new_nodes: List[onnx.NodeProto] = []
    nodes_to_remove: List[onnx.NodeProto] = []
    changed = 0

    for node in g.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            continue

        data, starts, ends = node.input[:3]
        axes = node.input[3] if len(node.input) >= 4 else None
        steps = node.input[4] if len(node.input) >= 5 else None

        s_val = consts.get(starts)
        e_val = consts.get(ends)
        ax_val = consts.get(axes) if axes else None
        st_val = consts.get(steps) if steps else None

        s = as_scalar_int(s_val)
        e = as_scalar_int(e_val)
        ax = as_scalar_int(ax_val) if ax_val is not None else 0
        st = as_scalar_int(st_val) if st_val is not None else 1

        # Require scalar constants and step==1
        if s is None or e is None:
            continue
        if st != 1:
            continue
        # Only handle forward, non-empty ranges with length >= 2
        if e is None or s is None or e <= s:
            continue
        if (e - s) < 2:
            # Let single-element handler manage this case
            continue
        # Avoid negative starts/ends (can't normalize without input shape)
        if s < 0 or e < 0:
            continue

        # Build indices initializer for Gather
        indices = np.arange(s, e, 1, dtype=np.int64)
        idx_name = unique_name(node.name + "_gather_indices")
        g.initializer.extend([numpy_helper.from_array(indices, name=idx_name)])

        gather_node = helper.make_node(
            "Gather",
            inputs=[data, idx_name],
            outputs=list(node.output),
            name=unique_name(node.name + "_to_GatherRange"),
            axis=int(ax) if ax is not None else 0,
        )
        new_nodes.append(gather_node)
        nodes_to_remove.append(node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    kept = [n for n in g.node if n not in nodes_to_remove]
    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote Slice (range, step=1) -> Gather: {changed} node(s)")
    return out_path


def rewrite_resize_to_static(model_path: str, out_path: str) -> str:
    """Rewrite Resize nodes to use static 'sizes' based on inferred output shapes.

    Rationale: CoreML EP with RequireStaticInputShapes prefers static shapes.
    If shape inference can determine the output shape of a Resize, we can
    replace dynamic scales/sizes with a constant 'sizes' initializer.

    Strategy:
    - Collect value_info shapes for all tensors.
    - For each Resize:
        - If its first (and only) output has fully-known NCHW dims (>0),
          inject a constant int64 sizes=[N,C,H,W] and rebuild inputs to use sizes.
        - Preserve attributes and ROI input if provided.
        - Leave scales empty (optional input) to ensure static behavior.
    - Skip nodes whose output shape is not fully static.
    """
    m = onnx.load(model_path)
    g = m.graph

    # Build a map of tensor name -> static shape list (ints) if known
    shape_map: Dict[str, List[int]] = {}

    def record_vi(vi):
        try:
            name = vi.name
            tt = vi.type.tensor_type
            dims = []
            for d in tt.shape.dim:
                if d.dim_value and int(d.dim_value) > 0:
                    dims.append(int(d.dim_value))
                else:
                    dims.append(None)
            shape_map[name] = dims
        except Exception:
            pass

    for vi in list(g.input) + list(g.value_info) + list(g.output):
        record_vi(vi)

    # Helper for unique names
    def unique_name(base: str) -> str:
        idx = 0
        existing = {n.name for n in g.node}
        existing.update({vi.name for vi in list(g.input) + list(g.output)})
        existing.update({init.name for init in g.initializer})
        name = f"{base}__{idx}"
        while name in existing:
            idx += 1
            name = f"{base}__{idx}"
        return name

    kept: List[onnx.NodeProto] = []
    new_inits = []
    changed = 0

    for node in g.node:
        if node.op_type != "Resize" or len(node.output) == 0:
            kept.append(node)
            continue
        out_name = node.output[0]
        out_shape = shape_map.get(out_name)
        # If output is not in value_info (rare), try to fall back to input shape with scales if constant
        if not out_shape:
            kept.append(node)
            continue
        # Need fully static 4D shape (NCHW)
        if len(out_shape) != 4 or any(d is None for d in out_shape):
            kept.append(node)
            continue

        # Create or reuse ROI input
        inputs = list(node.input)
        # Normalize to 4 inputs: [x, roi, scales, sizes]
        while len(inputs) < 4:
            inputs.append("")

        # Build constant sizes initializer
        sizes_arr = np.asarray(out_shape, dtype=np.int64)
        sizes_name = unique_name(node.name + "_static_sizes")
        new_inits.append(numpy_helper.from_array(sizes_arr, name=sizes_name))

        # Construct new inputs: keep X, keep ROI if provided, blank scales, provide sizes
        new_inputs = [inputs[0], inputs[1] if len(inputs) >= 2 else "", "", sizes_name]

        # Recreate the node to ensure clean optional inputs, preserving attributes
        new_node = helper.make_node(
            "Resize",
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name or unique_name("Resize"),
        )
        for a in node.attribute:
            new_node.attribute.extend([a])

        kept.append(new_node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    if new_inits:
        g.initializer.extend(new_inits)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote Resize -> static sizes: {changed} node(s)")
    return out_path


def rewrite_reduce_to_globalpool(model_path: str, out_path: str) -> str:
    """Rewrite ReduceMean/ReduceMax over spatial dims [2,3] with keepdims=1 to GlobalAveragePool/GlobalMaxPool.

    This typically improves NCNN portability because pooling ops map directly to Vulkan kernels.
    Safety: only for 4D inputs (N,C,H,W), axes exactly {2,3}, keepdims=1.
    """
    m = onnx.load(model_path)
    g = m.graph

    shape_map: Dict[str, List[int]] = {}

    def record_vi(vi: onnx.ValueInfoProto) -> None:
        try:
            name = vi.name
            tt = vi.type.tensor_type
            dims: List[int] = []
            for d in tt.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value else None)
            shape_map[name] = dims
        except Exception:
            pass

    for vi in list(g.input) + list(g.value_info) + list(g.output):
        record_vi(vi)

    def get_attr_ints(node: onnx.NodeProto, name: str) -> Optional[List[int]]:
        for a in node.attribute:
            if a.name == name and a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
        return None

    def get_attr_int(node: onnx.NodeProto, name: str, default: Optional[int] = None) -> Optional[int]:
        for a in node.attribute:
            if a.name == name and a.type == onnx.AttributeProto.INT:
                return int(a.i)
        return default

    kept: List[onnx.NodeProto] = []
    changed = 0

    for node in g.node:
        if node.op_type not in ("ReduceMean", "ReduceMax"):
            kept.append(node)
            continue
        if not node.input:
            kept.append(node)
            continue

        x = node.input[0]
        shp = shape_map.get(x)
        if not shp or len(shp) != 4:
            kept.append(node)
            continue

        axes = get_attr_ints(node, "axes")
        keepdims = get_attr_int(node, "keepdims", 1)
        if keepdims != 1:
            kept.append(node)
            continue
        if axes is None:
            kept.append(node)
            continue

        axes_c: List[int] = []
        for a in axes:
            aa = a if a >= 0 else (len(shp) + a)
            axes_c.append(int(aa))
        if sorted(axes_c) != [2, 3]:
            kept.append(node)
            continue

        op = "GlobalAveragePool" if node.op_type == "ReduceMean" else "GlobalMaxPool"
        new_node = helper.make_node(op, [x], list(node.output), name=node.name + "_toGlobalPool")
        kept.append(new_node)
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote Reduce -> GlobalPool: {changed} node(s)")
    return out_path


def remove_noop_slice(model_path: str, out_path: str) -> str:
    """Remove Slice that is effectively identity (full-range on specified axes with step=1).

    Conditions per axis: start==0, step==1, end>=dim_size or very large sentinel, and dim_size known.
    """
    m = onnx.load(model_path)
    g = m.graph

    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    shape_map: Dict[str, List[int]] = {}

    def record_vi(vi: onnx.ValueInfoProto) -> None:
        try:
            name = vi.name
            tt = vi.type.tensor_type
            dims: List[int] = []
            for d in tt.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value else None)
            shape_map[name] = dims
        except Exception:
            pass

    for vi in list(g.input) + list(g.value_info) + list(g.output):
        record_vi(vi)

    kept: List[onnx.NodeProto] = []
    changed = 0

    def read(name: str) -> Optional[np.ndarray]:
        arr = consts.get(name)
        if arr is None:
            return None
        return np.array(arr)

    for node in g.node:
        if node.op_type != "Slice" or len(node.input) < 3:
            kept.append(node)
            continue
        data, starts, ends = node.input[:3]
        axes = node.input[3] if len(node.input) >= 4 else None
        steps = node.input[4] if len(node.input) >= 5 else None

        s = read(starts)
        e = read(ends)
        ax = read(axes) if axes else None
        st = read(steps) if steps else None

        if s is None or e is None:
            kept.append(node)
            continue

        try:
            s = s.astype(np.int64).reshape(-1)
            e = e.astype(np.int64).reshape(-1)
            if ax is not None:
                ax = ax.astype(np.int64).reshape(-1)
            else:
                ax = np.arange(len(s), dtype=np.int64)
            if st is not None:
                st = st.astype(np.int64).reshape(-1)
            else:
                st = np.ones_like(s, dtype=np.int64)
        except Exception:
            kept.append(node)
            continue

        if not (len(s) == len(e) == len(ax) == len(st)):
            kept.append(node)
            continue

        in_shape = shape_map.get(data)
        if not in_shape:
            kept.append(node)
            continue

        noop = True
        rank = len(in_shape)
        for i in range(len(s)):
            axis = int(ax[i]) if int(ax[i]) >= 0 else (rank + int(ax[i]))
            dim = in_shape[axis]
            if dim is None:
                noop = False
                break
            start_i = int(s[i])
            end_i = int(e[i])
            step_i = int(st[i])
            if step_i != 1:
                noop = False
                break
            if end_i < 0:
                end_i = dim + end_i
            if end_i > dim:
                end_i = dim
            if not (start_i == 0 and end_i == dim):
                noop = False
                break

        if noop:
            id_node = helper.make_node("Identity", [data], list(node.output), name=node.name + "_IdNoopSlice")
            kept.append(id_node)
            changed += 1
        else:
            kept.append(node)

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Removed no-op Slice: {changed} node(s)")
    return out_path


def remove_identity_nodes(model_path: str, out_path: str) -> str:
    """Remove Identity nodes by rewiring consumers to their original inputs."""
    m = onnx.load(model_path)
    g = m.graph

    targets: List[onnx.NodeProto] = []
    mapping: Dict[str, str] = {}

    for node in g.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        src = node.input[0]
        dst = node.output[0]
        if src == dst:
            continue
        mapping[dst] = src
        targets.append(node)

    if not targets:
        onnx.save(m, out_path)
        return out_path

    def resolve(name: str) -> str:
        seen = set()
        cur = name
        while cur in mapping and cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        return cur

    for node in g.node:
        if node in targets:
            continue
        for idx, nm in enumerate(node.input):
            node.input[idx] = resolve(nm)

    for vi in list(g.value_info) + list(g.output):
        vi.name = resolve(vi.name)

    kept = [n for n in g.node if n not in targets]
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Removed Identity nodes: {len(targets)} node(s)")
    return out_path


def analyze_ncnn_compatibility(model_info: Dict[str, Any]) -> None:
    """Emit a small NCNN readiness report from the collected model info histogram."""
    ops_hist = model_info.get("ops_hist", []) or []
    if not ops_hist:
        return

    unsupported = {
        "NonMaxSuppression", "GridSample", "RoiAlign", "ScatterND", "ScatterElements",
        "Loop", "If", "Scan", "TopK", "Unique", "Range", "Where", "Multinomial",
        "OneHot", "NonZero", "CumSum",
    }
    needs_static = {
        "Resize", "Slice", "Gather", "Expand", "Pad", "Tile", "Unsqueeze",
        "Reshape", "ReduceMean", "ReduceMax", "ReduceSum",
    }

    flagged = [(op, cnt) for op, cnt in ops_hist if op in unsupported]
    cautions = [(op, cnt) for op, cnt in ops_hist if op in needs_static]

    print("\n=== NCNN Readiness Check ===")
    if flagged:
        print("Potential blockers (verify with pnnx, may need custom rewrites):")
        for op, cnt in flagged:
            print(f"  • {op}: {cnt} node(s)")
    else:
        print("No obvious NCNN-unsupported ops detected.")

    if cautions:
        print("Ops that prefer fully static shapes for Vulkan performance:")
        for op, cnt in cautions:
            print(f"  • {op}: {cnt} node(s)")
        print("  ↳ Consider enabling --rewrite-* and --remove-noop-slice helpers.")


def main():
    parser = argparse.ArgumentParser(
        description="Graph surgery for PP-YOLOE ONNX prior to NCNN/Vulkan export."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_embed.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,640,640",
        help="Static input shape as comma separated N,C,H,W",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="PP-YOLOE/build/models/surgery",
        help="Directory to store intermediate ONNX artifacts",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        help="Optional destination for the final NCNN-ready ONNX",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run onnxruntime benchmark before/after surgery",
    )
    parser.add_argument(
        "--img",
        type=str,
        help="Image path for benchmarking when --run-benchmark is enabled",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for benchmarking")
    parser.add_argument("--runs", type=int, default=50, help="Timed iterations for benchmarking")
    parser.add_argument(
        "--ort-provider",
        type=str,
        default="cpu",
        help="Execution provider for onnxruntime benchmarking (e.g. cpu, coreml, tensorrt)",
    )
    parser.add_argument(
        "--enable-ort-profile",
        action="store_true",
        help="Enable onnxruntime profiling during benchmarking",
    )
    parser.add_argument(
        "--ort-profile-dir",
        type=str,
        default="pipeline/PP-YOLOE/output",
        help="Directory for ORT profiling traces",
    )
    parser.add_argument("--split-concat", type=int, default=4, help="Split large Concat nodes before NCNN export")
    parser.add_argument("--fix-input-shapes", action="store_true", help="Rewrite graph inputs to static shapes")
    parser.add_argument("--fold-static-shapes", action="store_true", help="Fold shape computation chains")
    parser.add_argument(
        "--fold-iterations",
        type=int,
        default=15,
        help="Iterations for constant folding when --fold-static-shapes is set",
    )
    parser.add_argument(
        "--rewrite-resize-to-static",
        action="store_true",
        help="Rewrite dynamic Resize ops to static sizes",
    )
    parser.add_argument("--rewrite-div", action="store_true", help="Rewrite Div by constant into Mul")
    parser.add_argument("--rewrite-pow", action="store_true", help="Rewrite Pow patterns that NCNN lacks")
    parser.add_argument(
        "--fold-conv-bn",
        action="store_true",
        help="Fold Conv+BatchNormalization into Conv weights/bias",
    )
    parser.add_argument(
        "--fold-conv-mul",
        action="store_true",
        help="Fold Conv+Mul(per-channel scale) into Conv",
    )
    parser.add_argument(
        "--fold-conv-add",
        action="store_true",
        help="Fold Conv+Add(per-channel bias) into Conv",
    )
    parser.add_argument(
        "--fold-pad-conv",
        action="store_true",
        help="Fold zero Pad before Conv into Conv pads attribute",
    )
    parser.add_argument(
        "--simplify-transpose",
        action="store_true",
        help="Simplify Transpose chains (merge/remove identities)",
    )
    parser.add_argument(
        "--rewrite-slice-range-to-gather",
        action="store_true",
        help="Rewrite Slice(range) to Gather with explicit indices",
    )
    parser.add_argument(
        "--rewrite-slice-to-gather",
        action="store_true",
        help="Rewrite compatible Slice ops to Gather",
    )
    parser.add_argument(
        "--rewrite-reduce-to-globalpool",
        action="store_true",
        help="Rewrite ReduceMean/ReduceMax over H,W to GlobalAverage/MaxPool",
    )
    parser.add_argument(
        "--remove-noop-slice",
        action="store_true",
        help="Remove Slice ops that do not change tensor ranges",
    )
    parser.add_argument(
        "--remove-identity",
        action="store_true",
        help="Remove Identity nodes after rewrites",
    )
    parser.add_argument("--fp16", action="store_true", help="Attempt FP16 casting on supported ops")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    ishape = parse_shape(args.input_shape)

    img_path: Optional[str] = None
    enable_benchmark = bool(args.run_benchmark)
    enable_profiling = bool(args.enable_ort_profile)

    if enable_benchmark:
        if ort is None:
            print("[WARN] onnxruntime not available; skipping benchmarking.")
            enable_benchmark = False
        elif not args.img:
            print("[WARN] --run-benchmark requested but --img not provided; skipping benchmarking.")
            enable_benchmark = False
        else:
            candidate = args.img if os.path.isabs(args.img) else os.path.abspath(args.img)
            if os.path.exists(candidate):
                img_path = candidate
            else:
                print(f"[WARN] Benchmark image not found: {candidate}; skipping benchmarking.")
                enable_benchmark = False

    if enable_benchmark and enable_profiling:
        os.makedirs(args.ort_profile_dir, exist_ok=True)

    base_info = load_model_info(args.model)
    print("\n=== Source Model ===")
    print(f"Path: {os.path.abspath(args.model)}")
    print(f"Nodes: {base_info['node_count']}  Unique ops: {base_info['unique_ops']}")
    analyze_ncnn_compatibility(base_info)

    baseline_run = None
    if enable_benchmark:
        try:
            baseline_run = run_benchmark(
                args.model,
                ishape,
                args.ort_provider,
                args.warmup,
                args.runs,
                enable_profile=enable_profiling,
                profile_dir=args.ort_profile_dir if enable_profiling else None,
                img_path=img_path,
            )
        except Exception as exc:
            print(f"[WARN] Baseline benchmark failed: {exc}")
            baseline_run = None
            enable_benchmark = False

    if baseline_run:
        b = baseline_run["benchmark"]
        print("\n=== Baseline Benchmark ===")
        print(f"Provider: {args.ort_provider}")
        print(
            "Latency (ms) avg={:.2f} p50={:.2f} p90={:.2f} p95={:.2f}".format(
                b["latency_ms_avg"], b["latency_ms_p50"], b["latency_ms_p90"], b["latency_ms_p95"]
            )
        )

    base_name = os.path.splitext(os.path.basename(args.model))[0]

    def stage_path(tag: str) -> str:
        return os.path.join(args.outdir, f"{base_name}_{tag}.onnx")

    work_path = args.model

    if args.fix_input_shapes:
        if len(ishape) == 4:
            print("[stage] Fixing input shapes to static NCHW…")
            nchw = tuple(int(x) for x in ishape[:4])
            work_path = fix_input_shapes(work_path, stage_path("fixed"), nchw)
        else:
            print(f"[WARN] Cannot fix input shapes for non-4D input: {ishape}")

    print("[stage] Running initial shape inference…")
    work_path = shape_infer_model(work_path, stage_path("shape"))

    # Early fusions to reduce op count and improve kernel selection
    if args.fold_conv_bn:
        print("[stage] Folding Conv+BatchNorm…")
        work_path = fold_conv_batchnorm(work_path, stage_path("conv_bn_fold"))
        work_path = shape_infer_model(work_path, stage_path("shape_bn"))

    if args.split_concat and args.split_concat > 1:
        print(f"[stage] Splitting Concat nodes (max_inputs={args.split_concat})…")
        work_path = split_large_concats(
            work_path,
            stage_path("splitconcat"),
            max_inputs=int(args.split_concat),
        )
        work_path = shape_infer_model(work_path, stage_path("shape_split"))

    if args.fold_static_shapes:
        print(f"[stage] Folding static shape chains ({args.fold_iterations} iterations)…")
        work_path = fold_static_shape_chains(
            work_path,
            stage_path("foldshape"),
            iterations=int(args.fold_iterations),
        )

    if args.rewrite_div:
        print("[stage] Rewriting Div by constants…")
        work_path = rewrite_div_by_const(work_path, stage_path("div2mul"))

    if args.rewrite_pow:
        print("[stage] Rewriting Pow patterns…")
        work_path = rewrite_pow_patterns(work_path, stage_path("powrew"))

    if args.fold_conv_mul:
        print("[stage] Folding Conv+Mul(scale)…")
        work_path = fold_conv_mul_scale(work_path, stage_path("conv_mul_fold"))
        work_path = shape_infer_model(work_path, stage_path("shape_mul"))

    if args.fold_conv_add:
        print("[stage] Folding Conv+Add(bias)…")
        work_path = fold_conv_add_bias(work_path, stage_path("conv_add_fold"))
        work_path = shape_infer_model(work_path, stage_path("shape_add"))

    if args.fold_pad_conv:
        print("[stage] Folding Pad into Conv…")
        work_path = fold_pad_into_conv(work_path, stage_path("pad_conv_fold"))
        work_path = shape_infer_model(work_path, stage_path("shape_padconv"))

    if args.simplify_transpose:
        print("[stage] Simplifying Transpose chains…")
        work_path = simplify_transpose(work_path, stage_path("transpose_simplify"))
        work_path = shape_infer_model(work_path, stage_path("shape_transpose"))

    if args.rewrite_slice_range_to_gather:
        print("[stage] Rewriting range Slice → Gather…")
        work_path = rewrite_slice_range_to_gather(work_path, stage_path("slice2gather_range"))

    if args.rewrite_slice_to_gather:
        print("[stage] Rewriting Slice → Gather…")
        work_path = rewrite_slice_to_gather(work_path, stage_path("slice2gather"))

    if args.rewrite_resize_to_static:
        print("[stage] Rewriting Resize to static sizes…")
        work_path = rewrite_resize_to_static(work_path, stage_path("resize_static"))

    if args.remove_noop_slice:
        print("[stage] Removing no-op Slice ops…")
        work_path = remove_noop_slice(work_path, stage_path("slice_noop"))

    if args.rewrite_reduce_to_globalpool:
        print("[stage] Rewriting Reduce → GlobalPool…")
        work_path = rewrite_reduce_to_globalpool(work_path, stage_path("globalpool"))

    print("[stage] Running shape inference after rewrites…")
    work_path = shape_infer_model(work_path, stage_path("shape_post"))

    if args.remove_identity:
        print("[stage] Removing Identity nodes…")
        work_path = remove_identity_nodes(work_path, stage_path("noidentity"))
        work_path = shape_infer_model(work_path, stage_path("shape_clean"))

    if args.fp16:
        print("[stage] Attempting FP16 casting…")
        work_path = cast_graph_to_fp16(work_path, stage_path("fp16"))

    final_path = args.output_model or stage_path("ncnn_ready")
    final_dir = os.path.dirname(os.path.abspath(final_path))
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)

    if os.path.abspath(work_path) != os.path.abspath(final_path):
        shutil.copyfile(work_path, final_path)
    work_path = final_path

    final_info = load_model_info(work_path)
    print("\n=== Final Model ===")
    print(f"Path: {os.path.abspath(work_path)}")
    print(f"Nodes: {final_info['node_count']}  Unique ops: {final_info['unique_ops']}")
    analyze_ncnn_compatibility(final_info)

    optimized_run = None
    if enable_benchmark:
        try:
            optimized_run = run_benchmark(
                work_path,
                ishape,
                args.ort_provider,
                args.warmup,
                args.runs,
                enable_profile=enable_profiling,
                profile_dir=args.ort_profile_dir if enable_profiling else None,
                img_path=img_path,
            )
        except Exception as exc:
            print(f"[WARN] Benchmark on optimized model failed: {exc}")
            optimized_run = None

    if optimized_run:
        m = optimized_run["benchmark"]
        print("\n=== Optimized Benchmark ===")
        print(f"Provider: {args.ort_provider}")
        print(
            "Latency (ms) avg={:.2f} p50={:.2f} p90={:.2f} p95={:.2f}".format(
                m["latency_ms_avg"], m["latency_ms_p50"], m["latency_ms_p90"], m["latency_ms_p95"]
            )
        )

    if baseline_run and optimized_run:
        b = baseline_run["benchmark"]
        m = optimized_run["benchmark"]

        def pct_delta(a: float, b_val: float) -> float:
            return 100.0 * (b_val - a) / a if a and np.isfinite(a) else float("nan")

        print("\n=== Benchmark Delta (optimized vs baseline) ===")
        print(
            "Average latency: {0:.2f}ms → {1:.2f}ms ({2:+.2f}%)".format(
                b["latency_ms_avg"], m["latency_ms_avg"], pct_delta(b["latency_ms_avg"], m["latency_ms_avg"])
            )
        )
        print(
            "P50 latency: {0:.2f}ms → {1:.2f}ms ({2:+.2f}%)".format(
                b["latency_ms_p50"], m["latency_ms_p50"], pct_delta(b["latency_ms_p50"], m["latency_ms_p50"])
            )
        )
        print(
            "P90 latency: {0:.2f}ms → {1:.2f}ms ({2:+.2f}%)".format(
                b["latency_ms_p90"], m["latency_ms_p90"], pct_delta(b["latency_ms_p90"], m["latency_ms_p90"])
            )
        )
        print(
            "P95 latency: {0:.2f}ms → {1:.2f}ms ({2:+.2f}%)".format(
                b["latency_ms_p95"], m["latency_ms_p95"], pct_delta(b["latency_ms_p95"], m["latency_ms_p95"])
            )
        )


if __name__ == "__main__":
    main()
