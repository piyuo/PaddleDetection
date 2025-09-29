#!/usr/bin/env python3
"""
Experimental graph surgery for PP-YOLOE ONNX to explore ANE-friendly transforms, with baseline vs modified timing.

This script:
 1) Loads original model, benchmarks it.
 2) Applies optional transformations:
    - onnx-simplifier (fold constants, remove redundant ops)
    - onnx.shape_inference (infer shapes)
    - Cast clamp (e.g., f32->f16) if requested
 3) Saves modified model and benchmarks it.
 4) Prints side-by-side latency comparison.

Note: For actual ANE-targeted improvements, tailor transforms to CoreML EP supported ops.
"""
import argparse
import os
import shutil
from typing import Tuple, List
import sys
import subprocess

import onnx
import numpy as np
from onnx import helper, numpy_helper

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Reuse functions from profile_onnx
from profile_onnx import (
    load_model_info,
    parse_shape,
    run_benchmark,
)


def simplify_model(model_path: str, out_path: str) -> str:
    """Run onnx-simplifier in a subprocess to isolate potential segfaults; fallback to copy on failure."""
    # Check if onnxsim is importable; if not, fall back early
    try:
        import onnxsim  # noqa: F401
    except Exception:
        shutil.copyfile(model_path, out_path)
        return out_path

    cmd = [sys.executable, "-m", "onnxsim", model_path, out_path]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
    except Exception:
        # Fallback: copy original when simplification not applicable or crashed
        shutil.copyfile(model_path, out_path)
        return out_path


def shape_infer_model(model_path: str, out_path: str) -> str:
    try:
        inferred = onnx.shape_inference.infer_shapes_path(model_path)
        # Note: infer_shapes_path returns path in newer onnx; fallback to in-memory
        if isinstance(inferred, str) and os.path.exists(inferred):
            return inferred
    except Exception:
        pass
    # Try in-memory API
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
            # batch-like vectors
            tt.shape.dim[0].dim_param = ""
            tt.shape.dim[0].dim_value = int(n)
    onnx.save(m, out_path)
    return out_path


def run_onnxoptimizer(model_path: str, out_path: str) -> str:
    try:
        import onnxoptimizer
    except Exception:
        # Package not available; just copy
        import shutil as _sh
        _sh.copyfile(model_path, out_path)
        return out_path
    m = onnx.load(model_path)
    # List of reasonable passes for inference graphs
    available = set(getattr(onnxoptimizer, 'get_available_passes', lambda: [])())
    desired = [
        "eliminate_identity",
        "eliminate_deadend",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_nop_dropout",
        "eliminate_nop_cast",
        "eliminate_nop_reshape",
        "eliminate_nop_monotone_argmax",
        "fuse_consecutive_transposes",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_unsqueezes",
        "fuse_pad_into_conv",
        "eliminate_unused_initializer",
    ]
    passes = [p for p in desired if not available or p in available]
    try:
        m_opt = onnxoptimizer.optimize(m, passes)
        onnx.save(m_opt, out_path)
        return out_path
    except Exception:
        import shutil as _sh
        _sh.copyfile(model_path, out_path)
        return out_path


def cast_graph_to_fp16(model_path: str, out_path: str) -> str:
    """Best-effort FP16 casting while preserving I/O dtypes.
    Requires onnxmltools or float16 converter; implement a minimal fallback.
    """
    try:
        import importlib
        mod = importlib.import_module('onnxmltools.utils.float16_converter')
        convert_float_to_float16 = getattr(mod, 'convert_float_to_float16')
    except Exception:
        # Minimal fallback: just return original; caller may choose CoreML EP fp16 via provider settings.
        shutil.copyfile(model_path, out_path)
        return out_path
    m = onnx.load(model_path)
    # Keep inputs/outputs in original precision to avoid mismatch at runtime
    keep_io_types = {vi.name for vi in list(m.graph.input) + list(m.graph.output)}
    m_fp16 = convert_float_to_float16(m, keep_io_types=keep_io_types)
    onnx.save(m_fp16, out_path)
    return out_path


def rewrite_hardswish(model_path: str, out_path: str) -> str:
    """Replace HardSwish nodes with x * Clip(x + 3, 0, 6) * (1/6).

    This form uses only Add/Clip/Mul which are generally well-supported and ANE-friendly.
    Clip is created with min/max as inputs (opset >= 11).
    """
    m = onnx.load(model_path)
    g = m.graph
    changed = 0

    def unique_name(base: str) -> str:
        idx = 0
        existing = {n.name for n in g.node}
        existing.update({init.name for init in g.initializer})
        existing.update({vi.name for vi in list(g.input) + list(g.output)})
        name = f"{base}__{idx}"
        while name in existing:
            idx += 1
            name = f"{base}__{idx}"
        return name

    new_nodes: List[onnx.NodeProto] = []
    nodes_to_remove: List[onnx.NodeProto] = []

    for node in g.node:
        if node.op_type != "HardSwish":
            continue
        x = node.input[0]
        y = node.output[0]

        # Constants (0-D scalars for CoreML Clip min/max requirements)
        c3_name = unique_name("hardswish_c3")
        c3_tensor = numpy_helper.from_array(np.array(3.0, dtype=np.float32), name=c3_name)
        g.initializer.extend([c3_tensor])

        c0_name = unique_name("hardswish_c0")
        c0_tensor = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=c0_name)
        g.initializer.extend([c0_tensor])

        c6_name = unique_name("hardswish_c6")
        c6_tensor = numpy_helper.from_array(np.array(6.0, dtype=np.float32), name=c6_name)
        g.initializer.extend([c6_tensor])

        cscale_name = unique_name("hardswish_c1_div6")
        cscale_tensor = numpy_helper.from_array(np.array(1.0 / 6.0, dtype=np.float32), name=cscale_name)
        g.initializer.extend([cscale_tensor])

        # x + 3
        add_out = unique_name(node.name + "_add3")
        add_node = helper.make_node("Add", [x, c3_name], [add_out], name=unique_name(node.name + "_Add"))

        # Clip(x + 3, 0, 6) - use min/max inputs for opset >= 11
        clip_out = unique_name(node.name + "_clip")
        clip_node = helper.make_node("Clip", [add_out, c0_name, c6_name], [clip_out], name=unique_name(node.name + "_Clip"))

        # x * clip
        mul1_out = unique_name(node.name + "_mul1")
        mul1_node = helper.make_node("Mul", [x, clip_out], [mul1_out], name=unique_name(node.name + "_Mul1"))

        # * (1/6)
        mul2_node = helper.make_node("Mul", [mul1_out, cscale_name], [y], name=unique_name(node.name + "_Mul2"))

        new_nodes.extend([add_node, clip_node, mul1_node, mul2_node])
        nodes_to_remove.append(node)
        changed += 1

    if changed == 0:
        # Nothing to do
        onnx.save(m, out_path)
        return out_path

    # Rebuild node list: insert new nodes in place of removed HardSwish nodes preserving order
    rebuilt: List[onnx.NodeProto] = []
    for node in g.node:
        if node in nodes_to_remove:
            # append the corresponding sequence in the order they were created
            start = len(rebuilt)
            # We can't easily match per-node here without mapping, so just extend all new nodes at the end.
            # To keep relative order, extend new_nodes once at the end after loop.
            continue
        rebuilt.append(node)

    # Append all new nodes at the end to avoid complicated in-place mapping; shape infer will sort out types.
    rebuilt.extend(new_nodes)
    del g.node[:]  # clear
    g.node.extend(rebuilt)

    onnx.save(m, out_path)
    print(f"Rewrote HardSwish -> Add+Clip+Mul: {changed} node(s)")
    return out_path


def rewrite_swish_to_hardswish(model_path: str, out_path: str) -> str:
    """Detect Swish/SiLU patterns (x * Sigmoid(x)) and rewrite into Add+Clip+Mul style.

    Heuristic pattern:
      - Mul node where one input is Sigmoid of the other input (same tensor)
    Caveat:
      - This is an approximation; mathematically Swish != HardSwish, but ANE often prefers the latter form.
    """
    m = onnx.load(model_path)
    g = m.graph

    # Map tensor -> producer node
    producer = {}
    for node in g.node:
        for out in node.output:
            producer[out] = node

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
    changed = 0

    for node in g.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        a, b = node.input
        # Check if one side is Sigmoid of the other
        pa = producer.get(a)
        pb = producer.get(b)
        # pattern: Mul(x, Sigmoid(x)) or Mul(Sigmoid(x), x)
        def is_sigmoid_of_x(pnode, xname):
            return pnode is not None and pnode.op_type == "Sigmoid" and len(pnode.input) == 1 and pnode.input[0] == xname

        if is_sigmoid_of_x(pa, b):
            x = b
        elif is_sigmoid_of_x(pb, a):
            x = a
        else:
            continue

        y = node.output[0]

        # Inject Add+Clip+Mul*(1/6) sequence similar to HardSwish rewrite
        c3_name = unique_name("swish_c3")
        c0_name = unique_name("swish_c0")
        c6_name = unique_name("swish_c6")
        cscale_name = unique_name("swish_c1_div6")
        for nm, val in [
            (c3_name, 3.0),
            (c0_name, 0.0),
            (c6_name, 6.0),
            (cscale_name, 1.0/6.0),
        ]:
            # Create 0-D scalar initializers
            g.initializer.extend([numpy_helper.from_array(np.array(val, dtype=np.float32), name=nm)])

        add_out = unique_name(node.name + "_add3")
        add_node = helper.make_node("Add", [x, c3_name], [add_out], name=unique_name(node.name + "_Add"))

        clip_out = unique_name(node.name + "_clip")
        clip_node = helper.make_node("Clip", [add_out, c0_name, c6_name], [clip_out], name=unique_name(node.name + "_Clip"))

        mul1_out = unique_name(node.name + "_mul1")
        mul1_node = helper.make_node("Mul", [x, clip_out], [mul1_out], name=unique_name(node.name + "_Mul1"))

        mul2_node = helper.make_node("Mul", [mul1_out, cscale_name], [y], name=unique_name(node.name + "_Mul2"))

        new_nodes.extend([add_node, clip_node, mul1_node, mul2_node])
        nodes_to_remove.append(node)
        # If we consumed a Sigmoid predecessor and it's now dead, let optimizer prune it later
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    # Rebuild node list (remove replaced Mul nodes, append the new sequence)
    kept: List[onnx.NodeProto] = []
    for n in g.node:
        if n in nodes_to_remove:
            continue
        kept.append(n)
    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)

    onnx.save(m, out_path)
    print(f"Rewrote Swish/SiLU -> Add+Clip+Mul: {changed} node(s)")
    return out_path

def split_large_concats(model_path: str, out_path: str, max_inputs: int = 8) -> str:
    """Split Concat nodes with too many inputs into a tree of smaller Concat nodes.

    Keeps the input order. Useful when CoreML has limits on Concat input count.
    """
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
        level_outputs: List[str] = []
        # Build tree layers until a single tensor remains
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

        # current[0] is the final output tensor
        final_out = current[0]
        # Redirect downstream consumers by creating an Identity if names differ
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

    # Rebuild graph nodes list: retain original nodes except removed, then append new nodes
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


def fold_static_shape_chains(model_path: str, out_path: str) -> str:
    """Constant-fold common shape computation chains once input shapes are static.

    Handles a small subset of ops typically seen around shape building:
      - Shape(x) -> Constant (int64 dims)
      - Gather(const, axis=0) -> Constant
      - Unsqueeze(const, axes) -> Constant
      - Concat(consts, axis) -> Constant
      - Cast(const) -> Constant
    """
    m = onnx.load(model_path)
    g = m.graph

    # Collect static shapes per value (from value_info & inputs after shape inference)
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

    # Prepare constant map: name -> np.ndarray
    const_vals = {}
    for init in g.initializer:
        const_vals[init.name] = numpy_helper.to_array(init)

    def get_const(name: str):
        return const_vals.get(name)

    def set_const(target_name: str, arr: np.ndarray):
        # Add/replace initializer; keep the same tensor name
        # Remove any existing initializer with same name first
        for i, init in enumerate(list(g.initializer)):
            if init.name == target_name:
                del g.initializer[i]
                break
        g.initializer.extend([numpy_helper.from_array(arr, name=target_name)])
        const_vals[target_name] = arr

    nodes_to_remove: List[onnx.NodeProto] = []

    def try_fold(node: onnx.NodeProto) -> bool:
        op = node.op_type
        # Helper to fetch attribute
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

        # Shape: output dims of input tensor
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

        # Gather(const, indices) along axis (default 0)
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

        # Unsqueeze(const)
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

        # Concat of constants
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

        # Cast of constant
        if op == "Cast" and len(node.input) == 1:
            x = get_const(node.input[0])
            to = get_attr("to", None)
            if x is None or to is None:
                return False
            # Map ONNX tensor type to numpy dtype (limited to common ones)
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
    # Iterate a few times to fold chains
    for _ in range(4):
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

    # Remove folded nodes
    kept = [n for n in g.node if n not in nodes_to_remove]
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Folded static shape chains: {len(nodes_to_remove)} node(s) replaced by constants")
    return out_path


def prune_outputs(model_path: str, out_path: str, keep_outputs: List[str]) -> str:
    """Keep only a subset of outputs and prune unreachable nodes."""
    if not keep_outputs:
        shutil.copyfile(model_path, out_path)
        return out_path
    m = onnx.load(model_path)
    input_names = [vi.name for vi in m.graph.input]
    try:
        onnx.utils.extract_model(model_path, out_path, input_names, keep_outputs)
        return out_path
    except Exception:
        # Fallback: just copy if extract_model not available
        shutil.copyfile(model_path, out_path)
        return out_path

def rewrite_div_by_const(model_path: str, out_path: str) -> str:
    """Replace Div(x, c) where c is constant (scalar or broadcastable) with Mul(x, 1/c)."""
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
        # Avoid divide by zero
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

    def get_scalar(v):
        arr = consts.get(v)
        if arr is None:
            return None
        try:
            return float(arr.reshape(-1)[0])
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
        if abs(c - 0.5) < 1e-6:
            new_nodes.append(helper.make_node("Sqrt", [x], out, name=unique_name(node.name + "_Sqrt")))
            changed += 1
            continue
        if abs(c + 1.0) < 1e-6:
            new_nodes.append(helper.make_node("Reciprocal", [x], out, name=unique_name(node.name + "_Recip")))
            changed += 1
            continue
        if abs(c - 1.0) < 1e-6:
            new_nodes.append(helper.make_node("Identity", [x], out, name=unique_name(node.name + "_Id")))
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


def rewrite_hardsigmoid_linear(model_path: str, out_path: str) -> str:
    """Rewrite HardSigmoid to a mul+add+clip linear form using min/max inputs for Clip.

    HardSigmoid(x) = max(0, min(1, alpha * x + beta))
    We'll materialize alpha and beta as initializers and build Mul/Add/Clip.
    """
    m = onnx.load(model_path)
    g = m.graph
    new_nodes: List[onnx.NodeProto] = []
    nodes_to_remove: List[onnx.NodeProto] = []
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

    for node in g.node:
        if node.op_type != "HardSigmoid":
            continue
        x = node.input[0]
        y = node.output[0]
        alpha = 0.2
        beta = 0.5
        for a in node.attribute:
            if a.name == "alpha":
                alpha = float(a.f)
            if a.name == "beta":
                beta = float(a.f)

        a_name = unique_name("hsig_alpha")
        b_name = unique_name("hsig_beta")
        z_name = unique_name("zero")
        o_name = unique_name("one")
        for nm, val in [
            (a_name, alpha), (b_name, beta), (z_name, 0.0), (o_name, 1.0)
        ]:
            # Use 0-D scalars for CoreML Clip inputs
            g.initializer.extend([numpy_helper.from_array(np.array(val, dtype=np.float32), name=nm)])

        mul_out = unique_name(node.name + "_mul")
        mul_node = helper.make_node("Mul", [x, a_name], [mul_out], name=unique_name(node.name + "_Mul"))
        add_out = unique_name(node.name + "_add")
        add_node = helper.make_node("Add", [mul_out, b_name], [add_out], name=unique_name(node.name + "_Add"))
        clip_node = helper.make_node("Clip", [add_out, z_name, o_name], [y], name=unique_name(node.name + "_Clip"))
        new_nodes.extend([mul_node, add_node, clip_node])
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
    print(f"Rewrote HardSigmoid -> Mul+Add+Clip: {changed} node(s)")
    return out_path


def rewrite_slice_to_gather(model_path: str, out_path: str) -> str:
    """Rewrite simple Slice with fixed single-axis indices into Gather for better CoreML support.

    Pattern: Slice(data, starts, ends, axes=[k], steps=[1]) with starts/ends scalar constants
    Transforms into Gather along axis=k for a single index when the slice selects exactly one index.
    """
    m = onnx.load(model_path)
    g = m.graph
    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    def get_const_scalar(name):
        arr = consts.get(name)
        if arr is None:
            return None
        try:
            v = int(np.array(arr).reshape(-1)[0])
            return v
        except Exception:
            return None

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
        # Select exactly one index
        if int(e[0]) - int(s[0]) != 1:
            continue
        idx = int(s[0])
        axis = int(ax[0])
        # Build an initializer for the Gather index
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


def main():
    parser = argparse.ArgumentParser(description="Graph surgery and performance compare for PP-YOLOE ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument("--input-shape", type=str, default="1,3,640,640")
    parser.add_argument("--ep", type=str, default="coreml")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="pipeline/PP-YOLOE/models/surgery")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-shape-infer", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Attempt FP16 casting (if tools available)")
    parser.add_argument("--fix-input-shapes", action="store_true", help="Rewrite graph inputs to static [N,C,H,W] and common 2D helpers")
    parser.add_argument("--no-optimizer", action="store_true", help="Disable onnxoptimizer passes")
    parser.add_argument("--rewrite-hardswish", action="store_true", help="Rewrite HardSwish to Add+Clip+Mul for ANE")
    parser.add_argument(
        "--split-concat",
        type=int,
        default=0,
        help="If >0, split Concat nodes with more than N inputs into a concat tree",
    )
    parser.add_argument("--fold-static-shapes", action="store_true", help="Fold Shape/Gather/Unsqueeze/Concat chains to constants")
    parser.add_argument(
        "--keep-outputs",
        type=str,
        default="",
        help="Comma-separated list of outputs to keep (prune others)",
    )
    parser.add_argument("--rewrite-div", action="store_true", help="Rewrite Div by constant to Mul with reciprocal")
    parser.add_argument("--rewrite-pow", action="store_true", help="Rewrite Pow(x,c) with simpler ops (2, 0.5, -1, 1)")
    parser.add_argument("--rewrite-swish", action="store_true", help="Rewrite Swish/SiLU pattern x*Sigmoid(x) to HardSwish-style Add+Clip+Mul (ANE-friendly)")
    parser.add_argument("--rewrite-hardsigmoid", action="store_true", help="Rewrite HardSigmoid to Mul+Add+Clip with min/max inputs")
    parser.add_argument("--rewrite-slice-to-gather", action="store_true", help="Rewrite simple Slice with single index to Gather (fixed axes, step=1)")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ishape = parse_shape(args.input_shape)

    print("=== Baseline ===")
    if ort is None:
        print("onnxruntime not available; install 'onnxruntime' or 'onnxruntime-silicon'.")
        return
    print(f"[Debug] run config: ep={args.ep}, warmup={args.warmup}, runs={args.runs}")
    base = run_benchmark(args.model, ishape, args.ep, args.warmup, args.runs)
    b = base["benchmark"]
    print("Providers (baseline):", b.get("providers"))
    if base.get("coreml_capability"):
        cap = base["coreml_capability"]
        print(f"CoreML capability (baseline): partitions={cap.get('num_partitions')} nodes supported={cap.get('num_nodes')}")
    print(
        "Baseline Latency (ms) avg={:.2f} p50={:.2f} p90={:.2f} p95={:.2f}".format(
            b["latency_ms_avg"], b["latency_ms_p50"], b["latency_ms_p90"], b["latency_ms_p95"]
        )
    )

    # Build modified path
    mod_path = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_mod.onnx'))
    work_path = args.model

    if args.fix_input_shapes:
        print("[stage] Fix input shapes to static…")
        mod_fix = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fixed.onnx'))
        work_path = fix_input_shapes(work_path, mod_fix, ishape if len(ishape) == 4 else (1, 3, 640, 640))

    if not args.no_shape_infer:
        print("[stage] Shape inference…")
        mod2 = mod_path.replace("_mod.onnx", "_shape.onnx")
        work_path = shape_infer_model(work_path, mod2)
    else:
        work_path = mod_path

    if args.rewrite_hardswish:
        print("[stage] Rewriting HardSwish nodes…")
        modH = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_hardswish.onnx'))
        work_path = rewrite_hardswish(work_path, modH)

    if args.rewrite_swish:
        print("[stage] Rewriting Swish/SiLU (x*Sigmoid(x)) to HardSwish-style…")
        modHS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_swish2hs.onnx'))
        work_path = rewrite_swish_to_hardswish(work_path, modHS)

    if args.split_concat and args.split_concat > 0:
        print(f"[stage] Splitting large Concat nodes (max_inputs={args.split_concat})…")
        modC = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_splitconcat.onnx'))
        work_path = split_large_concats(work_path, modC, max_inputs=int(args.split_concat))

    # Re-run shape inference after structural rewrites so later passes have shape info
    if not args.no_shape_infer and (args.rewrite_hardswish or args.rewrite_swish or (args.split_concat and args.split_concat > 0)):
        print("[stage] Running shape inference (post-rewrite)…")
        mod2b = mod_path.replace("_mod.onnx", "_shape2.onnx")
        work_path = shape_infer_model(work_path, mod2b)

    if args.fold_static_shapes:
        print("[stage] Folding static shape chains…")
        modF = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_foldshape.onnx'))
        work_path = fold_static_shape_chains(work_path, modF)

    if args.rewrite_hardsigmoid:
        print("[stage] Rewriting HardSigmoid to Mul+Add+Clip…")
        modHSig = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_hardsig.onnx'))
        work_path = rewrite_hardsigmoid_linear(work_path, modHSig)

    if args.rewrite_div:
        print("[stage] Rewriting Div by constant…")
        modD = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_div2mul.onnx'))
        work_path = rewrite_div_by_const(work_path, modD)

    if args.rewrite_pow:
        print("[stage] Rewriting Pow patterns…")
        modW = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_powrew.onnx'))
        work_path = rewrite_pow_patterns(work_path, modW)

    if args.rewrite_slice_to_gather:
        print("[stage] Rewriting simple Slice -> Gather…")
        modSG = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice2gather.onnx'))
        work_path = rewrite_slice_to_gather(work_path, modSG)

    # Optional pruning of outputs to maximize CoreML partition sizes
    if args.keep_outputs:
        keep = [s.strip() for s in args.keep_outputs.split(",") if s.strip()]
        if keep:
            print("[stage] Pruning outputs; keeping:", keep)
            modP = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_pruned.onnx'))
            work_path = prune_outputs(work_path, modP, keep)

    # One more shape inference pass after folding/pruning
    if not args.no_shape_infer:
        print("[stage] Running shape inference (final)…")
        mod2c = mod_path.replace("_mod.onnx", "_shape3.onnx")
        work_path = shape_infer_model(work_path, mod2c)

    if not args.no_optimizer:
        print("[stage] Applying onnxoptimizer passes…")
        modO = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_opt.onnx'))
        work_path = run_onnxoptimizer(work_path, modO)

    if not args.no_simplify:
        print("[stage] Applying onnx-simplifier…")
        modS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_simp.onnx'))
        work_path = simplify_model(work_path, modS)
    else:
        shutil.copyfile(work_path, mod_path)

    if args.fp16:
        print("[stage] Attempting FP16 casting…")
        mod3 = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fp16.onnx'))
        work_path = cast_graph_to_fp16(work_path, mod3)

    # Also copy to a stable "final" name for convenience
    final_path = os.path.join(
        args.outdir, os.path.basename(args.model).replace('.onnx', '_final.onnx')
    )
    try:
        shutil.copyfile(work_path, final_path)
    except Exception:
        final_path = work_path
    print("Modified model:", work_path)
    print("Final model:", final_path)
    mod_info = load_model_info(work_path)
    print("Nodes:", mod_info["node_count"], "Unique ops:", mod_info["unique_ops"])

    print("=== Modified Benchmark ===")
    mod = run_benchmark(work_path, ishape, args.ep, args.warmup, args.runs)
    m = mod["benchmark"]
    print("Providers (modified):", m.get("providers"))
    if mod.get("coreml_capability"):
        cap = mod["coreml_capability"]
        print(f"CoreML capability (modified): partitions={cap.get('num_partitions')} nodes supported={cap.get('num_nodes')}")
    print(
        "Modified Latency (ms) avg={:.2f} p50={:.2f} p90={:.2f} p95={:.2f}".format(
            m["latency_ms_avg"], m["latency_ms_p50"], m["latency_ms_p90"], m["latency_ms_p95"]
        )
    )

    # Simple compare
    def pct_delta(a, b):
        return 100.0 * (b - a) / a if a and np.isfinite(a) else float('nan')

    print("\n=== Compare (Modified vs Baseline) ===")
    print("avg delta: {:.2f}%".format(pct_delta(b["latency_ms_avg"], m["latency_ms_avg"])) )
    print("p50 delta: {:.2f}%".format(pct_delta(b["latency_ms_p50"], m["latency_ms_p50"])) )
    print("p90 delta: {:.2f}%".format(pct_delta(b["latency_ms_p90"], m["latency_ms_p90"])) )
    print("p95 delta: {:.2f}%".format(pct_delta(b["latency_ms_p95"], m["latency_ms_p95"])) )


if __name__ == "__main__":
    main()
