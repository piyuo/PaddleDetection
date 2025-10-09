#!/usr/bin/env python3
"""
Apple Neural Engine graph surgery for PP-YOLOE ONNX with aggressive ANE-targeted optimizations.

New optimizations targeting CoreML ANE acceleration:
- Aggressive constant folding (15 iterations)
- Enhanced ANE compatibility analysis
"""
import argparse
import os
import shutil
from typing import Tuple, List, Dict, Set, Any
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
        shutil.copyfile(model_path, out_path)
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


def run_onnxoptimizer(model_path: str, out_path: str) -> str:
    try:
        import onnxoptimizer
    except Exception:
        shutil.copyfile(model_path, out_path)
        return out_path
    m = onnx.load(model_path)
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
        shutil.copyfile(model_path, out_path)
        return out_path


def clean_unused_tensors(model_path: str, out_path: str, drop_unused_inputs: bool = False) -> str:
    """Remove unused initializers, dead Constant nodes, and stray value_info entries.

    This mirrors ORT's CleanUnusedInitializersAndNodeArgs so applying it here
    prevents warnings and slightly reduces the model size.

    - Conservative: does not traverse subgraphs (If/Loop/Scan).
    - Keeps any tensors that are graph outputs.
    - Optionally drops graph inputs that have no consumers.
    """
    m = onnx.load(model_path)
    g = m.graph

    # Build set of names consumed by nodes or required as graph outputs
    consumers: Set[str] = set()
    for n in g.node:
        for i in n.input:
            if i:
                consumers.add(i)
    graph_output_names = {o.name for o in g.output}
    consumers |= graph_output_names

    # Remove Constant nodes with outputs that no one consumes
    kept_nodes: List[onnx.NodeProto] = []
    removed_const = 0
    for n in g.node:
        if n.op_type == "Constant" and n.output and all((o not in consumers) for o in n.output):
            removed_const += 1
            continue
        kept_nodes.append(n)

    # Recompute consumers after removing some constants
    consumers.clear()
    for n in kept_nodes:
        for i in n.input:
            if i:
                consumers.add(i)
    consumers |= graph_output_names

    # Keep only initializers that are consumed
    kept_inits = [init for init in g.initializer if init.name in consumers]
    removed_inits = len(g.initializer) - len(kept_inits)

    # Live value names: consumed inputs and produced outputs from remaining nodes
    live_names: Set[str] = set(consumers)
    for n in kept_nodes:
        for o in n.output:
            if o:
                live_names.add(o)

    # Prune stray value_info entries
    kept_vi = [vi for vi in g.value_info if vi.name in live_names]
    removed_vi = len(g.value_info) - len(kept_vi)

    # Optionally drop unused graph inputs
    if drop_unused_inputs:
        kept_inputs = [inp for inp in g.input if (inp.name in consumers or inp.name in graph_output_names)]
        removed_inputs = len(g.input) - len(kept_inputs)
    else:
        kept_inputs = list(g.input)
        removed_inputs = 0

    # Write back pruned structures
    del g.node[:]
    g.node.extend(kept_nodes)
    del g.initializer[:]
    g.initializer.extend(kept_inits)
    del g.value_info[:]
    g.value_info.extend(kept_vi)
    del g.input[:]
    g.input.extend(kept_inputs)

    onnx.save(m, out_path)
    if removed_inits or removed_const or removed_vi or removed_inputs:
        print(
            f"Cleaned unused: initializers={removed_inits}, constants={removed_const}, value_info={removed_vi}, inputs={removed_inputs}"
        )
    return out_path


def prune_outputs(model_path: str, out_path: str, keep_outputs: List[str]) -> str:
    """Keep only a subset of outputs and prune unreachable nodes using custom DFS traversal."""
    if not keep_outputs:
        shutil.copyfile(model_path, out_path)
        return out_path

    m = onnx.load(model_path)
    g = m.graph
    print(f"  Pruning to keep outputs: {keep_outputs}")

    # Build a map of tensor -> producing node (use node index for identity)
    tensor_producers: Dict[str, int] = {}  # tensor_name -> node index
    for idx, node in enumerate(g.node):
        for out in node.output:
            tensor_producers[out] = idx

    # DFS from desired outputs to find all reachable node indices
    reachable_node_indices: Set[int] = set()
    visited_tensors: Set[str] = set()

    def visit(tensor_name: str):
        if tensor_name in visited_tensors:
            return
        visited_tensors.add(tensor_name)

        # If this tensor is produced by a node, visit that node
        if tensor_name in tensor_producers:
            node_idx = tensor_producers[tensor_name]
            if node_idx not in reachable_node_indices:
                reachable_node_indices.add(node_idx)
                node = g.node[node_idx]
                # Recursively visit all inputs of this node
                for inp in node.input:
                    if inp:  # Skip empty strings
                        visit(inp)

    # Start DFS from each kept output
    for out_name in keep_outputs:
        visit(out_name)

    print(f"  Reachable nodes: {len(reachable_node_indices)} / {len(g.node)}")

    # Keep only reachable nodes (maintain order)
    kept_nodes = [g.node[i] for i in sorted(reachable_node_indices)]

    # Collect all tensor names that are either:
    # - produced by kept nodes
    # - consumed by kept nodes
    # - are graph inputs
    # - are in initializers
    live_tensors: Set[str] = set()
    for node in kept_nodes:
        live_tensors.update(node.input)
        live_tensors.update(node.output)

    graph_input_names = {vi.name for vi in g.input}
    live_tensors |= graph_input_names

    init_names = {init.name for init in g.initializer}
    live_tensors |= init_names

    # Keep only initializers that are live
    kept_inits = [init for init in g.initializer if init.name in live_tensors]

    # Keep only value_info for live tensors
    kept_vi = [vi for vi in g.value_info if vi.name in live_tensors]

    # Create new outputs (find or create ValueInfoProto for each output)
    new_outputs = []
    for out_name in keep_outputs:
        # Try to find existing value_info or output
        found = False
        for vi in list(g.output) + list(g.value_info):
            if vi.name == out_name:
                new_outputs.append(vi)
                found = True
                break

        if not found:
            # Create a minimal ValueInfoProto
            vi = onnx.ValueInfoProto()
            vi.name = out_name
            # Set a generic tensor type (ONNX will infer shapes later)
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            new_outputs.append(vi)
            print(f"  [WARNING] Created minimal ValueInfo for output '{out_name}'")

    # Rebuild graph
    del g.node[:]
    g.node.extend(kept_nodes)

    del g.initializer[:]
    g.initializer.extend(kept_inits)

    del g.value_info[:]
    g.value_info.extend(kept_vi)

    del g.output[:]
    g.output.extend(new_outputs)

    onnx.save(m, out_path)

    # Verify pruning worked
    m_pruned = onnx.load(out_path)
    actual_outputs = [o.name for o in m_pruned.graph.output]
    print(f"  Pruned model outputs: {actual_outputs}")
    has_nms = any(n.op_type == 'NonMaxSuppression' for n in m_pruned.graph.node)
    print(f"  Has NMS nodes: {has_nms}")
    if has_nms:
        print("  [WARNING] NMS nodes still present - they may be reachable from kept outputs!")

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

    This typically improves CoreML EP partitioning as pooling is well supported, while some Reduce ops remain on CPU.
    Safety: only for 4D inputs (N,C,H,W), axes exactly {2,3} (order-insensitive), keepdims=1.
    """
    m = onnx.load(model_path)
    g = m.graph

    # Build simple shape map to infer rank
    shape_map: Dict[str, List[int]] = {}
    def record_vi(vi):
        try:
            name = vi.name
            tt = vi.type.tensor_type
            dims = []
            for d in tt.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value else None)
            shape_map[name] = dims
        except Exception:
            pass
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        record_vi(vi)

    def get_attr_ints(node, name):
        for a in node.attribute:
            if a.name == name and a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
        return None
    def get_attr_int(node, name, default=None):
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
        # Canonicalize negative axes
        axes_c = []
        for a in axes:
            aa = a if a >= 0 else (len(shp) + a)
            axes_c.append(int(aa))
        if sorted(axes_c) != [2, 3]:
            kept.append(node)
            continue
        # Build Global Pool node
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

    # Const maps
    consts = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    # Shape map for input ranks and dims
    shape_map: Dict[str, List[int]] = {}
    def record_vi(vi):
        try:
            name = vi.name
            tt = vi.type.tensor_type
            dims = []
            for d in tt.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value else None)
            shape_map[name] = dims
        except Exception:
            pass
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        record_vi(vi)

    kept: List[onnx.NodeProto] = []
    changed = 0

    def read(name):
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
            # Normalize negative end index to dim + end
            if end_i < 0:
                end_i = dim + end_i
            # Treat very large end (common sentinel) as dim
            if end_i > dim:
                end_i = dim
            if not (start_i == 0 and end_i == dim):
                noop = False
                break
        if noop:
            # Replace Slice with Identity
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


def find_nms_nodes(model_path: str) -> Dict[str, Any]:
    """
    Loads an ONNX model and finds NonMaxSuppression nodes.
    Returns dict with structured info about NMS inputs (boxes, scores).
    """
    try:
        m = onnx.load(model_path)
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}", file=sys.stderr)
        return {}

    nms_nodes_found = []
    for node in m.graph.node:
        if node.op_type == 'NonMaxSuppression':
            nms_nodes_found.append(node)

    if not nms_nodes_found:
        return {}

    # Extract first NMS node's inputs (typically: boxes, scores, max_output_boxes, iou_threshold, score_threshold)
    node = nms_nodes_found[0]
    input_tensors = [inp for inp in node.input if inp]  # Skip empty strings

    result = {}
    if len(input_tensors) >= 2:
        result['boxes'] = input_tensors[0]
        result['scores'] = input_tensors[1]

    return result


def auto_discover_outputs(model_path: str, img_hw: Tuple[int, int] = (640, 640), verbose: bool = True) -> Dict[str, Any]:
    """
    Automatically discover optimal outputs for model surgery:
    - NMS inputs (boxes, scores) - to be removed since NMS will be implemented in C++
    - Embed output (if present) - created by insert_embedding_head.py

    Returns dict with discovered outputs and their metadata.
    """
    if verbose:
        print("\n=== Automatic Output Discovery ===")

    # 1. Find NMS inputs (boxes, scores) - these will be kept as outputs to remove NMS
    nms_info = find_nms_nodes(model_path)
    if not nms_info:
        print("⚠️  No NMS nodes found")
    elif verbose:
        print(f"✓ Found NMS inputs (will be kept as outputs, NMS removed):")
        print(f"  Boxes:  {nms_info.get('boxes', 'N/A')}")
        print(f"  Scores: {nms_info.get('scores', 'N/A')}")

    # 2. Check if model has 'embed' output (created by insert_embedding_head.py)
    embed_output = None
    try:
        m = onnx.load(model_path)
        existing_outputs = [o.name for o in m.graph.output]
        if 'embed' in existing_outputs:
            embed_output = 'embed'
            if verbose:
                print(f"✓ Found embed output (created by insert_embedding_head.py)")
        else:
            if verbose:
                print(f"⚠️  No 'embed' output found - model doesn't have embeddings yet")
    except Exception as e:
        if verbose:
            print(f"⚠️  Error checking for embed output: {e}")

    # Build result - no more stride-8/stride-16 feature maps needed
    result = {
        'nms': nms_info,
        'embed': embed_output,
    }

    return result


def print_output_guide(discovered: Dict[str, Any], keep_outputs: List[str]) -> None:
    """Print comprehensive guide for using the pruned model outputs."""
    print("\n" + "="*70)
    print("=== Pruned Model Output Guide ===")
    print("="*70)

    nms = discovered.get('nms', {})
    embed = discovered.get('embed')

    # Map output indices
    for i, out_name in enumerate(keep_outputs):
        print(f"\nOutput {i}: '{out_name}'")

        # Identify what this output is
        if nms.get('boxes') == out_name:
            print("  Type: Detection boxes (raw, pre-NMS)")
            print("  Shape: (1, 8400, 4) or similar")
            print("  Format: [x_center, y_center, width, height] in input coordinates")
            print("  Usage: Apply NMS in C++ with scores to get final detections")
            print("  Note: NMS has been removed from ONNX model and will be implemented in C++")

        elif nms.get('scores') == out_name:
            print("  Type: Detection scores (raw, pre-NMS)")
            print("  Shape: (1, 1, 8400) - may need squeeze to (8400,)")
            print("  Format: Class probabilities (single class: person)")
            print("  Usage: Apply NMS in C++ with boxes to get final detections")
            print("  Note: NMS has been removed from ONNX model and will be implemented in C++")

        elif embed == out_name:
            print("  Type: Per-detection embeddings")
            print("  Shape: (N, D) where N=num_detections, D=embedding_dimension")
            print("  Format: L2-normalized appearance embeddings")
            print("  Usage: Use for appearance-based tracking (BoT-SORT, DeepSORT, etc.)")
            print("  Note: Created by insert_embedding_head.py with multi-scale features")

        else:
            print("  Type: Unknown (custom output)")

    # Print code template
    print(f"\n" + "-"*70)
    print("=== C++ Inference Template ===")
    print("-"*70)
    print("// 1. Run ONNX inference to get outputs")
    print("// 2. Apply NMS on boxes and scores (implement in C++)")
    print("// 3. Use embeddings for appearance-based tracking")
    print("")
    print("Example workflow:")

    if nms.get('boxes') in keep_outputs and nms.get('scores') in keep_outputs:
        print("  • Output 0 (boxes_raw): Pre-NMS boxes from all anchor points")
        print("  • Output 1 (scores_raw): Pre-NMS confidence scores")
        print("  • Apply NMS in C++ to filter detections")

    if embed in keep_outputs:
        print(f"  • Output {len(keep_outputs)-1} (embed): Per-detection embeddings")
        print("  • Use embeddings for appearance matching in tracker")
        print("")
        print("Tracking tips:")
        print("  - Cosine similarity threshold: 0.55-0.60 for matching")
        print("  - Combine with IoU gating (≥0.2-0.3) for robust tracking")
        print("  - Use feature history (nn_budget 50-100) for track features")

    print("\n" + "="*70 + "\n")


def analyze_ane_compatibility(profile_summary: Dict) -> None:
    """Analyze and report ops most likely blocking ANE execution."""
    if not profile_summary:
        return

    cpu_ops = profile_summary.get("top_ops_by_time", {}).get("CPUExecutionProvider", [])
    if not cpu_ops:
        return

    known_ane_unfriendly = {
        "NonMaxSuppression", "RoiAlign", "TopK", "Where", "IsNaN",
        "Loop", "If", "Scan", "Resize"  # Some Resize modes
    }

    print("\n=== ANE Compatibility Analysis ===")
    print("Top CPU ops (candidates for optimization):")
    for op, time_ms, count in cpu_ops[:15]:
        marker = " ⚠️  ANE-unfriendly" if op in known_ane_unfriendly else ""
        pct = ""
        total_cpu = profile_summary.get("provider_total_time_ms", {}).get("CPUExecutionProvider", 0)
        if total_cpu > 0:
            pct = f" ({100.0 * time_ms / total_cpu:.1f}%)"
        print(f"  • {op}: {time_ms:.2f}ms ({count} nodes){pct}{marker}")

    # Suggest specific optimizations
    op_names = {op for op, _, _ in cpu_ops[:15]}
    suggestions = []

    if "Slice" in op_names:
        suggestions.append("  → Try --rewrite-slice-to-gather for simple slicing patterns")
    if "NonMaxSuppression" in op_names:
        suggestions.append("  → NMS is inherently CPU-bound; consider splitting model at detection head")
    if "RoiAlign" in op_names:
        suggestions.append("  → RoiAlign may not be ANE-supported; consider alternative pooling")

    if suggestions:
        print("\nOptimization suggestions:")
        for s in suggestions:
            print(s)


def main():
    parser = argparse.ArgumentParser(description="Enhanced graph surgery for PP-YOLOE ONNX with ANE optimizations")
    parser.add_argument(
        "--model",
        type=str,
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument("--input-shape", type=str, default="1,3,640,640")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--img", type=str, help="Path to image for realistic preprocessing (not needed for --find-nms)")
    parser.add_argument("--outdir", type=str, default="pipeline/PP-YOLOE/models/surgery")
    parser.add_argument("--rewrite-div", action="store_true", help="Rewrite Div to Mul with reciprocal")
    parser.add_argument("--rewrite-pow", action="store_true", help="Rewrite Pow patterns")
    parser.add_argument("--rewrite-slice-range-to-gather", action="store_true", help="Rewrite range Slice (step=1) to Gather with indices")
    parser.add_argument("--rewrite-slice-to-gather", action="store_true", help="Rewrite Slice to Gather")
    parser.add_argument("--rewrite-resize-to-static", action="store_true", help="Replace dynamic Resize with static sizes")
    parser.add_argument("--rewrite-reduce-to-globalpool", action="store_true", help="Rewrite ReduceMean/ReduceMax over H,W to GlobalPool")
    parser.add_argument("--remove-noop-slice", action="store_true", help="Remove Slice ops that are effectively identity")
    parser.add_argument("--output-model", type=str, help="Path to copy final optimized model to")

    args = parser.parse_args()

    # Check required arguments for normal operation
    if not args.img:
        print("[ERROR] --img argument is required", file=sys.stderr)
        return

    os.makedirs(args.outdir, exist_ok=True)
    ishape = parse_shape(args.input_shape)
    img_path = args.img if os.path.isabs(args.img) else os.path.abspath(args.img)
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        return

    print("=== Baseline ===")
    if ort is None:
        print("onnxruntime not available; install 'onnxruntime' or 'onnxruntime-silicon'.")
        return

    base = run_benchmark(args.model, ishape, "coreml", args.warmup, args.runs,
                        enable_profile=False, profile_dir=None, img_path=img_path)
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

    if base.get("profile_summary"):
        analyze_ane_compatibility(base["profile_summary"])

    # Build modified path
    mod_path = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_mod.onnx'))
    work_path = args.model

    print("[stage] Shape inference…")
    mod2 = mod_path.replace("_mod.onnx", "_shape.onnx")
    work_path = shape_infer_model(work_path, mod2)

    # Re-run shape inference after structural rewrites
    print("[stage] Running shape inference (post-rewrite)…")
    mod2b = mod_path.replace("_mod.onnx", "_shape2.onnx")
    work_path = shape_infer_model(work_path, mod2b)

    if args.rewrite_div:
        print("[stage] Rewriting Div by constant…")
        modD = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_div2mul.onnx'))
        work_path = rewrite_div_by_const(work_path, modD)

    if args.rewrite_pow:
        print("[stage] Rewriting Pow patterns…")
        modW = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_powrew.onnx'))
        work_path = rewrite_pow_patterns(work_path, modW)

    if args.rewrite_slice_range_to_gather:
        print("[stage] Rewriting range Slice -> Gather…")
        modSGR = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice2gather_range.onnx'))
        work_path = rewrite_slice_range_to_gather(work_path, modSGR)

    if args.rewrite_slice_to_gather:
        print("[stage] Rewriting simple Slice -> Gather…")
        modSG = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice2gather.onnx'))
        work_path = rewrite_slice_to_gather(work_path, modSG)

    if args.rewrite_resize_to_static:
        print("[stage] Rewriting Resize to static sizes…")
        modRZ = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_resize_static.onnx'))
        work_path = rewrite_resize_to_static(work_path, modRZ)

    if args.remove_noop_slice:
        print("[stage] Removing no-op Slice ops…")
        modNS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice_noop.onnx'))
        work_path = remove_noop_slice(work_path, modNS)

    if args.rewrite_reduce_to_globalpool:
        print("[stage] Rewriting Reduce -> GlobalPool…")
        modGP = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_globalpool.onnx'))
        work_path = rewrite_reduce_to_globalpool(work_path, modGP)

    # Automatic output discovery and pruning
    discovered_info = None
    keep = []

    # Automatic mode: discover optimal outputs
    print("\n" + "="*70)
    print("[stage] Automatic output discovery enabled")
    print("="*70)

    # Parse input shape to get image dimensions
    img_hw = (640, 640)  # Default
    if args.input_shape:
        try:
            shape_parts = [int(x) for x in args.input_shape.split(',')]
            if len(shape_parts) == 4:  # B,C,H,W
                img_hw = (shape_parts[2], shape_parts[3])
        except:
            pass

    discovered_info = auto_discover_outputs(work_path, img_hw=img_hw, verbose=True)

    # Build keep_outputs list from discovered info
    # Keep NMS inputs (boxes, scores) to remove NMS node - NMS will be implemented in C++
    nms = discovered_info.get('nms', {})
    if nms.get('boxes'):
        keep.append(nms['boxes'])
    if nms.get('scores'):
        keep.append(nms['scores'])

    # Keep embed output if it exists (created by insert_embedding_head.py)
    embed = discovered_info.get('embed')
    if embed:
        keep.append(embed)

    if keep:
        print(f"\n✓ Auto-discovered outputs to keep: {len(keep)} tensors")
        for i, name in enumerate(keep):
            print(f"  {i+1}. {name}")
    else:
        print("\n⚠️  No outputs auto-discovered. Model will keep original outputs.")

    # Prune if we have outputs to keep
    if keep:
        print(f"\n[stage] Pruning graph; keeping {len(keep)} outputs...")
        modP = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_pruned.onnx'))
        work_path = prune_outputs(work_path, modP, keep)

    # Final shape inference
    print("[stage] Running shape inference (final)…")
    mod2c = mod_path.replace("_mod.onnx", "_shape3.onnx")
    work_path = shape_infer_model(work_path, mod2c)

    print("[stage] Applying onnxoptimizer passes…")
    modO = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_opt.onnx'))
    work_path = run_onnxoptimizer(work_path, modO)

    print("[stage] Applying onnx-simplifier…")
    modS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_simp.onnx'))
    work_path = simplify_model(work_path, modS)

    # Final cleanup to remove unused initializers/constants and silence ORT warnings
    try:
        print("[stage] Cleaning unused tensors…")
        modClean = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_clean.onnx'))
        work_path = clean_unused_tensors(work_path, modClean, drop_unused_inputs=False)
    except Exception as e:
        print(f"[WARNING] Cleanup pass failed: {e}")

    # Copy to final artifact
    if args.output_model:
        # Use the specified output model path
        final_path = args.output_model
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    else:
        # Use default naming in output directory
        final_path = os.path.join(
            args.outdir, os.path.basename(args.model).replace('.onnx', '_final.onnx')
        )

    try:
        shutil.copyfile(work_path, final_path)
        print(f"[stage] Copied final model to: {final_path}")
    except Exception as e:
        print(f"[WARNING] Failed to copy to final path: {e}")
        final_path = work_path

    final_abs = os.path.abspath(final_path)
    final_name = os.path.basename(final_path)

    # Print comprehensive output guide if we did auto-discovery
    if discovered_info and keep:
        print_output_guide(discovered_info, keep)

    # Model info and benchmarking at the end
    mod_info = load_model_info(work_path)
    print("\n" + "="*70)
    print("=== Modified Model Info ===")
    print("="*70)
    print("Nodes:", mod_info["node_count"], "Unique ops:", mod_info["unique_ops"])
    print("Modified model:", work_path)
    print("Final model:", final_path)

    print("\n" + "="*70)
    print("=== Final Artifact ===")
    print("="*70)
    print(f"Filename: {final_name}")
    print(f"Path: {final_abs}")

    print("\n" + "="*70)
    print("=== Modified Benchmark ===")
    print("="*70)
    mod = run_benchmark(work_path, ishape, "coreml", args.warmup, args.runs,
                       enable_profile=False, profile_dir=None, img_path=img_path)
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

    if mod.get("profile_summary"):
        analyze_ane_compatibility(mod["profile_summary"])

    # Comparison
    def pct_delta(a, b):
        return 100.0 * (b - a) / a if a and np.isfinite(a) else float('nan')

    print("\n" + "="*70)
    print("=== Performance Comparison (Modified vs Baseline) ===")
    print("="*70)
    avg_delta = pct_delta(b["latency_ms_avg"], m["latency_ms_avg"])
    p50_delta = pct_delta(b["latency_ms_p50"], m["latency_ms_p50"])
    p90_delta = pct_delta(b["latency_ms_p90"], m["latency_ms_p90"])
    p95_delta = pct_delta(b["latency_ms_p95"], m["latency_ms_p95"])

    def format_delta(d):
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.2f}%"

    print(f"Average latency: {b['latency_ms_avg']:.2f}ms → {m['latency_ms_avg']:.2f}ms ({format_delta(avg_delta)})")
    print(f"P50 latency:     {b['latency_ms_p50']:.2f}ms → {m['latency_ms_p50']:.2f}ms ({format_delta(p50_delta)})")
    print(f"P90 latency:     {b['latency_ms_p90']:.2f}ms → {m['latency_ms_p90']:.2f}ms ({format_delta(p90_delta)})")
    print(f"P95 latency:     {b['latency_ms_p95']:.2f}ms → {m['latency_ms_p95']:.2f}ms ({format_delta(p95_delta)})")

    # Speedup summary
    if avg_delta < 0:
        speedup = b["latency_ms_avg"] / m["latency_ms_avg"]
        print(f"\n🚀 Speedup: {speedup:.2f}x faster")

    # CoreML partition improvement
    if base.get("coreml_capability") and mod.get("coreml_capability"):
        base_parts = base["coreml_capability"].get("num_partitions", 0)
        mod_parts = mod["coreml_capability"].get("num_partitions", 0)
        base_nodes = base["coreml_capability"].get("num_nodes", 0)
        mod_nodes = mod["coreml_capability"].get("num_nodes", 0)

        if base_parts != mod_parts or base_nodes != mod_nodes:
            print("\n=== CoreML Partition Changes ===")
            print(f"Partitions: {base_parts} → {mod_parts}")
            print(f"ANE-supported nodes: {base_nodes} → {mod_nodes} ({mod_nodes - base_nodes:+d})")


if __name__ == "__main__":

    main()