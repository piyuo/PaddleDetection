#!/usr/bin/env python3
"""
Apple Neural Engine Graph Surgery for PP-YOLOE ONNX

This script applies CoreML ANE-specific optimizations to ONNX models:
- Static shape fixing for ANE compatibility
- Graph rewrites for ANE-friendly operations (Div→Mul, Pow patterns, Slice→Gather, etc.)
- Constant folding and shape chain simplification
- ANE compatibility analysis and profiling

This script is focused ONLY on ANE optimizations. It expects a customized model
as input (with NMS removed added via onnx_customize.py).

Workflow:
1. export_to_onnx.sh → ppyoloe_crn_s_36e_pphuman.onnx (base model)
2. onnx_customize.py → ppyoloe_crn_s_36e_pphuman_cust.onnx (customized model)
3. ane_graph_surgery.py → ppyoloe_crn_s_36e_pphuman_cust_ane.onnx (ANE-optimized model)
4. onnx_cleanup.py → ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx (final cleaned model)
"""
import argparse
import os
import shutil
from typing import Tuple, List, Dict
import sys

import onnx
import numpy as np
from onnx import helper, numpy_helper

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Add parent directory to path to import onnx_profile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse functions from profile_onnx
from onnx_profile import (
    load_model_info,
    parse_shape,
    run_benchmark,
)
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
        # Check for "x" specifically for ReID models, or generic "image"/"input"
        if rank == 4 and ("image" in vi.name.lower() or "input" in vi.name.lower() or vi.name == "x"):
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


def rewrite_hardsigmoid_linear(model_path: str, out_path: str) -> str:
    """Rewrite HardSigmoid to a mul+add+clip linear form using min/max inputs for Clip."""
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


def fold_conv_batchnorm(model_path: str, out_path: str) -> str:
    """Fold BatchNormalization parameters into preceding Conv weights/bias when safe."""
    m = onnx.load(model_path)
    g = m.graph

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

    init_map = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    def set_initializer(name: str, arr: np.ndarray):
        replaced = False
        for i, init in enumerate(g.initializer):
            if init.name == name:
                g.initializer[i].CopyFrom(numpy_helper.from_array(arr, name=name))
                replaced = True
                break
        if not replaced:
            g.initializer.extend([numpy_helper.from_array(arr, name=name)])
        init_map[name] = arr

    output_to_node: Dict[str, onnx.NodeProto] = {}
    for node in g.node:
        for out in node.output:
            output_to_node[out] = node

    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in g.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    nodes_to_remove: List[onnx.NodeProto] = []
    fused = 0

    for node in g.node:
        if node.op_type != "BatchNormalization" or len(node.input) < 5 or len(node.output) != 1:
            continue
        bn_in = node.input[0]
        prod = output_to_node.get(bn_in)
        if prod is None or prod.op_type != "Conv":
            continue
        if len(consumers.get(bn_in, [])) != 1:
            continue
        conv = prod
        if len(conv.input) < 2:
            continue

        weight_name = conv.input[1]
        weight = init_map.get(weight_name)
        if weight is None:
            continue

        scale = init_map.get(node.input[1])
        bias = init_map.get(node.input[2])
        mean = init_map.get(node.input[3])
        var = init_map.get(node.input[4])
        if any(v is None for v in (scale, bias, mean, var)):
            continue

        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = float(attr.f)
                break

        conv_bias_name = None
        conv_bias = None
        if len(conv.input) >= 3 and conv.input[2]:
            conv_bias_name = conv.input[2]
            conv_bias = init_map.get(conv_bias_name)
        if conv_bias is None:
            conv_bias_name = conv_bias_name or unique_name(conv.name + "_bias")
            conv_bias = np.zeros(weight.shape[0], dtype=weight.dtype)
            if len(conv.input) >= 3:
                conv.input[2] = conv_bias_name
            else:
                conv.input.append(conv_bias_name)
        dtype = weight.dtype
        scale = scale.astype(np.float32)
        bias = bias.astype(np.float32)
        mean = mean.astype(np.float32)
        var = var.astype(np.float32)
        conv_bias = conv_bias.astype(np.float32)

        inv = scale / np.sqrt(var + epsilon)
        inv_shape = inv.reshape(-1, *([1] * (weight.ndim - 1)))
        new_weight = (weight.astype(np.float32) * inv_shape).astype(dtype)
        new_bias = (inv * (conv_bias - mean) + bias).astype(np.float32)

        set_initializer(weight_name, new_weight)
        set_initializer(conv_bias_name, new_bias.astype(dtype))

        conv.output[0] = node.output[0]
        nodes_to_remove.append(node)
        fused += 1

    if fused == 0:
        onnx.save(m, out_path)
        return out_path

    kept = [n for n in g.node if n not in nodes_to_remove]
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Folded Conv+BatchNorm: {fused} pair(s)")
    return out_path


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


def remove_unused_initializers(model_path: str, out_path: str) -> str:
    """Drop initializers not consumed by any node or graph IO."""
    m = onnx.load(model_path)
    g = m.graph

    used = set()
    for node in g.node:
        for inp in node.input:
            if inp:
                used.add(inp)
    for vi in list(g.input) + list(g.output):
        if vi.name:
            used.add(vi.name)

    kept = []
    removed = 0
    for init in g.initializer:
        if init.name in used:
            kept.append(init)
        else:
            removed += 1

    if removed == 0:
        onnx.save(m, out_path)
        return out_path

    del g.initializer[:]
    g.initializer.extend(kept)
    onnx.save(m, out_path)
    print(f"Removed unused initializers: {removed} tensor(s)")
    return out_path


def remove_unused_constants(model_path: str, out_path: str) -> str:
    """Remove Constant nodes whose outputs are unused."""
    m = onnx.load(model_path)
    g = m.graph

    consumers = set()
    for node in g.node:
        for inp in node.input:
            if inp:
                consumers.add(inp)
    graph_outputs = {out.name for out in g.output}

    kept: List[onnx.NodeProto] = []
    removed = 0
    for node in g.node:
        if node.op_type == "Constant" and node.output:
            unused = True
            for out_name in node.output:
                if not out_name:
                    continue
                if out_name in consumers or out_name in graph_outputs:
                    unused = False
                    break
            if unused:
                removed += 1
                continue
        kept.append(node)

    if removed == 0:
        onnx.save(m, out_path)
        return out_path

    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Removed unused Constant nodes: {removed}")
    return out_path


def remove_identity_nodes(model_path: str, out_path: str) -> str:
    """Remove Identity nodes whose outputs are not graph outputs."""
    m = onnx.load(model_path)
    g = m.graph
    graph_outputs = {out.name for out in g.output}

    replacements: Dict[str, str] = {}
    nodes_to_remove: List[onnx.NodeProto] = []
    for node in g.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        out_name = node.output[0]
        if out_name in graph_outputs:
            continue
        replacements[out_name] = node.input[0]
        nodes_to_remove.append(node)

    if not replacements:
        onnx.save(m, out_path)
        return out_path

    def resolve(name: str) -> str:
        seen = set()
        cur = name
        while cur in replacements and cur not in seen:
            seen.add(cur)
            cur = replacements[cur]
        return cur

    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp in replacements:
                node.input[i] = resolve(inp)

    kept = [n for n in g.node if n not in nodes_to_remove]
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Removed Identity nodes: {len(nodes_to_remove)} node(s)")
    return out_path




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
        default="PP-YOLOE/build/models/human_reid.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument("--input-shape", type=str, default="4,3,256,128")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--img", type=str, help="Path to image for realistic preprocessing (not needed for --find-nms)")
    parser.add_argument("--outdir", type=str, default="PP-YOLOE/build/models/surgery")
    parser.add_argument("--ort-profile-dir", type=str, help="Directory for ORT profiling (enables profiling automatically)")
    parser.add_argument("--fp16", action="store_true", help="Attempt FP16 casting")
    parser.add_argument("--split-concat", type=int, default=0, help="Split large Concat nodes (default: 0/disabled for ReID)")
    parser.add_argument("--fix-input-shapes", action="store_true", help="Rewrite graph inputs to static shapes")
    parser.add_argument("--fold-static-shapes", action="store_true", help="Fold shape computation chains")
    # parser.add_argument("--rewrite-resize-to-static", action="store_true", help="Replace dynamic Resize with static sizes")
    parser.add_argument("--fold-iterations", type=int, default=15, help="Iterations for constant folding (default: 15)")
    parser.add_argument("--rewrite-div", action="store_true", help="Rewrite Div to Mul with reciprocal")
    parser.add_argument("--rewrite-pow", action="store_true", help="Rewrite Pow patterns")
    parser.add_argument("--rewrite-hardsigmoid", action="store_true", help="Rewrite HardSigmoid to Mul+Add+Clip")
    parser.add_argument("--rewrite-slice-range-to-gather", action="store_true", help="Rewrite range Slice (step=1) to Gather with indices")
    parser.add_argument("--rewrite-slice-to-gather", action="store_true", help="Rewrite Slice to Gather")
    parser.add_argument("--rewrite-reduce-to-globalpool", action="store_true", help="Rewrite ReduceMean/ReduceMax over H,W to GlobalPool")
    parser.add_argument("--remove-noop-slice", action="store_true", help="Remove Slice ops that are effectively identity")
    parser.add_argument("--fold-conv-bn", action="store_true", help="Fold BatchNorm parameters into preceding Conv nodes")
    parser.add_argument("--remove-identity", action="store_true", help="Remove redundant Identity nodes")
    parser.add_argument("--remove-unused-inits", action="store_true", help="Drop orphan initializers left after rewrites")
    parser.add_argument("--remove-unused-const", action="store_true", help="Drop Constant nodes with no consumers")
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

    # Enable profiling automatically if profile directory is specified
    enable_profiling = bool(args.ort_profile_dir)

    print("=== Baseline ===")
    if ort is None:
        print("onnxruntime not available; install 'onnxruntime' or 'onnxruntime-silicon'.")
        return

    base = run_benchmark(args.model, ishape, "coreml", args.warmup, args.runs,
                        enable_profile=enable_profiling, profile_dir=args.ort_profile_dir, img_path=img_path)
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

    if args.fix_input_shapes:
        print("[stage] Fix input shapes to static…")
        mod_fix = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fixed.onnx'))
        work_path = fix_input_shapes(work_path, mod_fix, ishape if len(ishape) == 4 else (1, 3, 640, 640))

    print("[stage] Shape inference…")
    mod2 = mod_path.replace("_mod.onnx", "_shape.onnx")
    work_path = shape_infer_model(work_path, mod2)



    if args.fold_conv_bn:
        print("[stage] Folding Conv+BatchNorm…")
        modCB = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_convbn.onnx'))
        work_path = fold_conv_batchnorm(work_path, modCB)



    if args.split_concat and args.split_concat > 0:
        print(f"[stage] Splitting large Concat nodes (max_inputs={args.split_concat})…")
        modC = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_splitconcat.onnx'))
        work_path = split_large_concats(work_path, modC, max_inputs=int(args.split_concat))

    # Re-run shape inference after structural rewrites
    print("[stage] Running shape inference (post-rewrite)…")
    mod2b = mod_path.replace("_mod.onnx", "_shape2.onnx")
    work_path = shape_infer_model(work_path, mod2b)

    if args.fold_static_shapes:
        print(f"[stage] Folding static shape chains ({args.fold_iterations} iterations)…")
        modF = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_foldshape.onnx'))
        work_path = fold_static_shape_chains(work_path, modF, iterations=args.fold_iterations)

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

    if args.rewrite_slice_range_to_gather:
        print("[stage] Rewriting range Slice -> Gather…")
        modSGR = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice2gather_range.onnx'))
        work_path = rewrite_slice_range_to_gather(work_path, modSGR)

    if args.rewrite_slice_to_gather:
        print("[stage] Rewriting simple Slice -> Gather…")
        modSG = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice2gather.onnx'))
        work_path = rewrite_slice_to_gather(work_path, modSG)

    # if args.rewrite_resize_to_static:
    #     print("[stage] Rewriting Resize to static sizes…")
    #     modRZ = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_resize_static.onnx'))
    #     work_path = rewrite_resize_to_static(work_path, modRZ)

    if args.remove_noop_slice:
        print("[stage] Removing no-op Slice ops…")
        modNS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice_noop.onnx'))
        work_path = remove_noop_slice(work_path, modNS)

    if args.remove_identity:
        print("[stage] Removing Identity nodes…")
        modID = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_no_identity.onnx'))
        work_path = remove_identity_nodes(work_path, modID)

    if args.remove_unused_const:
        print("[stage] Removing unused Constant nodes…")
        modUC = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_no_const.onnx'))
        work_path = remove_unused_constants(work_path, modUC)

    if args.remove_unused_inits:
        print("[stage] Dropping unused initializers…")
        modUI = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_clean_inits.onnx'))
        work_path = remove_unused_initializers(work_path, modUI)

    if args.rewrite_reduce_to_globalpool:
        print("[stage] Rewriting Reduce -> GlobalPool…")
        modGP = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_globalpool.onnx'))
        work_path = rewrite_reduce_to_globalpool(work_path, modGP)

    # Final shape inference to refresh value_info after ANE rewrites
    print("[stage] Running shape inference (final)…")
    mod2c = mod_path.replace("_mod.onnx", "_shape3.onnx")
    work_path = shape_infer_model(work_path, mod2c)

    if args.fp16:
        print("[stage] Attempting FP16 casting…")
        mod3 = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fp16.onnx'))
        work_path = cast_graph_to_fp16(work_path, mod3)

    optimized_path = work_path

    # Copy to final artifact
    if args.output_model:
        # Use the specified output model path
        final_path = args.output_model
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    else:
        # Use default naming in output directory
        final_path = os.path.join(
            args.outdir, os.path.basename(args.model).replace('.onnx', '_ane.onnx')
        )

    try:
        shutil.copyfile(optimized_path, final_path)
        print(f"[stage] Copied ANE-optimized model to: {final_path}")
    except Exception as e:
        print(f"[WARNING] Failed to copy to final path: {e}")
        final_path = optimized_path

    final_abs = os.path.abspath(final_path)
    final_name = os.path.basename(final_path)

    # Model info and benchmarking at the end
    mod_info = load_model_info(final_path)
    print("\n" + "="*70)
    print("=== Modified Model Info ===")
    print("="*70)
    print("Nodes:", mod_info["node_count"], "Unique ops:", mod_info["unique_ops"])
    print("Modified model:", optimized_path)
    print("Final model:", final_path)

    print("\n" + "="*70)
    print("=== Final Artifact ===")
    print("="*70)
    print(f"Filename: {final_name}")
    print(f"Path: {final_abs}")

    print("\n" + "="*70)
    print("=== Modified Benchmark ===")
    print("="*70)
    mod = run_benchmark(final_path, ishape, "coreml", args.warmup, args.runs,
                       enable_profile=enable_profiling, profile_dir=args.ort_profile_dir, img_path=img_path)
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