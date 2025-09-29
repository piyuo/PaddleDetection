#!/usr/bin/env python3
"""
Coreml graph surgery for PP-YOLOE ONNX with aggressive ANE-targeted optimizations.

New optimizations targeting CoreML ANE acceleration:
- ReduceMean -> GlobalAveragePool/AvgPool conversion
- Aggressive constant folding (15 iterations)
- Identity chain elimination
- Reshape/Transpose chain simplification
- Enhanced ANE compatibility analysis
"""
import argparse
import os
import shutil
from typing import Tuple, List, Dict, Set
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


def rewrite_hardswish(model_path: str, out_path: str) -> str:
    """Replace HardSwish nodes with x * Clip(x + 3, 0, 6) * (1/6)."""
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

        add_out = unique_name(node.name + "_add3")
        add_node = helper.make_node("Add", [x, c3_name], [add_out], name=unique_name(node.name + "_Add"))

        clip_out = unique_name(node.name + "_clip")
        clip_node = helper.make_node("Clip", [add_out, c0_name, c6_name], [clip_out], name=unique_name(node.name + "_Clip"))

        mul1_out = unique_name(node.name + "_mul1")
        mul1_node = helper.make_node("Mul", [x, clip_out], [mul1_out], name=unique_name(node.name + "_Mul1"))

        mul2_node = helper.make_node("Mul", [mul1_out, cscale_name], [y], name=unique_name(node.name + "_Mul2"))

        new_nodes.extend([add_node, clip_node, mul1_node, mul2_node])
        nodes_to_remove.append(node)
        changed += 1

    if changed == 0:
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
    print(f"Rewrote HardSwish -> Add+Clip+Mul: {changed} node(s)")
    return out_path


def rewrite_swish_to_hardswish(model_path: str, out_path: str) -> str:
    """Detect Swish/SiLU patterns (x * Sigmoid(x)) and rewrite into Add+Clip+Mul style."""
    m = onnx.load(model_path)
    g = m.graph

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
        pa = producer.get(a)
        pb = producer.get(b)

        def is_sigmoid_of_x(pnode, xname):
            return pnode is not None and pnode.op_type == "Sigmoid" and len(pnode.input) == 1 and pnode.input[0] == xname

        if is_sigmoid_of_x(pa, b):
            x = b
        elif is_sigmoid_of_x(pb, a):
            x = a
        else:
            continue

        y = node.output[0]

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
        changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

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
        shutil.copyfile(model_path, out_path)
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

def rewrite_reducemean_to_avgpool(model_path: str, out_path: str) -> str:
    """Convert spatial ReduceMean to GlobalAveragePool or AvgPool (ANE-optimized)."""
    m = onnx.load(model_path)
    g = m.graph

    # Get value info for shapes
    value_shapes = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
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
        if node.op_type != "ReduceMean":
            continue

        axes = None
        keepdims = 1
        for a in node.attribute:
            if a.name == "axes":
                axes = list(a.ints) if a.ints else None
            if a.name == "keepdims":
                keepdims = int(a.i)

        if axes is None:
            continue

        input_name = node.input[0]
        output_name = node.output[0]
        input_shape = value_shapes.get(input_name)

        # Pattern 1: ReduceMean on axes [2,3] (spatial H,W) for 4D tensor -> GlobalAveragePool
        if input_shape and len(input_shape) == 4 and set(axes) == {2, 3}:
            gap_out = output_name if keepdims else unique_name(node.name + "_gap")
            gap_node = helper.make_node(
                "GlobalAveragePool",
                [input_name],
                [gap_out],
                name=unique_name(node.name + "_GAP")
            )
            new_nodes.append(gap_node)

            # If keepdims=0, need to squeeze out H,W dims
            if not keepdims:
                # Create axes constant for Squeeze (opset >= 13 uses input instead of attribute)
                axes_name = unique_name(node.name + "_squeeze_axes")
                axes_tensor = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name=axes_name)
                g.initializer.extend([axes_tensor])

                squeeze_node = helper.make_node(
                    "Squeeze",
                    [gap_out, axes_name],
                    [output_name],
                    name=unique_name(node.name + "_Squeeze")
                )
                new_nodes.append(squeeze_node)

            nodes_to_remove.append(node)
            changed += 1
            continue

        # Pattern 2: ReduceMean on single spatial axis -> AveragePool with kernel covering that dimension
        # This is more complex and may not always be beneficial, so we're conservative

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    kept = [n for n in g.node if n not in nodes_to_remove]
    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Rewrote ReduceMean -> GlobalAveragePool: {changed} node(s)")
    return out_path


def eliminate_identity_chains(model_path: str, out_path: str) -> str:
    """Eliminate long chains of Identity nodes by directly connecting producers to consumers."""
    m = onnx.load(model_path)
    g = m.graph

    # Build producer map
    producer = {}
    for node in g.node:
        for out in node.output:
            producer[out] = node

    # Find Identity chains
    def trace_identity_chain(tensor_name: str) -> str:
        """Follow Identity chain to find the original source tensor."""
        visited = set()
        current = tensor_name
        while current not in visited:
            visited.add(current)
            prod = producer.get(current)
            if prod is None or prod.op_type != "Identity":
                return current
            if len(prod.input) != 1:
                return current
            current = prod.input[0]
        return tensor_name  # Cycle detected, return original

    # Rewrite all tensor references
    tensor_map = {}
    for node in g.node:
        for inp in node.input:
            source = trace_identity_chain(inp)
            if source != inp:
                tensor_map[inp] = source

    # Apply rewrites
    changed_nodes = 0
    for node in g.node:
        rewrote = False
        new_inputs = []
        for inp in node.input:
            if inp in tensor_map:
                new_inputs.append(tensor_map[inp])
                rewrote = True
            else:
                new_inputs.append(inp)
        if rewrote:
            del node.input[:]
            node.input.extend(new_inputs)
            changed_nodes += 1

    # Remove dead Identity nodes
    output_names = {o.name for o in g.output}
    consumers = {}
    for node in g.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    kept = []
    removed_identities = 0
    for node in g.node:
        if node.op_type == "Identity":
            out = node.output[0]
            # Keep if it's a graph output
            if out in output_names:
                kept.append(node)
            # Keep if it still has consumers
            elif out in consumers and consumers[out]:
                kept.append(node)
            else:
                removed_identities += 1
        else:
            kept.append(node)

    if removed_identities == 0:
        onnx.save(m, out_path)
        return out_path

    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Eliminated Identity chains: {removed_identities} Identity node(s) removed, {changed_nodes} node(s) rewired")
    return out_path


def fuse_reshape_transpose_chains(model_path: str, out_path: str) -> str:
    """Eliminate redundant Reshape/Transpose sequences."""
    m = onnx.load(model_path)
    g = m.graph

    producer = {}
    for node in g.node:
        for out in node.output:
            producer[out] = node

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

    new_nodes: List[onnx.NodeProto] = []
    nodes_to_remove: List[onnx.NodeProto] = []
    changed = 0

    # Pattern: Transpose -> Transpose (inverse) = Identity
    for node in g.node:
        if node.op_type != "Transpose":
            continue

        perm1 = None
        for a in node.attribute:
            if a.name == "perm":
                perm1 = list(a.ints)
                break
        if perm1 is None:
            continue

        # Check if consumer is also Transpose
        consumers = [n for n in g.node if node.output[0] in n.input]
        if len(consumers) != 1:
            continue

        consumer = consumers[0]
        if consumer.op_type != "Transpose":
            continue

        perm2 = None
        for a in consumer.attribute:
            if a.name == "perm":
                perm2 = list(a.ints)
                break
        if perm2 is None:
            continue

        # Check if perm2 is inverse of perm1
        if len(perm1) != len(perm2):
            continue

        composed = [perm1[i] for i in perm2]
        if composed == list(range(len(composed))):
            # This is identity! Replace with direct connection
            identity_node = helper.make_node(
                "Identity",
                [node.input[0]],
                list(consumer.output),
                name=unique_name(node.name + "_fused_id")
            )
            new_nodes.append(identity_node)
            nodes_to_remove.extend([node, consumer])
            changed += 1

    if changed == 0:
        onnx.save(m, out_path)
        return out_path

    kept = [n for n in g.node if n not in nodes_to_remove]
    kept.extend(new_nodes)
    del g.node[:]
    g.node.extend(kept)
    onnx.save(m, out_path)
    print(f"Fused Reshape/Transpose chains: {changed} pattern(s)")
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

    if "ReduceMean" in op_names:
        suggestions.append("  → Try --reducemean-to-avgpool to convert spatial ReduceMean to GlobalAveragePool")
    if "Reshape" in op_names or "Transpose" in op_names:
        suggestions.append("  → Try --fuse-reshape-transpose to eliminate redundant layout changes")
    if "Identity" in op_names:
        suggestions.append("  → Identity chains detected (already using --eliminate-identity-chains)")
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
    parser.add_argument("--ep", type=str, default="coreml")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--img", type=str, required=True, help="Path to image for realistic preprocessing")
    parser.add_argument("--outdir", type=str, default="pipeline/PP-YOLOE/models/surgery")
    parser.add_argument("--ort-profile", action="store_true", help="Enable ORT timeline profiling")
    parser.add_argument("--ort-profile-dir", type=str, default="pipeline/PP-YOLOE/output")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-shape-infer", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Attempt FP16 casting")
    parser.add_argument("--fix-input-shapes", action="store_true", help="Rewrite graph inputs to static shapes")
    parser.add_argument("--no-optimizer", action="store_true", help="Disable onnxoptimizer passes")
    parser.add_argument("--rewrite-hardswish", action="store_true", help="Rewrite HardSwish to Add+Clip+Mul")
    parser.add_argument("--split-concat", type=int, default=0, help="Split large Concat nodes")
    parser.add_argument("--fold-static-shapes", action="store_true", help="Fold shape computation chains")
    parser.add_argument("--fold-iterations", type=int, default=15, help="Iterations for constant folding (default: 15)")
    parser.add_argument("--keep-outputs", type=str, default="", help="Comma-separated outputs to keep")
    parser.add_argument("--rewrite-div", action="store_true", help="Rewrite Div to Mul with reciprocal")
    parser.add_argument("--rewrite-pow", action="store_true", help="Rewrite Pow patterns")
    parser.add_argument("--rewrite-swish", action="store_true", help="Rewrite Swish to HardSwish-style")
    parser.add_argument("--rewrite-hardsigmoid", action="store_true", help="Rewrite HardSigmoid to Mul+Add+Clip")
    parser.add_argument("--rewrite-slice-to-gather", action="store_true", help="Rewrite Slice to Gather")

    # NEW FLAGS
    parser.add_argument("--reducemean-to-avgpool", action="store_true", help="Convert ReduceMean to GlobalAveragePool (ANE-optimized)")
    parser.add_argument("--eliminate-identity-chains", action="store_true", help="Remove redundant Identity node chains")
    parser.add_argument("--fuse-reshape-transpose", action="store_true", help="Fuse inverse Reshape/Transpose pairs")
    parser.add_argument("--aggressive-mode", action="store_true", help="Enable all ANE optimizations")
    parser.add_argument("--output-model", type=str, help="Path to copy final optimized model to")

    args = parser.parse_args()

    # Aggressive mode enables all optimizations
    if args.aggressive_mode:
        args.fix_input_shapes = True
        args.rewrite_hardswish = True
        args.rewrite_swish = True
        args.rewrite_hardsigmoid = True
        args.rewrite_div = True
        args.rewrite_pow = True
        args.rewrite_slice_to_gather = True
        args.fold_static_shapes = True
        args.reducemean_to_avgpool = True
        args.eliminate_identity_chains = True
        args.fuse_reshape_transpose = True
        if not args.split_concat:
            args.split_concat = 4
        print("[Aggressive mode enabled - all ANE optimizations active]")

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

    base = run_benchmark(args.model, ishape, args.ep, args.warmup, args.runs,
                        enable_profile=args.ort_profile, profile_dir=args.ort_profile_dir, img_path=img_path)
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

    if not args.no_shape_infer:
        print("[stage] Shape inference…")
        mod2 = mod_path.replace("_mod.onnx", "_shape.onnx")
        work_path = shape_infer_model(work_path, mod2)

    if args.rewrite_hardswish:
        print("[stage] Rewriting HardSwish nodes…")
        modH = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_hardswish.onnx'))
        work_path = rewrite_hardswish(work_path, modH)

    if args.rewrite_swish:
        print("[stage] Rewriting Swish/SiLU to HardSwish-style…")
        modHS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_swish2hs.onnx'))
        work_path = rewrite_swish_to_hardswish(work_path, modHS)

    if args.split_concat and args.split_concat > 0:
        print(f"[stage] Splitting large Concat nodes (max_inputs={args.split_concat})…")
        modC = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_splitconcat.onnx'))
        work_path = split_large_concats(work_path, modC, max_inputs=int(args.split_concat))

    # NEW: ReduceMean -> AvgPool
    if args.reducemean_to_avgpool:
        print("[stage] Converting ReduceMean to GlobalAveragePool…")
        modRM = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_reducemean.onnx'))
        work_path = rewrite_reducemean_to_avgpool(work_path, modRM)

    # NEW: Fuse Reshape/Transpose
    if args.fuse_reshape_transpose:
        print("[stage] Fusing Reshape/Transpose chains…")
        modRT = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fuseRT.onnx'))
        work_path = fuse_reshape_transpose_chains(work_path, modRT)

    # Re-run shape inference after structural rewrites
    if not args.no_shape_infer:
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

    if args.rewrite_slice_to_gather:
        print("[stage] Rewriting simple Slice -> Gather…")
        modSG = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_slice2gather.onnx'))
        work_path = rewrite_slice_to_gather(work_path, modSG)

    # NEW: Eliminate Identity chains
    if args.eliminate_identity_chains:
        print("[stage] Eliminating Identity chains…")
        modID = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_no_identity.onnx'))
        work_path = eliminate_identity_chains(work_path, modID)

    if args.keep_outputs:
        keep = [s.strip() for s in args.keep_outputs.split(",") if s.strip()]
        if keep:
            print("[stage] Pruning outputs; keeping:", keep)
            modP = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_pruned.onnx'))
            work_path = prune_outputs(work_path, modP, keep)

    # Final shape inference
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

    if args.fp16:
        print("[stage] Attempting FP16 casting…")
        mod3 = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fp16.onnx'))
        work_path = cast_graph_to_fp16(work_path, mod3)

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

    mod_info = load_model_info(work_path)
    print("\n=== Modified Model Info ===")
    print("Nodes:", mod_info["node_count"], "Unique ops:", mod_info["unique_ops"])
    print("Modified model:", work_path)
    print("Final model:", final_path)

    final_abs = os.path.abspath(final_path)
    final_name = os.path.basename(final_path)
    print("\n=== Final Artifact ===")
    print(f"Filename: {final_name}")
    print(f"Path: {final_abs}")

    print("\n=== Modified Benchmark ===")
    mod = run_benchmark(work_path, ishape, args.ep, args.warmup, args.runs,
                       enable_profile=args.ort_profile, profile_dir=args.ort_profile_dir, img_path=img_path)
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

    print("\n=== Performance Comparison (Modified vs Baseline) ===")
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