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
    passes = [
        "eliminate_identity",
        "eliminate_deadend",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "eliminate_nop_dropout",
        "fuse_consecutive_transposes",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "eliminate_unused_initializer",
    ]
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

        # Constants
        c3_name = unique_name("hardswish_c3")
        c3_tensor = numpy_helper.from_array(np.array([3.0], dtype=np.float32), name=c3_name)
        g.initializer.extend([c3_tensor])

        c0_name = unique_name("hardswish_c0")
        c0_tensor = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name=c0_name)
        g.initializer.extend([c0_tensor])

        c6_name = unique_name("hardswish_c6")
        c6_tensor = numpy_helper.from_array(np.array([6.0], dtype=np.float32), name=c6_name)
        g.initializer.extend([c6_tensor])

        cscale_name = unique_name("hardswish_c1_div6")
        cscale_tensor = numpy_helper.from_array(np.array([1.0 / 6.0], dtype=np.float32), name=cscale_name)
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

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ishape = parse_shape(args.input_shape)

    print("=== Baseline ===")
    if ort is None:
        print("onnxruntime not available; install 'onnxruntime' or 'onnxruntime-silicon'.")
        return
    base = run_benchmark(args.model, ishape, args.ep, args.warmup, args.runs)
    b = base["benchmark"]
    print(
        "Baseline Latency (ms) avg={:.2f} p50={:.2f} p90={:.2f} p95={:.2f}".format(
            b["latency_ms_avg"], b["latency_ms_p50"], b["latency_ms_p90"], b["latency_ms_p95"]
        )
    )

    # Build modified path
    mod_path = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_mod.onnx'))
    work_path = args.model

    if args.fix_input_shapes:
        print("Fixing input shapes to static…")
        mod_fix = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fixed.onnx'))
        work_path = fix_input_shapes(work_path, mod_fix, ishape if len(ishape) == 4 else (1, 3, 640, 640))

    if not args.no_shape_infer:
        print("Running shape inference…")
        mod2 = mod_path.replace("_mod.onnx", "_shape.onnx")
        work_path = shape_infer_model(work_path, mod2)
    else:
        work_path = mod_path

    if args.rewrite_hardswish:
        print("Rewriting HardSwish nodes…")
        modH = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_hardswish.onnx'))
        work_path = rewrite_hardswish(work_path, modH)

    if args.split_concat and args.split_concat > 0:
        print(f"Splitting large Concat nodes (max_inputs={args.split_concat})…")
        modC = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_splitconcat.onnx'))
        work_path = split_large_concats(work_path, modC, max_inputs=int(args.split_concat))

    # Re-run shape inference after structural rewrites so later passes have shape info
    if not args.no_shape_infer and (args.rewrite_hardswish or (args.split_concat and args.split_concat > 0)):
        print("Running shape inference (post-rewrite)…")
        mod2b = mod_path.replace("_mod.onnx", "_shape2.onnx")
        work_path = shape_infer_model(work_path, mod2b)

    if not args.no_optimizer:
        print("Applying onnxoptimizer passes…")
        modO = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_opt.onnx'))
        work_path = run_onnxoptimizer(work_path, modO)

    if not args.no_simplify:
        print("Applying onnx-simplifier…")
        modS = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_simp.onnx'))
        work_path = simplify_model(work_path, modS)
    else:
        shutil.copyfile(work_path, mod_path)

    if args.fp16:
        print("Attempting FP16 casting…")
        mod3 = os.path.join(args.outdir, os.path.basename(args.model).replace('.onnx', '_fp16.onnx'))
        work_path = cast_graph_to_fp16(work_path, mod3)

    print("Modified model:", work_path)
    mod_info = load_model_info(work_path)
    print("Nodes:", mod_info["node_count"], "Unique ops:", mod_info["unique_ops"])

    print("=== Modified Benchmark ===")
    mod = run_benchmark(work_path, ishape, args.ep, args.warmup, args.runs)
    m = mod["benchmark"]
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
