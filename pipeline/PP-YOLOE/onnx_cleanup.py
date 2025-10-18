#!/usr/bin/env python3
"""Finalize PP-YOLOE ONNX by running optimizer, simplifier, and cleanup passes."""
import argparse
import os
import shutil
import sys
from typing import List, Set

import onnx

# Optional dependencies are imported inside helper functions

from profile_onnx import load_model_info


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


def simplify_model(model_path: str, out_path: str) -> str:
    try:
        import onnxsim  # type: ignore
    except Exception:
        shutil.copyfile(model_path, out_path)
        return out_path

    cmd = [sys.executable, "-m", "onnxsim", model_path, out_path]
    try:
        import subprocess

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
    except Exception:
        shutil.copyfile(model_path, out_path)
        return out_path


def run_onnxoptimizer(model_path: str, out_path: str) -> str:
    try:
        import onnxoptimizer  # type: ignore
    except Exception:
        shutil.copyfile(model_path, out_path)
        return out_path

    m = onnx.load(model_path)
    available = set(getattr(onnxoptimizer, "get_available_passes", lambda: [])())
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
    m = onnx.load(model_path)
    g = m.graph

    consumers: Set[str] = set()
    for n in g.node:
        for i in n.input:
            if i:
                consumers.add(i)
    graph_output_names = {o.name for o in g.output}
    consumers |= graph_output_names

    kept_nodes: List[onnx.NodeProto] = []
    removed_const = 0
    for n in g.node:
        if n.op_type == "Constant" and n.output and all((o not in consumers) for o in n.output):
            removed_const += 1
            continue
        kept_nodes.append(n)

    consumers.clear()
    for n in kept_nodes:
        for i in n.input:
            if i:
                consumers.add(i)
    consumers |= graph_output_names

    kept_inits = [init for init in g.initializer if init.name in consumers]
    removed_inits = len(g.initializer) - len(kept_inits)

    live_names: Set[str] = set(consumers)
    for n in kept_nodes:
        for o in n.output:
            if o:
                live_names.add(o)

    kept_vi = [vi for vi in g.value_info if vi.name in live_names]
    removed_vi = len(g.value_info) - len(kept_vi)

    if drop_unused_inputs:
        kept_inputs = [inp for inp in g.input if (inp.name in consumers or inp.name in graph_output_names)]
        removed_inputs = len(g.input) - len(kept_inputs)
    else:
        kept_inputs = list(g.input)
        removed_inputs = 0

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
            f"Cleaned unused: initializers={removed_inits}, constants={removed_const}, "
            f"value_info={removed_vi}, inputs={removed_inputs}"
        )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run cleanup passes (optimizer, simplifier, clean unused tensors)")
    parser.add_argument("--model", required=True, help="Path to the ONNX model to clean")
    parser.add_argument("--outdir", default="pipeline/PP-YOLOE/models/surgery", help="Output directory")
    parser.add_argument("--output-model", help="Explicit final output path")
    parser.add_argument("--skip-optimizer", action="store_true", help="Skip onnxoptimizer passes")
    parser.add_argument("--skip-simplifier", action="store_true", help="Skip onnx-simplifier")
    parser.add_argument("--skip-clean", action="store_true", help="Skip unused tensor cleanup")
    parser.add_argument("--drop-unused-inputs", action="store_true", help="Remove graph inputs with no consumers")
    parser.add_argument("--shape-infer", action="store_true", help="Run shape inference before cleanup")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] ONNX model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    base_name = os.path.basename(args.model)
    work_path = args.model

    if args.shape_infer:
        mod_shape = os.path.join(args.outdir, base_name.replace('.onnx', '_shape.onnx'))
        print("[stage] Running shape inference…")
        work_path = shape_infer_model(work_path, mod_shape)

    if not args.skip_optimizer:
        mod_opt = os.path.join(args.outdir, base_name.replace('.onnx', '_opt.onnx'))
        print("[stage] Applying onnxoptimizer passes…")
        work_path = run_onnxoptimizer(work_path, mod_opt)

    if not args.skip_simplifier:
        mod_simp = os.path.join(args.outdir, base_name.replace('.onnx', '_simp.onnx'))
        print("[stage] Applying onnx-simplifier…")
        work_path = simplify_model(work_path, mod_simp)

    if not args.skip_clean:
        mod_clean = os.path.join(args.outdir, base_name.replace('.onnx', '_clean.onnx'))
        print("[stage] Cleaning unused tensors…")
        work_path = clean_unused_tensors(work_path, mod_clean, drop_unused_inputs=args.drop_unused_inputs)

    if args.output_model:
        final_path = args.output_model
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    else:
        final_path = os.path.join(args.outdir, base_name.replace('.onnx', '_cu.onnx'))

    try:
        shutil.copyfile(work_path, final_path)
        print(f"[stage] Copied cleaned model to: {final_path}")
    except Exception as e:
        print(f"[WARNING] Failed to copy cleaned model: {e}")
        final_path = work_path

    info = load_model_info(final_path)
    print("\n" + "=" * 70)
    print("=== Cleanup Summary ===")
    print("=" * 70)
    print("Nodes:", info["node_count"], "Unique ops:", info["unique_ops"])
    print("Final path:", os.path.abspath(final_path))
    print("Cleanup complete. Model is ready for deployment or further validation.")


if __name__ == "__main__":
    main()
