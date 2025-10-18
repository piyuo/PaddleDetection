#!/usr/bin/env python3
"""
ONNX Model Cleanup and Optimization Script

This script performs final cleanup and optimization on ONNX models:
1. Apply onnxoptimizer passes (eliminate dead nodes, fuse ops, etc.)
2. Run onnx-simplifier for constant folding and simplification
3. Clean unused tensors and initializers

Input: Optimized ONNX model (e.g., ppyoloe_crn_s_36e_pphuman_cust_ane.onnx)
Output: Cleaned ONNX model (e.g., ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx)
"""
import os
import sys
import argparse
import subprocess
import shutil
from typing import Set, List
import onnx
from onnx import numpy_helper

try:
    import onnxoptimizer
except Exception:
    onnxoptimizer = None


def simplify_model(model_path: str, out_path: str) -> str:
    """Run onnx-simplifier in a subprocess to isolate potential segfaults; fallback to copy on failure."""
    try:
        import onnxsim
    except Exception:
        print("⚠️  onnx-simplifier not available, skipping simplification")
        shutil.copy(model_path, out_path)
        return out_path

    cmd = [sys.executable, "-m", "onnxsim", model_path, out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
    except Exception:
        print("⚠️  onnx-simplifier failed, copying original model")
        shutil.copy(model_path, out_path)
        return out_path


def run_onnxoptimizer(model_path: str, out_path: str) -> str:
    """Apply onnxoptimizer passes to eliminate dead nodes and fuse operations."""
    try:
        import onnxoptimizer
    except Exception:
        print("⚠️  onnxoptimizer not available, skipping optimization")
        shutil.copy(model_path, out_path)
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
        print("⚠️  onnxoptimizer failed, copying original model")
        shutil.copy(model_path, out_path)
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
        for inp in n.input:
            if inp:
                consumers.add(inp)
    graph_output_names = {o.name for o in g.output}
    consumers |= graph_output_names

    # Remove Constant nodes with outputs that no one consumes
    kept_nodes: List[onnx.NodeProto] = []
    removed_const = 0
    for n in g.node:
        if n.op_type == 'Constant' and all(o not in consumers for o in n.output):
            removed_const += 1
        else:
            kept_nodes.append(n)

    # Recompute consumers after removing some constants
    consumers.clear()
    for n in kept_nodes:
        for inp in n.input:
            if inp:
                consumers.add(inp)
    consumers |= graph_output_names

    # Keep only initializers that are consumed
    kept_inits = [init for init in g.initializer if init.name in consumers]
    removed_inits = len(g.initializer) - len(kept_inits)

    # Live value names: consumed inputs and produced outputs from remaining nodes
    live_names: Set[str] = set(consumers)
    for n in kept_nodes:
        for out in n.output:
            if out:
                live_names.add(out)

    # Prune stray value_info entries
    kept_vi = [vi for vi in g.value_info if vi.name in live_names]
    removed_vi = len(g.value_info) - len(kept_vi)

    # Optionally drop unused graph inputs
    if drop_unused_inputs:
        kept_inputs = [vi for vi in g.input if vi.name in consumers]
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
        print(f"  Removed: {removed_const} Constant nodes, {removed_inits} initializers, "
              f"{removed_vi} value_info, {removed_inputs} unused inputs")
    return out_path


def shape_infer_model(model_path: str, out_path: str) -> str:
    """Run ONNX shape inference."""
    try:
        m = onnx.load(model_path)
        m2 = onnx.shape_inference.infer_shapes(m)
        onnx.save(m2, out_path)
        return out_path
    except Exception:
        print("⚠️  Shape inference failed, copying original model")
        shutil.copy(model_path, out_path)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Clean and optimize ONNX model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to source ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output cleaned ONNX model",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        help="Working directory for intermediate files (default: temp dir next to output)",
    )
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="Skip onnxoptimizer passes",
    )
    parser.add_argument(
        "--skip-simplifier",
        action="store_true",
        help="Skip onnx-simplifier",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip unused tensor cleanup",
    )
    parser.add_argument(
        "--shape-inference",
        action="store_true",
        help="Run shape inference before cleanup",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)

    # Setup working directory
    if args.workdir:
        workdir = args.workdir
    else:
        workdir = os.path.join(os.path.dirname(args.output), "cleanup_temp")
    os.makedirs(workdir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.model))[0]
    work_path = args.model

    # Stage 1: Shape inference (optional)
    if args.shape_inference:
        print("[stage] Running shape inference...")
        shape_model = os.path.join(workdir, f"{base_name}_shape.onnx")
        work_path = shape_infer_model(work_path, shape_model)

    # Stage 2: onnxoptimizer
    if not args.skip_optimizer:
        print("[stage] Applying onnxoptimizer passes...")
        opt_model = os.path.join(workdir, f"{base_name}_opt.onnx")
        work_path = run_onnxoptimizer(work_path, opt_model)

    # Stage 3: onnx-simplifier
    if not args.skip_simplifier:
        print("[stage] Applying onnx-simplifier...")
        simp_model = os.path.join(workdir, f"{base_name}_simp.onnx")
        work_path = simplify_model(work_path, simp_model)

    # Stage 4: Clean unused tensors
    if not args.skip_cleanup:
        print("[stage] Cleaning unused tensors...")
        clean_model = os.path.join(workdir, f"{base_name}_clean.onnx")
        work_path = clean_unused_tensors(work_path, clean_model, drop_unused_inputs=False)

    # Copy final result to output
    if work_path != args.output:
        shutil.copy(work_path, args.output)

    print(f"\n✓ Cleaned model saved to: {args.output}")

    # Cleanup temp directory if default was used
    if not args.workdir:
        try:
            shutil.rmtree(workdir)
        except:
            pass


if __name__ == "__main__":
    main()
