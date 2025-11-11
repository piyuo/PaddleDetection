#!/usr/bin/env python3
"""Customize PP-YOLOE ONNX model by exposing raw detections and removing NMS.

This script takes the base exported ONNX model and produces a customized variant
that keeps only the raw detection tensors (boxes/scores) required for downstream
post-processing. Feature-map outputs are no longer exported.

Workflow reminder:
1. export_to_onnx.sh -> base.onnx
2. onnx_customize.py -> base_cust.onnx (this script)
3. ncnn_graph_surgery.py / ane_graph_surgery.py -> base_cust_opt.onnx
4. onnx_cleanup.py -> base_cust_opt_cu.onnx
"""

import argparse
import os
import shutil
import sys
from typing import Dict, List, Set

import onnx
from onnx import helper

from onnx_profile import load_model_info, parse_shape


def shape_infer_model(model_path: str, out_path: str) -> str:
    """Run ONNX shape inference and persist the inferred model."""
    try:
        inferred = onnx.shape_inference.infer_shapes_path(model_path)
        if isinstance(inferred, str) and os.path.exists(inferred):
            return inferred
    except Exception:
        pass

    model = onnx.load(model_path)
    inferred_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred_model, out_path)
    return out_path


def prune_outputs(model_path: str, out_path: str, keep_outputs: List[str]) -> str:
    """Keep only selected outputs and prune unreachable nodes using DFS."""
    if not keep_outputs:
        shutil.copyfile(model_path, out_path)
        return out_path

    model = onnx.load(model_path)
    graph = model.graph
    print(f"  Pruning to keep outputs: {keep_outputs}")

    tensor_producers: Dict[str, int] = {}
    for idx, node in enumerate(graph.node):
        for out in node.output:
            tensor_producers[out] = idx

    reachable_node_indices: Set[int] = set()
    visited_tensors: Set[str] = set()

    def visit(tensor_name: str) -> None:
        if tensor_name in visited_tensors:
            return
        visited_tensors.add(tensor_name)
        if tensor_name in tensor_producers:
            node_idx = tensor_producers[tensor_name]
            if node_idx not in reachable_node_indices:
                reachable_node_indices.add(node_idx)
                node = graph.node[node_idx]
                for inp in node.input:
                    if inp:
                        visit(inp)

    for output_name in keep_outputs:
        visit(output_name)

    print(f"  Reachable nodes: {len(reachable_node_indices)} / {len(graph.node)}")
    kept_nodes = [graph.node[i] for i in sorted(reachable_node_indices)]

    live_tensors: Set[str] = set()
    for node in kept_nodes:
        live_tensors.update(inp for inp in node.input if inp)
        live_tensors.update(out for out in node.output if out)

    graph_input_names = {vi.name for vi in graph.input}
    init_names = {init.name for init in graph.initializer}
    live_tensors |= graph_input_names | init_names

    kept_initializers = [init for init in graph.initializer if init.name in live_tensors]
    kept_value_info = [vi for vi in graph.value_info if vi.name in live_tensors]

    new_outputs = []
    for output_name in keep_outputs:
        existing_vi = None
        for vi in list(graph.output) + list(graph.value_info):
            if vi.name == output_name:
                existing_vi = vi
                break
        if existing_vi is None:
            vi = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None)
            new_outputs.append(vi)
            print(f"  [WARNING] Created minimal ValueInfo for output '{output_name}'")
        else:
            new_outputs.append(existing_vi)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)
    del graph.output[:]
    graph.output.extend(new_outputs)

    onnx.save(model, out_path)

    pruned_model = onnx.load(out_path)
    actual_outputs = [output.name for output in pruned_model.graph.output]
    has_nms = any(node.op_type == "NonMaxSuppression" for node in pruned_model.graph.node)
    print(f"  Pruned model outputs: {actual_outputs}")
    print(f"  Has NMS nodes: {has_nms}")
    if has_nms:
        print("  [WARNING] NMS nodes still present - they may be reachable from kept outputs!")

    return out_path


def find_nms_nodes(model_path: str) -> Dict[str, str]:
    """Locate NonMaxSuppression nodes and return their input tensors."""
    try:
        model = onnx.load(model_path)
    except Exception as exc:
        print(f"[ERROR] Error loading ONNX model: {exc}", file=sys.stderr)
        return {}

    nms_nodes = [node for node in model.graph.node if node.op_type == "NonMaxSuppression"]
    if not nms_nodes:
        return {}

    node = nms_nodes[0]
    inputs = [inp for inp in node.input if inp]
    result: Dict[str, str] = {}
    if len(inputs) >= 2:
        result["boxes"] = inputs[0]
        result["scores"] = inputs[1]
    return result


def auto_discover_outputs(model_path: str) -> Dict[str, Dict[str, str]]:
    """Auto-discover detection tensors by inspecting NMS inputs."""
    discovered: Dict[str, Dict[str, str]] = {}
    nms_info = find_nms_nodes(model_path)
    if nms_info:
        print("\n=== Automatic Output Discovery ===")
        print("[INFO] Found NMS inputs:")
        print(f"  Boxes:  {nms_info.get('boxes', 'N/A')}")
        print(f"  Scores: {nms_info.get('scores', 'N/A')}")
        discovered["nms"] = nms_info
    else:
        print("\n=== Automatic Output Discovery ===")
        print("[WARN] No NMS nodes found - detection tensors must be supplied manually.")
    return discovered


def print_output_guide(discovered: Dict[str, Dict[str, str]], keep_outputs: List[str]) -> None:
    """Print a human-readable guide for the customized outputs."""
    print("\n" + "=" * 70)
    print("=== Customized Model Output Guide ===")
    print("=" * 70)

    nms = discovered.get("nms", {}) if discovered else {}

    for idx, output_name in enumerate(keep_outputs):
        print(f"\nOutput {idx}: '{output_name}'")
        if nms.get("boxes") == output_name:
            print("  Type: Detection boxes (raw)")
            print("  Usage: Pair with detection scores and run downstream NMS")
        elif nms.get("scores") == output_name:
            print("  Type: Detection scores (raw)")
            print("  Usage: Threshold and combine with boxes before NMS")
        else:
            print("  Type: Custom intermediate tensor")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Customize PP-YOLOE ONNX by exposing raw detections")
    parser.add_argument("--model", required=True, help="Path to the base ONNX model")
    parser.add_argument("--input-shape", type=str, default="1,3,640,640", help="Model input shape (N,C,H,W)")
    parser.add_argument("--outdir", type=str, default="PP-YOLOE/build/models/custom", help="Output directory")
    parser.add_argument("--output-model", type=str, help="Explicit output path for customized model")
    parser.add_argument("--keep-output", action="append", dest="keep_outputs", help="Extra output to keep (repeatable)")
    parser.add_argument("--skip-auto-discovery", action="store_true", help="Skip automatic discovery; requires --keep-output")
    parser.add_argument("--no-guide", action="store_true", help="Skip printing the output usage guide")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] ONNX model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    ishape = parse_shape(args.input_shape)
    if len(ishape) != 4:
        print("[WARNING] Unexpected input shape, defaulting to 640x640 image size")
    keep: List[str] = []
    discovered_info: Dict[str, Dict[str, str]] = {}

    if args.keep_outputs:
        keep.extend(args.keep_outputs)

    if not keep and not args.skip_auto_discovery:
        discovered_info = auto_discover_outputs(args.model)
        nms_info = discovered_info.get("nms", {})
        if nms_info.get("boxes"):
            keep.append(nms_info["boxes"])
        if nms_info.get("scores"):
            keep.append(nms_info["scores"])

    keep = list(dict.fromkeys(keep))

    if not keep:
        print("[ERROR] No outputs specified. Provide --keep-output or allow auto discovery.", file=sys.stderr)
        sys.exit(1)

    base_name = os.path.basename(args.model)
    pruned_path = os.path.join(args.outdir, base_name.replace(".onnx", "_cust_pruned.onnx"))
    work_path = prune_outputs(args.model, pruned_path, keep)

    shape_path = os.path.join(args.outdir, base_name.replace(".onnx", "_cust_shape.onnx"))
    work_path = shape_infer_model(work_path, shape_path)

    if args.output_model:
        final_path = args.output_model
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    else:
        final_path = os.path.join(args.outdir, base_name.replace(".onnx", "_cust.onnx"))

    try:
        shutil.copyfile(work_path, final_path)
        print(f"[stage] Copied customized model to: {final_path}")
    except Exception as exc:
        print(f"[WARNING] Failed to copy customized model: {exc}")
        final_path = work_path

    info = load_model_info(final_path)
    print("\n" + "=" * 70)
    print("=== Customized Model Summary ===")
    print("=" * 70)
    print("Nodes:", info["node_count"], "Unique ops:", info["unique_ops"])
    print("Final path:", os.path.abspath(final_path))

    if not args.no_guide and discovered_info:
        print_output_guide(discovered_info, keep)


if __name__ == "__main__":
    main()
