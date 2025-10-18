#!/usr/bin/env python3
"""
ONNX Model Customization Script for PP-YOLOE

This script transforms a base ONNX model by:
1. Discovering NMS nodes and their inputs (boxes, scores)
2. Finding optimal stride-8 and stride-16 feature maps for multi-scale embeddings
3. Pruning the model to keep only selected outputs (removing NMS)

Input: Base ONNX model (e.g., ppyoloe_crn_s_36e_pphuman.onnx)
Output: Customized ONNX model (e.g., ppyoloe_crn_s_36e_pphuman_cust.onnx)
"""
import os
import sys
import argparse
from typing import Tuple, List, Dict, Set, Any
import numpy as np
import onnx
from onnx import helper, numpy_helper

try:
    import onnxruntime as ort
except Exception:
    ort = None


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
    - NMS inputs (boxes, scores)
    - Stride-8 feature map for fine-grained embeddings
    - Stride-16 feature map for semantic embeddings

    Returns dict with discovered outputs and their metadata.
    """
    if verbose:
        print("\n=== Automatic Output Discovery ===")

    # 1. Find NMS inputs (boxes, scores)
    nms_info = find_nms_nodes(model_path)
    if not nms_info:
        print("⚠️  No NMS nodes found - will only discover feature maps")
    elif verbose:
        print(f"✓ Found NMS inputs:")
        print(f"  Boxes:  {nms_info.get('boxes', 'N/A')}")
        print(f"  Scores: {nms_info.get('scores', 'N/A')}")

    # 2. Find stride-8 and stride-16 feature maps
    if verbose:
        print(f"\n✓ Probing feature maps (input size: {img_hw})...")

    m = onnx.load(model_path)

    # Find candidate feature nodes (Conv, BN outputs)
    candidates = []
    for node in m.graph.node:
        if node.op_type in ['Conv', 'BatchNormalization']:
            for out in node.output:
                if not any(bad in out for bad in ['.w_', '.b_', 'constant', 'scale', 'bias']):
                    candidates.append((out, node.op_type))

    # Probe candidates with runtime inference
    probed = []
    H, W = img_hw

    for i, (name, op_type) in enumerate(candidates[:80]):  # Limit to 80 probes
        try:
            # Create temp model with extra output
            m_temp = onnx.load(model_path)
            found = False
            for vi in list(m_temp.graph.value_info) + list(m_temp.graph.output):
                if vi.name == name:
                    m_temp.graph.output.append(vi)
                    found = True
                    break

            if not found:
                vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [])
                m_temp.graph.output.append(vi)

            # Save and run
            temp_path = f'/tmp/probe_{i}.onnx'
            onnx.save(m_temp, temp_path)

            sess = ort.InferenceSession(temp_path, providers=['CPUExecutionProvider'])

            # Create dummy input
            feed = {}
            for inp in sess.get_inputs():
                if 'image' in inp.name.lower():
                    feed[inp.name] = np.random.randn(1, 3, H, W).astype(np.float32)
                elif 'shape' in inp.name.lower():
                    feed[inp.name] = np.array([[H, W]], dtype=np.float32)
                elif 'scale' in inp.name.lower():
                    feed[inp.name] = np.array([[1.0, 1.0]], dtype=np.float32)

            # Run inference
            outputs = sess.run(None, feed)

            # Find the probed output
            out_names = [o.name for o in sess.get_outputs()]
            if name in out_names:
                idx = out_names.index(name)
                shape = list(outputs[idx].shape)

                # Only keep 4D feature maps
                if len(shape) == 4 and shape[0] == 1:
                    _, C, Hf, Wf = shape
                    if 10 <= Hf <= 80 and 10 <= Wf <= 80 and 64 <= C <= 512:
                        stride_h = H / Hf
                        stride_w = W / Wf
                        stride_avg = (stride_h + stride_w) / 2
                        probed.append({
                            'name': name,
                            'shape': shape,
                            'channels': C,
                            'spatial': (Hf, Wf),
                            'stride': stride_avg,
                            'op': op_type
                        })
        except:
            pass

    # Categorize by stride
    stride8_candidates = [f for f in probed if 6 <= f['stride'] <= 10]
    stride16_candidates = [f for f in probed if 12 <= f['stride'] <= 20]

    if verbose:
        print(f"  Found {len(probed)} suitable feature maps")
        print(f"  Stride-8 candidates: {len(stride8_candidates)}")
        print(f"  Stride-16 candidates: {len(stride16_candidates)}")

    # Pick best candidates (prefer BatchNormalization outputs, higher channels)
    def select_best(candidates):
        if not candidates:
            return None
        # Prefer BN, then higher channels
        bn_cands = [c for c in candidates if c['op'] == 'BatchNormalization']
        pool = bn_cands if bn_cands else candidates
        return max(pool, key=lambda c: c['channels'])

    best_s8 = select_best(stride8_candidates)
    best_s16 = select_best(stride16_candidates)

    # Build result
    result = {
        'nms': nms_info,
        'stride_8': best_s8,
        'stride_16': best_s16,
    }

    if verbose:
        print("\n✓ Selected feature maps:")
        if best_s8:
            print(f"  Stride-8:  {best_s8['name']}")
            print(f"             Shape: {best_s8['shape']} ({best_s8['channels']} channels)")
        else:
            print(f"  Stride-8:  Not found")

        if best_s16:
            print(f"  Stride-16: {best_s16['name']}")
            print(f"             Shape: {best_s16['shape']} ({best_s16['channels']} channels)")
        else:
            print(f"  Stride-16: Not found")

    return result


def prune_outputs(model_path: str, out_path: str, keep_outputs: List[str]) -> str:
    """Keep only a subset of outputs and prune unreachable nodes using custom DFS traversal."""
    if not keep_outputs:
        print("⚠️  No outputs specified to keep, skipping pruning")
        return model_path

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
                # Visit all inputs of this node
                node = g.node[node_idx]
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
        existing_vi = None
        # Search in graph outputs first
        for vi in g.output:
            if vi.name == out_name:
                existing_vi = vi
                break
        # Then search in value_info
        if not existing_vi:
            for vi in g.value_info:
                if vi.name == out_name:
                    existing_vi = vi
                    break
        # Create minimal ValueInfoProto if not found
        if existing_vi:
            new_outputs.append(existing_vi)
        else:
            new_vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, [])
            new_outputs.append(new_vi)

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
        print("⚠️  Warning: NMS node still present after pruning (unreachable from kept outputs)")

    return out_path


def print_output_guide(discovered: Dict[str, Any], keep_outputs: List[str]) -> None:
    """Print comprehensive guide for using the customized model outputs."""
    print("\n" + "="*70)
    print("=== Customized Model Output Guide ===")
    print("="*70)

    nms = discovered.get('nms', {})
    s8 = discovered.get('stride_8')
    s16 = discovered.get('stride_16')

    # Map output indices
    for i, out_name in enumerate(keep_outputs):
        print(f"\nOutput {i}: '{out_name}'")

        # Identify what this output is
        if nms.get('boxes') == out_name:
            print("  Type: Detection boxes (raw)")
            print("  Shape: (1, 8400, 4) or similar")
            print("  Format: [x_center, y_center, width, height] in input coordinates")
            print("  Usage: Apply NMS with scores to get final detections")
            print("  Note: These are PRE-NMS boxes from all anchor points")

        elif nms.get('scores') == out_name:
            print("  Type: Detection scores (raw)")
            print("  Shape: (1, 1, 8400) - may need squeeze to (8400,)")
            print("  Format: Class probabilities (single class: person)")
            print("  Usage: Threshold and apply NMS with boxes")
            print("  Note: These are confidence scores for each anchor point")

        elif s8 and s8['name'] == out_name:
            print("  Type: Feature map (stride-8, fine-grained)")
            print(f"  Shape: {s8['shape']}")
            print(f"  Channels: {s8['channels']}")
            print(f"  Spatial: {s8['spatial'][0]}×{s8['spatial'][1]} (stride≈{s8['stride']:.1f})")
            print("  Usage: Pass as 'feat_s8' to roi_align_pool_multi_scale()")
            print("  Purpose: Fine-grained spatial features for small person instances")
            print("  Note: Higher resolution, good for precise localization")

        elif s16 and s16['name'] == out_name:
            print("  Type: Feature map (stride-16, semantic)")
            print(f"  Shape: {s16['shape']}")
            print(f"  Channels: {s16['channels']}")
            print(f"  Spatial: {s16['spatial'][0]}×{s16['spatial'][1]} (stride≈{s16['stride']:.1f})")
            print("  Usage: Pass as 'feat_s16' to roi_align_pool_multi_scale()")
            print("  Purpose: Semantic features for appearance discrimination")
            print("  Note: Lower resolution, stronger semantic information")

        else:
            print("  Type: Unknown (custom output)")

    # Print embedding info if both feature maps present
    if s8 and s16 and any(s8['name'] == o for o in keep_outputs) and any(s16['name'] == o for o in keep_outputs):
        total_dim = s8['channels'] + s16['channels']
        print(f"\n" + "-"*70)
        print("=== Multi-Scale Embedding Extraction ===")
        print(f"Total embedding dimension: {total_dim} ({s8['channels']} from s8 + {s16['channels']} from s16)")
        print("Recommended pooling config:")
        print("  - Global pooling weight: 0.2 (mix avg/max)")
        print("  - Part pooling weight: 0.8")
        print("  - Horizontal parts: 9 divisions × 2 stripes")
        print("  - Vertical parts: 2 divisions × 2 stripes")
        print("Normalization: InstanceNorm → power-law (α=0.35) → L2")
        print("Expected quality: median cosine < 0.15, p95 < 0.35")

    # Print code template
    print(f"\n" + "-"*70)
    print("=== Python Inference Template ===")
    print("-"*70)
    print("import onnxruntime as ort")
    print("import numpy as np")
    print("")
    print("# Load model")
    print("sess = ort.InferenceSession('model_cust.onnx', providers=['CPUExecutionProvider'])")
    print("")
    print("# Run inference")
    print("outputs = sess.run(None, {'image': img_tensor, ...})")
    print("")

    # Generate specific code based on discovered outputs
    for i, out_name in enumerate(keep_outputs):
        if nms.get('boxes') == out_name:
            print(f"boxes_raw = outputs[{i}]  # Shape: (1, N, 4)")
        elif nms.get('scores') == out_name:
            print(f"scores_raw = outputs[{i}].squeeze()  # Shape: (N,)")
        elif s8 and s8['name'] == out_name:
            print(f"feat_s8 = outputs[{i}]  # Shape: {s8['shape']}")
        elif s16 and s16['name'] == out_name:
            print(f"feat_s16 = outputs[{i}]  # Shape: {s16['shape']}")

    print("")
    print("# Apply NMS (if using raw boxes/scores)")
    if nms.get('boxes') in keep_outputs and nms.get('scores') in keep_outputs:
        print("import cv2")
        print("indices = cv2.dnn.NMSBoxes(boxes_raw[0], scores_raw, score_threshold=0.3, nms_threshold=0.5)")
        print("boxes_nms = boxes_raw[0][indices]")
        print("scores_nms = scores_raw[indices]")

    print("")
    if s8 and s16 and any(s8['name'] == o for o in keep_outputs) and any(s16['name'] == o for o in keep_outputs):
        print("# Extract embeddings")
        print("embeddings = roi_align_pool_multi_scale(")
        print("    feat_s8=feat_s8,")
        print("    feat_s16=feat_s16,")
        print("    boxes=boxes_nms,")
        print("    img_hw=(640, 640),")
        print("    gp_w=0.2, pp_w=0.8, pp_k=9, pp_stripe_h=2")
        print(")")
        if s8 and s16:
            total_dim = s8['channels'] + s16['channels']
            print(f"# Result: embeddings.shape = (num_detections, {total_dim})")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Customize PP-YOLOE ONNX model by removing NMS and adding feature outputs")
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
        help="Path to output customized ONNX model",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,640,640",
        help="Input shape for probing feature maps (default: 1,3,640,640)",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Automatically discover NMS inputs and feature maps",
    )
    parser.add_argument(
        "--keep-outputs",
        type=str,
        help="Comma-separated list of output names to keep (overrides auto-discovery)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)

    # Parse input shape
    try:
        shape_parts = [int(x) for x in args.input_shape.split(',')]
        if len(shape_parts) != 4:
            raise ValueError("Input shape must be N,C,H,W")
        img_hw = (shape_parts[2], shape_parts[3])
    except Exception as e:
        print(f"❌ Error parsing input shape: {e}")
        sys.exit(1)

    # Determine which outputs to keep
    if args.keep_outputs:
        # User-specified outputs
        keep_outputs = [x.strip() for x in args.keep_outputs.split(',')]
        discovered = {}
        print(f"Using user-specified outputs: {keep_outputs}")
    elif args.auto_discover:
        # Auto-discover outputs
        print("Auto-discovering optimal outputs...")
        discovered = auto_discover_outputs(args.model, img_hw=img_hw, verbose=True)

        # Build keep_outputs list
        keep_outputs = []
        nms = discovered.get('nms', {})
        if nms.get('boxes'):
            keep_outputs.append(nms['boxes'])
        if nms.get('scores'):
            keep_outputs.append(nms['scores'])

        s8 = discovered.get('stride_8')
        if s8:
            keep_outputs.append(s8['name'])

        s16 = discovered.get('stride_16')
        if s16:
            keep_outputs.append(s16['name'])

        if not keep_outputs:
            print("❌ Error: No outputs discovered")
            sys.exit(1)
    else:
        print("❌ Error: Must specify either --auto-discover or --keep-outputs")
        sys.exit(1)

    # Prune model
    print(f"\n[stage] Pruning model to keep outputs: {keep_outputs}")
    prune_outputs(args.model, args.output, keep_outputs)

    print(f"\n✓ Customized model saved to: {args.output}")

    # Print usage guide
    if discovered:
        print_output_guide(discovered, keep_outputs)


if __name__ == "__main__":
    main()
