#!/usr/bin/env python3
"""Customize PP-YOLOE ONNX model by exposing feature maps and removing NMS.

This script takes the base exported ONNX model and produces a customized variant
that keeps the raw detection tensors (boxes/scores) and exposes stride-8 and
stride-16 feature maps as graph outputs. The customized model is the expected
input for the ANE surgery stage.

Workflow reminder:
1. export_to_onnx.sh → base.onnx
2. onnx_customize.py → base_cust.onnx (this script)
3. ane_graph_surgery.py → base_cust_ane.onnx
4. onnx_cleanup.py → base_cust_ane_cu.onnx
"""
import argparse
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Set, Tuple, Any

import numpy as np
import onnx
from onnx import helper

try:
    import onnxruntime as ort
except Exception:
    ort = None

from onnx_profile import load_model_info, parse_shape


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


def prune_outputs(model_path: str, out_path: str, keep_outputs: List[str]) -> str:
    """Keep only a subset of outputs and prune unreachable nodes using DFS."""
    if not keep_outputs:
        shutil.copyfile(model_path, out_path)
        return out_path

    m = onnx.load(model_path)
    g = m.graph
    print(f"  Pruning to keep outputs: {keep_outputs}")

    tensor_producers: Dict[str, int] = {}
    for idx, node in enumerate(g.node):
        for out in node.output:
            tensor_producers[out] = idx

    reachable_node_indices: Set[int] = set()
    visited_tensors: Set[str] = set()

    def visit(tensor_name: str):
        if tensor_name in visited_tensors:
            return
        visited_tensors.add(tensor_name)
        if tensor_name in tensor_producers:
            node_idx = tensor_producers[tensor_name]
            if node_idx not in reachable_node_indices:
                reachable_node_indices.add(node_idx)
                node = g.node[node_idx]
                for inp in node.input:
                    if inp:
                        visit(inp)

    for out_name in keep_outputs:
        visit(out_name)

    print(f"  Reachable nodes: {len(reachable_node_indices)} / {len(g.node)}")
    kept_nodes = [g.node[i] for i in sorted(reachable_node_indices)]

    live_tensors: Set[str] = set()
    for node in kept_nodes:
        live_tensors.update([inp for inp in node.input if inp])
        live_tensors.update([out for out in node.output if out])

    graph_input_names = {vi.name for vi in g.input}
    init_names = {init.name for init in g.initializer}
    live_tensors |= graph_input_names | init_names

    kept_inits = [init for init in g.initializer if init.name in live_tensors]
    kept_vi = [vi for vi in g.value_info if vi.name in live_tensors]

    new_outputs = []
    for out_name in keep_outputs:
        found_vi = None
        for vi in list(g.output) + list(g.value_info):
            if vi.name == out_name:
                found_vi = vi
                break
        if found_vi is None:
            vi = onnx.ValueInfoProto()
            vi.name = out_name
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            new_outputs.append(vi)
            print(f"  [WARNING] Created minimal ValueInfo for output '{out_name}'")
        else:
            new_outputs.append(found_vi)

    del g.node[:]
    g.node.extend(kept_nodes)
    del g.initializer[:]
    g.initializer.extend(kept_inits)
    del g.value_info[:]
    g.value_info.extend(kept_vi)
    del g.output[:]
    g.output.extend(new_outputs)

    onnx.save(m, out_path)

    m_pruned = onnx.load(out_path)
    actual_outputs = [o.name for o in m_pruned.graph.output]
    print(f"  Pruned model outputs: {actual_outputs}")
    has_nms = any(n.op_type == 'NonMaxSuppression' for n in m_pruned.graph.node)
    print(f"  Has NMS nodes: {has_nms}")
    if has_nms:
        print("  [WARNING] NMS nodes still present - they may be reachable from kept outputs!")

    return out_path


def find_nms_nodes(model_path: str) -> Dict[str, Any]:
    try:
        m = onnx.load(model_path)
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}", file=sys.stderr)
        return {}

    nms_nodes = [node for node in m.graph.node if node.op_type == 'NonMaxSuppression']
    if not nms_nodes:
        return {}

    node = nms_nodes[0]
    input_tensors = [inp for inp in node.input if inp]
    result: Dict[str, Any] = {}
    if len(input_tensors) >= 2:
        result['boxes'] = input_tensors[0]
        result['scores'] = input_tensors[1]
    return result


def auto_discover_outputs(model_path: str, img_hw: Tuple[int, int], probe_limit: int = 80, verbose: bool = True) -> Dict[str, Any]:
    if ort is None:
        raise RuntimeError("onnxruntime is required for automatic discovery")

    if verbose:
        print("\n=== Automatic Output Discovery ===")

    nms_info = find_nms_nodes(model_path)
    if not nms_info:
        if verbose:
            print("⚠️  No NMS nodes found - will only discover feature maps")
    elif verbose:
        print("✓ Found NMS inputs:")
        print(f"  Boxes:  {nms_info.get('boxes', 'N/A')}")
        print(f"  Scores: {nms_info.get('scores', 'N/A')}")

    if verbose:
        print(f"\n✓ Probing feature maps (input size: {img_hw})…")

    m = onnx.load(model_path)
    candidates = []
    for node in m.graph.node:
        if node.op_type in ['Conv', 'BatchNormalization']:
            for out in node.output:
                if not any(token in out for token in ['.w_', '.b_', 'constant', 'scale', 'bias']):
                    candidates.append((out, node.op_type))

    probed: List[Dict[str, Any]] = []
    H, W = img_hw

    for idx, (name, op_type) in enumerate(candidates[: max(0, probe_limit)]):
        temp_path = None
        try:
            m_temp = onnx.load(model_path)
            found = False
            for vi in list(m_temp.graph.value_info) + list(m_temp.graph.output):
                if vi.name == name:
                    m_temp.graph.output.append(vi)
                    found = True
                    break
            if not found:
                vi = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [])
                m_temp.graph.output.append(vi)

            fd, temp_path = tempfile.mkstemp(suffix="_probe.onnx")
            os.close(fd)
            onnx.save(m_temp, temp_path)

            sess = ort.InferenceSession(temp_path, providers=['CPUExecutionProvider'])
            feed = {}
            for inp in sess.get_inputs():
                if 'image' in inp.name.lower():
                    feed[inp.name] = np.random.randn(1, 3, H, W).astype(np.float32)
                elif 'shape' in inp.name.lower():
                    feed[inp.name] = np.array([[H, W]], dtype=np.float32)
                elif 'scale' in inp.name.lower():
                    feed[inp.name] = np.array([[1.0, 1.0]], dtype=np.float32)

            outputs = sess.run(None, feed)
            out_names = [o.name for o in sess.get_outputs()]
            if name in out_names:
                out_idx = out_names.index(name)
                shape = list(outputs[out_idx].shape)
                if len(shape) == 4 and shape[0] == 1:
                    _, C, Hf, Wf = shape
                    if 10 <= Hf <= 80 and 10 <= Wf <= 80 and 64 <= C <= 512:
                        stride_h = H / Hf
                        stride_w = W / Wf
                        probed.append({
                            'name': name,
                            'shape': shape,
                            'channels': C,
                            'spatial': (Hf, Wf),
                            'stride': (stride_h + stride_w) / 2,
                            'op': op_type,
                        })
        except Exception:
            pass
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    stride8_candidates = [f for f in probed if 6 <= f['stride'] <= 10]
    stride16_candidates = [f for f in probed if 12 <= f['stride'] <= 20]

    if verbose:
        print(f"  Found {len(probed)} suitable feature maps")
        print(f"  Stride-8 candidates: {len(stride8_candidates)}")
        print(f"  Stride-16 candidates: {len(stride16_candidates)}")

    def select_best(candidates: List[Dict[str, Any]]):
        if not candidates:
            return None
        bn_candidates = [c for c in candidates if c['op'] == 'BatchNormalization']
        pool = bn_candidates if bn_candidates else candidates
        return max(pool, key=lambda c: c['channels'])

    best_s8 = select_best(stride8_candidates)
    best_s16 = select_best(stride16_candidates)

    if verbose:
        print("\n✓ Selected feature maps:")
        if best_s8:
            print(f"  Stride-8:  {best_s8['name']}")
            print(f"             Shape: {best_s8['shape']} ({best_s8['channels']} channels)")
        else:
            print("  Stride-8:  Not found")
        if best_s16:
            print(f"  Stride-16: {best_s16['name']}")
            print(f"             Shape: {best_s16['shape']} ({best_s16['channels']} channels)")
        else:
            print("  Stride-16: Not found")

    return {
        'nms': nms_info,
        'stride_8': best_s8,
        'stride_16': best_s16,
    }


def print_output_guide(discovered: Dict[str, Any], keep_outputs: List[str]) -> None:
    print("\n" + "=" * 70)
    print("=== Customized Model Output Guide ===")
    print("=" * 70)

    nms = discovered.get('nms', {}) if discovered else {}
    s8 = discovered.get('stride_8') if discovered else None
    s16 = discovered.get('stride_16') if discovered else None

    for idx, out_name in enumerate(keep_outputs):
        print(f"\nOutput {idx}: '{out_name}'")
        if nms.get('boxes') == out_name:
            print("  Type: Detection boxes (raw)")
            print("  Usage: Apply custom NMS downstream")
        elif nms.get('scores') == out_name:
            print("  Type: Detection scores (raw)")
            print("  Usage: Threshold and pair with boxes before NMS")
        elif s8 and s8['name'] == out_name:
            print("  Type: Feature map (stride-8)")
            print(f"  Shape: {s8['shape']}")
            print("  Purpose: Fine-grained spatial embeddings")
        elif s16 and s16['name'] == out_name:
            print("  Type: Feature map (stride-16)")
            print(f"  Shape: {s16['shape']}")
            print("  Purpose: Semantic embeddings")
        else:
            print("  Type: Custom intermediate tensor")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Customize PP-YOLOE ONNX by exposing raw detections and features")
    parser.add_argument("--model", required=True, help="Path to the base ONNX model")
    parser.add_argument("--input-shape", type=str, default="1,3,640,640", help="Model input shape (N,C,H,W)")
    parser.add_argument("--outdir", type=str, default="PP-YOLOE/build/models/custom", help="Output directory")
    parser.add_argument("--output-model", type=str, help="Explicit output path for customized model")
    parser.add_argument("--keep-output", action="append", dest="keep_outputs", help="Extra output to keep (repeatable)")
    parser.add_argument("--skip-auto-discovery", action="store_true", help="Skip automatic discovery; requires --keep-output")
    parser.add_argument("--probe-limit", type=int, default=80, help="Limit of feature probes for auto discovery")
    parser.add_argument("--no-guide", action="store_true", help="Skip printing the output usage guide")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] ONNX model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    ishape = parse_shape(args.input_shape)
    img_hw = (640, 640)
    if len(ishape) == 4:
        img_hw = (int(ishape[2]), int(ishape[3]))

    keep: List[str] = []
    discovered_info: Dict[str, Any] = {}

    if args.keep_outputs:
        keep.extend(args.keep_outputs)

    if not keep and not args.skip_auto_discovery:
        try:
            discovered_info = auto_discover_outputs(
                args.model, img_hw=img_hw, probe_limit=args.probe_limit, verbose=True
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
        nms = discovered_info.get('nms', {})
        if nms.get('boxes'):
            keep.append(nms['boxes'])
        if nms.get('scores'):
            keep.append(nms['scores'])
        if discovered_info.get('stride_8'):
            keep.append(discovered_info['stride_8']['name'])
        if discovered_info.get('stride_16'):
            keep.append(discovered_info['stride_16']['name'])

    keep = list(dict.fromkeys(keep))  # preserve order, drop duplicates

    if not keep:
        print("[ERROR] No outputs specified. Provide --keep-output or enable auto discovery.", file=sys.stderr)
        sys.exit(1)

    base_name = os.path.basename(args.model)
    mod_pruned = os.path.join(args.outdir, base_name.replace('.onnx', '_cust_pruned.onnx'))
    work_path = prune_outputs(args.model, mod_pruned, keep)

    mod_shape = os.path.join(args.outdir, base_name.replace('.onnx', '_cust_shape.onnx'))
    work_path = shape_infer_model(work_path, mod_shape)

    if args.output_model:
        final_path = args.output_model
        os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    else:
        final_path = os.path.join(args.outdir, base_name.replace('.onnx', '_cust.onnx'))

    try:
        shutil.copyfile(work_path, final_path)
        print(f"[stage] Copied customized model to: {final_path}")
    except Exception as e:
        print(f"[WARNING] Failed to copy customized model: {e}")
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
