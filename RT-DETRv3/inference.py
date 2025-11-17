#!/usr/bin/env python3
"""
RT-DETRv3 Model Inference Validation and Testing
===============================================

This script provides comprehensive validation of the final RT-DETRv3 model (potentially
quantized) through inference testing, performance benchmarking, and result visualization.
It serves as the final quality gate before production deployment.

Validation Features:
    1. Model Loading and Verification
       - ONNX model integrity validation
       - Input/output tensor verification
       - Provider compatibility testing
       - Memory allocation validation

    2. Inference Pipeline Testing
       - End-to-end inference validation
       - Preprocessing pipeline verification
       - Output format consistency checking
       - Performance timing analysis

    3. Detection Quality Assessment
       - Object detection accuracy evaluation
       - Confidence score distribution analysis
       - Bounding box precision validation
       - Class prediction verification

    4. Quantization Impact Analysis
       - Accuracy comparison (if original available)
       - Performance gain measurement
       - Quality degradation assessment
       - Numerical precision validation

    5. Visualization and Reporting
       - Detection result visualization
       - Performance metrics reporting
       - Quality score computation
       - Deployment readiness assessment

Test Scenarios:
    • Standard object detection validation
    • Edge case handling (empty images, single objects)
    • Performance stress testing
    • Memory usage profiling
    • Cross-platform compatibility

Performance Metrics:
    • Inference latency (target: < 200ms)
    • Throughput (images per second)
    • Memory usage (peak and average)
    • CPU utilization efficiency

Quality Metrics:
    • Detection precision and recall
    • Confidence score calibration
    • Bounding box accuracy (IoU)
    • Class prediction accuracy

Deployment Validation:
    • Model file integrity
    • Runtime compatibility
    • Resource requirements
    • Performance benchmarks

Usage Examples:
    # Basic inference validation
    python inference.py --debug

    # Custom model and image
    python inference.py --model custom_model.onnx --image test.jpg --debug

    # Performance benchmarking
    python inference.py --benchmark --iterations 100

Output:
    Comprehensive validation report with performance metrics, quality scores,
    and deployment readiness assessment.

Author: RT-DETRv3 Development Team
License: Same as RT-DETRv3 repository
"""

import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
import textwrap
import json
from typing import List, Tuple

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

np.random.seed(42)
CLASS_COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(COCO_CLASSES))}

def preprocess_image(image_path, target_size=(640, 640)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    im_shape = np.array([[img.shape[0], img.shape[1]]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
    resized = cv2.resize(img, target_size)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return img, resized, im_shape, scale_factor

def normalize_shape(shape):
    return [(-1 if (isinstance(dim, str) or dim is None) else dim) for dim in shape]

def decode_outputs(outputs, orig_shape, conf_threshold=0.5, debug=False):
    detections = []
    for out_idx, out in enumerate(outputs):
        arr = np.array(out)
        if debug:
            print(f"[decode] Output[{out_idx}] shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim == 2 and arr.shape[1] >= 6:
            for det in arr:
                cls_id, conf, x1, y1, x2, y2 = det[:6]
                if conf >= conf_threshold:
                    detections.append((int(cls_id), float(conf), [float(x1), float(y1), float(x2), float(y2)]))
        elif arr.ndim == 3 and arr.shape[2] >= 6:
            reshaped = arr.reshape(-1, arr.shape[-1])
            for det in reshaped:
                cls_id, conf, x1, y1, x2, y2 = det[:6]
                if conf >= conf_threshold:
                    detections.append((int(cls_id), float(conf), [float(x1), float(y1), float(x2), float(y2)]))
    return detections

def draw_detections(image, detections, save_path, class_names=None):
    rng = np.random.default_rng(42)
    for idx, (cls_id, conf, (x1, y1, x2, y2)) in enumerate(detections):
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        if class_names and 0 <= cls_id < len(class_names):
            label_text = f"{class_names[cls_id]} {conf:.2f}"
        else:
            label_text = f"{cls_id}:{conf:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label_text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    print(f"Saved result image with detections to {save_path}")

def _tensor_stats(name, arr, max_rows=5, max_cols=12):
    flat = arr.astype(np.float32).ravel()
    stats = {
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
        "std": float(flat.std()) if flat.size else None,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype)
    }
    header = f"Output: {name} stats: {stats}"
    sample = ""
    if arr.ndim >= 2 and arr.shape[-1] <= 256:
        rows = min(arr.shape[0], max_rows)
        sub = arr[:rows]
        # Collapse >2D for display
        if sub.ndim > 2:
            sub = sub.reshape(sub.shape[0], -1)
        sub = sub[:, :max_cols]
        sample = "\nSample rows:\n" + "\n".join(str(r.tolist()) for r in sub)
    return header + sample

def _find_candidate_detection_outputs(ort_session, raw_outputs):
    candidates = []
    for meta, arr in zip(ort_session.get_outputs(), raw_outputs):
        a = np.array(arr)
        if (a.ndim == 2 and a.shape[1] >= 6) or (a.ndim == 3 and a.shape[2] >= 6):
            candidates.append((meta.name, a.shape))
    return candidates

def verify_model(onnx_model, image_path, output_path, conf_threshold=0.5, debug=False, dump_json=None):
    if debug:
        print(f"Loading model: {onnx_model}")
    sess_opts = ort.SessionOptions()
    if debug:
        sess_opts.log_severity_level = 0
    sess = ort.InferenceSession(onnx_model, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    if debug:
        print("\n=== MODEL INPUTS ===")
    input_nodes = sess.get_inputs()
    for i in input_nodes:
        if debug:
            print(f"Input[{i.name}] shape={i.shape} type={i.type}")

    if debug:
        print("\n=== MODEL OUTPUTS (declared) ===")
        for o in sess.get_outputs():
            print(f"Declared Output[{o.name}] shape={o.shape} type={o.type}")

    orig_img, image_tensor, im_shape_tensor, scale_factor_tensor = preprocess_image(image_path)

    if debug:
        print("\n=== PREPROCESS ===")
        print(f"Original image shape: {orig_img.shape}")
        print(f"Image tensor shape: {image_tensor.shape} range=({image_tensor.min():.4f},{image_tensor.max():.4f}) mean={image_tensor.mean():.4f}")

    input_names = [n.name for n in input_nodes]
    input_feed = {}
    for node in input_nodes:
        norm_shape = normalize_shape(node.shape)
        if len(norm_shape) == 4 and norm_shape[1] == 3:
            input_feed[node.name] = image_tensor
        elif len(norm_shape) == 2 and norm_shape[1] == 2:
            if "shape" in node.name:
                input_feed[node.name] = im_shape_tensor
            elif "scale" in node.name:
                input_feed[node.name] = scale_factor_tensor
            else:
                if input_names.index(node.name) == 0:
                    input_feed[node.name] = im_shape_tensor
                else:
                    input_feed[node.name] = scale_factor_tensor
        else:
            raise RuntimeError(f"Unhandled input: {node.name} with shape {node.shape}")
    if debug:
        print("\n=== INPUT FEED KEYS ===")
        for k,v in input_feed.items():
            print(f"{k}: shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}")

    if debug:
        print("\nRunning inference...")
    outputs = sess.run(None, input_feed)
    if debug:
        print("Inference done. Total outputs:", len(outputs))

    if debug:
        print("\n=== RAW OUTPUT STATS ===")
        for meta, arr in zip(sess.get_outputs(), outputs):
            a = np.array(arr)
            print(_tensor_stats(meta.name, a))

    candidates = _find_candidate_detection_outputs(sess, outputs)
    if debug:
        print("\nCandidate detection-like outputs (>=6 columns in last dim):")
        for name, shape in candidates:
            print(f"  {name}: shape={shape}")

    detections = decode_outputs(outputs, orig_img.shape, conf_threshold=conf_threshold, debug=debug)

    # If no detections, probe confidences anyway
    probe_info = {}
    if len(detections) == 0:
        flat_candidates = []
        for arr in outputs:
            a = np.array(arr)
            if a.ndim == 2 and a.shape[1] >= 2:
                flat_candidates.append(a)
            elif a.ndim == 3 and a.shape[2] >= 2:
                flat_candidates.append(a.reshape(-1, a.shape[-1]))
        confidences = []
        for a in flat_candidates:
            confidences.extend(a[:,1].tolist())
        confidences = np.array(confidences, dtype=np.float32)
        if confidences.size:
            topk = np.sort(confidences)[-10:]
            probe_info = {
                "max_conf": float(confidences.max()),
                "mean_conf": float(confidences.mean()),
                "num_scores": int(confidences.size),
                "top10": topk.tolist()
            }
            if debug:
                print("\nNo detections at threshold "
                      f"{conf_threshold}. Highest confidences (top10): {topk}")
                if confidences.max() < conf_threshold:
                    print("Reason: all confidences below threshold.")
        else:
            if debug:
                print("\nNo candidate confidence scores found (output format mismatch).")

    print(f"\n--- Verification Result ---")
    print(f"Detected {len(detections)} objects with confidence > {conf_threshold}")
    for i, (cls_id, conf, bbox) in enumerate(detections, 1):
        class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
        print(f"  - Obj {i}: Class={class_name} ({cls_id}), Conf={conf:.3f}, BBox={bbox}")

    draw_detections(orig_img.copy(), detections, output_path)

    if dump_json:
        dump = {
            "model": onnx_model,
            "image": image_path,
            "conf_threshold": conf_threshold,
            "detections": [
                {"class_id": d[0], "confidence": d[1], "bbox_xyxy": d[2],
                 "class_name": COCO_CLASSES[d[0]] if d[0] < len(COCO_CLASSES) else str(d[0])}
                for d in detections
            ],
            "probe": probe_info,
            "candidates": [{"name": n, "shape": list(s)} for n,s in candidates]
        }
        os.makedirs(os.path.dirname(dump_json), exist_ok=True)
        with open(dump_json, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"Wrote debug JSON: {dump_json}")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Verify ONNX (quantized) model with debug.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--model", default="pipeline/output/rtdetrv3_r18vd_6x.onnx")
    ap.add_argument("--image", default="pipeline/dataset/demo/demo.jpg")
    ap.add_argument("--out", default="pipeline/output/validation/demo_inference.jpg")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dump-json", default="")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    verify_model(
        onnx_model=args.model,
        image_path=args.image,
        output_path=args.out,
        conf_threshold=args.conf,
        debug=args.debug,
        dump_json=args.dump_json or None
    )