#!/usr/bin/env python3
"""
Run object detection + ReID inference on a single image using ONNX Runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onnx_inference_ane_model import run_ane_inference
from onnx_inference_original_model import run_original_inference
from onnx_inference_utils import (
    default_paths,
    detect_model_type,
    get_hardcoded_preprocess,
    preprocess_image,
    get_coreml_version_info,
)

def get_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(
            "[ERROR] onnxruntime not installed. Install with: pip install onnxruntime",
            file=sys.stderr,
        )
        raise exc

    available = ort.get_available_providers()
    providers = None
    if "CoreMLExecutionProvider" in available:
        coreml_opts = {
            "ModelFormat": "MLProgram",
            "EnableOnSubgraphs": "1",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
        }
        providers = [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"]
        print(f"[INFO] Loading {os.path.basename(onnx_path)} with CoreMLExecutionProvider...")
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        except Exception as exc:
            print("[ERROR] Failed to initialize CoreMLExecutionProvider session:", exc, file=sys.stderr)
            raise
    else:
        print(f"[INFO] Loading {os.path.basename(onnx_path)} with default providers: {available}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess


def build_feed_dict(sess, inputs_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    feed = {}
    input_names = [i.name for i in sess.get_inputs()]
    for name in input_names:
        if name == "image":
            feed[name] = inputs_map["image"][None, :]
        elif name in inputs_map:
            feed[name] = inputs_map[name][None, :]
        else:
            # Some models might have optional inputs or different naming
            pass
    return feed

def preprocess_reid(img_crop: np.ndarray, target_size: Tuple[int, int] = (256, 128)) -> np.ndarray:
    """
    Preprocess a crop for ReID model.
    Standard PaddleClas ReID preprocessing:
    - Resize to (H=256, W=128)
    - BGR to RGB
    - Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    - NCHW layout
    """
    # Resize (cv2.resize takes (W, H))
    img = cv2.resize(img_crop, (target_size[1], target_size[0]))

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    # Add batch dimension: NCHW
    img = img[np.newaxis, :].astype(np.float32)
    return img

def main():
    # Default paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_det_onnx = os.path.join(root, "PP-YOLOE", "build", "models", "ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx")
    default_reid_onnx = os.path.join(root, "PP-YOLOE", "build", "models", "human_reid.onnx")
    default_img = os.path.join(root, "PP-YOLOE", "build", "dataset", "demo", "demo.jpg")
    default_out = os.path.join(root, "PP-YOLOE", "build", "output")

    parser = argparse.ArgumentParser(description="ONNX Runtime inference for Object Detection + ReID")
    parser.add_argument("--img", default=default_img, help="Path to input image")
    parser.add_argument("--det_onnx", default=default_det_onnx, help="Path to Detection ONNX model file")
    parser.add_argument("--reid_onnx", default=default_reid_onnx, help="Path to ReID ONNX model file")
    parser.add_argument("--out", default=default_out, help="Directory to save visualization")
    parser.add_argument("--thresh", type=float, default=0.5, help="Score threshold for detection (default: 0.5)")
    args = parser.parse_args()

    # 1. Check files
    for path, label in ((args.img, "Input image"), (args.det_onnx, "Detection model"), (args.reid_onnx, "ReID model")):
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # 2. Load Detection Model
    print("\n[1/3] Loading Detection Model...")
    det_sess = get_session(args.det_onnx)

    # 3. Run Detection
    print("\n[2/3] Running Detection...")
    draw_threshold = args.thresh

    # Preprocess for detection
    inputs_map = preprocess_image(args.img, target_size=(640, 640), keep_ratio=False)
    feed = build_feed_dict(det_sess, inputs_map)

    # Detect model type and run inference
    out_names = [o.name for o in det_sess.get_outputs()]
    model_type = detect_model_type(out_names)

    if model_type == "original":
        boxes_valid, _ = run_original_inference(det_sess, feed, draw_threshold, warmup_runs=1)
    else:
        boxes_valid, _ = run_ane_inference(det_sess, feed, args.img, draw_threshold, warmup_runs=1)

    print(f"  -> Found {len(boxes_valid)} valid detections.")

    if len(boxes_valid) == 0:
        print("No objects detected. Exiting.")
        return

    # 4. Load ReID Model
    print("\n[3/3] Loading ReID Model and Extracting Features...")
    reid_sess = get_session(args.reid_onnx)
    reid_input_name = reid_sess.get_inputs()[0].name

    # Load original image for cropping
    orig_img = cv2.imread(args.img)
    if orig_img is None:
        print(f"[ERROR] Failed to read image: {args.img}")
        sys.exit(1)

    h, w = orig_img.shape[:2]

    # Prepare output directory
    os.makedirs(args.out, exist_ok=True)

    # Visualize
    vis_img = orig_img.copy()

    print(f"{'ID':<5} {'Class':<5} {'Score':<10} {'Box':<25} {'Embedding Shape'}")
    print("-" * 70)

    for i, box in enumerate(boxes_valid):
        cls_id, score, x0, y0, x1, y1 = box

        # Clip coordinates
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(w, int(x1))
        y1 = min(h, int(y1))

        if x1 <= x0 or y1 <= y0:
            print(f"[WARN] Invalid box {i}: {x0, y0, x1, y1}, skipping.")
            continue

        # Crop
        crop = orig_img[y0:y1, x0:x1]

        # ReID Inference
        reid_input = preprocess_reid(crop)
        reid_output = reid_sess.run(None, {reid_input_name: reid_input})[0]

        # Normalize embedding (optional, but standard for ReID)
        # embedding = reid_output / np.linalg.norm(reid_output)

        print(f"{i:<5} {int(cls_id):<5} {score:.4f}     [{x0}, {y0}, {x1}, {y1}]     {reid_output.shape}")

        # Draw on image
        color = (0, 255, 0)
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis_img, f"ID:{i}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vis_path = os.path.join(args.out, "result_reid.jpg")
    cv2.imwrite(vis_path, vis_img)
    print(f"\n✅ Visualization saved to: {vis_path}")

if __name__ == "__main__":
    main()
