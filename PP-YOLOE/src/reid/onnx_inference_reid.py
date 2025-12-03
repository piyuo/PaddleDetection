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

def get_session(onnx_path: str, force_cpu: bool = False):
    try:
        import onnxruntime as ort
        import platform
    except Exception as exc:
        print(
            "[ERROR] onnxruntime not installed. Install with: pip install onnxruntime",
            file=sys.stderr,
        )
        raise exc

    print(f"[INFO] System: {platform.system()} {platform.machine()}")
    print(f"[INFO] Python: {sys.version.split()[0]}")
    print(f"[INFO] ONNX Runtime Version: {ort.__version__}")
    available = ort.get_available_providers()
    print(f"[INFO] Available Providers: {available}")

    providers = None
    if not force_cpu and "CoreMLExecutionProvider" in available:
        coreml_opts = {
            "ModelFormat": "MLProgram",
            "EnableOnSubgraphs": "1",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "0", # Relaxed constraint
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
        print(f"[INFO] Loading {os.path.basename(onnx_path)} with default providers (CPU forced: {force_cpu}): {available}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

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
    # Based on provided infer_cfg.yml: mean=[0,0,0], std=[1,1,1]
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
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
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (disable CoreML)")
    args = parser.parse_args()

    # 1. Check files
    for path, label in ((args.img, "Input image"), (args.det_onnx, "Detection model"), (args.reid_onnx, "ReID model")):
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # 2. Load Detection Model
    print("\n[1/3] Loading Detection Model...")
    det_sess = get_session(args.det_onnx, args.cpu)

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
    reid_sess = get_session(args.reid_onnx, args.cpu)
    reid_input_name = reid_sess.get_inputs()[0].name

    # Warmup ReID
    if not args.cpu:
        print("[INFO] Warming up ReID model...")
        try:
            # Create a dummy input matching the expected shape (1, 3, 256, 128)
            dummy_input = np.zeros((1, 3, 256, 128), dtype=np.float32)
            reid_sess.run(None, {reid_input_name: dummy_input})
        except Exception as e:
            print(f"[WARN] ReID warmup failed: {e}")

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

    embeddings = []
    valid_ids = []

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
        t0 = time.time()
        reid_output = reid_sess.run(None, {reid_input_name: reid_input})[0]
        t1 = time.time()
        print(f"     [Time] ReID Inference: {(t1 - t0) * 1000:.2f} ms")

        # Store for analysis
        embeddings.append(reid_output.flatten())
        valid_ids.append(i)

        print(f"{i:<5} {int(cls_id):<5} {score:.4f}     [{x0}, {y0}, {x1}, {y1}]     {reid_output.shape}")

        # Draw on image
        color = (0, 255, 0)
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis_img, f"ID:{i}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vis_path = os.path.join(args.out, "result_reid.jpg")
    cv2.imwrite(vis_path, vis_img)
    print(f"\n✅ Visualization saved to: {vis_path}")

    # --- BoT-SORT Suitability Analysis ---
    if len(embeddings) > 1:
        print("\n" + "="*60)
        print("🔍 BoT-SORT Suitability Analysis (Discriminative Power)")
        print("="*60)

        # Stack and Normalize
        feats = np.stack(embeddings) # (N, 256)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_norm = feats / (norms + 1e-6)

        print(f"[Info] Embedding Norms (should be non-zero): {norms.flatten().round(2)}")

        # Compute Cosine Similarity Matrix (N x N)
        # Sim(A, B) = (A . B) / (|A|*|B|)
        sim_matrix = np.dot(feats_norm, feats_norm.T)

        # We are looking for LOW similarity between different IDs (off-diagonal elements)
        # Mask the diagonal (self-similarity is always 1.0)
        np.fill_diagonal(sim_matrix, -1.0)

        max_sim = np.max(sim_matrix)
        avg_sim = np.mean(sim_matrix[sim_matrix > -1.0])

        print(f"\n[Stats] Similarity between DIFFERENT people (Lower is better):")
        print(f"  • Max Similarity: {max_sim:.4f}")
        print(f"  • Avg Similarity: {avg_sim:.4f}")

        # Thresholds for BoT-SORT
        # Typically, a match is rejected if distance > 0.2~0.4 (Similarity < 0.8~0.6)
        # Conversely, if different people have similarity > 0.7, tracking might fail.

        print("\n[Top Confusing Pairs] (High similarity between different IDs):")
        print(f"  {'ID_A':<5} {'ID_B':<5} {'Similarity':<10} {'Status'}")
        print("-" * 45)

        # Find pairs with high similarity
        count_high = 0
        # Only check upper triangle to avoid duplicates
        for r in range(len(valid_ids)):
            for c in range(r + 1, len(valid_ids)):
                sim = sim_matrix[r, c]
                id_a = valid_ids[r]
                id_b = valid_ids[c]

                status = ""
                if sim > 0.8:
                    status = "🔴 CRITICAL (Likely ID Switch)"
                    count_high += 1
                elif sim > 0.6:
                    status = "🟡 WARN (Risk of Switch)"
                else:
                    status = "🟢 OK"

                if sim > 0.5: # Only print relevant ones
                    print(f"  {id_a:<5} {id_b:<5} {sim:.4f}     {status}")

        if count_high == 0 and max_sim < 0.6:
            print("\n✅ RESULT: Embeddings look GOOD for BoT-SORT.")
            print("   Different people are well separated in the embedding space.")
        elif count_high > 0:
            print("\n❌ RESULT: Embeddings might cause ID SWITCHES.")
            print("   Some different people look too similar to the model.")
        else:
            print("\n⚠️ RESULT: Embeddings are ACCEPTABLE but not perfect.")
if __name__ == "__main__":
    main()
