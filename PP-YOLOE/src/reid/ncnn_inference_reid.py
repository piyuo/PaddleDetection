#!/usr/bin/env python3
"""
Run object detection (NCNN) + ReID inference (NCNN) on a single image.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    import ncnn
except ImportError:
    print("ERROR: ncnn not installed. Install with: pip install ncnn", file=sys.stderr)
    sys.exit(1)

# Constants for Detection Model
DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)
DET_SIZE = (640, 640)

def get_ncnn_net(param_path: str, bin_path: str, num_threads: int = 4):
    net = ncnn.Net()

    # Core optimizations
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_arithmetic = True
    net.opt.use_packing_layout = True
    net.opt.num_threads = num_threads

    if net.load_param(param_path) != 0:
        raise RuntimeError(f"Failed to load NCNN param: {param_path}")
    if net.load_model(bin_path) != 0:
        raise RuntimeError(f"Failed to load NCNN bin: {bin_path}")

    print(f"[INFO] Loaded NCNN model: {os.path.basename(param_path)}")
    return net

def load_and_preprocess_det(img_path, target_size=DET_SIZE):
    """Load image and create NCNN Mat with normalization for Detection."""
    with open(img_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Store original size for scaling
    orig_h, orig_w = im.shape[:2]

    # Resize to model input size
    im_resized = cv2.resize(im, target_size)

    # Create NCNN Mat (BGR to RGB)
    mat = ncnn.Mat.from_pixels(im_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, target_size[0], target_size[1])

    # Apply normalization: (x/255 - mean) / std
    mean_vals = [m * 255.0 for m in DET_MEAN]
    norm_vals = [1.0 / (s * 255.0) for s in DET_STD]
    mat.substract_mean_normalize(mean_vals, norm_vals)

    return mat, (orig_w, orig_h), im

def run_det_inference(net, img_mat, input_names=["in0", "in1"], output_names=["out0", "out1"]):
    """Run NCNN detection inference."""
    # Create scale_factor input (required by PP-YOLOE)
    scale_factor = ncnn.Mat(np.array([1.0, 1.0], dtype=np.float32))

    ex = net.create_extractor()
    ex.set_light_mode(True)

    if "in0" in input_names:
        ex.input("in0", scale_factor)
    if "in1" in input_names:
        ex.input("in1", img_mat)
    else:
        ex.input(input_names[0], img_mat)

    outputs = {}
    for name in output_names:
        ret, out_mat = ex.extract(name)
        if ret == 0:
            outputs[name] = out_mat.numpy()

    return outputs

def apply_nms(boxes_raw, scores_raw, score_thresh=0.5, nms_thresh=0.5):
    """Apply NMS to raw detection outputs."""
    if scores_raw.ndim == 2 and scores_raw.shape[1] == 1:
        person_scores = scores_raw[:, 0]
    elif scores_raw.ndim == 2:
        person_scores = scores_raw[:, 0]
    else:
        person_scores = scores_raw

    x0, y0, x1, y1 = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    w = x1 - x0
    h = y1 - y0
    nms_boxes = np.column_stack([x0, y0, w, h]).tolist()

    score_threshold = float(score_thresh)
    nms_threshold = float(nms_thresh)
    selected_indices = cv2.dnn.NMSBoxes(nms_boxes, person_scores.tolist(), score_threshold, nms_threshold)

    if len(selected_indices) > 0:
        selected_indices = np.array(selected_indices).flatten()
        boxes_nms = boxes_raw[selected_indices]
        scores_nms = person_scores[selected_indices]
        class_ids = np.zeros_like(scores_nms)
        return np.column_stack([class_ids, scores_nms, boxes_nms]).astype(np.float32)
    else:
        return np.zeros((0, 6), dtype=np.float32)

def run_reid_ncnn(net, img_crop: np.ndarray, target_size: Tuple[int, int] = (256, 128)) -> np.ndarray:
    """
    Run ReID inference using NCNN.
    target_size: (H, W) -> (256, 128)
    """
    h, w = img_crop.shape[:2]
    target_h, target_w = target_size

    # Use cv2.resize to match ONNX script exactly
    img_resized = cv2.resize(img_crop, (target_w, target_h))

    in_mat = ncnn.Mat.from_pixels(
        img_resized,
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        target_w,
        target_h
    )

    # Preprocessing: Mean=[0,0,0], Std=[1,1,1], Scale=1/255.0
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1/255.0, 1/255.0, 1/255.0]

    in_mat.substract_mean_normalize(mean_vals, norm_vals)

    ex = net.create_extractor()
    ex.input("in0", in_mat)

    ret, out_mat = ex.extract("out0")
    if ret != 0:
        raise RuntimeError("NCNN ReID inference failed")

    output = np.array(out_mat)
    return output

def main():
    # Default paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_det_param = os.path.join(root, "PP-YOLOE", "build", "models", "ppyoloe_crn_s_36e_pphuman_ncnn.param")
    default_det_bin = os.path.join(root, "PP-YOLOE", "build", "models", "ppyoloe_crn_s_36e_pphuman_ncnn.bin")
    default_reid_param = os.path.join(root, "PP-YOLOE", "build", "models", "human_reid.param")
    default_reid_bin = os.path.join(root, "PP-YOLOE", "build", "models", "human_reid.bin")
    default_img = os.path.join(root, "PP-YOLOE", "build", "dataset", "demo", "demo.jpg")
    default_out = os.path.join(root, "PP-YOLOE", "build", "output_ncnn_full")

    parser = argparse.ArgumentParser(description="Inference: NCNN Detection + NCNN ReID")
    parser.add_argument("--img", default=default_img, help="Path to input image")
    parser.add_argument("--det_param", default=default_det_param, help="Path to Detection NCNN param file")
    parser.add_argument("--det_bin", default=default_det_bin, help="Path to Detection NCNN bin file")
    parser.add_argument("--reid_param", default=default_reid_param, help="Path to ReID NCNN param file")
    parser.add_argument("--reid_bin", default=default_reid_bin, help="Path to ReID NCNN bin file")
    parser.add_argument("--out", default=default_out, help="Directory to save visualization")
    parser.add_argument("--thresh", type=float, default=0.5, help="Score threshold for detection (default: 0.5)")
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    args = parser.parse_args()

    # 1. Check files
    for path, label in ((args.img, "Input image"), (args.det_param, "Detection param"), (args.det_bin, "Detection bin"),
                        (args.reid_param, "ReID param"), (args.reid_bin, "ReID bin")):
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # 2. Load Detection Model (NCNN)
    print("\n[1/4] Loading Detection Model (NCNN)...")
    det_net = get_ncnn_net(args.det_param, args.det_bin, args.threads)

    # 3. Run Detection
    print("\n[2/4] Running Detection...")
    img_mat, orig_size, orig_img = load_and_preprocess_det(args.img, DET_SIZE)
    orig_w, orig_h = orig_size

    outputs = run_det_inference(det_net, img_mat)
    boxes_raw = outputs.get("out0")
    scores_raw = outputs.get("out1")

    if boxes_raw is None or scores_raw is None:
        print("[ERROR] Missing required outputs (out0, out1) from detection model", file=sys.stderr)
        sys.exit(1)

    scores_raw = scores_raw.squeeze()
    boxes_valid = apply_nms(boxes_raw, scores_raw, args.thresh, args.nms_thresh)

    print(f"  -> Found {len(boxes_valid)} valid detections.")

    if len(boxes_valid) == 0:
        print("No objects detected. Exiting.")
        return

    # 4. Load ReID Model (NCNN)
    print("\n[3/4] Loading ReID Model (NCNN) and Extracting Features...")
    reid_net = get_ncnn_net(args.reid_param, args.reid_bin, args.threads)

    os.makedirs(args.out, exist_ok=True)
    vis_img = orig_img.copy()

    print(f"{'ID':<5} {'Class':<5} {'Score':<10} {'Box':<25} {'Embedding Shape'}")
    print("-" * 70)

    embeddings = []
    valid_ids = []

    # Scale factors for mapping boxes back to original image
    scale_x = orig_w / float(DET_SIZE[0])
    scale_y = orig_h / float(DET_SIZE[1])

    for i, box in enumerate(boxes_valid):
        cls_id, score, x0, y0, x1, y1 = box

        # Scale coordinates
        x0 = int(x0 * scale_x)
        y0 = int(y0 * scale_y)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        # Clip to image bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(orig_w, x1)
        y1 = min(orig_h, y1)

        if x1 <= x0 or y1 <= y0:
            continue

        crop = orig_img[y0:y1, x0:x1]

        # NCNN Inference
        reid_output = run_reid_ncnn(reid_net, crop)

        embeddings.append(reid_output.flatten())
        valid_ids.append(i)

        print(f"{i:<5} {int(cls_id):<5} {score:.4f}     [{x0}, {y0}, {x1}, {y1}]     {reid_output.shape}")

        color = (0, 255, 0)
        cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis_img, f"ID:{i}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vis_path = os.path.join(args.out, "result_ncnn_full.jpg")
    cv2.imwrite(vis_path, vis_img)
    print(f"\n✅ Visualization saved to: {vis_path}")

    # --- BoT-SORT Suitability Analysis ---
    if len(embeddings) > 1:
        print("\n" + "="*60)
        print("🔍 BoT-SORT Suitability Analysis (NCNN)")
        print("="*60)

        feats = np.stack(embeddings)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats_norm = feats / (norms + 1e-6)

        print(f"[Info] Embedding Norms: {norms.flatten().round(2)}")

        sim_matrix = np.dot(feats_norm, feats_norm.T)
        np.fill_diagonal(sim_matrix, -1.0)

        max_sim = np.max(sim_matrix)
        avg_sim = np.mean(sim_matrix[sim_matrix > -1.0])

        print(f"\n[Stats] Similarity between DIFFERENT people:")
        print(f"  • Max Similarity: {max_sim:.4f}")
        print(f"  • Avg Similarity: {avg_sim:.4f}")

        print("\n[Top Confusing Pairs]:")
        print(f"  {'ID_A':<5} {'ID_B':<5} {'Similarity':<10} {'Status'}")
        print("-" * 45)

        count_high = 0
        for r in range(len(valid_ids)):
            for c in range(r + 1, len(valid_ids)):
                sim = sim_matrix[r, c]
                id_a = valid_ids[r]
                id_b = valid_ids[c]

                status = ""
                if sim > 0.8:
                    status = "🔴 CRITICAL"
                    count_high += 1
                elif sim > 0.6:
                    status = "🟡 WARN"
                else:
                    status = "🟢 OK"

                if sim > 0.5:
                    print(f"  {id_a:<5} {id_b:<5} {sim:.4f}     {status}")

        if count_high == 0 and max_sim < 0.6:
            print("\n✅ RESULT: NCNN Embeddings look GOOD.")
        elif count_high > 0:
            print("\n❌ RESULT: NCNN Embeddings might cause ID SWITCHES.")
        else:
            print("\n⚠️ RESULT: NCNN Embeddings are ACCEPTABLE.")

if __name__ == "__main__":
    main()
