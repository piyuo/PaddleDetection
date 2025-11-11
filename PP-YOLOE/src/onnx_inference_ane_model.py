#!/usr/bin/env python3
"""ANE-optimized (automatic surgery) inference pipeline for PP-YOLOE ONNX models."""

from __future__ import annotations

from typing import Dict, Tuple

import time

import cv2
import numpy as np

from onnx_inference_utils import (
    print_embedding_diagnostics,
    print_outputs_header,
    run_session_with_warmup,
)


def roi_align_pool_multi_scale(
    feat_s8: np.ndarray,
    feat_s16: np.ndarray,
    boxes_xyxy: np.ndarray,
    img_hw: Tuple[int, int],
    input_size_hw: Tuple[int, int] = (640, 640),
    gp_w: float = 0.2,
    avg_w: float = 1.0,
    max_w: float = 0.0,
    pp_w: float = 0.8,
    pp_k: int = 9,
    pp_stripe_h: int = 2,
    pp_vertical_k: int = 2,
    pp_vertical_stripe_w: int = 2,
    use_inst_norm: bool = True,
    pl_alpha: float = 0.35,
) -> np.ndarray:
    assert feat_s8.ndim == 4 and feat_s8.shape[0] == 1
    assert feat_s16.ndim == 4 and feat_s16.shape[0] == 1

    _, c_s8, h_s8, w_s8 = feat_s8.shape
    _, c_s16, h_s16, w_s16 = feat_s16.shape
    h_img, w_img = img_hw
    h_input, w_input = input_size_hw

    def compute_scales(h_feat, w_feat):
        scale_y = (h_input / float(h_img)) * (h_feat / float(h_input))
        scale_x = (w_input / float(w_img)) * (w_feat / float(w_input))
        return scale_x, scale_y

    scale_x_s8, scale_y_s8 = compute_scales(h_s8, w_s8)
    scale_x_s16, scale_y_s16 = compute_scales(h_s16, w_s16)

    def extract_roi(feat_map, x0, y0, x1, y1, scale_x, scale_y, h_feat, w_feat, channels):
        fx0 = int(max(0, np.floor(x0 * scale_x)))
        fy0 = int(max(0, np.floor(y0 * scale_y)))
        fx1 = int(min(w_feat, np.ceil(x1 * scale_x)))
        fy1 = int(min(h_feat, np.ceil(y1 * scale_y)))
        if fx1 <= fx0 or fy1 <= fy0:
            return np.zeros((channels,), dtype=np.float32)

        roi = feat_map[0, :, fy0:fy1, fx0:fx1]
        _, h_roi, w_roi = roi.shape
        features = []

        if gp_w > 0:
            global_feat = np.zeros((channels,), dtype=np.float32)
            if avg_w > 0:
                global_feat += avg_w * roi.mean(axis=(1, 2))
            if max_w > 0:
                global_feat += max_w * roi.max(axis=(1, 2))
            features.append(global_feat * gp_w)

        if pp_w > 0 and pp_k > 0 and pp_stripe_h > 0:
            stripe_size = max(1, h_roi // pp_k)
            for i in range(pp_k):
                y_start = i * stripe_size
                y_end = min(h_roi, (i + 1) * stripe_size)
                if y_end <= y_start:
                    continue
                sub_stripe_size = max(1, (y_end - y_start) // pp_stripe_h)
                for j in range(pp_stripe_h):
                    sub_y_start = y_start + j * sub_stripe_size
                    sub_y_end = min(y_end, y_start + (j + 1) * sub_stripe_size)
                    if sub_y_end <= sub_y_start:
                        continue
                    stripe = roi[:, sub_y_start:sub_y_end, :]
                    stripe_feat = stripe.mean(axis=(1, 2))
                    features.append(stripe_feat * pp_w / (pp_k * pp_stripe_h))

        if pp_w > 0 and pp_vertical_k > 0 and pp_vertical_stripe_w > 0:
            stripe_size = max(1, w_roi // pp_vertical_k)
            for i in range(pp_vertical_k):
                x_start = i * stripe_size
                x_end = min(w_roi, (i + 1) * stripe_size)
                if x_end <= x_start:
                    continue
                sub_stripe_size = max(1, (x_end - x_start) // pp_vertical_stripe_w)
                for j in range(pp_vertical_stripe_w):
                    sub_x_start = x_start + j * sub_stripe_size
                    sub_x_end = min(x_end, x_start + (j + 1) * sub_stripe_size)
                    if sub_x_end <= sub_x_start:
                        continue
                    stripe = roi[:, :, sub_x_start:sub_x_end]
                    stripe_feat = stripe.mean(axis=(1, 2))
                    features.append(stripe_feat * pp_w / (pp_vertical_k * pp_vertical_stripe_w))

        if not features:
            return np.zeros((channels,), dtype=np.float32)

        combined = np.sum(features, axis=0)
        if use_inst_norm:
            mean = combined.mean()
            std = combined.std()
            if std > 1e-6:
                combined = (combined - mean) / std
        if pl_alpha != 1.0:
            sign = np.sign(combined)
            combined = sign * np.power(np.abs(combined), pl_alpha)
        return combined

    embs = []
    for x0, y0, x1, y1 in boxes_xyxy:
        feat_s8_vec = extract_roi(feat_s8, x0, y0, x1, y1, scale_x_s8, scale_y_s8, h_s8, w_s8, c_s8)
        feat_s16_vec = extract_roi(feat_s16, x0, y0, x1, y1, scale_x_s16, scale_y_s16, h_s16, w_s16, c_s16)
        combined = np.concatenate([feat_s8_vec, feat_s16_vec])
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            combined = combined / norm
        embs.append(combined)

    if not embs:
        return np.zeros((0, c_s8 + c_s16), dtype=np.float32)
    return np.stack(embs, axis=0)


def run_ane_inference(
    sess,
    feed: Dict[str, np.ndarray],
    img_path: str,
    draw_threshold: float,
    warmup_runs: int = 3,
):
    outputs, out_names, inference_ms = run_session_with_warmup(sess, feed, warmup_runs)
    post_start = time.perf_counter()
    print_outputs_header("ane", out_names, outputs, draw_threshold)

    name_to_out = {name: arr for name, arr in zip(out_names, outputs)}

    raw_boxes_key = "p2o.pd_op.divide.0.0"
    raw_scores_key = "p2o.pd_op.concat.14.0"

    if raw_boxes_key not in name_to_out or raw_scores_key not in name_to_out:
        raise RuntimeError(
            "Expected pruned model outputs not found."
            f" Available outputs: {list(name_to_out.keys())}"
        )

    raw_boxes = name_to_out[raw_boxes_key]
    raw_scores = name_to_out[raw_scores_key]

    if raw_boxes.ndim == 3:
        raw_boxes = raw_boxes.squeeze(0)
    elif raw_boxes.ndim == 4:
        raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])

    raw_boxes = np.squeeze(raw_boxes)
    raw_scores = np.squeeze(raw_scores)

    print("\n[INFO] Applying custom NMS post-processing (pruned model detected)...")
    print(f"  Raw boxes shape: {raw_boxes.shape}")
    print(f"  Raw scores shape: {raw_scores.shape}")

    if raw_scores.ndim == 1:
        person_scores = raw_scores
    elif raw_scores.ndim == 2 and raw_scores.shape[1] == 1:
        person_scores = raw_scores[:, 0]
    elif raw_scores.ndim == 2:
        person_scores = raw_scores[:, 0]
    else:
        raise RuntimeError(f"Unexpected scores shape: {raw_scores.shape}")

    x0, y0, x1, y1 = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
    w = x1 - x0
    h = y1 - y0
    nms_boxes = np.column_stack([x0, y0, w, h]).tolist()

    score_threshold = float(draw_threshold)
    nms_threshold = 0.5
    print(
        f"  Applying NMS with score_threshold={score_threshold:.2f}, nms_threshold={nms_threshold:.2f}"
    )
    selected_indices = cv2.dnn.NMSBoxes(nms_boxes, person_scores.tolist(), score_threshold, nms_threshold)

    if len(selected_indices) > 0:
        selected_indices = np.array(selected_indices).flatten()
        print(f"  NMS kept {len(selected_indices)} detections from {len(person_scores)} proposals")
        boxes_nms = raw_boxes[selected_indices]
        scores_nms = person_scores[selected_indices]
        class_ids = np.zeros_like(scores_nms)
        bboxes = np.column_stack([class_ids, scores_nms, boxes_nms])
    else:
        print(f"  NMS found no boxes above threshold {score_threshold:.2f}")
        bboxes = np.zeros((0, 6), dtype=np.float32)

    feat_s8 = None
    feat_s16 = None
    feat_s8_key = None
    feat_s16_key = None

    for name, arr in name_to_out.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 1:
            _, channels, height, width = arr.shape
            if 70 <= height <= 90 and 70 <= width <= 90 and channels >= 64:
                if feat_s8 is None or channels > feat_s8.shape[1]:
                    feat_s8 = arr
                    feat_s8_key = name
            elif 30 <= height <= 50 and 30 <= width <= 50 and channels >= 64:
                if feat_s16 is None or channels > feat_s16.shape[1]:
                    feat_s16 = arr
                    feat_s16_key = name

    if feat_s8 is not None and feat_s16 is not None and len(bboxes) > 0:
        print("  [INFO] Multi-scale embedding extraction:")
        print(f"         Stride-8:  {feat_s8_key} (shape={feat_s8.shape})")
        print(f"         Stride-16: {feat_s16_key} (shape={feat_s16.shape})")
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            orig_h, orig_w = orig_img.shape[:2]
            embs_nms = roi_align_pool_multi_scale(
                feat_s8,
                feat_s16,
                bboxes[:, 2:6],
                (orig_h, orig_w),
                input_size_hw=(640, 640),
                gp_w=0.2,
                pp_w=0.8,
                pp_k=9,
                pp_stripe_h=2,
                pp_vertical_k=2,
                pp_vertical_stripe_w=2,
                use_inst_norm=True,
                pl_alpha=0.35,
            )
            print(
                f"  [INFO] Extracted multi-scale embeddings (shape={embs_nms.shape}, dim={embs_nms.shape[1] if embs_nms.ndim == 2 else 0})"
            )
        else:
            print("  [WARN] Could not load image to get original size, using placeholder embeddings")
            s8_ch = feat_s8.shape[1]
            s16_ch = feat_s16.shape[1]
            embs_nms = np.zeros((len(bboxes), s8_ch + s16_ch), dtype=np.float32)
    else:
        if len(bboxes) > 0:
            expected_dim = 224
            if len(outputs) >= 4:
                expected_dim = 0
                for arr in outputs:
                    if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 1:
                        if 70 <= arr.shape[2] <= 90:
                            expected_dim += arr.shape[1]
                        elif 30 <= arr.shape[2] <= 50:
                            expected_dim += arr.shape[1]
            print("  [WARN] Could not auto-detect stride-8 and stride-16 feature maps")
            print(f"         Available outputs: {list(name_to_out.keys())}")
            print(f"         Using placeholder embeddings (all zeros, dim={expected_dim})")
            embs_nms = np.zeros((len(bboxes), expected_dim), dtype=np.float32)
        else:
            embs_nms = np.zeros((0, 224), dtype=np.float32)

    print("\nDetections after NMS (class score x0 y0 x1 y1):")
    if len(bboxes) > 0:
        for b in bboxes:
            print(f"{int(b[0])} {b[1]:.4f} {b[2]:.1f} {b[3]:.1f} {b[4]:.1f} {b[5]:.1f}")
    else:
        print(f"No boxes above threshold {draw_threshold} after NMS.")

    det_embs = embs_nms.astype(np.float32)
    embs_valid = det_embs
    boxes_valid = bboxes

    print_embedding_diagnostics(det_embs, embs_valid, boxes_valid)

    post_ms = (time.perf_counter() - post_start) * 1000.0

    benchmark = {
        "model_type": "ane",
        "warmup_runs": warmup_runs,
        "inference_ms": inference_ms,
        "num_detections": int(boxes_valid.shape[0]),
        "output_names": list(out_names),
        "post_ms": post_ms,
        "total_ms": inference_ms + post_ms,
    }

    return boxes_valid, embs_valid, benchmark
