#!/usr/bin/env python3
"""ANE-optimized (automatic surgery) inference pipeline for PP-YOLOE ONNX models."""

from __future__ import annotations

from typing import Dict

import time

import cv2
import numpy as np

from onnx_inference_utils import (
    print_embedding_diagnostics,
    print_outputs_header,
    run_session_with_warmup,
)


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

    if len(bboxes) > 0:
        print("  [INFO] Embedding extraction disabled (feature maps not exported)")
        embs_nms = np.zeros((len(bboxes), 0), dtype=np.float32)
    else:
        embs_nms = np.zeros((0, 0), dtype=np.float32)

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
