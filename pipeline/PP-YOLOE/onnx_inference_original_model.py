#!/usr/bin/env python3
"""Original PP-YOLOE ONNX model inference path (with built-in NMS)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from onnx_inference_utils import print_outputs_header, run_session_with_warmup


def run_original_inference(
    sess,
    feed: Dict[str, np.ndarray],
    draw_threshold: float,
    warmup_runs: int = 3,
):
    outputs, out_names, inference_ms = run_session_with_warmup(sess, feed, warmup_runs)
    print_outputs_header("original", out_names, outputs, draw_threshold)

    if not outputs:
        raise RuntimeError("Model returned no outputs.")

    bboxes = np.array(outputs[0])
    print("\n[INFO] Model has NMS built-in (using existing detections)")
    print("Detections (class score x0 y0 x1 y1):")
    kept = 0
    for b in bboxes:
        if int(b[0]) > -1 and float(b[1]) >= float(draw_threshold):
            kept += 1
            print(f"{int(b[0])} {b[1]:.4f} {b[2]:.1f} {b[3]:.1f} {b[4]:.1f} {b[5]:.1f}")
    if kept == 0:
        print(f"No boxes above threshold {draw_threshold}. Try lowering --thresh.")

    valid_mask = (bboxes[:, 0] > -1) & (bboxes[:, 1] >= float(draw_threshold))
    boxes_valid = bboxes[valid_mask]
    embs_valid = np.zeros((boxes_valid.shape[0], 0), dtype=np.float32)
    if boxes_valid.size:
        print("[INFO] Original model does not provide embeddings; returning empty features.")

    benchmark = {
        "model_type": "original",
        "warmup_runs": warmup_runs,
        "inference_ms": inference_ms,
        "num_detections": int(boxes_valid.shape[0]),
        "output_names": list(out_names),
    }

    return boxes_valid, embs_valid, benchmark
