#!/usr/bin/env python3
"""Original PP-YOLOE ONNX model inference path (with built-in NMS)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from onnx_inference_utils import (
    print_embedding_diagnostics,
    print_outputs_header,
    run_session_with_warmup,
)


def run_original_inference(
    sess,
    feed: Dict[str, np.ndarray],
    draw_threshold: float,
    warmup_runs: int = 3,
):
    outputs, out_names, inference_ms = run_session_with_warmup(sess, feed, warmup_runs)
    print_outputs_header("original", out_names, outputs, draw_threshold)

    name_to_out = {name: arr for name, arr in zip(out_names, outputs)}

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

    if "embed" not in name_to_out or not isinstance(name_to_out["embed"], np.ndarray):
        raise RuntimeError(
            'Model does not expose per-detection embeddings "embed". '
            "Use pipeline/PP-YOLOE/insert_embedding_head.py to augment your model."
        )

    det_embs = name_to_out["embed"].astype(np.float32)
    if det_embs.ndim != 2 or det_embs.shape[0] == 0:
        raise RuntimeError(
            f'"embed" must be a non-empty 2D array shaped (N, D). Got {det_embs.shape}.'
        )

    if det_embs.shape[0] != bboxes.shape[0]:
        raise RuntimeError(
            "Row count mismatch between detections and embeddings: "
            f"detections={bboxes.shape}, embeddings={det_embs.shape}."
        )

    det_embs = det_embs / (np.linalg.norm(det_embs, axis=1, keepdims=True) + 1e-8)
    valid_mask = (bboxes[:, 0] > -1) & (bboxes[:, 1] >= float(draw_threshold))
    boxes_valid = bboxes[valid_mask]
    embs_valid = det_embs[valid_mask]

    print_embedding_diagnostics(det_embs, embs_valid, boxes_valid)

    benchmark = {
        "model_type": "original",
        "warmup_runs": warmup_runs,
        "inference_ms": inference_ms,
        "num_detections": int(boxes_valid.shape[0]),
        "output_names": list(out_names),
    }

    return boxes_valid, embs_valid, benchmark
