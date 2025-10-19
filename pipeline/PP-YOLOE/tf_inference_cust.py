#!/usr/bin/env python3
"""Inference helper for customized/pruned PP-YOLOE TFLite models."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from onnx_inference_utils import (
    draw_and_save,
    preprocess_image,
    print_embedding_diagnostics,
)
from onnx_inference_ane_model import roi_align_pool_multi_scale


def _as_list(shape: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in np.array(shape, dtype=np.int64).flatten().tolist())


def _matches_shape(shape: Tuple[int, ...], target: Tuple[int, ...]) -> bool:
    if len(shape) != len(target):
        return False
    for dim, tgt in zip(shape, target):
        if dim == -1 or tgt == -1:
            continue
        if dim != tgt:
            return False
    return True


def _has_negative(shape: Tuple[int, ...]) -> bool:
    return any(dim < 0 for dim in shape)


def _infer_layout(shape: Tuple[int, ...], default: str = "nhwc") -> str:
    if len(shape) != 4:
        return default
    batch, dim1, dim2, dim3 = shape
    if dim1 == 3:
        return "nchw"
    if dim3 == 3:
        return "nhwc"
    if dim1 == -1 and dim3 == 3:
        return "nhwc"
    if dim3 == -1 and dim1 == 3:
        return "nchw"
    if dim1 in (1, 3) and dim3 not in (1, 3):
        return "nchw"
    if dim3 in (1, 3) and dim1 not in (1, 3):
        return "nhwc"
    return default


def _ensure_nchw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4 or arr.shape[0] != 1:
        return arr
    b, d1, d2, d3 = arr.shape
    if d1 == d2 and d3 not in (d1, d2):
        return arr.transpose(0, 3, 1, 2)
    if d1 in (1, 3) and d3 not in (1, 3):
        return arr.transpose(0, 3, 1, 2)
    if d3 in (1, 3) and d1 not in (1, 3):
        return arr
    if d3 > d1:
        return arr.transpose(0, 3, 1, 2)
    return arr


def _create_interpreter(model_path: str, threads: Optional[int], delegate: Optional[str]):
    errors = []

    def _format_error(source: str, exc: Exception) -> str:
        return f"{source} ({type(exc).__name__}: {exc})"

    # Try tflite_runtime first.
    try:
        from tflite_runtime.interpreter import Interpreter as RtInterpreter  # type: ignore
        from tflite_runtime.interpreter import load_delegate as rt_load_delegate  # type: ignore

        delegates = None
        if delegate:
            try:
                delegates = [rt_load_delegate(delegate)]
            except Exception as exc:  # pragma: no cover - delegate failures are informational
                errors.append(_format_error("tflite_runtime.load_delegate", exc))
                delegates = None
        interpreter = RtInterpreter(
            model_path=model_path,
            num_threads=threads,
            experimental_delegates=delegates,
        )
        return interpreter
    except ImportError as exc:
        errors.append(_format_error("tflite_runtime", exc))
    except Exception as exc:  # pragma: no cover - runtime errors are passed through
        errors.append(_format_error("tflite_runtime.Interpreter", exc))

    # Fallback to tensorflow.lite
    try:
        import tensorflow as tf  # type: ignore

        delegates = None
        if delegate:
            load_delegate_fn = None
            try:
                from tensorflow.lite.experimental import load_delegate as tf_load_delegate  # type: ignore

                load_delegate_fn = tf_load_delegate
            except Exception:
                try:
                    from tensorflow.lite import load_delegate as tf_load_delegate  # type: ignore

                    load_delegate_fn = tf_load_delegate
                except Exception as exc:
                    errors.append(_format_error("tensorflow.lite.load_delegate", exc))
            if load_delegate_fn:
                try:
                    delegates = [load_delegate_fn(delegate)]
                except Exception as exc:  # pragma: no cover
                    errors.append(_format_error("tensorflow.lite.load_delegate", exc))
                    delegates = None

        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=threads,
            experimental_delegates=delegates,
        )
        return interpreter
    except ImportError as exc:
        errors.append(_format_error("tensorflow", exc))
    except Exception as exc:
        errors.append(_format_error("tf.lite.Interpreter", exc))

    error_msg = "Failed to create a TFLite interpreter. Attempts:\n - " + "\n - ".join(errors)
    raise RuntimeError(error_msg)


def _prepare_inputs(interpreter, prep: Dict[str, np.ndarray], layout: str) -> None:
    image_chw = prep["image"].astype(np.float32)
    image_nchw = image_chw[np.newaxis, :]
    image_nhwc = image_chw.transpose(1, 2, 0)[np.newaxis, :]
    scale = prep["scale_factor"].astype(np.float32)[np.newaxis, :]
    im_shape = prep["im_shape"].astype(np.float32)[np.newaxis, :]

    plans = []
    scale_assigned = False
    im_shape_assigned = False

    input_details = interpreter.get_input_details()
    for detail in input_details:
        name = detail.get("name", f"input_{detail['index']}")
        lower = name.lower()
        shape = _as_list(detail.get("shape_signature", detail["shape"]))

        desired_value = None
        desired_shape = None

        if "scale" in lower:
            desired_value = scale
            desired_shape = scale.shape
            scale_assigned = True
        elif "im_shape" in lower or "imshape" in lower or "image_shape" in lower:
            desired_value = im_shape
            desired_shape = im_shape.shape
            im_shape_assigned = True
        elif len(shape) == 2 and shape[-1] == 2 and not scale_assigned:
            desired_value = scale
            desired_shape = scale.shape
            scale_assigned = True
        elif len(shape) == 2 and shape[-1] == 2 and not im_shape_assigned:
            desired_value = im_shape
            desired_shape = im_shape.shape
            im_shape_assigned = True
        else:
            if layout == "auto":
                if _matches_shape(shape, image_nchw.shape):
                    chosen_layout = "nchw"
                elif _matches_shape(shape, image_nhwc.shape):
                    chosen_layout = "nhwc"
                else:
                    chosen_layout = _infer_layout(shape)
            else:
                chosen_layout = layout

            if chosen_layout == "nchw":
                desired_value = image_nchw
                desired_shape = image_nchw.shape
            else:
                desired_value = image_nhwc
                desired_shape = image_nhwc.shape

        if desired_value is None:
            raise RuntimeError(f"Could not determine input mapping for tensor '{name}' with shape {shape}.")

        if desired_shape is not None:
            if _has_negative(shape) or not _matches_shape(shape, desired_shape):
                interpreter.resize_tensor_input(detail["index"], desired_shape)

        plans.append({
            "index": detail["index"],
            "name": name,
            "value": desired_value,
        })

    interpreter.allocate_tensors()

    post_details = interpreter.get_input_details()
    detail_by_index = {detail["index"]: detail for detail in post_details}

    for plan in plans:
        detail = detail_by_index[plan["index"]]
        dtype = detail["dtype"]
        value = plan["value"].astype(dtype, copy=False)
        interpreter.set_tensor(plan["index"], value)


def _select_boxes_and_scores(outputs: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, str, np.ndarray]:
    boxes_candidate = None
    scores_candidate = None
    multi_class_candidates = []

    for name, raw in outputs.items():
        arr = np.array(raw)
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 0:
            continue
        if squeezed.ndim == 1:
            if squeezed.size > 4:
                continue
            candidate_scores = squeezed.reshape(-1, 1)
            if scores_candidate is None or candidate_scores.shape[0] > scores_candidate[1].shape[0]:
                scores_candidate = (name, candidate_scores)
            continue
        if squeezed.ndim >= 3:
            squeezed = squeezed.reshape(-1, squeezed.shape[-1])

        if squeezed.ndim != 2:
            continue

        last_dim = squeezed.shape[1]
        if last_dim == 4:
            if boxes_candidate is None or squeezed.shape[0] > boxes_candidate[1].shape[0]:
                boxes_candidate = (name, squeezed)
        elif last_dim == 1:
            if scores_candidate is None or squeezed.shape[0] > scores_candidate[1].shape[0]:
                scores_candidate = (name, squeezed)
        elif last_dim <= 80:
            multi_class_candidates.append((name, squeezed))

    if boxes_candidate is None:
        raise RuntimeError("Could not find a detection boxes tensor in TFLite outputs.")

    boxes_name, boxes_arr = boxes_candidate

    if scores_candidate is None:
        for name, arr in multi_class_candidates:
            if arr.shape[0] == boxes_arr.shape[0]:
                scores_candidate = (name, arr[:, 0:1])
                break

    if scores_candidate is None:
        raise RuntimeError("Could not find detection scores tensor matching the boxes output.")

    scores_name, scores_arr = scores_candidate
    return boxes_name, boxes_arr, scores_name, scores_arr


def _ensure_xyxy(boxes: np.ndarray) -> Tuple[np.ndarray, str]:
    if boxes.shape[1] < 4:
        raise RuntimeError("Boxes tensor must have at least 4 columns.")
    trimmed = boxes[:, :4].astype(np.float32, copy=False)
    diff_x = trimmed[:, 2] - trimmed[:, 0]
    diff_y = trimmed[:, 3] - trimmed[:, 1]
    neg_x = float((diff_x < 0).sum()) / max(1, diff_x.size)
    neg_y = float((diff_y < 0).sum()) / max(1, diff_y.size)
    if neg_x > 0.6 or neg_y > 0.6:
        x, y, w, h = trimmed[:, 0], trimmed[:, 1], trimmed[:, 2], trimmed[:, 3]
        converted = np.column_stack((x, y, x + w, y + h)).astype(np.float32, copy=False)
        return converted, "xywh"
    return trimmed, "xyxy"


def _extract_feature_maps(outputs: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
    feat_s8 = None
    feat_s16 = None
    key_s8 = None
    key_s16 = None

    for name, raw in outputs.items():
        arr = np.array(raw)
        if arr.ndim != 4 or arr.shape[0] != 1:
            continue
        arr_nchw = _ensure_nchw(arr.astype(np.float32, copy=False))
        _, c, h, w = arr_nchw.shape
        if 70 <= h <= 90 and 70 <= w <= 90:
            if feat_s8 is None or c > feat_s8.shape[1]:
                feat_s8 = arr_nchw
                key_s8 = name
        elif 30 <= h <= 50 and 30 <= w <= 50:
            if feat_s16 is None or c > feat_s16.shape[1]:
                feat_s16 = arr_nchw
                key_s16 = name

    return feat_s8, feat_s16, key_s8, key_s16


def _postprocess(
    outputs: Dict[str, np.ndarray],
    draw_threshold: float,
    img_path: str,
    warmup_runs: int,
    inference_ms: float,
    input_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    post_start = time.perf_counter()

    boxes_name, boxes_arr, scores_name, scores_arr = _select_boxes_and_scores(outputs)

    boxes_xyxy, origin_format = _ensure_xyxy(boxes_arr)
    scores = np.squeeze(scores_arr.astype(np.float32, copy=False))
    if scores.ndim == 0:
        scores = scores[np.newaxis]
    if boxes_xyxy.shape[0] != scores.shape[0]:
        raise RuntimeError(
            f"Boxes and scores have different lengths ({boxes_xyxy.shape[0]} vs {scores.shape[0]})."
        )

    print("Detected tensors:")
    print(f"  boxes : {boxes_name} shape={boxes_arr.shape} (interpreted as {origin_format})")
    print(f"  scores: {scores_name} shape={scores_arr.shape}")

    x0, y0, x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = np.clip(x1 - x0, a_min=0.0, a_max=None)
    h = np.clip(y1 - y0, a_min=0.0, a_max=None)
    boxes_xywh = np.column_stack((x0, y0, w, h)).astype(np.float32, copy=False)

    score_threshold = float(draw_threshold)
    nms_threshold = 0.5
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), score_threshold, nms_threshold)

    if len(indices) == 0:
        detections = np.zeros((0, 6), dtype=np.float32)
        print(f"NMS kept 0 detections (threshold={score_threshold:.2f}).")
    else:
        selected = np.array(indices).flatten()
        kept_boxes = boxes_xyxy[selected]
        kept_scores = scores[selected]
        detections = np.column_stack(
            (np.zeros_like(kept_scores), kept_scores, kept_boxes)
        ).astype(np.float32, copy=False)
        print(f"NMS kept {len(selected)} detections (threshold={score_threshold:.2f}).")

    feat_s8, feat_s16, key_s8, key_s16 = _extract_feature_maps(outputs)

    if feat_s8 is not None and feat_s16 is not None and detections.shape[0] > 0:
        print("Embedding feature maps detected:")
        print(f"  stride-8 : {key_s8} shape={feat_s8.shape}")
        print(f"  stride-16: {key_s16} shape={feat_s16.shape}")
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            h_img, w_img = orig_img.shape[:2]
            embs = roi_align_pool_multi_scale(
                feat_s8,
                feat_s16,
                detections[:, 2:6],
                (h_img, w_img),
                input_size_hw=input_hw,
                gp_w=0.2,
                pp_w=0.8,
                pp_k=9,
                pp_stripe_h=2,
                pp_vertical_k=2,
                pp_vertical_stripe_w=2,
                use_inst_norm=True,
                pl_alpha=0.35,
            )
        else:
            print("Warning: unable to read original image for embedding extraction; returning zero embeddings.")
            embs = np.zeros((detections.shape[0], feat_s8.shape[1] + feat_s16.shape[1]), dtype=np.float32)
    else:
        if detections.shape[0] > 0:
            print("Embedding feature maps not found; returning zero embeddings.")
    embs = np.zeros((detections.shape[0], 0), dtype=np.float32)

    print("Detections (class score x0 y0 x1 y1):")
    if detections.shape[0] == 0:
        print("  <none>")
    else:
        for row in detections:
            cls_id, score, bx0, by0, bx1, by1 = row
            print(f"  {int(cls_id)} {score:.4f} {bx0:.1f} {by0:.1f} {bx1:.1f} {by1:.1f}")

    print_embedding_diagnostics(embs, embs, detections)

    post_ms = (time.perf_counter() - post_start) * 1000.0

    benchmark = {
        "warmup_runs": warmup_runs,
        "inference_ms": inference_ms,
        "post_ms": post_ms,
        "total_ms": inference_ms + post_ms,
        "num_detections": int(detections.shape[0]),
        "boxes_tensor": boxes_name,
        "scores_tensor": scores_name,
        "feature_stride8": key_s8,
        "feature_stride16": key_s16,
    }

    return detections, embs, benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on customized/pruned PP-YOLOE TFLite model")
    parser.add_argument("--img", default="pipeline/dataset/demo/demo.jpg", help="Path to input image")
    parser.add_argument(
        "--tflite",
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_f32.tflite",
        help="Path to customized TFLite model",
    )
    parser.add_argument("--out", default="pipeline/output", help="Directory to save visualization")
    parser.add_argument("--thresh", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument(
        "--network-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Network input size (width height)",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs before timing")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads for TFLite interpreter")
    parser.add_argument("--delegate", default=None, help="Optional custom delegate shared library path")
    parser.add_argument(
        "--layout",
        choices=["auto", "nchw", "nhwc"],
        default="auto",
        help="Override input layout detection (default: auto)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.img):
        print(f"Error: image not found: {args.img}")
        sys.exit(1)
    if not os.path.exists(args.tflite):
        print(f"Error: TFLite model not found: {args.tflite}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print("================================================================================")
    print("PP-YOLOE Pruned Model TFLite Inference")
    print("================================================================================")
    print(f"Image      : {args.img}")
    print(f"Model      : {args.tflite}")
    print(f"Threshold  : {args.thresh}")
    print(f"Network sz : {args.network_size}")
    print(f"Threads    : {args.threads if args.threads else 'default'}")
    print(f"Layout     : {args.layout}")
    print()

    interpreter = _create_interpreter(args.tflite, args.threads, args.delegate)

    prep = preprocess_image(args.img, target_size=tuple(args.network_size))
    _prepare_inputs(interpreter, prep, args.layout)

    warmup_runs = max(args.warmup, 0)
    for _ in range(warmup_runs):
        interpreter.invoke()

    t0 = time.perf_counter()
    interpreter.invoke()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    output_details = interpreter.get_output_details()
    outputs = {}
    for idx, detail in enumerate(output_details):
        name = detail.get("name", f"output_{idx}")
        if name in outputs:
            name = f"{name}_{idx}"
        outputs[name] = interpreter.get_tensor(detail["index"])

    print("Outputs summary:")
    for name, arr in outputs.items():
        print(f"  - {name}: shape={np.array(arr).shape}, dtype={np.array(arr).dtype}")
    print()

    input_hw = (prep["image"].shape[1], prep["image"].shape[2])
    detections, embeddings, benchmark = _postprocess(
        outputs,
        float(args.thresh),
        args.img,
        warmup_runs,
        inference_ms,
        input_hw,
    )

    out_path = os.path.join(args.out, f"{Path(args.img).stem}_cust_tf.jpg")
    draw_and_save(args.img, detections, float(args.thresh), out_path, label="person")
    print(f"Saved visualization to: {out_path}")

    print("Benchmark summary:")
    for key, value in benchmark.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    if embeddings.size > 0:
        print(f"Embeddings available: {embeddings.shape}")
    else:
        print("Embeddings: none or zero detections")

    print("\nInference complete.")


if __name__ == "__main__":
    main()