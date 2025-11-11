#!/usr/bin/env python3
"""Common utilities for PP-YOLOE ONNX inference scripts."""

from __future__ import annotations

import os
import plistlib
import platform
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def preprocess_image(
    img_path: str,
    target_size: Tuple[int, int] = (640, 640),
    keep_ratio: bool = False,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    is_scale: bool = True,
) -> dict:
    """Standalone image preprocessing for PP-YOLOE ONNX inference."""
    with open(img_path, "rb") as f:
        im_read = f.read()
    data = np.frombuffer(im_read, dtype="uint8")
    im = cv2.imdecode(data, 1)
    if im is None:
        raise ValueError(f"Failed to decode image: {img_path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    original_shape = im.shape[:2]

    if keep_ratio:
        im_size_min = min(original_shape)
        im_size_max = max(original_shape)
        target_size_min = min(target_size)
        target_size_max = max(target_size)
        im_scale = float(target_size_min) / float(im_size_min)
        if round(im_scale * im_size_max) > target_size_max:
            im_scale = float(target_size_max) / float(im_size_max)
        im_scale_y = im_scale
        im_scale_x = im_scale
    else:
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(original_shape[0])
        im_scale_x = resize_w / float(original_shape[1])

    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=cv2.INTER_LINEAR,
    )

    im = im.astype(np.float32, copy=False)
    if is_scale:
        im *= 1.0 / 255.0

    mean_arr = np.array(mean)[np.newaxis, np.newaxis, :]
    std_arr = np.array(std)[np.newaxis, np.newaxis, :]
    im -= mean_arr
    im /= std_arr

    im = im.transpose((2, 0, 1))

    return {
        "image": im,
        "im_shape": np.array(im.shape[1:], dtype=np.float32),
        # scale_factor: [scale_y, scale_x] = [resized_h/orig_h, resized_w/orig_w]
        # Used by PP-YOLOE post-process to convert boxes: network_coords → original_coords
        # For embedding ROI alignment: multiply boxes by scale_factor to get back to network coords
        "scale_factor": np.array([im_scale_y, im_scale_x], dtype=np.float32),
    }


def draw_and_save(
    img_path: str,
    boxes: np.ndarray,
    thresh: float,
    out_path: str,
    label: Optional[str] = "object",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im = cv2.imread(img_path)
    if im is None:
        print(f"[WARN] Could not load image to draw: {img_path}")
        return
    for b in boxes:
        cls_id, score, x0, y0, x1, y1 = b
        if cls_id < 0 or score < thresh:
            continue
        p1 = (int(x0), int(y0))
        p2 = (int(x1), int(y1))
        color = (0, 255, 0)
        cv2.rectangle(im, p1, p2, color, 2)
        if label:
            label_text = label
        else:
            label_text = f"cls{int(cls_id)}"
        text = f"{label_text}:{score:.2f}"
        cv2.putText(
            im,
            text,
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(out_path, im)


def draw_and_save_with_ids(
    img_path: str,
    boxes: np.ndarray,
    ids: np.ndarray,
    thresh: float,
    out_path: str,
    label: Optional[str] = "object",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im = cv2.imread(img_path)
    if im is None:
        print(f"[WARN] Could not load image to draw: {img_path}")
        return
    for k, b in enumerate(boxes):
        cls_id, score, x0, y0, x1, y1 = b
        if cls_id < 0 or score < thresh:
            continue
        p1 = (int(x0), int(y0))
        p2 = (int(x1), int(y1))
        color = (0, 200, 255)
        cv2.rectangle(im, p1, p2, color, 2)
        if label:
            label_text = label
        else:
            label_text = f"cls{int(cls_id)}"
        text = f"id{int(ids[k])}:{label_text}:{score:.2f}"
        cv2.putText(
            im,
            text,
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(out_path, im)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_coreml_version_info() -> dict:
    info = {}
    try:
        import coremltools as ct  # type: ignore

        info["coremltools"] = getattr(ct, "__version__", "unknown")
    except Exception:
        info["coremltools"] = None

    plist_candidates = [
        "/System/Library/Frameworks/CoreML.framework/Resources/Info.plist",
        "/System/Library/Frameworks/CoreML.framework/Versions/Current/Resources/Info.plist",
    ]
    framework = None
    for path in plist_candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    pl = plistlib.load(f)
                framework = {
                    "CFBundleShortVersionString": pl.get("CFBundleShortVersionString"),
                    "CFBundleVersion": pl.get("CFBundleVersion"),
                    "path": path,
                }
                break
            except Exception:
                continue
    info["framework"] = framework
    try:
        info["macOS"] = platform.mac_ver()[0]
    except Exception:
        info["macOS"] = None
    return info


def default_paths() -> Tuple[str, str, str]:
    root = repo_root()
    model_name = "ppyoloe_crn_s_36e_pphuman"
    onnx_candidates = [
        os.path.join(root, "pipeline", "PP-YOLOE", "models", f"{model_name}_embed_det.onnx"),
        os.path.join(root, "pipeline", "PP-YOLOE", "models", f"{model_name}_embed.onnx"),
        os.path.join(root, "pipeline", "PP-YOLOE", "models", f"{model_name}.onnx"),
        os.path.join(root, "pipeline", "output", "onnx", f"{model_name}.onnx"),
        os.path.join(root, "pipeline", "output", f"{model_name}.onnx"),
    ]
    onnx_path = next((p for p in onnx_candidates if os.path.exists(p)), onnx_candidates[0])

    img_path = os.path.join(root, "pipeline", "dataset", "demo", "demo.jpg")
    out_dir = os.path.join(root, "pipeline", "output", "onnx_vis")
    return onnx_path, img_path, out_dir


def get_hardcoded_preprocess() -> Tuple[float, str, str]:
    draw_threshold = 0.5
    arch = "YOLO"
    label_name = "object"
    return draw_threshold, arch, label_name


def run_session_with_warmup(sess, feed: dict, warmup_runs: int = 3, verbose: bool = True):
    if verbose:
        print(f"[INFO] Warming up model ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        sess.run(None, feed)
    t0 = time.perf_counter()
    outputs = sess.run(None, feed)
    t1 = time.perf_counter()
    out_names = [o.name for o in sess.get_outputs()]
    inference_ms = (t1 - t0) * 1000.0
    return outputs, out_names, inference_ms


def detect_model_type(out_names: Sequence[str]) -> str:
    lowered = [name.lower() for name in out_names]
    if any(name.startswith("fetch_name_") for name in out_names):
        return "original"
    if "embed" in out_names:
        return "original"
    if any("divide" in name for name in lowered) and any("concat" in name for name in lowered):
        return "ane"
    if len(out_names) >= 4:
        return "ane"
    return "unknown"


def print_outputs_header(model_type: str, out_names: Sequence[str], outputs: Sequence[np.ndarray], draw_threshold: float):
    print("\nModel outputs:", list(out_names))
    if model_type == "original":
        print(" - outputs[0]: detections (N,6) [class_id, score, x0, y0, x1, y1]")
        if "embed" in out_names:
            print(" - 'embed': per-detection embeddings (N, D) - float32")
        else:
            print(" - Original model with NMS (no embeddings)")
    elif model_type == "ane":
        if outputs:
            print(f" ✓ ANE-optimized model (automatic surgery) with {len(outputs)} outputs")
        if len(outputs) >= 1:
            print(f" - outputs[0]: raw boxes {outputs[0].shape} [x_center, y_center, w, h]")
        if len(outputs) >= 2:
            print(f" - outputs[1]: raw scores {outputs[1].shape} - needs squeeze and NMS")
        if len(outputs) >= 3:
            print(f" - outputs[2]: stride-8 features {outputs[2].shape} - fine-grained")
        if len(outputs) >= 4:
            s8_ch = outputs[2].shape[1]
            s16_ch = outputs[3].shape[1]
            total_dim = s8_ch + s16_ch
            print(f" - outputs[3]: stride-16 features {outputs[3].shape} - semantic")
            print(f" ✓ Multi-scale embeddings: {total_dim}D ({s8_ch} + {s16_ch})")
    else:
        print(" - Unknown model structure, see output details below")

    print("\n[Debug] Model outputs (names, shapes, and quick notes):")
    for i, name in enumerate(out_names):
        arr = outputs[i]
        shape = getattr(arr, "shape", None)
        print(f"  - {name}: {shape}")
        try:
            if (
                model_type == "original"
                and i == 0
                and isinstance(arr, np.ndarray)
                and arr.ndim == 2
                and arr.shape[1] == 6
            ):
                dets = arr
                num = dets.shape[0]
                num_valid = int(((dets[:, 0] > -1) & (dets[:, 1] >= float(draw_threshold))).sum()) if num else 0
                print("      • detections (N,6) = [class, score, x0, y0, x1, y1]")
                print(f"      • N={num}, valid(≥{float(draw_threshold):.2f})={num_valid}")
                if num:
                    k = min(3, num)
                    print("      • samples:")
                    for r in range(k):
                        cls, sc, x0, y0, x1, y1 = dets[r]
                        print(f"         {int(cls)} {sc:.4f} {x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}")
            elif name == "embed" and isinstance(arr, np.ndarray) and arr.ndim == 2:
                n_rows, dim = arr.shape
                print("      • embeddings (N,D), L2-normalized expected downstream")
                print(f"      • N={n_rows}, D={dim}")
                if n_rows > 0:
                    preview = arr[0, : min(8, dim)]
                    pv_str = " ".join(f"{v:.2f}" for v in preview.tolist())
                    print(f"      • first emb[:{min(8, dim)}]=[{pv_str}]")
            elif isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size <= 4:
                vals = " ".join(f"{float(v):.3f}" for v in arr.tolist())
                print(f"      • auxiliary vector: [{vals}] (often metadata)")
            else:
                flat = arr.ravel() if isinstance(arr, np.ndarray) else []
                if isinstance(flat, np.ndarray) and flat.size:
                    k = min(8, flat.size)
                    vals = " ".join(f"{float(v):.3f}" for v in flat[:k].tolist())
                    more = " …" if flat.size > k else ""
                    print(f"      • sample: [{vals}]{more}")
        except Exception:
            continue


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = box_a[2], box_a[3], box_a[4], box_a[5]
    bx0, by0, bx1, by1 = box_b[2], box_b[3], box_b[4], box_b[5]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def print_embedding_diagnostics(
    det_embs: np.ndarray,
    embs_valid: np.ndarray,
    boxes_valid: np.ndarray,
):
    print("\n[BoT-SORT] Per-detection embeddings ready (console only):")
    print("  - embed (all):", det_embs.shape)
    print("  - embed_valid:", embs_valid.shape)
    print("  - detections_valid:", boxes_valid.shape)

    d = det_embs.shape[1] if det_embs.ndim == 2 else 0
    print("\n[Embeddings check] Basic stats:")
    print(f"  - embedding dim: {d}, total N: {det_embs.shape[0] if det_embs.ndim == 2 else 0}, valid N: {embs_valid.shape[0] if embs_valid.ndim == 2 else 0}")
    if det_embs.size:
        norms_all = np.linalg.norm(det_embs, axis=1)
        print(f"  - L2 norms (all, after normalization): min={norms_all.min():.4f} mean={norms_all.mean():.4f} max={norms_all.max():.4f}")
    else:
        print("  - L2 norms (all, after normalization): n/a")

    if embs_valid.size:
        norms_valid = np.linalg.norm(embs_valid, axis=1)
        print(f"  - L2 norms (valid): min={norms_valid.min():.4f} mean={norms_valid.mean():.4f} max={norms_valid.max():.4f}")
        zero_rate = float((norms_valid < 1e-8).sum()) / float(norms_valid.size)
        print(f"  - zero-vector rate (valid): {zero_rate * 100:.1f}%")

        if embs_valid.shape[0] >= 2:
            cos = embs_valid @ embs_valid.T
            cos_off = cos.copy()
            np.fill_diagonal(cos_off, -np.inf)
            finite_mask = np.isfinite(cos_off)
            flat = cos_off[finite_mask]
            if flat.size:
                p5 = np.percentile(flat, 5)
                p50 = np.percentile(flat, 50)
                p95 = np.percentile(flat, 95)
                print(f"  - pairwise cosine (valid): min={flat.min():.3f} p5={p5:.3f} median={p50:.3f} p95={p95:.3f} max={flat.max():.3f}")

            iu = np.triu_indices(cos.shape[0], k=1)
            cos_pairs = cos[iu]
            K = min(5, iu[0].size)
            order = np.argsort(-cos_pairs)[:K]
            print("  - top similar pairs (idx_i, idx_j, cosine, IoU):")
            for idx in order:
                i = iu[0][idx]
                j = iu[1][idx]
                cosine = float(cos_pairs[idx])
                iou_val = iou_xyxy(boxes_valid[i], boxes_valid[j]) if len(boxes_valid) > max(i, j) else 0.0
                print(f"     ({i:2d}, {j:2d})  cos={cosine:.3f}  IoU={iou_val:.3f}")

            high_mask = cos_pairs > 0.9
            if high_mask.any():
                high_i = iu[0][high_mask]
                high_j = iu[1][high_mask]
                ious = np.array([iou_xyxy(boxes_valid[i], boxes_valid[j]) for i, j in zip(high_i, high_j)], dtype=np.float32)
                non_overlap = (ious < 0.1).mean() if ious.size else 0.0
                print(f"  - high-cos (>0.90) non-overlapping pair rate: {non_overlap * 100:.1f}%")

            nn_idx = np.argmax(cos_off, axis=1)
            nn_cos = cos_off[np.arange(cos.shape[0]), nn_idx]
            order_nn = np.argsort(-nn_cos)
            print("  - per-detection nearest neighbor (sorted by cosine):")
            for t in order_nn[: min(10, cos.shape[0])]:
                i = int(t)
                j = int(nn_idx[i])
                iou_val = iou_xyxy(boxes_valid[i], boxes_valid[j]) if len(boxes_valid) > max(i, j) else 0.0
                print(
                    f"     i={i:2d} -> j={j:2d}  cos={nn_cos[i]:.3f}  IoU={iou_val:.3f}  score={boxes_valid[i, 1]:.3f}"
                )

            dims_preview = min(8, embs_valid.shape[1])
            print(f"  - embedding previews (first {dims_preview} dims):")
            for idx in range(embs_valid.shape[0]):
                cls_id, score, x0, y0, x1, y1 = boxes_valid[idx]
                nn_j = int(nn_idx[idx])
                preview = embs_valid[idx, :dims_preview]
                pv_str = " ".join(f"{v:.2f}" for v in preview.tolist())
                is_zero = " ZERO" if np.linalg.norm(embs_valid[idx]) < 1e-8 else ""
                iou_val = iou_xyxy(boxes_valid[idx], boxes_valid[nn_j]) if len(boxes_valid) > nn_j else 0.0
                print(
                    f"     id={idx:2d} cls={int(cls_id)} score={score:.3f} box=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]{is_zero}"
                )
                print(
                    f"        emb[:{dims_preview}]=[{pv_str}]  NN-> id={nn_j:2d} cos={nn_cos[idx]:.3f} IoU={iou_val:.3f}"
                )
        else:
            print("  - Not enough valid detections for pairwise comparison.")
    else:
        print("  - Not enough valid detections for pairwise comparison.")
