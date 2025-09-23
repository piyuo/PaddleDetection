#!/usr/bin/env python3
"""
Run inference on a single image using ONNX Runtime with the exported PP-YOLOE Human model.

Usage:
    python pipeline/PP-YOLOE/onnx_inference_image.py \
        [--img pipeline/dataset/demo/demo.jpg] \
        [--onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx] \
        [--out pipeline/output/onnx_vis] \
        [--thresh 0.5] [--gpu]

Requirement:
    - The ONNX model must expose per-detection embeddings named "embed" with shape (N, D),
        where N matches the number of rows in the detection output (top-K). This script will error out
        if "embed" is not present.

Notes:
    - This script reuses PaddleDetection's ONNX preprocess (deploy/third_engine/onnx/preprocess.py).
    - Outputs are printed to stdout, with optional visualization and .npy files in --out.
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np

def roi_pool_average(feat_map: np.ndarray, boxes_xyxy: np.ndarray, img_hw: Tuple[int, int]) -> np.ndarray:
    """
    Simple ROI average pooling on a feature map.
    - feat_map: (1, C, Hf, Wf)
    - boxes_xyxy: (N, 4) in original image coordinates
    - img_hw: (Himg, Wimg)
    Returns:
      embeddings: (N, C)
    """
    assert feat_map.ndim == 4 and feat_map.shape[0] == 1
    _, C, Hf, Wf = feat_map.shape
    Himg, Wimg = img_hw
    scale_x = Wf / float(Wimg)
    scale_y = Hf / float(Himg)

    embs = []
    for x0, y0, x1, y1 in boxes_xyxy:
        fx0 = int(max(0, np.floor(x0 * scale_x)))
        fy0 = int(max(0, np.floor(y0 * scale_y)))
        fx1 = int(min(Wf, np.ceil(x1 * scale_x)))
        fy1 = int(min(Hf, np.ceil(y1 * scale_y)))
        if fx1 <= fx0 or fy1 <= fy0:
            embs.append(np.zeros((C,), dtype=np.float32))
            continue
        region = feat_map[0, :, fy0:fy1, fx0:fx1]
        vec = region.reshape(C, -1).mean(axis=1) if region.size > 0 else np.zeros((C,), dtype=np.float32)
        embs.append(vec)
    return np.stack(embs, axis=0) if embs else np.zeros((0, C), dtype=np.float32)


def repo_root() -> str:
    # This file is at <repo>/pipeline/PP-YOLOE/onnx_inference_image.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def default_paths() -> Tuple[str, str, str, str]:
    root = repo_root()
    model_name = 'ppyoloe_crn_s_36e_pphuman'
    # Try likely ONNX locations in priority order
    onnx_candidates = [
        os.path.join(root, 'pipeline', 'PP-YOLOE', 'models', f'{model_name}_embed_det.onnx'),
        os.path.join(root, 'pipeline', 'PP-YOLOE', 'models', f'{model_name}_embed.onnx'),
        os.path.join(root, 'pipeline', 'PP-YOLOE', 'models', f'{model_name}.onnx'),
        os.path.join(root, 'pipeline', 'output', 'onnx', f'{model_name}.onnx'),
        os.path.join(root, 'pipeline', 'output', f'{model_name}.onnx'),
    ]
    onnx_path = next((p for p in onnx_candidates if os.path.exists(p)), onnx_candidates[0])

    img_path = os.path.join(root, 'pipeline', 'dataset', 'demo', 'demo.jpg')
    out_dir = os.path.join(root, 'pipeline', 'output', 'onnx_vis')
    pd_onnx_preprocess_dir = os.path.join(root, 'deploy', 'third_engine', 'onnx')
    return onnx_path, img_path, out_dir, pd_onnx_preprocess_dir


def get_session(onnx_path: str, use_gpu: bool):
    try:
        import onnxruntime as ort
    except Exception as e:
        print('[ERROR] onnxruntime not installed. Install with: pip install onnxruntime', file=sys.stderr)
        raise

    providers = None
    if use_gpu:
        # Use CUDA if available
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print('[WARN] CUDAExecutionProvider not available; falling back to CPU.')
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess


def get_hardcoded_preprocess(preprocess_dir: str):
    # Make PaddleDetection ONNX preprocess importable
    if preprocess_dir not in sys.path:
        sys.path.insert(0, preprocess_dir)
    from preprocess import Compose  # type: ignore

    # Hardcoded preprocessing config from infer_cfg.yml
    preprocess_infos = [
        {
            'type': 'Resize',
            'interp': 2,
            'keep_ratio': False,
            'target_size': [640, 640]
        },
        {
            'type': 'NormalizeImage',
            'is_scale': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        {
            'type': 'Permute'
        }
    ]

    draw_threshold = 0.5
    arch = 'YOLO'

    # COCO label list (80 classes)
    label_list = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    transforms = Compose(preprocess_infos)
    return transforms, draw_threshold, arch, label_list


def draw_and_save(img_path: str, boxes: np.ndarray, thresh: float, out_path: str, labels):
    import cv2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # load in BGR for drawing
    im = cv2.imread(img_path)
    if im is None:
        print('[WARN] Could not load image to draw:', img_path)
        return
    for b in boxes:
        cls_id, score, x0, y0, x1, y1 = b
        if cls_id < 0 or score < thresh:
            continue
        p1 = (int(x0), int(y0))
        p2 = (int(x1), int(y1))
        color = (0, 255, 0)
        cv2.rectangle(im, p1, p2, color, 2)
        label = labels[int(cls_id)] if isinstance(labels, (list, tuple)) and int(cls_id) < len(labels) else f'cls{int(cls_id)}'
        text = f'{label}:{score:.2f}'
        cv2.putText(im, text, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, im)


def draw_and_save_with_ids(img_path: str, boxes: np.ndarray, ids: np.ndarray, thresh: float, out_path: str, labels):
    """Draw boxes with an integer id prefix (e.g., id0, id1) for easier matching with printed tables."""
    import cv2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im = cv2.imread(img_path)
    if im is None:
        print('[WARN] Could not load image to draw:', img_path)
        return
    for k, b in enumerate(boxes):
        cls_id, score, x0, y0, x1, y1 = b
        if cls_id < 0 or score < thresh:
            continue
        p1 = (int(x0), int(y0))
        p2 = (int(x1), int(y1))
        color = (0, 200, 255)  # orange-ish for id view
        cv2.rectangle(im, p1, p2, color, 2)
        label = labels[int(cls_id)] if isinstance(labels, (list, tuple)) and int(cls_id) < len(labels) else f'cls{int(cls_id)}'
        text = f'id{int(ids[k])}:{label}:{score:.2f}'
        cv2.putText(im, text, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, im)


def main():
    d_onnx, d_img, d_out, d_preproc_dir = default_paths()

    parser = argparse.ArgumentParser(description='ONNX Runtime inference for PP-YOLOE Human on one image')
    parser.add_argument('--img', default=d_img, help='Path to input image')
    parser.add_argument('--onnx', default=d_onnx, help='Path to ONNX model file')
    parser.add_argument('--out', default=d_out, help='Directory to save visualization')
    parser.add_argument('--thresh', type=float, default=None, help='Score threshold for printing/drawing (default: 0.5)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if onnxruntime-gpu is available')
    args = parser.parse_args()

    # Validate inputs
    for p, label in [
        (args.img, 'Input image'),
        (args.onnx, 'ONNX model'),
    ]:
        if not os.path.exists(p):
            print(f'[ERROR] {label} not found: {p}', file=sys.stderr)
            sys.exit(1)

    # Load preprocess and session
    transforms, draw_threshold, arch, label_list = get_hardcoded_preprocess(d_preproc_dir)
    if args.thresh is not None:
        draw_threshold = args.thresh
    sess = get_session(args.onnx, args.gpu)

    # Prepare inputs using Compose. It will return a dict keyed by model input names.
    inputs_map = transforms(args.img)
    input_names = [i.name for i in sess.get_inputs()]
    feed = {name: inputs_map[name][None, ] for name in input_names}

    # Run
    outputs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]

    # Combined model outputs summary (names + shapes) and expectations
    print('Model outputs:', out_names)
    print(" - expected outputs[0]: detections (N,6) [class, score, x0, y0, x1, y1]")
    if 'embed' in out_names:
        print(" - 'embed': per-detection embeddings (N, D)")
    else:
        print(" - 'embed' not present: run insert_embedding_head.py to add embeddings or use *_embed.onnx")
    print('\n[Debug] Model outputs (names, shapes, and quick notes):')
    for i, n in enumerate(out_names):
        arr = outputs[i]
        shape = getattr(arr, 'shape', None)
        print(f'  - {n}: {shape}')
        try:
            # Heuristic descriptions and small samples
            if i == 0 and isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 6:
                # Detection head
                dets = arr
                num = dets.shape[0]
                num_valid = int(((dets[:, 0] > -1) & (dets[:, 1] >= float(draw_threshold))).sum()) if num else 0
                print('      • detections (N,6) = [class, score, x0, y0, x1, y1]')
                print(f'      • N={num}, valid(≥{float(draw_threshold):.2f})={num_valid}')
                if num:
                    k = min(3, num)
                    print('      • samples:')
                    for r in range(k):
                        cls, sc, x0, y0, x1, y1 = dets[r]
                        print(f'         {int(cls)} {sc:.4f} {x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}')
            elif n == 'embed' and isinstance(arr, np.ndarray) and arr.ndim == 2:
                N, D = arr.shape
                print('      • embeddings (N,D), L2-normalized expected downstream')
                print(f'      • N={N}, D={D}')
                if N > 0:
                    pv = arr[0, :min(8, D)]
                    pv_str = ' '.join(f'{v:.2f}' for v in pv.tolist())
                    print(f'      • first emb[:{min(8,D)}]=[{pv_str}]')
            elif isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size <= 4:
                # Likely auxiliary scalar/vector (often count or image meta); safe to ignore for tracking
                vals = ' '.join(f'{float(v):.3f}' for v in arr.tolist())
                hint = ' (often count/image meta; usually safe to ignore)'
                print(f'      • auxiliary vector: [{vals}]' + hint)
            else:
                # Generic small sample
                flat = arr.ravel() if isinstance(arr, np.ndarray) else []
                if isinstance(flat, np.ndarray) and flat.size:
                    k = min(8, flat.size)
                    vals = ' '.join(f'{float(v):.3f}' for v in flat[:k].tolist())
                    more = ' …' if flat.size > k else ''
                    print(f'      • sample: [{vals}]{more}')
        except Exception:
            # best-effort only
            pass

    # Post-process for PP-YOLOE: first output is [N,6] -> [class_id, score, x0, y0, x1, y1]
    bboxes = np.array(outputs[0])

    print('Detections (class score x0 y0 x1 y1):')
    kept = 0
    for b in bboxes:
        if int(b[0]) > -1 and float(b[1]) >= float(draw_threshold):
            kept += 1
            print(f'{int(b[0])} {b[1]:.4f} {b[2]:.1f} {b[3]:.1f} {b[4]:.1f} {b[5]:.1f}')
    if kept == 0:
        print(f'No boxes above threshold {draw_threshold}. Try lowering --thresh.')

    base = os.path.splitext(os.path.basename(args.img))[0]
    vis_path = os.path.join(args.out, f'{base}.jpg')

    # --- Require per-detection embeddings for BoT-SORT ---
    name_to_out = {out_names[i]: outputs[i] for i in range(len(out_names))}
    # Enforce presence of per-detection embeddings
    if 'embed' not in name_to_out or not isinstance(name_to_out['embed'], np.ndarray):
        print('\n[ERROR] Model does not expose per-detection embeddings "embed".', file=sys.stderr)
        print('        Use pipeline/PP-YOLOE/insert_embedding_head.py to augment your model, or load the *_embed.onnx.', file=sys.stderr)
        sys.exit(2)

    det_embs = name_to_out['embed']
    if det_embs.ndim != 2 or det_embs.shape[0] == 0:
        print('\n[ERROR] "embed" must be a 2D array shaped (N, D) with N>0. Got:', det_embs.shape, file=sys.stderr)
        sys.exit(2)

    if det_embs.shape[0] != bboxes.shape[0]:
        print('\n[ERROR] Row count mismatch between detections and embeddings:', file=sys.stderr)
        print('        detections:', bboxes.shape, ' embed:', det_embs.shape, file=sys.stderr)
        print('        Ensure your model outputs align. Regenerate with insert_embedding_head.py if needed.', file=sys.stderr)
        sys.exit(2)

    # Normalize per-detection embeddings
    det_embs = det_embs.astype(np.float32)
    det_embs = det_embs / (np.linalg.norm(det_embs, axis=1, keepdims=True) + 1e-8)

    # Filter by threshold to match drawn/kept detections
    valid_mask = (bboxes[:, 0] > -1) & (bboxes[:, 1] >= float(draw_threshold))
    boxes_valid = bboxes[valid_mask]
    embs_valid = det_embs[valid_mask]

    base = os.path.splitext(os.path.basename(args.img))[0]
    os.makedirs(args.out, exist_ok=True)

    # Save a single visualization with valid detection ids
    try:
        ids_valid = np.arange(boxes_valid.shape[0])
        draw_and_save_with_ids(args.img, boxes_valid, ids_valid, float(draw_threshold), vis_path, label_list)
        print('Saved visualization to:', vis_path)
    except Exception as e:
        print('[WARN] Failed to save visualization:', e)

    print('\n[BoT-SORT] Per-detection embeddings ready (console only):')
    print('  - embed (all):', det_embs.shape)
    print('  - embed_valid:', embs_valid.shape)
    print('  - detections_valid:', boxes_valid.shape)

    # Embedding sanity checks to ensure values are informative per detection (always on)
    print('\n[Embeddings check] Basic stats:')
    D = det_embs.shape[1]
    print(f'  - embedding dim: {D}, total N: {det_embs.shape[0]}, valid N: {embs_valid.shape[0]}')
    norms = np.linalg.norm(det_embs, axis=1)
    print(f'  - L2 norms (all, after normalization): min={norms.min():.4f} mean={norms.mean():.4f} max={norms.max():.4f}')
    norms_v = np.linalg.norm(embs_valid, axis=1) if embs_valid.size else np.array([])
    if norms_v.size:
        print(f'  - L2 norms (valid): min={norms_v.min():.4f} mean={norms_v.mean():.4f} max={norms_v.max():.4f}')
        # Zero-vector rate (after normalization, zero means original vector was zero)
        zero_rate = float((norms_v < 1e-8).sum()) / float(norms_v.size)
        print(f'  - zero-vector rate (valid): {zero_rate*100:.1f}%')

        def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
            # a,b: (6,) [cls,score,x0,y0,x1,y1]
            ax0, ay0, ax1, ay1 = a[2], a[3], a[4], a[5]
            bx0, by0, bx1, by1 = b[2], b[3], b[4], b[5]
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
            inter = iw * ih
            area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
            area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
            union = area_a + area_b - inter + 1e-6
            return float(inter / union)

        # Report cosine similarities on valid detections
        if embs_valid.shape[0] >= 2:
            # embeddings are L2-normalized; cosine = dot product
            cos = embs_valid @ embs_valid.T
            nv = cos.shape[0]
            # build an off-diagonal view for stats and NN (avoid self=1.0)
            cos_off = cos.copy()
            np.fill_diagonal(cos_off, -np.inf)
            # pairwise stats (ignore -inf placeholders)
            flat = cos_off[np.isfinite(cos_off)].ravel()
            if flat.size:
                p5 = np.percentile(flat, 5)
                p50 = np.percentile(flat, 50)
                p95 = np.percentile(flat, 95)
                print(f'  - pairwise cosine (valid): min={flat.min():.3f} p5={p5:.3f} median={p50:.3f} p95={p95:.3f} max={flat.max():.3f}')

            # Top-K most similar pairs (to spot potential duplicates)
            K = min(5, nv * (nv - 1) // 2)
            iu = np.triu_indices(nv, k=1)
            cos_pairs = cos[iu]
            order = np.argsort(-cos_pairs)[:K]
            print('  - top similar pairs (idx_i, idx_j, cosine, IoU):')
            for r in order:
                i, j = iu[0][r], iu[1][r]
                c = float(cos[i, j])
                iou = iou_xyxy(boxes_valid[i], boxes_valid[j])
                print(f'     ({i:2d}, {j:2d})  cos={c:.3f}  IoU={iou:.3f}')

            # Health summary: fraction of high-cos pairs that don't overlap (potential ID confusion)
            high = cos_pairs > 0.9
            if high.any():
                i_idx, j_idx = iu[0][high], iu[1][high]
                ious = np.array([iou_xyxy(boxes_valid[i], boxes_valid[j]) for i, j in zip(i_idx, j_idx)], dtype=np.float32)
                non_overlap = (ious < 0.1).mean() if ious.size else 0.0
                print(f'  - high-cos (>0.90) non-overlapping pair rate: {non_overlap*100:.1f}%')

            # Console-only mode: do not save report or arrays

            # Per-detection nearest neighbor summary (valid only)
            nn_idx = np.argmax(cos_off, axis=1)
            nn_cos = cos_off[np.arange(nv), nn_idx]
            # Print a compact table (top few with highest nn cosine)
            order_nn = np.argsort(-nn_cos)
            print('  - per-detection nearest neighbor (sorted by cosine):')
            for t in order_nn[:min(10, nv)]:
                i = int(t)
                j = int(nn_idx[i])
                iou = iou_xyxy(boxes_valid[i], boxes_valid[j])
                print(f'     i={i:2d} -> j={j:2d}  cos={nn_cos[i]:.3f}  IoU={iou:.3f}  score={boxes_valid[i,1]:.3f}')

            # Also print a short embedding preview per valid detection (first 8 dims)
            dims_preview = min(8, D)
            print('  - embedding previews (first', dims_preview, 'dims):')
            for i in range(nv):
                cls_id, score, x0, y0, x1, y1 = boxes_valid[i]
                j = int(nn_idx[i])
                pv = embs_valid[i, :dims_preview]
                pv_str = ' '.join([f'{v:.2f}' for v in pv.tolist()])
                is_zero = ' ZERO' if np.linalg.norm(embs_valid[i]) < 1e-8 else ''
                print(f'     id={i:2d} cls={int(cls_id)} score={score:.3f} box=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]{is_zero}')
                print(f'        emb[:{dims_preview}]=[{pv_str}]  NN-> id={j:2d} cos={nn_cos[i]:.3f} IoU={iou_xyxy(boxes_valid[i], boxes_valid[j]):.3f}')

        else:
            print('  - Not enough valid detections for pairwise comparison.')

    # Final success message with specific file path
    print(f'\n✅ ONNX inference completed! Generated visualization: {vis_path}')
    return


if __name__ == '__main__':
    main()
