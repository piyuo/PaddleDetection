#!/usr/bin/env python3
"""
Run inference on a single image using ONNX Runtime with the exported PP-YOLOE Human model.

Usage:
    python pipeline/PP-YOLOE/onnx_inference_image.py \
        [--img pipeline/dataset/demo/demo.jpg] \
        [--onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_det.onnx] \
        [--infer_cfg pipeline/PP-YOLOE/backbone/inference_model/ppyoloe_crn_s_36e_pphuman/infer_cfg.yml] \
        [--out pipeline/output/onnx_vis] \
        [--thresh 0.5] [--gpu]

Requirement:
    - The ONNX model must expose per-detection embeddings named "embeddings_det" with shape (N, D),
        where N matches the number of rows in the detection output (top-K). This script will error out
        if "embeddings_det" is not present.

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


def default_paths() -> Tuple[str, str, str, str, str]:
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

    # Try likely infer_cfg locations in priority order
    infer_cfg_candidates = [
        os.path.join(root, 'pipeline', 'PP-YOLOE', 'backbone', 'inference_model', model_name, 'infer_cfg.yml'),
        os.path.join(root, 'pipeline', 'output', 'inference_model', model_name, 'infer_cfg.yml'),
    ]
    infer_cfg = next((p for p in infer_cfg_candidates if os.path.exists(p)), infer_cfg_candidates[0])
    img_path = os.path.join(root, 'pipeline', 'dataset', 'demo', 'demo.jpg')
    out_dir = os.path.join(root, 'pipeline', 'output', 'onnx_vis')
    pd_onnx_preprocess_dir = os.path.join(root, 'deploy', 'third_engine', 'onnx')
    return onnx_path, infer_cfg, img_path, out_dir, pd_onnx_preprocess_dir


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


def load_preprocess(infer_cfg_path: str, preprocess_dir: str):
    # Make PaddleDetection ONNX preprocess importable
    if preprocess_dir not in sys.path:
        sys.path.insert(0, preprocess_dir)
    # The preprocess module expects YAML loading done in their infer.py PredictConfig
    import yaml
    from preprocess import Compose  # type: ignore

    with open(infer_cfg_path, 'r') as f:
        yml_conf = yaml.safe_load(f)
    preprocess_infos = yml_conf['Preprocess']
    draw_threshold = float(yml_conf.get('draw_threshold', 0.5))
    arch = yml_conf.get('arch', '')
    label_list = yml_conf.get('label_list', [])
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
    d_onnx, d_infer_cfg, d_img, d_out, d_preproc_dir = default_paths()

    parser = argparse.ArgumentParser(description='ONNX Runtime inference for PP-YOLOE Human on one image')
    parser.add_argument('--img', default=d_img, help='Path to input image')
    parser.add_argument('--onnx', default=d_onnx, help='Path to ONNX model file')
    parser.add_argument('--infer_cfg', default=d_infer_cfg, help='Path to infer_cfg.yml from exported Paddle model')
    parser.add_argument('--out', default=d_out, help='Directory to save visualization')
    parser.add_argument('--thresh', type=float, default=None, help='Score threshold for printing/drawing (default from infer_cfg)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if onnxruntime-gpu is available')
    parser.add_argument('--list_outputs', action='store_true', help='Print ONNX output names and shapes')
    parser.add_argument('--check_embed', action='store_true',
                        help='Run embedding sanity checks (cosine similarity stats, top similar pairs) and save a brief report')
    args = parser.parse_args()

    # Validate inputs
    for p, label in [
        (args.img, 'Input image'),
        (args.onnx, 'ONNX model'),
        (args.infer_cfg, 'infer_cfg.yml'),
    ]:
        if not os.path.exists(p):
            print(f'[ERROR] {label} not found: {p}', file=sys.stderr)
            sys.exit(1)

    # Load preprocess and session
    transforms, draw_threshold, arch, label_list = load_preprocess(args.infer_cfg, d_preproc_dir)
    if args.thresh is not None:
        draw_threshold = args.thresh
    sess = get_session(args.onnx, args.gpu)

    # Prepare inputs using Compose. It will return a dict keyed by model input names.
    inputs_map = transforms(args.img)
    input_names = [i.name for i in sess.get_inputs()]
    feed = {name: inputs_map[name][None, ] for name in input_names}

    # Run
    outputs = sess.run(None, feed)

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

    # Save visualization
    base = os.path.splitext(os.path.basename(args.img))[0]
    vis_path = os.path.join(args.out, f'{base}.jpg')
    try:
        draw_and_save(args.img, bboxes, draw_threshold, vis_path, label_list)
        print('Saved visualization to:', vis_path)
    except Exception as e:
        print('[WARN] Failed to save visualization:', e)

    # --- Require per-detection embeddings for BoT-SORT ---
    out_names = [o.name for o in sess.get_outputs()]
    name_to_out = {out_names[i]: outputs[i] for i in range(len(out_names))}
    if args.list_outputs:
        print('\n[Debug] Model outputs:')
        for i, n in enumerate(out_names):
            arr = outputs[i]
            shape = getattr(arr, 'shape', None)
            print(f'  - {n}: {shape}')
    # Enforce presence of per-detection embeddings
    if 'embeddings_det' not in name_to_out or not isinstance(name_to_out['embeddings_det'], np.ndarray):
        print('\n[ERROR] Model does not expose per-detection embeddings "embeddings_det".', file=sys.stderr)
        if not args.list_outputs:
            print('        Tip: run with --list_outputs to see available outputs.', file=sys.stderr)
        print('        Use pipeline/PP-YOLOE/insert_embedding_head.py to augment your model, or load the *_embed_det.onnx.', file=sys.stderr)
        sys.exit(2)

    det_embs = name_to_out['embeddings_det']
    if det_embs.ndim != 2 or det_embs.shape[0] == 0:
        print('\n[ERROR] "embeddings_det" must be a 2D array shaped (N, D) with N>0. Got:', det_embs.shape, file=sys.stderr)
        sys.exit(2)

    if det_embs.shape[0] != bboxes.shape[0]:
        print('\n[ERROR] Row count mismatch between detections and embeddings:', file=sys.stderr)
        print('        detections:', bboxes.shape, ' embeddings_det:', det_embs.shape, file=sys.stderr)
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
    path_det_all = os.path.join(args.out, f'{base}_embeddings_det.npy')
    path_det_valid = os.path.join(args.out, f'{base}_embeddings_det_valid.npy')
    path_boxes_valid = os.path.join(args.out, f'{base}_detections_valid.npy')
    path_npz = os.path.join(args.out, f'{base}_botsort_inputs.npz')
    np.save(path_det_all, det_embs)
    np.save(path_det_valid, embs_valid)
    np.save(path_boxes_valid, boxes_valid)
    np.savez(path_npz, boxes=boxes_valid, embeddings=embs_valid)

    print('\n[BoT-SORT] Per-detection embeddings ready:')
    print('  - embeddings_det (all):', det_embs.shape, ' →', path_det_all)
    print('  - embeddings_det_valid:', embs_valid.shape, ' →', path_det_valid)
    print('  - detections_valid:', boxes_valid.shape, ' →', path_boxes_valid)
    print('  - combined (npz):', path_npz, ' (keys: boxes, embeddings)')

    # Optional: Embedding sanity checks to ensure values are informative per detection
    if args.check_embed:
        print('\n[Embeddings check] Basic stats:')
        D = det_embs.shape[1]
        print(f'  - embedding dim: {D}, total N: {det_embs.shape[0]}, valid N: {embs_valid.shape[0]}')
        norms = np.linalg.norm(det_embs, axis=1)
        print(f'  - L2 norms (all, after normalization): min={norms.min():.4f} mean={norms.mean():.4f} max={norms.max():.4f}')
        norms_v = np.linalg.norm(embs_valid, axis=1) if embs_valid.size else np.array([])
        if norms_v.size:
            print(f'  - L2 norms (valid): min={norms_v.min():.4f} mean={norms_v.mean():.4f} max={norms_v.max():.4f}')

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
            # exclude self-similarity
            nv = cos.shape[0]
            cos_nodiag = cos.copy()
            np.fill_diagonal(cos_nodiag, np.nan)
            # pairwise stats
            flat = cos_nodiag[~np.isnan(cos_nodiag)].ravel()
            if flat.size:
                p5 = np.percentile(flat, 5)
                p50 = np.percentile(flat, 50)
                p95 = np.percentile(flat, 95)
                print(f'  - pairwise cosine (valid): min={np.nanmin(cos_nodiag):.3f} p5={p5:.3f} median={p50:.3f} p95={p95:.3f} max={np.nanmax(cos_nodiag):.3f}')

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

            # Save a brief report
            report_txt = os.path.join(args.out, f'{base}_embed_check.txt')
            with open(report_txt, 'w') as f:
                f.write('Embedding Sanity Report\n')
                f.write(f'image: {args.img}\n')
                f.write(f'valid detections: {nv}\n')
                f.write(f'embedding dim: {D}\n')
                f.write(f'norms (min/mean/max): {norms.min():.6f}/{norms.mean():.6f}/{norms.max():.6f}\n')
                if flat.size:
                    f.write(f'pairwise cosine (min/median/max): {np.nanmin(cos_nodiag):.6f}/{p50:.6f}/{np.nanmax(cos_nodiag):.6f}\n')
                f.write('top similar pairs (i,j,cos,IoU):\n')
                for r in order:
                    i, j = iu[0][r], iu[1][r]
                    c = float(cos[i, j])
                    iou = iou_xyxy(boxes_valid[i], boxes_valid[j])
                    f.write(f'{i},{j},{c:.6f},{iou:.6f}\n')
            np.save(os.path.join(args.out, f'{base}_cosine_valid.npy'), cos)
            print('  - saved:', report_txt)

            # Per-detection nearest neighbor summary (valid only)
            nn_idx = np.argmax(cos_nodiag, axis=1)
            nn_cos = cos[np.arange(nv), nn_idx]
            # Print a compact table (top few with highest nn cosine)
            order_nn = np.argsort(-nn_cos)
            print('  - per-detection nearest neighbor (sorted by cosine):')
            for t in order_nn[:min(10, nv)]:
                i = int(t)
                j = int(nn_idx[i])
                iou = iou_xyxy(boxes_valid[i], boxes_valid[j])
                print(f'     i={i:2d} -> j={j:2d}  cos={nn_cos[i]:.3f}  IoU={iou:.3f}  score={boxes_valid[i,1]:.3f}')

            # Save CSVs for easier inspection
            ids_valid = np.arange(nv)
            nn_csv = os.path.join(args.out, f'{base}_embed_nn.csv')
            with open(nn_csv, 'w') as f:
                f.write('vidx,class,score,x0,y0,x1,y1,nn_vidx,nn_cos,nn_iou\n')
                for i in range(nv):
                    j = int(nn_idx[i])
                    iou = iou_xyxy(boxes_valid[i], boxes_valid[j])
                    cls_id, score, x0, y0, x1, y1 = boxes_valid[i]
                    f.write(f'{i},{int(cls_id)},{score:.6f},{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f},{j},{nn_cos[i]:.6f},{iou:.6f}\n')

            emb_csv = os.path.join(args.out, f'{base}_embeddings_valid.csv')
            with open(emb_csv, 'w') as f:
                header = ','.join(['vidx'] + [f'e{k}' for k in range(D)])
                f.write(header + '\n')
                for i in range(nv):
                    row = ','.join([str(i)] + [f'{v:.6f}' for v in embs_valid[i].tolist()])
                    f.write(row + '\n')
            print('  - saved:', nn_csv)
            print('  - saved:', emb_csv)

            # Save an additional visualization with valid detection ids
            vis_idx = os.path.join(args.out, f'{base}_idx.jpg')
            try:
                draw_and_save_with_ids(args.img, boxes_valid, ids_valid, float(draw_threshold), vis_idx, label_list)
                print('  - saved:', vis_idx)
            except Exception as e:
                print('[WARN] Failed to save id visualization:', e)
        else:
            print('  - Not enough valid detections for pairwise comparison.')
    return


if __name__ == '__main__':
    main()
