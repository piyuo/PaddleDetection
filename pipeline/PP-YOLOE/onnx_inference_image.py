#!/usr/bin/env python3
"""
Run inference on a single image using ONNX Runtime with the exported PP-YOLOE Human model.

Usage:
    python pipeline/PP-YOLOE/onnx_inference_image.py \
        [--img pipeline/dataset/demo/demo.jpg] \
        [--onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx] \
        [--out pipeline/output/onnx_vis] \
        [--thresh 0.5]

Requirement:
    - The ONNX model must expose per-detection embeddings named "embed" with shape (N, D),
        where N matches the number of rows in the detection output (top-K). This script will error out
        if "embed" is not present.

Notes:
    - This script implements standalone preprocessing (no PaddleDetection dependency required).
    - Preprocessing matches the original PaddleDetection ONNX pipeline: resize to 640x640, normalize, permute.
    - Outputs are printed to stdout, with optional visualization and .npy files in --out.
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import cv2

def preprocess_image(img_path: str, target_size: Tuple[int, int] = (640, 640),
                     keep_ratio: bool = False,
                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                     is_scale: bool = True) -> dict:
    """
    Standalone image preprocessing for PP-YOLOE ONNX inference.

    Args:
        img_path: Path to input image
        target_size: Target size (height, width) for resizing
        keep_ratio: Whether to keep aspect ratio during resize
        mean: RGB mean values for normalization
        std: RGB standard deviation values for normalization
        is_scale: Whether to scale pixel values by 1/255.0

    Returns:
        Dictionary with preprocessed image tensor and metadata
    """
    # Load image in RGB format
    with open(img_path, 'rb') as f:
        im_read = f.read()
    data = np.frombuffer(im_read, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert to RGB

    original_shape = im.shape[:2]  # (H, W)

    # Resize
    if keep_ratio:
        # Keep aspect ratio resize
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
        # Direct resize without keeping ratio
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(original_shape[0])
        im_scale_x = resize_w / float(original_shape[1])

    im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                    interpolation=cv2.INTER_LINEAR)

    # Normalize
    im = im.astype(np.float32, copy=False)
    if is_scale:
        im *= (1.0 / 255.0)

    # Apply mean and std normalization
    mean_arr = np.array(mean)[np.newaxis, np.newaxis, :]
    std_arr = np.array(std)[np.newaxis, np.newaxis, :]
    im -= mean_arr
    im /= std_arr

    # Permute from HWC to CHW
    im = im.transpose((2, 0, 1))

    # Return in the format expected by ONNX model
    return {
        'image': im,
        'im_shape': np.array(im.shape[1:], dtype=np.float32),  # (H, W)
        'scale_factor': np.array([im_scale_y, im_scale_x], dtype=np.float32)
    }
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


def default_paths() -> Tuple[str, str, str]:
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
    return onnx_path, img_path, out_dir


def get_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except Exception as e:
        print('[ERROR] onnxruntime not installed. Install with: pip install onnxruntime', file=sys.stderr)
        raise

    available = ort.get_available_providers()
    # Always attempt CoreML first (macOS). Fallback to CPU (default) if unavailable.
    providers = None
    provider_options = None
    if 'CoreMLExecutionProvider' in available:
        coreml_opts = {
            #'mlprogram': '1',
            #'enable_on_subgraph': '1',
            #'only_allow_static_input_shapes': '1',
        }
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        provider_options = [coreml_opts, {}]
        print('[INFO] Using CoreMLExecutionProvider (Apple Core ML) with options:', coreml_opts)
    else:
        print(f'[WARN] CoreMLExecutionProvider not available. Using default providers: {available}')

    # Create session with provider options when supported; gracefully fallback otherwise
    def try_build(providers, provider_options):
        return ort.InferenceSession(
            onnx_path,
            providers=providers,
            provider_options=provider_options,
        )

    try:
        if providers is None:
            # CPU/default path
            sess = ort.InferenceSession(onnx_path, providers=providers)
        else:
            # Try progressively less strict CoreML options to match the installed ORT version
            option_variants = []
            if provider_options is not None:
                full = provider_options[0].copy()
                option_variants.append(full)
                v2 = full.copy(); v2.pop('only_allow_static_input_shapes', None); option_variants.append(v2)
                v1 = {'mlprogram': full.get('mlprogram', '1')}
                # keep enable_on_subgraph if present; otherwise just mlprogram
                if 'enable_on_subgraph' in full:
                    v1['enable_on_subgraph'] = full['enable_on_subgraph']
                option_variants.append(v1)
                option_variants.append({})  # default CoreML options
            else:
                option_variants.append({})

            last_error = None
            for opts in option_variants:
                try:
                    po = [opts, {}]
                    sess = try_build(providers, po)
                    print('[INFO] CoreML provider initialized with options:', opts if opts else '(default)')
                    break
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    if 'Unknown option' in msg or 'EP Error' in msg:
                        # Try next variant
                        continue
                    # Non-option error; break and fallback to CPU
                    break
            else:
                # Exhausted variants
                raise last_error
    except TypeError:
        # Older onnxruntime may not support provider_options parameter
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print('[WARN] CoreML session creation failed:', e)
        print("[WARN] Falling back to CPUExecutionProvider.")
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return sess


def get_hardcoded_preprocess() -> Tuple[float, str, list]:
    """
    Return hardcoded preprocessing parameters without PaddleDetection dependency.
    """
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

    return draw_threshold, arch, label_list


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
    d_onnx, d_img, d_out = default_paths()

    parser = argparse.ArgumentParser(description='ONNX Runtime inference for PP-YOLOE Human on one image')
    parser.add_argument('--img', default=d_img, help='Path to input image')
    parser.add_argument('--onnx', default=d_onnx, help='Path to ONNX model file')
    parser.add_argument('--out', default=d_out, help='Directory to save visualization')
    parser.add_argument('--thresh', type=float, default=None, help='Score threshold for printing/drawing (default: 0.5)')
    args = parser.parse_args()

    # Validate inputs
    for p, label in [
        (args.img, 'Input image'),
        (args.onnx, 'ONNX model'),
    ]:
        if not os.path.exists(p):
            print(f'[ERROR] {label} not found: {p}', file=sys.stderr)
            sys.exit(1)

    # Load preprocess parameters and session
    draw_threshold, arch, label_list = get_hardcoded_preprocess()
    if args.thresh is not None:
        draw_threshold = args.thresh
    sess = get_session(args.onnx)

    # Preprocess image using our standalone function
    inputs_map = preprocess_image(args.img, target_size=(640, 640), keep_ratio=False)
    input_names = [i.name for i in sess.get_inputs()]

    # Prepare feed dictionary - the ONNX model expects 'image' input
    feed = {}
    for name in input_names:
        if name == 'image':
            feed[name] = inputs_map['image'][None, :]  # Add batch dimension
        elif name in inputs_map:
            feed[name] = inputs_map[name][None, :]  # Add batch dimension for other inputs
        else:
            print(f'[WARN] Model input "{name}" not found in preprocessed data', file=sys.stderr)

    # Run
    outputs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]

    # C++ Porting Guide: Critical preprocessing and model info
    print('[C++ Porting Info] Model and preprocessing details:')
    print(f'  • Input image size: {args.img} -> resized to 640x640 (keep_ratio=False)')
    print('    ↳ Why keep_ratio=False? Model was trained this way, handles distortion well for humans')
    print('  • Normalization: RGB values /255.0, then (x - mean) / std')
    print('    - mean = [0.485, 0.456, 0.406]')
    print('    - std = [0.229, 0.224, 0.225]')
    print('  • Channel order: RGB (not BGR)')
    print('  • Input tensor: (1, 3, 640, 640) NCHW format, float32')
    print(f'  • Input tensor name: "{input_names[0] if input_names else "unknown"}"')
    print(f'  • Score threshold: {draw_threshold} (filter detections below this)')
    print('  • Post-processing: L2-normalize embeddings, filter detections by score')
    print('  • No dependency on PaddleDetection - standalone preprocessing implementation')

    # Combined model outputs summary (names + shapes) and expectations
    print('\nModel outputs:', out_names)
    print(" - outputs[0]: detections (N,6) [class_id, score, x0, y0, x1, y1] - float32")
    if 'embed' in out_names:
        print(" - 'embed': per-detection embeddings (N, D) - float32, requires L2-normalization")
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
