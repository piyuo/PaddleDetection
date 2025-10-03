#!/usr/bin/env python3
"""
Run inference on a single image using ONNX Runtime with the exported PP-YOLOE Human model.

Usage:
    python pipeline/PP-YOLOE/onnx_inference_image.py \
        [--img pipeline/dataset/demo/demo.jpg] \
        [--onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ane.onnx] \
        [--out pipeline/output/onnx_vis] \
        [--thresh 0.5]

Model Support:
    - ANE-optimized model (automatic surgery): 4 outputs
      * outputs[0]: raw boxes (8400×4) [x_center, y_center, w, h]
      * outputs[1]: raw scores (1×1×8400) - needs squeeze and NMS
      * outputs[2]: stride-8 features (1×C8×80×80) - fine-grained features
      * outputs[3]: stride-16 features (1×C16×40×40) - semantic features
      → Multi-scale embeddings extracted via roi_align_pool_multi_scale()

    - Original model (with NMS): 2-3 outputs
      * outputs[0]: detections (N×6) [class_id, score, x0, y0, x1, y1]
      * outputs[1]: count
      * outputs[2]: embed (optional, N×D) - per-detection embeddings

Notes:
    - This script implements standalone preprocessing (no PaddleDetection dependency required).
    - Preprocessing matches the original PaddleDetection ONNX pipeline: resize to 640x640, normalize, permute.
    - Outputs are printed to stdout, with optional visualization and .npy files in --out.
    - For ANE-optimized models, NMS is applied in Python and embeddings are extracted from feature maps.
"""

import argparse
import os
import sys
import time
import plistlib
import platform
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


def roi_align_pool_multi_scale(
    feat_s8: np.ndarray,
    feat_s16: np.ndarray,
    boxes_xyxy: np.ndarray,
    img_hw: tuple,
    input_size_hw: tuple = (640, 640),
    # Global pooling params (recommended: gp_w=0.2)
    gp_w: float = 0.2,
    avg_w: float = 1.0,
    max_w: float = 0.0,
    # Part pooling params (recommended: pp_w=0.8, pp_k=9, pp_stripe_h=2)
    pp_w: float = 0.8,
    pp_k: int = 9,
    pp_stripe_h: int = 2,
    pp_vertical_k: int = 2,
    pp_vertical_stripe_w: int = 2,
    # Normalization params
    use_inst_norm: bool = True,
    pl_alpha: float = 0.35,
) -> np.ndarray:
    """
    Multi-scale ROI pooling with global and part-based pooling (sophisticated embedding extraction).

    Based on insert_embedding_head.py production-grade implementation:
    - Multi-scale: Combines stride-8 and stride-16 features
    - Global pooling: Weighted mix of avg/max pooling
    - Part pooling: Horizontal/vertical stripes for body part discrimination
    - Normalization: InstanceNorm → power-law → L2

    Args:
        feat_s8: (1, C_s8, H_s8, W_s8) - stride-8 feature map (~80×80)
        feat_s16: (1, C_s16, H_s16, W_s16) - stride-16 feature map (~40×40)
        boxes_xyxy: (N, 4) - boxes in original image coordinates [x0, y0, x1, y1]
        img_hw: (Himg, Wimg) - original image size
        input_size_hw: (H_input, W_input) - model input size (default 640x640)
        gp_w: Global pooling weight (0.2 recommended)
        avg_w: Average pooling weight (1.0 recommended)
        max_w: Max pooling weight (0.0 recommended)
        pp_w: Part pooling weight (0.8 recommended)
        pp_k: Number of horizontal part divisions (9 recommended)
        pp_stripe_h: Height of horizontal stripes (2 recommended)
        pp_vertical_k: Number of vertical part divisions (2 recommended)
        pp_vertical_stripe_w: Width of vertical stripes (2 recommended)
        use_inst_norm: Apply instance normalization (True recommended)
        pl_alpha: Power-law normalization exponent (0.35 recommended)

    Returns:
        embeddings: (N, C_s8 + C_s16) - L2-normalized embeddings

    Expected quality metrics (from insert_embedding_head.py):
        - Median cosine: 0.05-0.10
        - P95 cosine: < 0.35
    """
    assert feat_s8.ndim == 4 and feat_s8.shape[0] == 1
    assert feat_s16.ndim == 4 and feat_s16.shape[0] == 1

    _, C_s8, H_s8, W_s8 = feat_s8.shape
    _, C_s16, H_s16, W_s16 = feat_s16.shape
    Himg, Wimg = img_hw
    H_input, W_input = input_size_hw

    # Coordinate scaling for each feature map
    def compute_scales(Hf, Wf):
        scale_y = (H_input / float(Himg)) * (Hf / float(H_input))
        scale_x = (W_input / float(Wimg)) * (Wf / float(W_input))
        return scale_x, scale_y

    scale_x_s8, scale_y_s8 = compute_scales(H_s8, W_s8)
    scale_x_s16, scale_y_s16 = compute_scales(H_s16, W_s16)

    def extract_roi_advanced(feat_map, x0, y0, x1, y1, scale_x, scale_y, Hf, Wf, C):
        """Extract and process ROI with global and part pooling."""
        # Map to feature coordinates
        fx0 = int(max(0, np.floor(x0 * scale_x)))
        fy0 = int(max(0, np.floor(y0 * scale_y)))
        fx1 = int(min(Wf, np.ceil(x1 * scale_x)))
        fy1 = int(min(Hf, np.ceil(y1 * scale_y)))

        if fx1 <= fx0 or fy1 <= fy0:
            return np.zeros((C,), dtype=np.float32)

        # Extract ROI
        roi = feat_map[0, :, fy0:fy1, fx0:fx1]  # (C, h, w)
        _, h, w = roi.shape

        features = []

        # 1. Global pooling (weighted avg + max)
        if gp_w > 0:
            global_feat = np.zeros((C,), dtype=np.float32)
            if avg_w > 0:
                global_feat += avg_w * roi.mean(axis=(1, 2))
            if max_w > 0:
                global_feat += max_w * roi.max(axis=(1, 2))
            features.append(global_feat * gp_w)

        # 2. Horizontal part pooling (body parts: head, torso, legs, etc.)
        if pp_w > 0 and pp_k > 0 and pp_stripe_h > 0:
            stripe_size = max(1, h // pp_k)
            for i in range(pp_k):
                y_start = i * stripe_size
                y_end = min(h, (i + 1) * stripe_size)
                if y_end <= y_start:
                    continue

                # Each stripe has pp_stripe_h sub-divisions
                sub_stripe_size = max(1, (y_end - y_start) // pp_stripe_h)
                for j in range(pp_stripe_h):
                    sub_y_start = y_start + j * sub_stripe_size
                    sub_y_end = min(y_end, y_start + (j + 1) * sub_stripe_size)
                    if sub_y_end <= sub_y_start:
                        continue

                    stripe = roi[:, sub_y_start:sub_y_end, :]  # (C, sub_h, w)
                    stripe_feat = stripe.mean(axis=(1, 2))
                    features.append(stripe_feat * pp_w / (pp_k * pp_stripe_h))

        # 3. Vertical part pooling (left/right symmetry)
        if pp_w > 0 and pp_vertical_k > 0 and pp_vertical_stripe_w > 0:
            stripe_size = max(1, w // pp_vertical_k)
            for i in range(pp_vertical_k):
                x_start = i * stripe_size
                x_end = min(w, (i + 1) * stripe_size)
                if x_end <= x_start:
                    continue

                sub_stripe_size = max(1, (x_end - x_start) // pp_vertical_stripe_w)
                for j in range(pp_vertical_stripe_w):
                    sub_x_start = x_start + j * sub_stripe_size
                    sub_x_end = min(x_end, x_start + (j + 1) * sub_stripe_size)
                    if sub_x_end <= sub_x_start:
                        continue

                    stripe = roi[:, :, sub_x_start:sub_x_end]  # (C, h, sub_w)
                    stripe_feat = stripe.mean(axis=(1, 2))
                    features.append(stripe_feat * pp_w / (pp_vertical_k * pp_vertical_stripe_w))

        # Combine all features
        if not features:
            return np.zeros((C,), dtype=np.float32)

        combined = np.sum(features, axis=0)

        # Instance normalization (channel-wise standardization)
        if use_inst_norm:
            mean = combined.mean()
            std = combined.std()
            if std > 1e-6:
                combined = (combined - mean) / std

        # Power-law normalization: sign(x) * |x|^alpha
        if pl_alpha != 1.0:
            sign = np.sign(combined)
            combined = sign * np.power(np.abs(combined), pl_alpha)

        return combined

    # Process each box on both feature maps
    embs = []
    for x0, y0, x1, y1 in boxes_xyxy:
        # Extract from stride-8 feature
        feat_s8_vec = extract_roi_advanced(
            feat_s8, x0, y0, x1, y1, scale_x_s8, scale_y_s8, H_s8, W_s8, C_s8
        )

        # Extract from stride-16 feature
        feat_s16_vec = extract_roi_advanced(
            feat_s16, x0, y0, x1, y1, scale_x_s16, scale_y_s16, H_s16, W_s16, C_s16
        )

        # Concatenate multi-scale features
        combined = np.concatenate([feat_s8_vec, feat_s16_vec])

        # Final L2 normalization
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            combined = combined / norm

        embs.append(combined)

    return np.stack(embs, axis=0) if embs else np.zeros((0, C_s8 + C_s16), dtype=np.float32)


def repo_root() -> str:
    # This file is at <repo>/pipeline/PP-YOLOE/onnx_inference_image.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_coreml_version_info() -> dict:
    """Best-effort CoreML environment info for debugging on macOS.
    Returns dict with keys: coremltools (str|None), framework (dict|None), macOS (str)
    framework dict contains CFBundleShortVersionString, CFBundleVersion, path when available.
    """
    info = {}
    # Python package (conversion toolkit), may be absent
    try:
        import coremltools as ct  # type: ignore
        info['coremltools'] = getattr(ct, '__version__', 'unknown')
    except Exception:
        info['coremltools'] = None

    # System CoreML framework Info.plist (runtime)
    plist_candidates = [
        '/System/Library/Frameworks/CoreML.framework/Resources/Info.plist',
        '/System/Library/Frameworks/CoreML.framework/Versions/Current/Resources/Info.plist',
    ]
    framework = None
    for p in plist_candidates:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    pl = plistlib.load(f)
                framework = {
                    'CFBundleShortVersionString': pl.get('CFBundleShortVersionString'),
                    'CFBundleVersion': pl.get('CFBundleVersion'),
                    'path': p,
                }
                break
            except Exception:
                continue
    info['framework'] = framework
    try:
        info['macOS'] = platform.mac_ver()[0]
    except Exception:
        info['macOS'] = None
    return info


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
    providers = None
    if 'CoreMLExecutionProvider' in available:
        coreml_opts = {
            "ModelFormat": "MLProgram",
            "EnableOnSubgraphs": "1",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
       }
        providers = [
            ('CoreMLExecutionProvider', coreml_opts),
            'CPUExecutionProvider'
            ]
        print('[INFO] Using CoreMLExecutionProvider (Apple Core ML) with options:', coreml_opts)
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            sess = ort.InferenceSession(
                onnx_path, sess_options=so,
                providers=providers,
            )
        except Exception as e:
            # Fail fast as requested (no complex retry of options)
            print('[ERROR] Failed to initialize CoreMLExecutionProvider session:', e, file=sys.stderr)
            raise
    else:
        print(f'[WARN] CoreMLExecutionProvider not available. Using default providers: {available}')
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
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

    # Environment and provider info
    try:
        import onnxruntime as ort  # local import for version
        ort_ver = getattr(ort, '__version__', 'unknown')
        avail = ort.get_available_providers()
    except Exception:
        ort_ver = 'unknown'
        avail = []
    print('\n[Env] onnxruntime:', ort_ver)
    if avail:
        print('      available providers:', avail)
    try:
        print('      session providers:', sess.get_providers())
    except Exception:
        pass
    cm = get_coreml_version_info()
    print('[Env] CoreML:', end=' ')
    cmtools = cm.get('coremltools')
    print(f"coremltools={cmtools if cmtools else 'not installed'}", end='; ')
    fw = cm.get('framework')
    if fw:
        sv = fw.get('CFBundleShortVersionString') or 'unknown'
        bv = fw.get('CFBundleVersion') or 'unknown'
        print(f'framework={sv} (bundle {bv})')
    else:
        print('framework version: unknown')
    if cm.get('macOS'):
        print(f"[Env] macOS: {cm['macOS']}")

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

    # Warmup (3 runs to match profile_onnx.py)
    print('[INFO] Warming up model (3 runs)...')
    for i in range(3):
        sess.run(None, feed)

    # Run (measure after warmup)
    t0 = time.perf_counter()
    outputs = sess.run(None, feed)
    t1 = time.perf_counter()
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

    # Detect model type and explain outputs
    has_nms = 'fetch_name_0' in out_names or (len(outputs) > 0 and isinstance(outputs[0], np.ndarray) and outputs[0].ndim == 2 and outputs[0].shape[1] == 6)
    has_raw_boxes = any('divide' in n or 'box' in n.lower() for n in out_names)
    has_features = len(out_names) >= 4 or any('batch_norm' in n or 'conv2d' in n for n in out_names)

    if has_nms:
        print(" - outputs[0]: detections (N,6) [class_id, score, x0, y0, x1, y1] - float32")
        if 'embed' in out_names:
            print(" - 'embed': per-detection embeddings (N, D) - float32, requires L2-normalization")
        else:
            print(" - Original model with NMS (no embeddings)")
    elif has_raw_boxes and has_features:
        print(" ✓ ANE-optimized model (automatic surgery)")
        print(f" - outputs[0]: raw boxes ({outputs[0].shape}) [x_center, y_center, w, h]")
        print(f" - outputs[1]: raw scores ({outputs[1].shape}) - needs squeeze and NMS")
        if len(out_names) >= 3:
            print(f" - outputs[2]: stride-8 features ({outputs[2].shape}) - fine-grained")
        if len(out_names) >= 4:
            s8_ch = outputs[2].shape[1] if len(out_names) >= 3 else 0
            s16_ch = outputs[3].shape[1] if len(out_names) >= 4 else 0
            total_dim = s8_ch + s16_ch
            print(f" - outputs[3]: stride-16 features ({outputs[3].shape}) - semantic")
            print(f" ✓ Multi-scale embeddings: {total_dim}D ({s8_ch} + {s16_ch})")
    else:
        print(" - Unknown model structure, see output details below")
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

    # =========================================================================
    # Post-process for PP-YOLOE with custom NMS
    # =========================================================================
    # Map output names to the output tensors for clarity
    name_to_out = {name: arr for name, arr in zip(out_names, outputs)}

    # Check if this is a pruned model (without NMS) or original model (with NMS)
    # ANE-optimized model (automatic surgery):
    #   - outputs[0]: p2o.pd_op.divide.0.0 (raw boxes, 8400×4)
    #   - outputs[1]: p2o.pd_op.concat.14.0 (raw scores, 1×1×8400)
    #   - outputs[2]: p2o.pd_op.batch_norm_.13.0 (stride-8 features, 1×128×80×80)
    #   - outputs[3]: p2o.pd_op.batch_norm_.19.0 (stride-16 features, 1×256×40×40)
    # Original model (with NMS):
    #   - outputs[0]: fetch_name_0 (detections, N×6)
    #   - outputs[1]: fetch_name_1 (count)
    #   - outputs[2]: embed (optional, N×D)
    has_nms = 'fetch_name_0' in name_to_out or (len(outputs) > 0 and isinstance(outputs[0], np.ndarray) and outputs[0].ndim == 2 and outputs[0].shape[1] == 6)

    if has_nms:
        # Original model with NMS already applied
        print('\n[INFO] Model has NMS built-in (using existing detections)')
        bboxes = np.array(outputs[0])

        print('Detections (class score x0 y0 x1 y1):')
        kept = 0
        for b in bboxes:
            if int(b[0]) > -1 and float(b[1]) >= float(draw_threshold):
                kept += 1
                print(f'{int(b[0])} {b[1]:.4f} {b[2]:.1f} {b[3]:.1f} {b[4]:.1f} {b[5]:.1f}')
        if kept == 0:
            print(f'No boxes above threshold {draw_threshold}. Try lowering --thresh.')

        # --- Require per-detection embeddings for BoT-SORT ---
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

    else:
        # Pruned model without NMS - we need to apply custom NMS
        print('\n[INFO] Applying custom NMS post-processing (pruned model detected)...')

        # Get raw model outputs (adjust names based on your pruned model)
        # Expected outputs: boxes (N, 4), scores (N, num_classes or N,1)
        raw_boxes_key = 'p2o.pd_op.divide.0.0'
        raw_scores_key = 'p2o.pd_op.concat.14.0'

        if raw_boxes_key not in name_to_out or raw_scores_key not in name_to_out:
            print(f'\n[ERROR] Pruned model expected outputs not found!', file=sys.stderr)
            print(f'        Expected: {raw_boxes_key}, {raw_scores_key}', file=sys.stderr)
            print(f'        Found: {list(name_to_out.keys())}', file=sys.stderr)
            sys.exit(2)

        raw_boxes = np.squeeze(name_to_out[raw_boxes_key], axis=0) if name_to_out[raw_boxes_key].ndim > 2 else name_to_out[raw_boxes_key]
        raw_scores = np.squeeze(name_to_out[raw_scores_key])  # Squeeze all batch dimensions

        # Ensure boxes are 2D (N, 4)
        if raw_boxes.ndim == 3:
            raw_boxes = raw_boxes.squeeze(0)

        print(f'  Raw boxes shape: {raw_boxes.shape}')
        print(f'  Raw scores shape: {raw_scores.shape}')        # PP-YOLOE outputs boxes in [x0, y0, x1, y1] format and scores for each class
        # Scores shape is typically (num_proposals, num_classes)
        # For single-class (person), we use class 0
        if raw_scores.ndim == 1:
            person_scores = raw_scores
        elif raw_scores.ndim == 2 and raw_scores.shape[1] == 1:
            person_scores = raw_scores[:, 0]
        elif raw_scores.ndim == 2:
            # Multi-class: use class 0 (person)
            person_scores = raw_scores[:, 0]
        else:
            print(f'\n[ERROR] Unexpected scores shape: {raw_scores.shape}', file=sys.stderr)
            sys.exit(2)

        # Boxes are already in [x0, y0, x1, y1] format from PP-YOLOE
        # Convert to [x, y, w, h] for cv2.dnn.NMSBoxes
        x0, y0, x1, y1 = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
        w = x1 - x0
        h = y1 - y0
        nms_boxes = np.column_stack([x0, y0, w, h]).tolist()

        # Run NMS
        score_threshold = float(draw_threshold)
        nms_threshold = 0.5  # IoU threshold for NMS
        print(f'  Applying NMS with score_threshold={score_threshold:.2f}, nms_threshold={nms_threshold:.2f}')
        selected_indices = cv2.dnn.NMSBoxes(nms_boxes, person_scores.tolist(), score_threshold, nms_threshold)

        # Assemble the final filtered outputs
        if len(selected_indices) > 0:
            # Flatten the indices array if it's nested
            selected_indices = selected_indices.flatten()
            print(f'  NMS kept {len(selected_indices)} detections from {len(person_scores)} proposals')

            # Gather the final boxes and scores using the selected indices
            boxes_nms = raw_boxes[selected_indices]
            scores_nms = person_scores[selected_indices]

            # Reconstruct the final (N, 6) bboxes array: [class_id, score, x0, y0, x1, y1]
            # Class ID is 0 for "person"
            class_ids = np.zeros_like(scores_nms)

            # Boxes are already in (x0, y0, x1, y1) format
            bboxes = np.column_stack([class_ids, scores_nms, boxes_nms])

            # Extract real embeddings from multi-scale feature maps
            # Automatically detect stride-8 and stride-16 feature maps from model outputs
            # Expected patterns:
            #  - Stride-8: ~80x80 spatial size (640/8=80)
            #  - Stride-16: ~40x40 spatial size (640/16=40)
            feat_s8_key = None
            feat_s16_key = None
            feat_s8 = None
            feat_s16 = None

            for name, arr in name_to_out.items():
                if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 1:
                    _, C, H, W = arr.shape
                    # Detect stride-8 (spatial ~80x80, channels >= 64)
                    if 70 <= H <= 90 and 70 <= W <= 90 and C >= 64:
                        if feat_s8_key is None or C > name_to_out[feat_s8_key].shape[1]:
                            feat_s8_key = name
                            feat_s8 = arr
                    # Detect stride-16 (spatial ~40x40, channels >= 64)
                    elif 30 <= H <= 50 and 30 <= W <= 50 and C >= 64:
                        if feat_s16_key is None or C > name_to_out[feat_s16_key].shape[1]:
                            feat_s16_key = name
                            feat_s16 = arr

            if feat_s8 is not None and feat_s16 is not None:
                print(f'  [INFO] Multi-scale embedding extraction:')
                print(f'         Stride-8:  {feat_s8_key} (shape={feat_s8.shape})')
                print(f'         Stride-16: {feat_s16_key} (shape={feat_s16.shape})')

                # Get original image size from the loaded image
                orig_img = cv2.imread(args.img)
                if orig_img is not None:
                    orig_h, orig_w = orig_img.shape[:2]
                    img_hw = (orig_h, orig_w)

                    # Extract embeddings using multi-scale ROI pooling with sophisticated features
                    # Recommended settings from insert_embedding_head.py:
                    # gp_w=0.2, pp_w=0.8, pp_k=9, pp_stripe_h=2
                    embs_nms = roi_align_pool_multi_scale(
                        feat_s8, feat_s16, boxes_nms, img_hw,
                        input_size_hw=(640, 640),
                        gp_w=0.2, pp_w=0.8, pp_k=9, pp_stripe_h=2,
                        pp_vertical_k=2, pp_vertical_stripe_w=2,
                        use_inst_norm=True, pl_alpha=0.35
                    )
                    emb_dim = embs_nms.shape[1]
                    print(f'  [INFO] Extracted multi-scale embeddings (shape={embs_nms.shape}, dim={emb_dim})')
                    print(f'         Expected quality: median cosine < 0.15, p95 < 0.35')
                else:
                    print(f'  [WARN] Could not load image to get original size, using placeholder embeddings')
                    s8_ch = feat_s8.shape[1]
                    s16_ch = feat_s16.shape[1]
                    embs_nms = np.zeros((len(bboxes), s8_ch + s16_ch), dtype=np.float32)
            else:
                # Feature maps not found - determine expected embedding dimension from outputs
                expected_dim = 224  # Default fallback
                if len(out_names) >= 4:
                    # Try to get actual dimension from feature map shapes
                    for name, arr in name_to_out.items():
                        if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 1:
                            if 70 <= arr.shape[2] <= 90:  # stride-8
                                expected_dim = arr.shape[1]
                            elif 30 <= arr.shape[2] <= 50:  # stride-16
                                expected_dim += arr.shape[1]

                print(f'  [WARN] Could not auto-detect stride-8 and stride-16 feature maps')
                print(f'         Available outputs: {list(name_to_out.keys())}')
                print(f'         Using placeholder embeddings (all zeros, dim={expected_dim})')
                embs_nms = np.zeros((len(bboxes), expected_dim), dtype=np.float32)
        else:
            print(f'  NMS found no boxes above threshold {score_threshold:.2f}')
            bboxes = np.zeros((0, 6), dtype=np.float32)
            embs_nms = np.zeros((0, 224), dtype=np.float32)  # 128+96 multi-scale

        # Print detections
        print('\nDetections after NMS (class score x0 y0 x1 y1):')
        if len(bboxes) > 0:
            for b in bboxes:
                print(f'{int(b[0])} {b[1]:.4f} {b[2]:.1f} {b[3]:.1f} {b[4]:.1f} {b[5]:.1f}')
        else:
            print(f'No boxes above threshold {draw_threshold} after NMS.')

        # Set final variables for downstream use
        det_embs = embs_nms
        boxes_valid = bboxes
        embs_valid = embs_nms

    # =========================================================================
    # Common code for both paths (continues from here)
    # =========================================================================

    base = os.path.splitext(os.path.basename(args.img))[0]
    vis_path = os.path.join(args.out, f'{base}.jpg')
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
    print(f'[Timing] onnxruntime sess.run (after warmup): {(t1 - t0)*1000.0:.2f} ms')
    print(f'\n✅ ONNX inference completed! Generated visualization: {vis_path}')
    return


if __name__ == '__main__':
    main()
