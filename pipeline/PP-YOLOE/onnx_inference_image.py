#!/usr/bin/env python3
"""
Run inference on a single image using ONNX Runtime with the exported PP-YOLOE Human model.

Defaults assume you exported via pipeline/PP-YOLOE/export_to_onnx.sh.

Usage:
  python pipeline/PP-YOLOE/onnx_inference_image.py \
    [--img pipeline/dataset/demo/demo.jpg] \
    [--onnx pipeline/output/ppyoloe_crn_s_36e_pphuman.onnx] \
    [--infer_cfg pipeline/output/inference_model/ppyoloe_crn_s_36e_pphuman/infer_cfg.yml] \
    [--out pipeline/output_vis] \
    [--thresh 0.5] [--gpu]

Notes:
  - This script reuses PaddleDetection's ONNX preprocess (deploy/third_engine/onnx/preprocess.py)
  - Outputs printed to stdout and an optional visualization image in --out
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np


def repo_root() -> str:
    # This file is at <repo>/pipeline/PP-YOLOE/onnx_inference_image.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def default_paths() -> Tuple[str, str, str, str, str]:
    root = repo_root()
    model_name = 'ppyoloe_crn_s_36e_pphuman'
    onnx_a = os.path.join(root, 'pipeline', 'output', 'onnx', f'{model_name}.onnx')
    onnx_b = os.path.join(root, 'pipeline', 'output', f'{model_name}.onnx')
    onnx_path = onnx_a if os.path.exists(onnx_a) else onnx_b

    infer_cfg = os.path.join(
        root, 'pipeline', 'output', 'inference_model', model_name, 'infer_cfg.yml'
    )
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


def main():
    d_onnx, d_infer_cfg, d_img, d_out, d_preproc_dir = default_paths()

    parser = argparse.ArgumentParser(description='ONNX Runtime inference for PP-YOLOE Human on one image')
    parser.add_argument('--img', default=d_img, help='Path to input image')
    parser.add_argument('--onnx', default=d_onnx, help='Path to ONNX model file')
    parser.add_argument('--infer_cfg', default=d_infer_cfg, help='Path to infer_cfg.yml from exported Paddle model')
    parser.add_argument('--out', default=d_out, help='Directory to save visualization')
    parser.add_argument('--thresh', type=float, default=None, help='Score threshold for printing/drawing (default from infer_cfg)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if onnxruntime-gpu is available')
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
    vis_path = os.path.join(args.out, f'{base}_onnx_vis.jpg')
    try:
        draw_and_save(args.img, bboxes, draw_threshold, vis_path, label_list)
        print('Saved visualization to:', vis_path)
    except Exception as e:
        print('[WARN] Failed to save visualization:', e)


if __name__ == '__main__':
    main()
