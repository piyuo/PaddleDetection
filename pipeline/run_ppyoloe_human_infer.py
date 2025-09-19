#!/usr/bin/env python3
"""
Run PP-YOLOE Human model inference on the demo image and save results to pipeline/output.

This is a thin wrapper around tools/infer.py that:
- Resolves paths relative to the repository root
- Checks model/config existence
- Auto-selects GPU if available (falls back to CPU)

Usage:
  python pipeline/run_ppyoloe_human_infer.py

Optional args:
  --img <path>      Override the input image (default: pipeline/dataset/demo/demo.jpg)
  --out <dir>       Override the output directory (default: pipeline/output)
  --thresh <float>  Draw threshold (default: 0.5)
"""

import argparse
import os
import subprocess
import sys


def repo_root() -> str:
    # This file lives in <repo>/pipeline/, so parent dir is the repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def has_gpu() -> bool:
    try:
        import paddle  # type: ignore
        return bool(
            getattr(paddle.device, 'is_compiled_with_cuda', lambda: False)()
            and getattr(paddle.device.cuda, 'device_count', lambda: 0)() > 0
        )
    except Exception:
        # If Paddle isn't available at import time, assume no GPU to avoid failing checks
        return False


def main():
    parser = argparse.ArgumentParser(description='Run PP-YOLOE Human inference on demo image')
    parser.add_argument('--img', default=None, help='Path to input image (default: pipeline/dataset/demo/demo.jpg)')
    parser.add_argument('--out', default=None, help='Path to output directory (default: pipeline/output)')
    parser.add_argument('--thresh', type=float, default=0.5, help='Draw threshold')
    args = parser.parse_args()

    root = repo_root()

    config_path = os.path.join(root, 'pipeline', 'PP-YOLOE', 'models', 'ppyoloe_crn_s_36e_pphuman.yml')
    weights_path = os.path.join(root, 'pipeline', 'PP-YOLOE', 'models', 'ppyoloe_crn_s_36e_pphuman.pdparams')
    img_path = args.img or os.path.join(root, 'pipeline', 'dataset', 'demo', 'demo.jpg')
    out_dir = args.out or os.path.join(root, 'pipeline', 'output')

    # Basic validations
    for p, label in [
        (config_path, 'Config file'),
        (weights_path, 'Weights file'),
        (img_path, 'Input image'),
    ]:
        if not os.path.exists(p):
            print(f'[ERROR] {label} not found: {p}', file=sys.stderr)
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    gpu = has_gpu()
    device_flag = 'use_gpu=true' if gpu else 'use_gpu=false'

    cmd = [
        sys.executable,
        os.path.join(root, 'tools', 'infer.py'),
        '-c', config_path,
        '-o', f'weights={weights_path}',
        '-o', device_flag,
        '--infer_img', img_path,
        '--output_dir', out_dir,
        '--draw_threshold', str(args.thresh),
    ]

    print('Running inference with:')
    print('  Config :', config_path)
    print('  Weights:', weights_path)
    print('  Image  :', img_path)
    print('  Output :', out_dir)
    print('  Device :', 'GPU' if gpu else 'CPU')
    print()

    try:
        proc = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('[ERROR] Inference failed with exit code', e.returncode, file=sys.stderr)
        sys.exit(e.returncode)

    print('\nDone. Visualized results should be saved under:', out_dir)
    print('Tip: If you do not see any boxes, try lowering --thresh (e.g., 0.3).')


if __name__ == '__main__':
    main()
