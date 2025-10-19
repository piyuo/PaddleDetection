#!/usr/bin/env python3
"""Convert PP-YOLOE ONNX to TensorFlow WITHOUT custom transposes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from onnx2tf import convert
except ModuleNotFoundError as exc:
    print("[ERROR] Missing onnx2tf. Install with: pip install onnx2tf")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PP-YOLOE ONNX to TensorFlow (simple mode)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline/PP-YOLOE/models/tensorflow_simple",
        help="Directory where the TensorFlow SavedModel will be written",
    )
    parser.add_argument(
        "--saved-model-name",
        type=str,
        default="ppyoloe_saved_model",
        help="Subdirectory name for the SavedModel export",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debug information during conversion.",
    )
    parser.add_argument(
        "--keep-nchw",
        action="store_true",
        help="Try to keep NCHW layout (experimental)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.model).expanduser().resolve()
    if not onnx_path.exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    saved_dir = output_dir / args.saved_model_name

    if saved_dir.exists():
        if args.verbose:
            print(f"[INFO] Clearing existing directory: {saved_dir}")
        shutil.rmtree(saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Converting ONNX → TensorFlow (SIMPLE MODE - no custom transposes)")
    print(f"[INFO] Input: {onnx_path}")
    print(f"[INFO] Output: {saved_dir}")

    try:
        convert_kwargs = dict(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_dir),
            output_signaturedefs=True,
            non_verbose=not args.verbose,
            # NO param_replacement_file - let onnx2tf handle layout automatically
        )

        if args.keep_nchw:
            print("[INFO] Attempting to preserve NCHW layout")
            convert_kwargs["keep_ncw_or_nchw_or_ncdhw_input_names"] = ["image"]
            convert_kwargs["keep_nwc_or_nhwc_or_ndhwc_input_names"] = []

        convert(**convert_kwargs)
    except Exception as exc:
        print(f"[ERROR] Conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] SavedModel exported to: {saved_dir}")
    print("[DONE] Simple conversion complete.")
    print()
    print("Next steps:")
    print("1. Convert SavedModel to TFLite:")
    print(f"   python3 -c \"")
    print(f"import tensorflow as tf")
    print(f"converter = tf.lite.TFLiteConverter.from_saved_model('{saved_dir}')")
    print(f"converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]")
    print(f"tflite_model = converter.convert()")
    print(f"open('{output_dir}/model_simple.tflite', 'wb').write(tflite_model)")
    print(f"\"")
    print()
    print("2. Test with diagnostic script:")
    print(f"   python3 pipeline/PP-YOLOE/diagnose_conversion.py --tflite {output_dir}/model_simple.tflite")


if __name__ == "__main__":
    main()
