#!/usr/bin/env python3
"""Convert PP-YOLOE ONNX models to TensorFlow SavedModel (and optional TFLite)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    import onnx  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    print("[ERROR] Missing dependency: onnx. Install with 'pip install onnx'.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from onnx2tf import convert  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    missing = exc.name or "onnx2tf"
    suggestion = f"pip install {missing}" if missing != "onnx2tf" else "pip install onnx2tf"
    hint = " (required by onnx2tf)" if missing != "onnx2tf" else ""
    print(f"[ERROR] Missing dependency: {missing}.{hint} Install with '{suggestion}'.", file=sys.stderr)
    raise SystemExit(1) from exc
except ImportError as exc:  # pragma: no cover - import guard
    print(f"[ERROR] Failed to import onnx2tf: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import tensorflow as tf  # type: ignore
except ImportError:
    tf = None  # Optional; only needed for TFLite conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PP-YOLOE ONNX model into TensorFlow SavedModel format (and optional TFLite)."
    )
    default_model = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx"
    parser.add_argument("--model", type=str, default=default_model, help="Path to input ONNX model")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline/PP-YOLOE/models/tensorflow",
        help="Directory where the TensorFlow SavedModel will be written",
    )
    parser.add_argument(
        "--saved-model-name",
        type=str,
        default="ppyoloe_saved_model",
        help="Subdirectory name for the SavedModel export",
    )
    parser.add_argument(
        "--convert-tflite",
        action="store_true",
        help="Also convert the SavedModel to TFLite (requires TensorFlow).",
    )
    parser.add_argument(
        "--tflite-output",
        type=str,
        default="ppyoloe_model.tflite",
        help="Filename for the generated TFLite model (when --convert-tflite is set)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply float16 quantization when producing the TFLite model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debug information during conversion.",
    )
    return parser.parse_args()


def _build_param_replacement_file(onnx_path: Path, verbose: bool) -> Optional[Path]:
    """Generate an onnx2tf parameter replacement file that flips bias tensors to NHWC."""

    try:
        import numpy as np
        import onnx_graphsurgeon as gs
    except ImportError as exc:  # pragma: no cover - optional tooling
        if verbose:
            print(f"[WARN] Skipping bias layout fixups; onnx_graphsurgeon unavailable ({exc}).")
        return None

    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)

    from collections import deque

    def resolve_constant_values(tensor: gs.Tensor) -> Optional["np.ndarray"]:
        queue: deque[gs.Tensor] = deque([tensor])
        visited_tensors: set[int] = set()
        visited_nodes: set[int] = set()

        while queue:
            current = queue.popleft()
            if isinstance(current, gs.Constant):
                return current.values
            if isinstance(current, gs.Variable):
                tensor_id = id(current)
                if tensor_id in visited_tensors:
                    continue
                visited_tensors.add(tensor_id)
                for producer in current.inputs or []:
                    node_id = id(producer)
                    if node_id in visited_nodes:
                        continue
                    visited_nodes.add(node_id)
                    if producer.op == "Constant":
                        return producer.attrs["value"].values
                    if producer.op in {"Identity", "Cast"}:
                        for upstream in producer.inputs:
                            queue.append(upstream)
        return None

    operations = []
    for node in graph.nodes:
        if node.op != "Reshape" or len(node.outputs) != 1:
            continue
        shape_values = resolve_constant_values(node.inputs[1])
        if shape_values is None:
            continue
        arr = np.asarray(shape_values)
        if arr.shape == (4,) and arr[0] == 1 and arr[2] == 1 and arr[3] == 1:
            operations.append(
                {
                    "op_name": node.name,
                    "param_target": "outputs",
                    "param_name": node.outputs[0].name,
                    "post_process_transpose_perm": [0, 2, 3, 1],
                }
            )

    if not operations:
        return None

    tmp_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump({"operations": operations}, tmp_handle, indent=2)
    finally:
        tmp_handle.flush()
        tmp_handle.close()

    tmp_path = Path(tmp_handle.name)
    if verbose:
        print(f"[INFO] Generated parameter replacements for {len(operations)} reshape nodes")
    return tmp_path


def export_to_saved_model(onnx_path: Path, saved_dir: Path, verbose: bool = False) -> Path:
    if verbose:
        print(f"[INFO] Preparing to convert ONNX model: {onnx_path}")
    if saved_dir.exists():
        if verbose:
            print(f"[INFO] Clearing existing directory: {saved_dir}")
        shutil.rmtree(saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[INFO] Converting ONNX → TensorFlow with onnx2tf…")
    param_file: Optional[Path] = None
    try:
        param_file = _build_param_replacement_file(onnx_path, verbose=verbose)
        convert_kwargs = dict(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_dir),
            output_signaturedefs=True,
            non_verbose=not verbose,
        )
        if param_file is not None:
            if verbose:
                print(f"[INFO] Applying channel-last bias transposes via {param_file}")
            convert_kwargs["param_replacement_file"] = str(param_file)
        convert(**convert_kwargs)
    finally:
        if param_file is not None and param_file.exists():
            param_file.unlink(missing_ok=True)

    print(f"[OK] SavedModel exported to: {saved_dir}")
    return saved_dir


def convert_saved_model_to_tflite(saved_dir: Path, out_path: Path, quantize: bool = False, verbose: bool = False) -> Path:
    if tf is None:
        raise RuntimeError("TensorFlow is required for TFLite conversion. Install with 'pip install tensorflow'.")

    if verbose:
        print(f"[INFO] Converting SavedModel to TFLite: {saved_dir} → {out_path}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))

    if quantize:
        if verbose:
            print("[INFO] Enabling float16 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    print(f"[OK] TFLite model written to: {out_path}")
    return out_path


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.model).expanduser().resolve()
    if not onnx_path.exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    saved_dir = output_dir / args.saved_model_name

    try:
        export_to_saved_model(onnx_path, saved_dir, verbose=args.verbose)
    except Exception as exc:
        print(f"[ERROR] Failed to convert ONNX to TensorFlow SavedModel: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.convert_tflite:
        tflite_path = Path(args.tflite_output)
        if not tflite_path.is_absolute():
            tflite_path = output_dir / tflite_path
        try:
            convert_saved_model_to_tflite(saved_dir, tflite_path, quantize=args.quantize, verbose=args.verbose)
        except Exception as exc:
            print(f"[ERROR] Failed to create TFLite model: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    print("[DONE] Conversion pipeline completed successfully.")


if __name__ == "__main__":
    main()
