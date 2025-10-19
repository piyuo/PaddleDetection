#!/usr/bin/env python3
"""Convert PP-YOLOE ONNX models to TensorFlow SavedModel."""

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PP-YOLOE ONNX model into TensorFlow SavedModel format."
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
        "--verbose",
        action="store_true",
        help="Print additional debug information during conversion.",
    )
    parser.add_argument(
        "--keep-nchw",
        action="store_true",
        help="Keep NCHW layout for image input (disable auto NHWC conversion)",
    )
    parser.add_argument(
        "--use-nms-dynamic",
        action="store_true",
        help="Use dynamic tensor output for NMS (variable number of detections)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fix dynamic batch size to specified value (default: 1)",
    )
    parser.add_argument(
        "--disable-heuristics",
        action="store_true",
        help="Disable all custom transpose heuristics (use onnx2tf defaults)",
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

    operations: list[dict] = []
    # Track unique ops to avoid duplicates when heuristics overlap
    seen_ops: set[tuple[str, str, str]] = set()
    for node in graph.nodes:
        if node.op != "Reshape" or len(node.outputs) != 1:
            continue
        shape_values = resolve_constant_values(node.inputs[1])
        if shape_values is None:
            continue
        arr = np.asarray(shape_values)
        if arr.shape == (4,) and arr[0] == 1 and arr[2] == 1 and arr[3] == 1:
            key = (node.name, "outputs", node.outputs[0].name)
            if key not in seen_ops:
                operations.append(
                    {
                        "op_name": node.name,
                        "param_target": "outputs",
                        "param_name": node.outputs[0].name,
                        "post_process_transpose_perm": [0, 2, 3, 1],
                    }
                )
                seen_ops.add(key)

    # Heuristic fix for PP-YOLOE Softmax head layout
    # The ONNX → TF transposition can produce Softmax outputs with shape [N, 4, 17, 6400]
    # where the last two dims are swapped for downstream 1x1 Conv expecting C=17.
    # We detect Softmax nodes with [*, 4, 17, *] and inject a transpose [0,1,3,2].
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        shape_map: dict[str, list[Optional[int]]] = {}
        initializer_map: dict[str, list[int]] = {}
        for init in inferred.graph.initializer:
            dims = [int(d) for d in init.dims]
            initializer_map[init.name] = dims
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
            if not vi.type.HasField("tensor_type"):
                continue
            dims: list[Optional[int]] = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    # dynamic dim
                    dims.append(None)
            shape_map[vi.name] = dims
        softmax_fixes = 0
        for node in graph.nodes:
            if node.op != "Softmax" or not node.outputs:
                continue
            out_name = node.outputs[0].name
            dims = shape_map.get(out_name, None)
            if dims and len(dims) == 4:
                # Match [N, 4, 17, *]
                second = dims[1]
                third = dims[2]
                if (second == 4) and (third == 17):
                    key = (node.name, "outputs", out_name)
                    if key not in seen_ops:
                        operations.append(
                            {
                                "op_name": node.name,
                                "param_target": "outputs",
                                "param_name": out_name,
                                "post_process_transpose_perm": [0, 1, 3, 2],
                            }
                        )
                        seen_ops.add(key)
                        softmax_fixes += 1
        # Additionally, ensure Conv immediately consuming Softmax applies the same swap on its input.
        conv_input_fixes = 0
        for node in graph.nodes:
            if node.op != "Conv" or len(node.inputs) < 2:
                continue
            x, w = node.inputs[0], node.inputs[1]
            wshape = initializer_map.get(getattr(w, "name", ""), None)
            if not wshape:
                continue
            # ONNX conv weights: [out_c, in_c/groups, kH, kW]
            if len(wshape) == 4 and wshape[1] == 17 and wshape[2] == 1 and wshape[3] == 1:
                key = (node.name, "inputs", getattr(x, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(x, "name", ""),
                            "pre_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    conv_input_fixes += 1

        # Graph-connectivity based fallback: if a Softmax feeds a 1x1 Conv with in_c=17,
        # force post_transpose on Softmax output and pre_transpose on Conv input.
        connectivity_softmax_fixes = 0
        connectivity_conv_fixes = 0
        for node in graph.nodes:
            if node.op != "Softmax" or not node.outputs:
                continue
            softmax_out = node.outputs[0]
            for consumer in list(getattr(softmax_out, "outputs", []) or []):
                if consumer.op != "Conv" or len(consumer.inputs) < 2:
                    continue
                w = consumer.inputs[1]
                wshape = initializer_map.get(getattr(w, "name", ""), None)
                if not (wshape and len(wshape) == 4 and wshape[1] == 17 and wshape[2] == 1 and wshape[3] == 1):
                    continue
                # Enforce Softmax NHWC swap [H,W,C] => [H,C,W]
                key_s = (node.name, "outputs", softmax_out.name)
                if key_s not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "outputs",
                            "param_name": softmax_out.name,
                            "post_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key_s)
                    connectivity_softmax_fixes += 1
                # Match Conv input pre_transpose
                key_c = (consumer.name, "inputs", consumer.inputs[0].name)
                if key_c not in seen_ops:
                    operations.append(
                        {
                            "op_name": consumer.name,
                            "param_target": "inputs",
                            "param_name": consumer.inputs[0].name,
                            "pre_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key_c)
                    connectivity_conv_fixes += 1
        # Heuristic for Squeeze: if axis includes 1 but input is NHWC with C=1 at dim=3,
        # move channels to axis=1 so squeeze can remove it.
        squeeze_fixes = 0
        for node in graph.nodes:
            if node.op != "Squeeze" or not node.inputs:
                continue
            x = node.inputs[0]
            axes_vals = None
            if len(node.inputs) > 1:
                try:
                    axes_vals = resolve_constant_values(node.inputs[1])
                except Exception:
                    axes_vals = None
            # Normalize axes to Python list of ints if available
            if axes_vals is not None:
                try:
                    axes_list = list(map(int, list(axes_vals)))
                except Exception:
                    axes_list = None
            else:
                axes_list = None
            if axes_list is None or 1 not in axes_list:
                continue
            dims = shape_map.get(getattr(x, "name", ""), None)
            if dims and len(dims) == 4:
                # Expect NHWC with C=1, H not 1
                h_dim = dims[1]
                c_dim = dims[3]
                if (h_dim is None or h_dim != 1) and (c_dim == 1 or c_dim is None):
                    key = (node.name, "inputs", getattr(x, "name", ""))
                    if key not in seen_ops:
                        operations.append(
                            {
                                "op_name": node.name,
                                "param_target": "inputs",
                                "param_name": getattr(x, "name", ""),
                                "pre_process_transpose_perm": [0, 3, 1, 2],
                            }
                        )
                        seen_ops.add(key)
                        squeeze_fixes += 1
        # Heuristic for Add broadcasting with a 2D constant whose last dim is 2
        # In PP-YOLOE head post-processing, tensors of shape [N, 2, K]
        # are added with a constant grid of shape [K, 2]. For NHWC paths this
        # needs to be [2, K] to broadcast correctly. Insert a 2D transpose [1, 0]
        # on the constant input. Even if the other input's rank is unknown,
        # safely flipping any 2D [K,2] constant used by Add ops near the head
        # avoids the broadcast shape error (observed at Add.76).
        add_const_transpose_fixes = 0
        for node in graph.nodes:
            if node.op != "Add" or len(node.inputs) < 2:
                continue
            a, b = node.inputs[0], node.inputs[1]
            # Helper to get dims for a tensor input
            def get_dims(t: gs.Tensor) -> Optional[list[int]]:
                name = getattr(t, "name", None)
                # Prefer initializer dims if available
                if name and name in initializer_map:
                    return [int(d) for d in initializer_map[name]]
                # Try shape inference map
                if name and name in shape_map:
                    dims = shape_map[name]
                    # If any None present, keep it as is
                    return [(-1 if d is None else int(d)) for d in dims]
                # Try resolving constant values (Constant/Identity/Cast of Constant)
                try:
                    vals = resolve_constant_values(t)
                    if vals is not None:
                        return list(map(int, list(vals.shape)))
                except Exception:
                    pass
                return None

            dims_a = get_dims(a) or []
            dims_b = get_dims(b) or []
            # Normalize negative/unknown
            def is_n2_8400(dims: list[int]) -> bool:
                return (
                    isinstance(dims, list)
                    and len(dims) == 3
                    and (dims[0] == -1 or dims[0] is None or isinstance(dims[0], int))
                    and dims[1] == 2
                    and dims[2] == 8400
                )
            def is_8400_2(dims: list[int]) -> bool:
                return isinstance(dims, list) and len(dims) == 2 and dims[0] == 8400 and dims[1] == 2
            def is_2_8400(dims: list[int]) -> bool:
                return isinstance(dims, list) and len(dims) == 2 and dims[0] == 2 and dims[1] == 8400

            # If one input is [N,2,8400] and the other is [8400,2], transpose the 2D one
            if is_n2_8400(dims_a) and is_8400_2(dims_b) and not is_2_8400(dims_b):
                key = (node.name, "inputs", getattr(b, "name", ""))
                if key not in seen_ops and getattr(b, "name", None):
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(b, "name", ""),
                            "pre_process_transpose_perm": [1, 0],
                        }
                    )
                    seen_ops.add(key)
                    add_const_transpose_fixes += 1
            elif is_n2_8400(dims_b) and is_8400_2(dims_a) and not is_2_8400(dims_a):
                key = (node.name, "inputs", getattr(a, "name", ""))
                if key not in seen_ops and getattr(a, "name", None):
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(a, "name", ""),
                            "pre_process_transpose_perm": [1, 0],
                        }
                    )
                    seen_ops.add(key)
                    add_const_transpose_fixes += 1
            # Fallback: if either input is 2D with trailing dim==2 and appears
            # to be a constant/initializer, flip it regardless of the other input's rank.
            # This covers cases where the dynamic [N,2,K] input shape is not inferred.
            else:
                def is_2d_trailing_two(dims: list[int]) -> bool:
                    return isinstance(dims, list) and len(dims) == 2 and dims[1] == 2

                # a is a 2D constant [K,2]
                if is_2d_trailing_two(dims_a) and getattr(a, "name", None):
                    key = (node.name, "inputs", getattr(a, "name", ""))
                    if key not in seen_ops:
                        operations.append(
                            {
                                "op_name": node.name,
                                "param_target": "inputs",
                                "param_name": getattr(a, "name", ""),
                                "pre_process_transpose_perm": [1, 0],
                            }
                        )
                        seen_ops.add(key)
                        add_const_transpose_fixes += 1
                # b is a 2D constant [K,2]
                if is_2d_trailing_two(dims_b) and getattr(b, "name", None):
                    key = (node.name, "inputs", getattr(b, "name", ""))
                    if key not in seen_ops:
                        operations.append(
                            {
                                "op_name": node.name,
                                "param_target": "inputs",
                                "param_name": getattr(b, "name", ""),
                                "pre_process_transpose_perm": [1, 0],
                            }
                        )
                        seen_ops.add(key)
                        add_const_transpose_fixes += 1
        # Fallback by specific node names seen in PP-YOLOE graphs
        name_based_fixes = 0
        # Heuristic for Div broadcasting: when dividing a [N,4,K] tensor by a factor
        # shaped [N,1,4] or [1,4], swap the last two dims so it becomes [N,4,1] or [4,1],
        # enabling broadcast across K. This addresses errors like Div.0 expecting
        # y shape to align with (None, 4, 8400).
        div_transpose_fixes = 0
        for node in graph.nodes:
            if node.op != "Div" or len(node.inputs) < 2:
                continue
            a, b = node.inputs[0], node.inputs[1]

            def get_dims_div(t: gs.Tensor) -> Optional[list[int]]:
                name = getattr(t, "name", None)
                if name and name in initializer_map:
                    return [int(d) for d in initializer_map[name]]
                if name and name in shape_map:
                    dims = shape_map[name]
                    return [(-1 if d is None else int(d)) for d in dims]
                try:
                    vals = resolve_constant_values(t)
                    if vals is not None:
                        return list(map(int, list(vals.shape)))
                except Exception:
                    pass
                return None

            dims_a = get_dims_div(a) or []
            dims_b = get_dims_div(b) or []

            def is_n4k(dims: list[int]) -> bool:
                return isinstance(dims, list) and len(dims) == 3 and dims[1] == 4

            def is_n14(dims: list[int]) -> bool:
                return isinstance(dims, list) and len(dims) == 3 and dims[1] == 1 and dims[2] == 4

            def is_14(dims: list[int]) -> bool:
                return isinstance(dims, list) and len(dims) == 2 and dims[0] == 1 and dims[1] == 4

            # If b is [N,1,4] -> transpose b [0,2,1] to [N,4,1]
            if is_n14(dims_b) and getattr(b, "name", None):
                key = (node.name, "inputs", getattr(b, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(b, "name", ""),
                            "pre_process_transpose_perm": [0, 2, 1],
                        }
                    )
                    seen_ops.add(key)
                    div_transpose_fixes += 1
            # If b is [1,4] -> transpose b [1,0] to [4,1]
            elif is_14(dims_b) and getattr(b, "name", None):
                key = (node.name, "inputs", getattr(b, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(b, "name", ""),
                            "pre_process_transpose_perm": [1, 0],
                        }
                    )
                    seen_ops.add(key)
                    div_transpose_fixes += 1
            # Mirror cases for a
            elif is_n14(dims_a) and getattr(a, "name", None):
                key = (node.name, "inputs", getattr(a, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(a, "name", ""),
                            "pre_process_transpose_perm": [0, 2, 1],
                        }
                    )
                    seen_ops.add(key)
                    div_transpose_fixes += 1
            elif is_14(dims_a) and getattr(a, "name", None):
                key = (node.name, "inputs", getattr(a, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(a, "name", ""),
                            "pre_process_transpose_perm": [1, 0],
                        }
                    )
                    seen_ops.add(key)
                    div_transpose_fixes += 1
        # Heuristic for Mul broadcasting with a 2D constant whose last dim is 1
        # In PP-YOLOE post-processing, tensors of shape [N, 4, K]
        # are multiplied with a constant of shape [K, 1] (anchor grid scales).
        # For NHWC-like paths this needs to be [1, K] to broadcast correctly
        # across the last dimension. Insert a 2D transpose [1, 0] on the
        # constant input. Include a specific pattern matcher and a generic fallback
        # that flips any 2D [K,1] constant used by Mul ops near the head.
        mul_const_transpose_fixes = 0
        for node in graph.nodes:
            if node.op != "Mul" or len(node.inputs) < 2:
                continue
            a, b = node.inputs[0], node.inputs[1]

            # Helper to get dims for a tensor input
            def get_dims_mul(t: gs.Tensor) -> Optional[list[int]]:
                name = getattr(t, "name", None)
                # Prefer initializers
                if name and name in initializer_map:
                    return [int(d) for d in initializer_map[name]]
                if name and name in shape_map:
                    dims = shape_map[name]
                    return [(-1 if d is None else int(d)) for d in dims]
                try:
                    vals = resolve_constant_values(t)
                    if vals is not None:
                        return list(map(int, list(vals.shape)))
                except Exception:
                    pass
                return None

            dims_a = get_dims_mul(a) or []
            dims_b = get_dims_mul(b) or []

            def is_n4k(dims: list[int]) -> bool:
                return (
                    isinstance(dims, list)
                    and len(dims) == 3
                    and (dims[0] == -1 or isinstance(dims[0], int))
                    and dims[1] == 4
                )

            def is_k1(dims: list[int]) -> bool:
                return isinstance(dims, list) and len(dims) == 2 and dims[1] == 1

            # Specific pattern: [N,4,K] * [K,1] -> flip the 2D constant to [1,K]
            if is_n4k(dims_a) and is_k1(dims_b) and getattr(b, "name", None):
                key = (node.name, "inputs", getattr(b, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(b, "name", ""),
                            "pre_process_transpose_perm": [1, 0],
                        }
                    )
                    seen_ops.add(key)
                    mul_const_transpose_fixes += 1
            elif is_n4k(dims_b) and is_k1(dims_a) and getattr(a, "name", None):
                key = (node.name, "inputs", getattr(a, "name", ""))
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": getattr(a, "name", ""),
                            "pre_process_transpose_perm": [1, 0],
                        }
                    )
                    seen_ops.add(key)
                    mul_const_transpose_fixes += 1
            else:
                # Generic fallback: flip any 2D [K,1] constant feeding Mul
                if is_k1(dims_a) and getattr(a, "name", None):
                    key = (node.name, "inputs", getattr(a, "name", ""))
                    if key not in seen_ops:
                        operations.append(
                            {
                                "op_name": node.name,
                                "param_target": "inputs",
                                "param_name": getattr(a, "name", ""),
                                "pre_process_transpose_perm": [1, 0],
                            }
                        )
                        seen_ops.add(key)
                        mul_const_transpose_fixes += 1
                if is_k1(dims_b) and getattr(b, "name", None):
                    key = (node.name, "inputs", getattr(b, "name", ""))
                    if key not in seen_ops:
                        operations.append(
                            {
                                "op_name": node.name,
                                "param_target": "inputs",
                                "param_name": getattr(b, "name", ""),
                                "pre_process_transpose_perm": [1, 0],
                            }
                        )
                        seen_ops.add(key)
                        mul_const_transpose_fixes += 1
        for node in graph.nodes:
            if node.name == "Softmax.2" and node.outputs:
                key = (node.name, "outputs", node.outputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "outputs",
                            "param_name": node.outputs[0].name,
                            "post_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            # Additional detection heads (e.g., 20x20)
            if node.name == "Softmax.0" and node.outputs:
                key = (node.name, "outputs", node.outputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "outputs",
                            "param_name": node.outputs[0].name,
                            "post_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            if node.name == "Conv.85" and node.inputs:
                key = (node.name, "inputs", node.inputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": node.inputs[0].name,
                            "pre_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            # Additional detection heads (e.g., 40x40, 20x20)
            if node.name == "Softmax.1" and node.outputs:
                key = (node.name, "outputs", node.outputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "outputs",
                            "param_name": node.outputs[0].name,
                            "post_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            if node.name == "Conv.78" and node.inputs:
                key = (node.name, "inputs", node.inputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": node.inputs[0].name,
                            "pre_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            if node.name == "Conv.71" and node.inputs:
                key = (node.name, "inputs", node.inputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": node.inputs[0].name,
                            "pre_process_transpose_perm": [0, 1, 3, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            # After Conv.85, Squeeze.2 tries to squeeze axis=1 (channel),
            # but the size-1 dim is at the end (NHWC). Move it to channel first.
            if node.name == "Squeeze.2" and node.inputs:
                key = (node.name, "inputs", node.inputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": node.inputs[0].name,
                            "pre_process_transpose_perm": [0, 3, 1, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            # Same issue can occur for Squeeze.1 (other detection head)
            if node.name == "Squeeze.1" and node.inputs:
                key = (node.name, "inputs", node.inputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": node.inputs[0].name,
                            "pre_process_transpose_perm": [0, 3, 1, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            # And for Squeeze.0 (third detection head)
            if node.name == "Squeeze.0" and node.inputs:
                key = (node.name, "inputs", node.inputs[0].name)
                if key not in seen_ops:
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": node.inputs[0].name,
                            "pre_process_transpose_perm": [0, 3, 1, 2],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
            # Name-based fallback for final division in PP-YOLOE head
            # Div.0 typically divides [N,4,K] by a per-channel factor shaped [N,1,4].
            # When shapes are dynamic, heuristics may not trigger. Force a swap of the
            # last two dims on the second input so it becomes [N,4,1] and broadcasts over K.
            if node.name == "Div.0" and len(node.inputs) >= 2:
                div_rhs = node.inputs[1]
                key = (node.name, "inputs", getattr(div_rhs, "name", ""))
                if key not in seen_ops and getattr(div_rhs, "name", None):
                    operations.append(
                        {
                            "op_name": node.name,
                            "param_target": "inputs",
                            "param_name": div_rhs.name,
                            "pre_process_transpose_perm": [0, 2, 1],
                        }
                    )
                    seen_ops.add(key)
                    name_based_fixes += 1
        if verbose and softmax_fixes > 0:
            print(f"[INFO] Applied Softmax post-transpose fix to {softmax_fixes} node(s) with [N,4,17,*] output")
        if verbose and conv_input_fixes > 0:
            print(f"[INFO] Applied Conv input pre-transpose fix to {conv_input_fixes} node(s) with in_channels=17")
        if verbose and (connectivity_softmax_fixes > 0 or connectivity_conv_fixes > 0):
            print(
                f"[INFO] Applied connectivity-based layout fixes: Softmax={connectivity_softmax_fixes}, Conv={connectivity_conv_fixes}"
            )
        if verbose and squeeze_fixes > 0:
            print(f"[INFO] Applied Squeeze pre-transpose fix to {squeeze_fixes} node(s) targeting axis=1")
        if verbose and add_const_transpose_fixes > 0:
            print(f"[INFO] Applied Add-const 2D transpose fix to {add_const_transpose_fixes} node(s)")
        if verbose and mul_const_transpose_fixes > 0:
            print(f"[INFO] Applied Mul-const 2D transpose fix to {mul_const_transpose_fixes} node(s)")
        if verbose and div_transpose_fixes > 0:
            print(f"[INFO] Applied Div broadcast transpose fix to {div_transpose_fixes} node(s)")
        if verbose and name_based_fixes > 0:
            print(
                f"[INFO] Applied name-based layout fixes to {name_based_fixes} node(s) [Softmax.2/Softmax.0/Conv.85/Conv.71/Squeeze.2/Softmax.1/Conv.78/Squeeze.1/Squeeze.0]"
            )
    except Exception as exc:  # pragma: no cover - best-effort shape inference
        if verbose:
            print(f"[WARN] Softmax layout heuristic skipped (shape inference failed: {exc})")

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
        print(f"[INFO] Generated parameter replacements for {len(operations)} operation(s)")
    return tmp_path


def export_to_saved_model(
    onnx_path: Path,
    saved_dir: Path,
    verbose: bool = False,
    keep_nchw: bool = True,
    use_nms_dynamic: bool = True,
    batch_size: int = 1,
    disable_heuristics: bool = True,
) -> Path:
    if verbose:
        print(f"[INFO] Preparing to convert ONNX model: {onnx_path}")
    if saved_dir.exists():
        if verbose:
            print(f"[INFO] Clearing existing directory: {saved_dir}")
        shutil.rmtree(saved_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    keep_nchw = True
    disable_heuristics = True


    if verbose:
        print("[INFO] Converting ONNX → TensorFlow with onnx2tf…")
        print(f"[INFO] Options: keep_nchw={keep_nchw}, nms_dynamic={use_nms_dynamic}, batch={batch_size}")

    param_file: Optional[Path] = None
    try:
        # Build parameter replacement file (custom heuristics)
        if not disable_heuristics:
            param_file = _build_param_replacement_file(onnx_path, verbose=verbose)
        else:
            if verbose:
                print("[INFO] Custom heuristics disabled - using onnx2tf defaults")

        # Base conversion arguments
        convert_kwargs = dict(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_dir),
            output_signaturedefs=True,
            non_verbose=not verbose,
        )

        # Add custom parameter file if generated
        if param_file is not None:
            if verbose:
                print(f"[INFO] Applying channel-last bias transposes via {param_file}")
            convert_kwargs["param_replacement_file"] = str(param_file)

        # Add batch size if specified
        if batch_size > 0:
            convert_kwargs["batch_size"] = batch_size
            if verbose:
                print(f"[INFO] Fixed batch size: {batch_size}")

        # Add layout preservation options
        if keep_nchw:
            # Keep NCHW layout - useful if your model is already optimized for NCHW
            convert_kwargs["keep_ncw_or_nchw_or_ncdhw_input_names"] = ["image"]
            if verbose:
                print("[INFO] Keeping NCHW layout for 'image' input")

        # Add NMS dynamic output option
        if use_nms_dynamic:
            convert_kwargs["output_nms_with_dynamic_tensor"] = True
            if verbose:
                print("[INFO] Using dynamic tensor output for NMS")

        # Perform conversion
        convert(**convert_kwargs)
    finally:
        if param_file is not None and param_file.exists():
            param_file.unlink(missing_ok=True)

    print(f"[OK] SavedModel exported to: {saved_dir}")
    return saved_dir


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.model).expanduser().resolve()
    if not onnx_path.exists():
        print(f"[ERROR] ONNX model not found: {onnx_path}", file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    saved_dir = output_dir / args.saved_model_name

    try:
        export_to_saved_model(
            onnx_path,
            saved_dir,
            verbose=args.verbose,
            keep_nchw=args.keep_nchw,
            use_nms_dynamic=args.use_nms_dynamic,
            batch_size=args.batch_size,
            disable_heuristics=args.disable_heuristics,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to convert ONNX to TensorFlow SavedModel: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("[DONE] Conversion pipeline completed successfully.")


if __name__ == "__main__":
    main()
