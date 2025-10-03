#!/usr/bin/env python3
import argparse
import time
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import onnx

try:
    import onnxruntime as ort
except Exception as e:
    ort = None


def _elem_type_to_dtype(tensor_type: int):
    # https://onnx.ai/onnx/api/mapping.html#onnx-mapping-tensor-dtype
    import onnx

    m = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.BOOL: np.bool_,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.COMPLEX64: np.complex64,
        onnx.TensorProto.COMPLEX128: np.complex128,
        onnx.TensorProto.STRING: np.object_,
        onnx.TensorProto.UINT32: np.uint32 if hasattr(np, 'uint32') else np.uint64,
        onnx.TensorProto.UINT64: np.uint64,
        onnx.TensorProto.BFLOAT16: np.float16,  # best effort
    }
    return m.get(tensor_type, np.float32)


def load_model_info(model_path: str) -> Dict[str, Any]:
    m = onnx.load(model_path)
    info: Dict[str, Any] = {}
    info["ir_version"] = m.ir_version
    info["opsets"] = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]

    def shape_of(value_info) -> List[Any]:
        t = value_info.type.tensor_type
        shp = []
        for d in t.shape.dim:
            if d.dim_value:
                shp.append(int(d.dim_value))
            else:
                shp.append(d.dim_param or "?")
        return shp

    inputs = []
    for i in m.graph.input:
        t = i.type.tensor_type
        inputs.append({
            "name": i.name,
            "dtype": t.elem_type,
            "shape": shape_of(i),
        })
    outputs = []
    for o in m.graph.output:
        t = o.type.tensor_type
        outputs.append({
            "name": o.name,
            "dtype": t.elem_type,
            "shape": shape_of(o),
        })

    ops: Dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1

    initializers = [(init.name, list(init.dims), init.data_type) for init in m.graph.initializer]
    total_params = 0
    for _, dims, _ in initializers:
        if dims:
            prod = 1
            for d in dims:
                prod *= int(d)
            total_params += prod

    info.update({
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(m.graph.node),
        "unique_ops": len(ops),
        "ops_hist": sorted(ops.items(), key=lambda kv: -kv[1]),
        "initializers": len(initializers),
        "total_params": int(total_params),
    })
    return info


def parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x)


# Removed synthetic input generation; benchmarking now requires a real image via --img/--use-demo.


def _repo_root() -> str:
    # This file is at <repo>/pipeline/PP-YOLOE/profile_onnx.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _default_demo_image() -> str:
    root = _repo_root()
    return os.path.join(root, 'pipeline', 'dataset', 'demo', 'demo.jpg')


def make_image_inputs(model_info: Dict[str, Any], input_shape: Tuple[int, ...], img_path: str) -> Dict[str, np.ndarray]:
    """Build feeds using real image preprocessing to better reflect real-world latency.

    Falls back to dummy inputs if preprocessing import fails.
    """
    feeds = None
    try:
        # Reuse the exact preprocessing from onnx_inference_image.py for parity
        from onnx_inference_image import preprocess_image  # type: ignore
        # Determine target size from provided input_shape if possible
        target_hw = None
        if len(input_shape) == 4 and input_shape[2] and input_shape[3]:
            target_hw = (int(input_shape[2]), int(input_shape[3]))
        if target_hw is None:
            target_hw = (640, 640)
        prep = preprocess_image(img_path, target_size=target_hw, keep_ratio=False)
        # Map by input names declared in the model
        feeds = {}
        for inp in model_info["inputs"]:
            name = inp["name"]
            if name == 'image' and 'image' in prep:
                feeds[name] = prep['image'][None, :]
            elif name in ('im_shape', 'scale_factor') and name in prep:
                feeds[name] = prep[name][None, :]
        # Ensure all inputs are covered without synthetic fallbacks
        expected = {i["name"] for i in model_info["inputs"]}
        missing = expected - set(feeds.keys())
        if missing:
            raise RuntimeError(f"Missing required inputs for image preprocessing: {sorted(missing)}")
    except Exception as e:
        raise RuntimeError(f"Failed to build image-based inputs: {e}")
    return feeds


def available_providers() -> List[str]:
    if ort is None:
        return []
    return list(ort.get_available_providers())


def pick_providers(pref: str) -> List[Any]:
    provs = available_providers()
    pref = (pref or "cpu").lower()
    if pref in ("coreml", "ane") and "CoreMLExecutionProvider" in provs:
        # Use the same CoreML options as onnx_inference_image.py for realistic results
        coreml_opts: Dict[str, str] = {
            "ModelFormat": "MLProgram",
            "EnableOnSubgraphs": "1",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
        }
        return [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"]
    if pref in ("cpu", "default"):
        return ["CPUExecutionProvider"]
    # Generic: try exact match tokenizing by 'ExecutionProvider'
    matches = [p for p in provs if pref.lower() in p.lower()]
    if matches:
        return [matches[0], "CPUExecutionProvider"]
    # Fallback
    return ["CPUExecutionProvider"]


def run_benchmark(model_path: str, input_shape: Tuple[int, ...], ep: str, warmup: int, runs: int, enable_profile: bool = False, profile_dir: str = "", img_path: Optional[str] = None) -> Dict[str, Any]:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Please install 'onnxruntime' or 'onnxruntime-silicon'.")

    model_info = load_model_info(model_path)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if enable_profile:
        so.enable_profiling = True

    providers = pick_providers(ep)
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)

    # Build feeds: require real image inputs
    if not img_path:
        raise RuntimeError("img_path is required for benchmarking; pass --img or --use-demo")
    feeds = make_image_inputs(model_info, input_shape, img_path)

    # Warmup
    for _ in range(max(0, warmup)):
        sess.run(None, feeds)

    # Timed runs
    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    def pct(p: float) -> float:
        arr = np.array(times)
        return float(np.percentile(arr, p)) if len(arr) else float("nan")

    # Normalize providers for reporting: expand tuples to display options
    shown_providers: List[Any] = []
    for p in providers:
        if isinstance(p, tuple):
            shown_providers.append(p)
        else:
            shown_providers.append(p)

    result = {
        "providers": shown_providers,
        "warmup": warmup,
        "runs": runs,
        "img_path": img_path or "",
        "latency_ms_avg": float(np.mean(times)) if times else float("nan"),
        "latency_ms_p50": pct(50),
        "latency_ms_p90": pct(90),
        "latency_ms_p95": pct(95),
        "latency_ms_min": float(np.min(times)) if times else float("nan"),
        "latency_ms_max": float(np.max(times)) if times else float("nan"),
    }
    profile_path = ""
    profile_summary: Dict[str, Any] = {}
    if enable_profile:
        try:
            prof_file = sess.end_profiling()
            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
                dest = os.path.join(profile_dir, os.path.basename(prof_file))
                try:
                    import shutil
                    shutil.move(prof_file, dest)
                    profile_path = dest
                except Exception:
                    profile_path = prof_file
            else:
                profile_path = prof_file
            # Parse a brief provider/node/time summary from ORT profile JSON
            try:
                import json as _json
                from collections import defaultdict
                with open(profile_path, "r") as pf:
                    trace = _json.load(pf)
                # ORT profile is a Chrome trace format; node events typically have category 'Node'
                provider_counts: Dict[str, int] = {}
                provider_time_ms: Dict[str, float] = {}
                op_type_counts: Dict[str, int] = {}
                provider_op_time_ms: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
                provider_op_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
                for ev in trace:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("cat") != "Node":
                        continue
                    args = ev.get("args", {}) or {}
                    prov = args.get("provider") or args.get("execution_provider") or "UNKNOWN"
                    op = args.get("op_name") or args.get("op") or "?"
                    dur_us = float(ev.get("dur", 0.0))
                    dur_ms = dur_us / 1000.0
                    provider_counts[prov] = provider_counts.get(prov, 0) + 1
                    provider_time_ms[prov] = provider_time_ms.get(prov, 0.0) + dur_ms
                    op_type_counts[op] = op_type_counts.get(op, 0) + 1
                    provider_op_time_ms[prov][op] += dur_ms
                    provider_op_counts[prov][op] += 1

                # Compute top ops by time per provider (limit to top 10)
                top_ops_by_time: Dict[str, List[Tuple[str, float, int]]] = {}
                for prov, op_times in provider_op_time_ms.items():
                    items = sorted(op_times.items(), key=lambda kv: -kv[1])
                    top = []
                    for op, t in items[:10]:
                        top.append((op, round(t, 3), int(provider_op_counts[prov].get(op, 0))))
                    top_ops_by_time[prov] = top

                profile_summary = {
                    "provider_node_counts": provider_counts,
                    "provider_total_time_ms": {k: round(v, 3) for k, v in provider_time_ms.items()},
                    "unique_op_types": len(op_type_counts),
                    "provider_op_time_ms": {p: {op: round(t, 3) for op, t in d.items()} for p, d in provider_op_time_ms.items()},
                    "provider_op_counts": {p: dict(d) for p, d in provider_op_counts.items()},
                    "top_ops_by_time": top_ops_by_time,
                }
            except Exception:
                profile_summary = {}
        except Exception:
            profile_path = ""
    return {"model": model_info, "benchmark": result, "ort_profile": profile_path, "profile_summary": profile_summary}


def main():
    parser = argparse.ArgumentParser(description="Inspect and benchmark an ONNX model with onnxruntime")
    parser.add_argument(
        "--model",
        type=str,
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,640,640",
        help="Input shape for the main 4D input (N,C,H,W)",
    )
    parser.add_argument(
        "--ep",
        type=str,
        default="coreml",
        help="Execution provider preference: coreml|cpu|…",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--json", type=str, default="", help="Optional path to write JSON report")
    parser.add_argument("--img", type=str, default="", help="Path to image for realistic preprocessing (required, unless --use-demo provided)")
    parser.add_argument("--use-demo", action="store_true", help="Use pipeline/dataset/demo/demo.jpg for preprocessing")
    parser.add_argument("--ort-profile", action="store_true", help="Enable onnxruntime profiling and save timeline JSON")
    parser.add_argument("--ort-profile-dir", type=str, default="pipeline/PP-YOLOE/output", help="Directory to place ORT profile JSON")

    args = parser.parse_args()
    ishape = parse_shape(args.input_shape)

    info = load_model_info(args.model)
    print("=== Model Info ===")
    print("ir_version:", info["ir_version"])
    print("opset_import:", info["opsets"])
    print("inputs:")
    for i in info["inputs"]:
        print(" -", i["name"], i["shape"], i["dtype"])  # dtype is TensorProto enum
    print("outputs:")
    for o in info["outputs"]:
        print(" -", o["name"], o["shape"], o["dtype"])  # dtype is TensorProto enum
    print("node_count:", info["node_count"], "unique_ops:", info["unique_ops"], "initializers:", info["initializers"], "total_params:", info["total_params"])
    print("top ops:")
    for k, v in info["ops_hist"][:25]:
        print(f"  {k}: {v}")

    if ort is None:
        print("onnxruntime not available; skipping benchmark. Install 'onnxruntime' or 'onnxruntime-silicon'.")
        return


    print("\n=== Benchmark ===")
    print("Available providers:", available_providers())
    use_img = args.img or (_default_demo_image() if args.use_demo else "")
    if not use_img:
        print("[ERROR] --img is required (or use --use-demo). No synthetic inputs are supported.")
        return
    # Resolve to absolute path and ensure it exists
    img_path = use_img if os.path.isabs(use_img) else os.path.abspath(use_img)
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        return
    else:
        print(f"Using real image inputs: {img_path}")
    res = run_benchmark(args.model, ishape, args.ep, args.warmup, args.runs, enable_profile=args.ort_profile, profile_dir=args.ort_profile_dir, img_path=img_path)
    bench = res["benchmark"]
    print("Providers:", bench["providers"])
    print("EP:",args.ep,"Runs:", bench["runs"], "Warmup:", bench["warmup"], "Image:", bench.get("img_path","(synthetic)"))
    print(
        "Latency (ms): avg={avg:.2f} p50={p50:.2f} p90={p90:.2f} p95={p95:.2f} min={min:.2f} max={max:.2f}".format(
            avg=bench["latency_ms_avg"],
            p50=bench["latency_ms_p50"],
            p90=bench["latency_ms_p90"],
            p95=bench["latency_ms_p95"],
            min=bench["latency_ms_min"],
            max=bench["latency_ms_max"],
        )
    )

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print("Saved JSON report to:", args.json)
    if res.get("ort_profile"):
        print("Saved ORT profile to:", res["ort_profile"])
        if res.get("profile_summary"):
            ps = res["profile_summary"]
            print("ORT provider summary: node_counts=", ps.get("provider_node_counts", {}), " time_ms=", ps.get("provider_total_time_ms", {}))
            # Print top CPU ops by total time for quick insight
            tops = (ps.get("top_ops_by_time") or {}).get("CPUExecutionProvider")
            if tops:
                print("Top CPU ops by time:")
                for op, t, cnt in tops[:8]:
                    print(f" - {op}: {t} ms ({cnt} nodes)")


if __name__ == "__main__":
    main()
