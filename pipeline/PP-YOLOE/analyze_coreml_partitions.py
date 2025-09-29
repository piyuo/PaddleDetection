#!/usr/bin/env python3
"""
Analyze CoreML partition boundaries in ONNX models to identify bottlenecks.

This script identifies:
1. Which nodes CoreML EP accepts vs rejects
2. Where partition boundaries occur (CPU ↔ CoreML transitions)
3. The ops immediately before/after partition breaks
4. Suggestions for improving CoreML coverage
"""
import argparse
import os
import json
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

import onnx
from onnx import numpy_helper
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def get_node_chain(model: onnx.ModelProto) -> Dict[str, Any]:
    """Build graph connectivity information."""
    g = model.graph

    # Map tensor name to producer node
    producer: Dict[str, onnx.NodeProto] = {}
    for node in g.node:
        for out in node.output:
            producer[out] = node

    # Map tensor name to consumer nodes
    consumers: Dict[str, List[onnx.NodeProto]] = defaultdict(list)
    for node in g.node:
        for inp in node.input:
            consumers[inp].append(node)

    # Find initializers (constants)
    initializers = {init.name for init in g.initializer}

    # Find graph inputs/outputs
    inputs = {vi.name for vi in g.input}
    outputs = {vo.name for vo in g.output}

    return {
        "producer": producer,
        "consumers": consumers,
        "initializers": initializers,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": list(g.node)
    }


def profile_with_coreml(model_path: str, img_path: str) -> Dict[str, Any]:
    """Run model with CoreML EP profiling to identify which nodes run where."""
    if ort is None:
        raise RuntimeError("onnxruntime not available")

    # Import preprocessing from profile_onnx
    try:
        from onnx_inference_image import preprocess_image
    except ImportError:
        raise RuntimeError("Cannot import preprocessing - ensure onnx_inference_image.py is available")

    # Load model to get input info
    model = onnx.load(model_path)

    # Prepare inputs
    prep = preprocess_image(img_path, target_size=(640, 640), keep_ratio=False)
    feeds = {}
    for inp in model.graph.input:
        name = inp.name
        if name == 'image' and 'image' in prep:
            feeds[name] = prep['image'][None, :]
        elif name in ('im_shape', 'scale_factor') and name in prep:
            feeds[name] = prep[name][None, :]

    # Create session with profiling enabled
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True

    coreml_opts = {
        "ModelFormat": "MLProgram",
        "EnableOnSubgraphs": "1",
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1",
    }
    providers = [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)

    # Warmup
    for _ in range(3):
        sess.run(None, feeds)

    # Profile run
    sess.run(None, feeds)
    prof_file = sess.end_profiling()

    # Parse profile
    with open(prof_file, 'r') as f:
        trace = json.load(f)

    # Extract node execution info
    node_provider: Dict[str, str] = {}
    node_time_ms: Dict[str, float] = {}
    provider_counts: Dict[str, int] = defaultdict(int)

    for ev in trace:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "Node":
            continue

        args = ev.get("args", {}) or {}
        node_name = args.get("op_name", "")
        provider = args.get("provider") or args.get("execution_provider") or "UNKNOWN"
        dur_us = float(ev.get("dur", 0.0))
        dur_ms = dur_us / 1000.0

        if node_name:
            node_provider[node_name] = provider
            node_time_ms[node_name] = node_time_ms.get(node_name, 0.0) + dur_ms
            provider_counts[provider] += 1

    # Clean up profile file
    try:
        os.remove(prof_file)
    except:
        pass

    return {
        "node_provider": node_provider,
        "node_time_ms": node_time_ms,
        "provider_counts": dict(provider_counts)
    }


def analyze_partitions(model_path: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze partition boundaries and identify bottleneck ops."""
    model = onnx.load(model_path)
    chain = get_node_chain(model)

    node_provider = profile_data["node_provider"]
    node_time_ms = profile_data["node_time_ms"]

    # Identify CoreML vs CPU nodes
    coreml_nodes: Set[str] = set()
    cpu_nodes: Set[str] = set()

    for node in chain["nodes"]:
        provider = node_provider.get(node.name, "UNKNOWN")
        if "CoreML" in provider:
            coreml_nodes.add(node.name)
        elif "CPU" in provider:
            cpu_nodes.add(node.name)

    # Find partition boundaries (transitions between CoreML and CPU)
    boundaries: List[Dict[str, Any]] = []

    for node in chain["nodes"]:
        if node.name not in coreml_nodes:
            continue

        # Check if any consumer is CPU
        for out_tensor in node.output:
            for consumer in chain["consumers"].get(out_tensor, []):
                if consumer.name in cpu_nodes:
                    boundaries.append({
                        "coreml_node": node.name,
                        "coreml_op": node.op_type,
                        "cpu_node": consumer.name,
                        "cpu_op": consumer.op_type,
                        "connecting_tensor": out_tensor,
                        "type": "CoreML→CPU"
                    })

        # Check if any input producer is CPU
        for inp_tensor in node.input:
            if inp_tensor in chain["initializers"] or inp_tensor in chain["inputs"]:
                continue
            producer = chain["producer"].get(inp_tensor)
            if producer and producer.name in cpu_nodes:
                boundaries.append({
                    "cpu_node": producer.name,
                    "cpu_op": producer.op_type,
                    "coreml_node": node.name,
                    "coreml_op": node.op_type,
                    "connecting_tensor": inp_tensor,
                    "type": "CPU→CoreML"
                })

    # Analyze CPU bottleneck ops
    cpu_op_time: Dict[str, float] = defaultdict(float)
    cpu_op_count: Dict[str, int] = defaultdict(int)

    for node in chain["nodes"]:
        if node.name in cpu_nodes:
            time_ms = node_time_ms.get(node.name, 0.0)
            cpu_op_time[node.op_type] += time_ms
            cpu_op_count[node.op_type] += 1

    cpu_ops_sorted = sorted(cpu_op_time.items(), key=lambda x: -x[1])

    # Identify CoreML op types
    coreml_op_types: Set[str] = set()
    for node in chain["nodes"]:
        if node.name in coreml_nodes:
            coreml_op_types.add(node.op_type)

    # Identify ops that appear ONLY on CPU (never on CoreML)
    all_op_types = {node.op_type for node in chain["nodes"]}
    cpu_only_ops = all_op_types - coreml_op_types

    return {
        "total_nodes": len(chain["nodes"]),
        "coreml_nodes": len(coreml_nodes),
        "cpu_nodes": len(cpu_nodes),
        "partition_boundaries": boundaries,
        "coreml_op_types": sorted(coreml_op_types),
        "cpu_only_ops": sorted(cpu_only_ops),
        "cpu_bottleneck_ops": [(op, round(time, 2), cpu_op_count[op]) for op, time in cpu_ops_sorted[:20]],
        "provider_stats": profile_data["provider_counts"]
    }


def suggest_optimizations(analysis: Dict[str, Any]) -> List[str]:
    """Generate optimization suggestions based on partition analysis."""
    suggestions = []

    cpu_only = set(analysis["cpu_only_ops"])
    boundaries = analysis["partition_boundaries"]
    bottlenecks = [op for op, _, _ in analysis["cpu_bottleneck_ops"][:10]]

    # Identify problematic ops at boundaries
    boundary_cpu_ops = set()
    for b in boundaries:
        if "cpu_op" in b:
            boundary_cpu_ops.add(b["cpu_op"])

    # Known ANE-unfriendly ops
    known_issues = {
        "NonMaxSuppression": "NMS is CPU-only; consider splitting model before/after NMS",
        "RoiAlign": "RoiAlign may not be ANE-supported; try alternative pooling approaches",
        "TopK": "TopK often forces CPU fallback; check if replaceable",
        "Resize": "Some Resize modes are CPU-only; ensure using 'nearest' or 'linear' modes",
        "Where": "Where op typically CPU-only; check if conditional logic can be simplified",
        "IsNaN": "IsNaN forces CPU; consider preprocessing to avoid NaN checks",
        "Loop": "Control flow ops (Loop/If) are CPU-only",
        "If": "Control flow ops (Loop/If) are CPU-only",
    }

    for op in boundary_cpu_ops:
        if op in known_issues:
            suggestions.append(f"⚠️  {op} at partition boundary: {known_issues[op]}")

    for op in bottlenecks:
        if op in cpu_only and op not in boundary_cpu_ops:
            if op in known_issues:
                suggestions.append(f"🔴 {op} is CPU bottleneck: {known_issues[op]}")
            else:
                suggestions.append(f"🔴 {op} is CPU-only and taking significant time")

    # General suggestions
    coreml_pct = 100.0 * analysis["coreml_nodes"] / analysis["total_nodes"]
    if coreml_pct < 10:
        suggestions.append(f"⚠️  Only {coreml_pct:.1f}% of nodes on CoreML - major architecture incompatibility")
        suggestions.append("💡 Consider splitting model at natural boundaries (backbone vs head)")

    if len(boundaries) > 5:
        suggestions.append(f"⚠️  {len(boundaries)} partition transitions detected - each transition has overhead")
        suggestions.append("💡 Try to consolidate operations to reduce CPU↔CoreML data transfers")

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Analyze CoreML partition boundaries in ONNX models")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--img", type=str, required=True, help="Path to test image")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed node-by-node breakdown")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return

    if not os.path.exists(args.img):
        print(f"Error: Image not found: {args.img}")
        return

    print("=== CoreML Partition Analysis ===\n")
    print(f"Model: {args.model}")
    print(f"Image: {args.img}\n")

    print("Running profiled inference...")
    profile_data = profile_with_coreml(args.model, args.img)

    print("Analyzing partitions...\n")
    analysis = analyze_partitions(args.model, profile_data)

    print("=== Summary ===")
    print(f"Total nodes: {analysis['total_nodes']}")
    print(f"CoreML nodes: {analysis['coreml_nodes']} ({100.0 * analysis['coreml_nodes'] / analysis['total_nodes']:.1f}%)")
    print(f"CPU nodes: {analysis['cpu_nodes']} ({100.0 * analysis['cpu_nodes'] / analysis['total_nodes']:.1f}%)")
    print(f"Partition boundaries: {len(analysis['partition_boundaries'])}")

    print(f"\n=== CoreML-Supported Op Types ({len(analysis['coreml_op_types'])}) ===")
    for op in analysis['coreml_op_types'][:20]:
        print(f"  • {op}")
    if len(analysis['coreml_op_types']) > 20:
        print(f"  ... and {len(analysis['coreml_op_types']) - 20} more")

    print(f"\n=== CPU-Only Op Types ({len(analysis['cpu_only_ops'])}) ===")
    for op in analysis['cpu_only_ops'][:20]:
        print(f"  • {op}")
    if len(analysis['cpu_only_ops']) > 20:
        print(f"  ... and {len(analysis['cpu_only_ops']) - 20} more")

    print("\n=== CPU Bottleneck Ops (by time) ===")
    for op, time_ms, count in analysis['cpu_bottleneck_ops'][:15]:
        print(f"  • {op}: {time_ms:.2f}ms ({count} nodes)")

    print("\n=== Partition Boundaries ===")
    if len(analysis['partition_boundaries']) == 0:
        print("  No boundaries detected (single partition)")
    else:
        # Group by type
        coreml_to_cpu = [b for b in analysis['partition_boundaries'] if b['type'] == 'CoreML→CPU']
        cpu_to_coreml = [b for b in analysis['partition_boundaries'] if b['type'] == 'CPU→CoreML']

        if coreml_to_cpu:
            print(f"\n  CoreML → CPU transitions ({len(coreml_to_cpu)}):")
            for b in coreml_to_cpu[:10]:
                print(f"    {b['coreml_op']} → {b['cpu_op']}")
            if len(coreml_to_cpu) > 10:
                print(f"    ... and {len(coreml_to_cpu) - 10} more")

        if cpu_to_coreml:
            print(f"\n  CPU → CoreML transitions ({len(cpu_to_coreml)}):")
            for b in cpu_to_coreml[:10]:
                print(f"    {b['cpu_op']} → {b['coreml_op']}")
            if len(cpu_to_coreml) > 10:
                print(f"    ... and {len(cpu_to_coreml) - 10} more")

    print("\n=== Optimization Suggestions ===")
    suggestions = suggest_optimizations(analysis)
    if suggestions:
        for s in suggestions:
            print(f"  {s}")
    else:
        print("  No specific issues detected")

    if args.verbose:
        print("\n=== Detailed Node Breakdown ===")
        profile_data_nodes = profile_data["node_provider"]
        model = onnx.load(args.model)
        for node in model.graph.node[:50]:  # Limit to first 50 nodes
            provider = profile_data_nodes.get(node.name, "UNKNOWN")
            time_ms = profile_data["node_time_ms"].get(node.name, 0.0)
            print(f"  {node.name} ({node.op_type}): {provider} - {time_ms:.3f}ms")
        if len(model.graph.node) > 50:
            print(f"  ... and {len(model.graph.node) - 50} more nodes")

    if args.output:
        output_data = {
            "model": args.model,
            "analysis": analysis,
            "suggestions": suggestions
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved detailed analysis to: {args.output}")


if __name__ == "__main__":
    main()