import sys
import argparse
import collections
from typing import Dict, List, Tuple, Optional, Any
import json

import onnx
from onnx import numpy_helper, helper, shape_inference, TensorProto

try:
    import numpy as np
except Exception:
    np = None  # Only used for sizes; degrade gracefully

try:
    import onnxruntime as ort  # optional, for provider info only
except Exception:
    ort = None


DTYPE_NAME = {v: k for k, v in TensorProto.DataType.items()}


def dtype_name(dtype: int) -> str:
    return DTYPE_NAME.get(dtype, str(dtype))


def fmt_shape(value_info) -> List[str]:
    t = value_info.type.tensor_type
    shp = []
    if not t.HasField('shape'):
        return ['?']
    for d in t.shape.dim:
        if d.HasField('dim_value') and d.dim_value != 0:
            shp.append(str(d.dim_value))
        else:
            shp.append(str(d.dim_param or '?'))
    return shp


def graph_inputs_outputs(m: onnx.ModelProto) -> Tuple[List[Tuple[str, List[str], str]], List[Tuple[str, List[str], str]]]:
    ins = []
    for i in m.graph.input:
        shp = fmt_shape(i)
        t = i.type.tensor_type
        ins.append((i.name, shp, dtype_name(t.elem_type)))
    outs = []
    for o in m.graph.output:
        shp = fmt_shape(o)
        t = o.type.tensor_type
        outs.append((o.name, shp, dtype_name(t.elem_type)))
    return ins, outs


def collect_constant_values(m: onnx.ModelProto) -> Dict[str, Any]:
    consts: Dict[str, Any] = {}
    # Initializers
    name_to_init = {init.name: init for init in m.graph.initializer}
    if np is not None:
        for name, init in name_to_init.items():
            try:
                consts[name] = numpy_helper.to_array(init)
            except Exception:
                pass
    # Constant nodes (value attribute)
    for n in m.graph.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value' and a.HasField('t'):
                    try:
                        arr = numpy_helper.to_array(a.t)
                        # A Constant node typically has a single output
                        if n.output:
                            consts[n.output[0]] = arr
                    except Exception:
                        pass
    return consts


def bytes_of_array(arr: Any) -> int:
    if np is None:
        # fallback: estimate from dtype size map if possible
        return int(arr.size) * 4
    return int(arr.nbytes)


def summarize_initializers(m: onnx.ModelProto) -> Tuple[int, int, Dict[str, int], List[Tuple[str, int, str]]]:
    total_params = 0
    total_bytes = 0
    dtypes = collections.Counter()
    largest: List[Tuple[str, int, str]] = []
    for init in m.graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
            sz = arr.size
            by = bytes_of_array(arr)
            total_params += sz
            total_bytes += by
            dtypes[dtype_name(init.data_type)] += sz
            largest.append((init.name, by, f"{arr.shape} {arr.dtype}"))
        except Exception:
            # Fallback if numpy_helper fails
            sz = 1
            for d in init.dims:
                sz *= d if d > 0 else 1
            total_params += sz
            # approximate bytes
            total_bytes += sz * 4
            dtypes[dtype_name(init.data_type)] += sz
            largest.append((init.name, sz * 4, f"{list(init.dims)} {dtype_name(init.data_type)} (approx)"))
    largest.sort(key=lambda x: -x[1])
    return total_params, total_bytes, dtypes, largest


def op_histogram(m: onnx.ModelProto) -> Dict[str, int]:
    ops: Dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return ops


def infer_shapes_safe(m: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(m)
    except Exception:
        return m


def is_scalar_like(arr: Any) -> bool:
    if arr is None:
        return False
    return arr.shape == () or arr.size == 1


def analyze_coreml_surgery_hints(m: onnx.ModelProto) -> Dict[str, object]:
    consts = collect_constant_values(m)
    hints = {
        'clip_non_scalar_bounds': [],
        'pow_constant_histogram': collections.Counter(),
        'pow_non_const_count': 0,
        'div_by_const': 0,
        'div_by_var': 0,
        'slice_gather_candidates': 0,
        'slice_details': [],
        'concat_max_inputs': 0,
        'concat_large_nodes': [],
        'swish_patterns': 0,
        'hardsigmoid_count': 0,
        'nms_count': 0,
        'roialign_count': 0,
        'resize_attrs': [],
        'pad_dynamic_count': 0,
    }

    # Build quick map from output name to producing node (for simple pattern checks)
    out_to_node: Dict[str, onnx.NodeProto] = {}
    for n in m.graph.node:
        for o in n.output:
            out_to_node[o] = n

    # Attribute/inputs analysis
    for n in m.graph.node:
        if n.op_type == 'Clip':
            # Inputs: X, min, max
            # CoreML prefers scalar min/max (0-D). Flag non-scalars.
            for idx, role in [(1, 'min'), (2, 'max')]:
                if idx < len(n.input) and n.input[idx]:
                    name = n.input[idx]
                    arr = consts.get(name)
                    if arr is not None:
                        if not is_scalar_like(arr):
                            hints['clip_non_scalar_bounds'].append({'node': n.name or '(unnamed)', 'bound': role, 'input': name, 'shape': list(arr.shape)})
        elif n.op_type == 'Pow':
            # Find exponent as constant
            exp_val = None
            if len(n.input) >= 2:
                b = consts.get(n.input[1])
                a = consts.get(n.input[0])
                # Prefer exponent on second input (X^Y)
                if b is not None and is_scalar_like(b):
                    try:
                        exp_val = float(b.reshape(()))
                    except Exception:
                        pass
                elif a is not None and is_scalar_like(a):
                    # Rare case c^x, not typical
                    try:
                        exp_val = float(a.reshape(()))
                    except Exception:
                        pass
            if exp_val is None:
                hints['pow_non_const_count'] += 1
            else:
                # Bucket exponents into friendly text
                key = repr(round(exp_val, 6))
                hints['pow_constant_histogram'][key] += 1
        elif n.op_type == 'Div':
            if len(n.input) >= 2 and n.input[1] in consts and is_scalar_like(consts[n.input[1]]):
                hints['div_by_const'] += 1
            else:
                hints['div_by_var'] += 1
        elif n.op_type == 'Slice':
            # Heuristic: single-axis slice with size 1 -> Gather candidate
            starts = consts.get(n.input[1]) if len(n.input) > 1 else None
            ends = consts.get(n.input[2]) if len(n.input) > 2 else None
            axes = consts.get(n.input[3]) if len(n.input) > 3 else None
            steps = consts.get(n.input[4]) if len(n.input) > 4 else None
            try:
                if starts is not None and ends is not None:
                    s = np.array(starts).flatten().tolist()
                    e = np.array(ends).flatten().tolist()
                    ax = np.array(axes).flatten().tolist() if axes is not None else [0]
                    st = np.array(steps).flatten().tolist() if steps is not None else [1] * len(s)
                    if len(s) == len(e) == len(ax) == len(st) == 1 and st[0] == 1 and (e[0] - s[0] == 1):
                        hints['slice_gather_candidates'] += 1
                        hints['slice_details'].append({'node': n.name or '(unnamed)', 'axis': ax[0], 'index': s[0]})
            except Exception:
                pass
        elif n.op_type == 'Concat':
            cnt = len(n.input)
            if cnt > hints['concat_max_inputs']:
                hints['concat_max_inputs'] = cnt
            if cnt >= 8:
                hints['concat_large_nodes'].append({'node': n.name or '(unnamed)', 'inputs': cnt})
        elif n.op_type == 'Mul':
            # crude swish pattern: Mul(x, Sigmoid(x))
            if len(n.input) == 2:
                a, b = n.input
                na = out_to_node.get(a)
                nb = out_to_node.get(b)
                if na and na.op_type == 'Sigmoid' and na.input and (na.input[0] == b):
                    hints['swish_patterns'] += 1
                if nb and nb.op_type == 'Sigmoid' and nb.input and (nb.input[0] == a):
                    hints['swish_patterns'] += 1
        elif n.op_type == 'HardSigmoid':
            hints['hardsigmoid_count'] += 1
        elif n.op_type == 'NonMaxSuppression':
            hints['nms_count'] += 1
        elif n.op_type == 'RoiAlign':
            hints['roialign_count'] += 1
        elif n.op_type == 'Resize':
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            hints['resize_attrs'].append(attrs)
        elif n.op_type == 'Pad':
            # dynamic pads?
            if len(n.input) >= 2 and n.input[1] not in consts:
                hints['pad_dynamic_count'] += 1

    return hints


def coreml_provider_info() -> Optional[Dict[str, object]]:
    if ort is None:
        return None
    try:
        provs = ort.get_available_providers()
    except Exception:
        provs = []
    info = {
        'ort_version': getattr(ort, '__version__', 'unknown'),
        'available_providers': provs,
        'coreml_opts': {'ModelFormat': 'MLProgram', 'EnableOnSubgraphs': '1', 'MLComputeUnits': 'ALL', 'RequireStaticInputShapes': '1'},
    }
    return info


def human_bytes(n: int) -> str:
    units = ['B', 'KB', 'MB', 'GB']
    f = float(n)
    for u in units:
        if f < 1024.0:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description='Inspect an ONNX model with CoreML/ANE surgery hints')
    parser.add_argument('--model', '-m', default='pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx', help='Path to ONNX model')
    parser.add_argument('--infer-shapes', action='store_true', help='Run ONNX shape inference before reporting')
    parser.add_argument('--list-ops', action='store_true', help='List all ops and counts')
    parser.add_argument('--dump-initializers', type=int, default=15, help='Show top-N largest initializers by size (0 to disable)')
    parser.add_argument('--verbose-attrs', action='store_true', help='Show verbose op attributes for selected ops')
    parser.add_argument('--coreml-info', action='store_true', help='Print ONNX Runtime CoreML provider availability/options')
    parser.add_argument('--json-out', type=str, default=None, help='Optional path to write a JSON report with all findings')
    args = parser.parse_args()

    # Load
    try:
        m = onnx.load(args.model)
    except Exception as e:
        print('ERR loading', e)
        sys.exit(1)

    if args.infer_shapes:
        m = infer_shapes_safe(m)

    print('== Model Meta ==')
    print(' - ir_version:', m.ir_version)
    print(' - opset_import:', [(o.domain or 'ai.onnx', o.version) for o in m.opset_import])

    ins, outs = graph_inputs_outputs(m)
    print('== IO ==')
    print('inputs:')
    for n, shp, dt in ins:
        print(f' - {n}: shape={shp}, dtype={dt}, dynamic={any(s=="?" for s in shp)}')
    print('outputs:')
    for n, shp, dt in outs:
        print(f' - {n}: shape={shp}, dtype={dt}, dynamic={any(s=="?" for s in shp)}')

    # Graph stats
    ops = op_histogram(m)
    print('== Graph ==')
    print(' - node_count:', len(m.graph.node))
    print(' - unique_ops:', len(ops))
    print(' - top ops:')
    for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:25]:
        print(f'   * {k}: {v}')

    # Initializers
    total_params, total_bytes, dtypes, largest = summarize_initializers(m)
    print('== Weights ==')
    print(' - initializer_count:', len(m.graph.initializer))
    print(' - total_params:', total_params)
    print(' - total_bytes:', f"{total_bytes} ({human_bytes(total_bytes)})")
    print(' - dtype_params:', dict(dtypes))
    dump_n = args.dump_initializers
    if dump_n:
        if dump_n > 0:
            print(f' - largest_initializers top {dump_n}:')
            for name, by, desc in largest[: dump_n]:
                print(f'   * {name}: {human_bytes(by)} [{desc}]')

    # Surgery hints
    print('== CoreML/ANE Surgery Hints ==')
    hints = analyze_coreml_surgery_hints(m)
    if hints['clip_non_scalar_bounds']:
        print(f" - Clip with non-scalar bounds: {len(hints['clip_non_scalar_bounds'])} (convert min/max to 0-D scalars)")
    print(f" - Pow exponents (const histogram): {dict(hints['pow_constant_histogram'])}")
    print(f" - Pow with non-const exponent: {hints['pow_non_const_count']}")
    print(f" - Div by const: {hints['div_by_const']}, Div by var: {hints['div_by_var']}")
    if hints['slice_gather_candidates']:
        print(f" - Slice→Gather candidates: {hints['slice_gather_candidates']} (single index slices)")
    print(f" - Concat max inputs: {hints['concat_max_inputs']}")
    if hints['concat_large_nodes']:
        print(f" - Concat nodes with >=8 inputs: {len(hints['concat_large_nodes'])}")
    print(f" - Swish (x*Sigmoid(x)) patterns: {hints['swish_patterns']}")
    print(f" - HardSigmoid count: {hints['hardsigmoid_count']}")
    print(f" - NonMaxSuppression count: {hints['nms_count']}")
    print(f" - RoiAlign count: {hints['roialign_count']}")
    if hints['pad_dynamic_count']:
        print(f" - Pad with dynamic pads: {hints['pad_dynamic_count']}")
    if hints['resize_attrs'] and args.verbose_attrs:
        print(' - Resize attrs (sample up to 5):')
        for a in hints['resize_attrs'][:5]:
            print('   *', a)
    if hints['clip_non_scalar_bounds'] and args.verbose_attrs:
        print(' - Clip non-scalar bound details (up to 10):')
        for d in hints['clip_non_scalar_bounds'][:10]:
            print('   *', d)
    if hints['slice_details'] and args.verbose_attrs:
        print(' - Slice→Gather details (up to 10):')
        for d in hints['slice_details'][:10]:
            print('   *', d)

    if args.list_ops:
        print('== All Ops ==')
        for k in sorted(ops.keys()):
            print(f' - {k}: {ops[k]}')

    ort_info = None
    if args.coreml_info:
        info = coreml_provider_info()
        if info is None:
            print('== ORT/CoreML ==\n - onnxruntime not available')
        else:
            print('== ORT/CoreML ==')
            print(' - ort_version:', info['ort_version'])
            print(' - available_providers:', info['available_providers'])
            print(' - standard_coreml_opts:', info['coreml_opts'])
            ort_info = info

    # JSON report if requested
    if args.json_out:
        # convert bytes (from attribute values) to strings for JSON
        def to_jsonable(x: Any):
            if isinstance(x, bytes):
                try:
                    return x.decode('utf-8')
                except Exception:
                    return x.hex()
            if isinstance(x, dict):
                return {k: to_jsonable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [to_jsonable(v) for v in x]
            return x
        report = {
            'meta': {
                'ir_version': m.ir_version,
                'opset_import': [(o.domain or 'ai.onnx', o.version) for o in m.opset_import],
            },
            'io': {
                'inputs': [{'name': n, 'shape': shp, 'dtype': dt} for n, shp, dt in ins],
                'outputs': [{'name': n, 'shape': shp, 'dtype': dt} for n, shp, dt in outs],
            },
            'graph': {
                'node_count': len(m.graph.node),
                'unique_ops': len(ops),
                'ops': ops,
            },
            'weights': {
                'initializer_count': len(m.graph.initializer),
                'total_params': total_params,
                'total_bytes': total_bytes,
                'dtype_params': dict(dtypes),
                'largest_initializers': [
                    {'name': name, 'bytes': by, 'desc': desc} for name, by, desc in largest[: max(0, args.dump_initializers or 0)]
                ],
            },
            'hints': to_jsonable(hints),
        }
        if ort_info is not None:
            report['ort'] = ort_info
        try:
            with open(args.json_out, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"== Wrote JSON report: {args.json_out}")
        except Exception as e:
            print('ERR writing json report:', e)


if __name__ == '__main__':
    main()