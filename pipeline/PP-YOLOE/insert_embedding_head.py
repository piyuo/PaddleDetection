#!/usr/bin/env python3
"""
Insert per-detection appearance embeddings into a PP-YOLOE ONNX model.

What it does:
- Auto-pick two backbone/neck feature maps around strides s8 and s16
- Add a multi-scale (s8+s16) ROIAlign + mean + L2 head that outputs per-detection embeddings -> 'embeddings_det'

Notes:
- No projection layer is added (no learned weights). This is a solid baseline for BoT-SORT-style appearance features.
- The original model outputs are preserved.

Usage:
    python pipeline/PP-YOLOE/insert_embedding_head.py \
    --onnx_in pipeline/PP-YOLOE/backbone/ppyoloe_crn_s_36e_pphuman.onnx

Optional:
    --s8_node <tensor_name>  --s16_node <tensor_name>   (override auto-pick)
    --det_out <tensor_name>  Detection output (Nx6) to source boxes from (auto-pick if omitted)

"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import tempfile
import numpy as np


def import_onnx_modules():
    try:
        import onnx  # type: ignore
        from onnx import helper, TensorProto  # type: ignore
    except Exception:
        print('[ERROR] onnx package not installed. pip install onnx', file=sys.stderr)
        raise
    return onnx, helper, TensorProto


def get_input_hw(model) -> Tuple[int, int]:
    # Try to read H,W from first input's shape
    g = model.graph
    if g.input:
        tp = g.input[0].type.tensor_type
        if tp and tp.shape and tp.shape.dim and len(tp.shape.dim) >= 3:
            dims = tp.shape.dim
            # Expect NCHW
            try:
                H = int(dims[-2].dim_value) if dims[-2].HasField('dim_value') else 640
                W = int(dims[-1].dim_value) if dims[-1].HasField('dim_value') else 640
                return H, W
            except Exception:
                pass
    return 640, 640


def value_info_shapes(onnx):
    def extract_shape(vi) -> Optional[List[int]]:
        if not vi.type or not vi.type.tensor_type or not vi.type.tensor_type.shape:
            return None
        dims = vi.type.tensor_type.shape.dim
        s: List[int] = []
        for d in dims:
            if d.HasField('dim_value'):
                s.append(int(d.dim_value))
            else:
                return None
        return s
    return extract_shape


def list_rank4_candidates(onnx, model) -> List[Tuple[str, Optional[List[int]]]]:
    # Try shape inference for better shapes
    try:
        model_inf = onnx.shape_inference.infer_shapes(model)
    except Exception:
        model_inf = model
    g = model_inf.graph
    extract_shape = value_info_shapes(onnx)
    init_names = {init.name for init in g.initializer}
    input_names = {i.name for i in g.input}

    vi_map: Dict[str, Any] = {}
    for vi in list(g.value_info) + list(g.output) + list(g.input):
        vi_map[vi.name] = vi

    cands: List[Tuple[str, Optional[List[int]]]] = []
    for node in g.node:
        for out in node.output:
            if out in init_names or out in input_names:
                continue
            vi = vi_map.get(out)
            shp = extract_shape(vi) if vi is not None else None
            if shp is not None and len(shp) == 4 and shp[0] in (1,):
                bad = ['.w_', '.b_', 'constant', 'full', 'scale', 'bias']
                if any(sub in out for sub in bad):
                    continue
                cands.append((out, shp))
    # Dedup keep order
    seen = set()
    uniq: List[Tuple[str, Optional[List[int]]]] = []
    for n, s in cands:
        if n not in seen:
            seen.add(n)
            uniq.append((n, s))
    return uniq


def pick_stride(cands: List[Tuple[str, List[int]]], Himg: int, Wimg: int, stride: int, exclude: Optional[set] = None) -> Optional[str]:
    """
    Select a feature map close to target stride with additional heuristics:
    - Prefer spatial sizes close to Himg/stride, Wimg/stride
    - Reject very small spatial maps (Hf or Wf <= 2) which often come from global pooling
    - Prefer reasonable channel counts (64..1024)
    - Penalize nodes that look like reductions/reshapes/pooling
    - Favor nodes that look like conv/concat/add/relu typical of neck features
    """
    if not cands:
        return None
    exclude = exclude or set()
    target = (max(1, Himg // stride), max(1, Wimg // stride))

    def score(name: str, s: List[int]) -> float:
        # Base: L1 distance on spatial dims
        _, C, Hf, Wf = s
        d = abs(Hf - target[0]) + abs(Wf - target[1])

        # Hard rejects
        if name in exclude:
            return float('inf')
        if Hf <= 2 or Wf <= 2:
            return float('inf')

        # Channel sanity: prefer mid/high channels
        ch_pen = 0.0
        if C < 32:
            ch_pen += 50.0
        elif C < 64:
            ch_pen += 10.0
        elif C > 1536:
            ch_pen += 20.0

        # Name-based penalties and bonuses
        lname = name.lower()
        bad_keys = ['reduce', 'mean', 'avgpool', 'maxpool', 'pool', 'global', 'reshape', 'flatten']
        good_keys = ['conv', 'concat', 'add', 'relu', 'bn', 'res', 'stage', 'neck']
        name_pen = 0.0
        if any(k in lname for k in bad_keys):
            name_pen += 20.0
        if any(k in lname for k in good_keys):
            name_pen -= 5.0

        return float(d + ch_pen + name_pen)

    best, best_s = None, float('inf')
    for name, s in cands:
        sc = score(name, s)
        if sc < best_s:
            best, best_s = name, sc
    return best


def add_output_to_model(model, value_name: str, onnx, helper, TensorProto):
    g = model.graph
    # If already an output, do nothing
    if any(o.name == value_name for o in g.output):
        return model
    # Try to find an existing value_info to attach; otherwise create generic
    vi = None
    for v in list(g.value_info) + list(g.output) + list(g.input):
        if v.name == value_name:
            vi = v
            break
    if vi is not None:
        g.output.extend([vi])
    else:
        out_vi = helper.make_tensor_value_info(value_name, TensorProto.FLOAT, None)
        g.output.extend([out_vi])
    return model


def guess_feed_from_inputs(sess):
    feed = {}
    for i in sess.get_inputs():
        shape = []
        # Normalize NCHW-ish inputs; fallback to 1x3x640x640
        for idx, d in enumerate(i.shape):
            if d is None or d == 'None' or (isinstance(d, str) and not d.isdigit()):
                # Default guesses
                if idx == 0:
                    shape.append(1)
                elif idx == 1:
                    shape.append(3)
                else:
                    shape.append(640)
            else:
                shape.append(int(d))
        arr = np.zeros(shape, dtype=np.float32)
        feed[i.name] = arr
    return feed


def probe_rank4_candidates(onnx, helper, TensorProto, model, max_probe: int = 150) -> List[Tuple[str, List[int]]]:
    # Build a list of graph value names to consider
    g = model.graph
    init_names = {init.name for init in g.initializer}
    input_names = {i.name for i in g.input}
    seen = set()
    names: List[str] = []
    bad_subs = ['.w_', '.b_', 'constant', 'full', 'scale', 'bias']
    for node in g.node:
        for out in node.output:
            if out in seen or out in init_names or out in input_names:
                continue
            if any(sub in out for sub in bad_subs):
                continue
            seen.add(out)
            names.append(out)

    probed: List[Tuple[str, List[int]]] = []
    # We'll probe in small batches by creating temp models with a single extra output
    for name in names[:max_probe]:
        try:
            tmp_model = onnx.load_from_string(model.SerializeToString())
            tmp_model = add_output_to_model(tmp_model, name, onnx, helper, TensorProto)
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tf:
                onnx.save(tmp_model, tf.name)
                import onnxruntime as ort  # type: ignore
                sess = ort.InferenceSession(tf.name)
                feed = guess_feed_from_inputs(sess)
                outs = sess.run(None, feed)
                out_names = [o.name for o in sess.get_outputs()]
                m = {out_names[i]: outs[i] for i in range(len(out_names))}
                if name in m:
                    arr = m[name]
                    if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] in (1,):
                        probed.append((name, list(arr.shape)))
        except Exception:
            continue
    return probed


# (Removed image-level embedding head to simplify the script; we only produce per-detection embeddings_det.)


def find_detection_output(onnx, model) -> Optional[str]:
    # Try static shape inference: look for 2D output with second dim == 6
    try:
        model_inf = onnx.shape_inference.infer_shapes(model)
    except Exception:
        model_inf = model
    g = model_inf.graph
    vi_map = {vi.name: vi for vi in list(g.value_info) + list(g.output)}
    ext = value_info_shapes(onnx)
    for o in g.output:
        shp = ext(vi_map.get(o.name)) if o.name in vi_map else None
        if shp is not None and len(shp) == 2 and shp[1] == 6:
            return o.name
    # Fallback: runtime probe
    try:
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(model.SerializeToString())
        # Build dummy feed
        feed = {}
        for i in sess.get_inputs():
            shape = []
            for idx, d in enumerate(i.shape):
                if d is None or d == 'None' or (isinstance(d, str) and not str(d).isdigit()):
                    shape.append([1, 3, 640, 640][idx] if idx < 4 else 1)
                else:
                    shape.append(int(d))
            feed[i.name] = np.zeros(shape, dtype=np.float32)
        outs = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()]
        for n, a in zip(names, outs):
            if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 6:
                return n
    except Exception:
        pass
    return None


def add_roi_head(onnx, helper, TensorProto, model, feat_name: str, det_out_name: str, stride: int, out_name: str = 'embeddings_det', pooled_hw: int = 14):
    g = model.graph

    def make_name(base):
        idx = 0
        while any(n.name == f'{base}_{idx}' for n in g.node):
            idx += 1
        return f'{base}_{idx}'

    # Slice boxes from det_out (Nx6): columns [2:6] -> (Nx4)
    # ONNX Slice with dynamic rows and fixed columns
    starts = helper.make_tensor(name='roi_slice_starts', data_type=TensorProto.INT64, dims=[2], vals=[0, 2])
    ends = helper.make_tensor(name='roi_slice_ends', data_type=TensorProto.INT64, dims=[2], vals=[9223372036854775807, 6])
    axes = helper.make_tensor(name='roi_slice_axes', data_type=TensorProto.INT64, dims=[2], vals=[0, 1])
    steps = helper.make_tensor(name='roi_slice_steps', data_type=TensorProto.INT64, dims=[2], vals=[1, 1])
    init_names = {init.name for init in g.initializer}
    for t in (starts, ends, axes, steps):
        if t.name not in init_names:
            g.initializer.extend([t])
    boxes = det_out_name + '_boxes'
    g.node.extend([
        helper.make_node('Slice', inputs=[det_out_name, 'roi_slice_starts', 'roi_slice_ends', 'roi_slice_axes', 'roi_slice_steps'], outputs=[boxes], name=make_name('Slice'))
    ])

    # Attempt to rescale boxes from original image coords -> network input coords using scale_factor
    # scale_factor is typically a model input of shape [1,2] = [scale_y, scale_x]
    scale_inp = None
    for inp in g.input:
        n = inp.name.lower()
        if 'scale' in n:  # matches 'scale_factor'
            scale_inp = inp.name
            break
    if scale_inp is not None:
        # reshape to (2,), then gather sx, sy and build [sx, sy, sx, sy]
        shape2 = 'roi_scale_shape2'
        if not any(init.name == shape2 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=shape2, data_type=TensorProto.INT64, dims=[1], vals=[2])])
        scale_flat = boxes + '_scale_flat'
        g.node.extend([helper.make_node('Reshape', inputs=[scale_inp, shape2], outputs=[scale_flat], name=make_name('Reshape'))])
        # indices
        idx0 = 'const_idx0'
        if not any(init.name == idx0 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=idx0, data_type=TensorProto.INT64, dims=[1], vals=[0])])
        idx1 = 'const_idx1'
        if not any(init.name == idx1 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=idx1, data_type=TensorProto.INT64, dims=[1], vals=[1])])
        sy = boxes + '_sy'
        sx = boxes + '_sx'
        g.node.extend([helper.make_node('Gather', inputs=[scale_flat, idx0], outputs=[sy], name=make_name('Gather'), axis=0)])
        g.node.extend([helper.make_node('Gather', inputs=[scale_flat, idx1], outputs=[sx], name=make_name('Gather'), axis=0)])
        scale4 = boxes + '_scale4'
        g.node.extend([helper.make_node('Concat', inputs=[sx, sy, sx, sy], outputs=[scale4], name=make_name('Concat'), axis=0)])
        boxes_scaled = boxes + '_scaled'
        g.node.extend([helper.make_node('Mul', inputs=[boxes, scale4], outputs=[boxes_scaled], name=make_name('Mul'))])
        boxes = boxes_scaled

    # Build batch_indices (num_rois,) int64 zeros dynamically
    shape_boxes = boxes + '_shape'
    g.node.extend([helper.make_node('Shape', inputs=[boxes], outputs=[shape_boxes], name=make_name('Shape'))])
    # gather dim0
    gather_idx0 = 'const_idx0'
    if not any(init.name == gather_idx0 for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=gather_idx0, data_type=TensorProto.INT64, dims=[1], vals=[0])])
    num_rois = boxes + '_n'
    g.node.extend([helper.make_node('Gather', inputs=[shape_boxes, gather_idx0], outputs=[num_rois], name=make_name('Gather'), axis=0)])
    # Ensure 1D shape
    batch_shape = boxes + '_batch_shape'
    g.node.extend([helper.make_node('Cast', inputs=[num_rois], outputs=[batch_shape], name=make_name('Cast'), to=TensorProto.INT64)])
    # Create zero scalar int64
    zero_i64 = 'zero_i64'
    if not any(init.name == zero_i64 for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=zero_i64, data_type=TensorProto.INT64, dims=[], vals=[0])])
    # Expand to [num_rois]
    batch_idx = boxes + '_batch_idx'
    g.node.extend([helper.make_node('Expand', inputs=[zero_i64, batch_shape], outputs=[batch_idx], name=make_name('Expand'))])

    # RoiAlign on selected feature map
    roi_out = feat_name + '_roi'
    g.node.extend([
        helper.make_node('RoiAlign', inputs=[feat_name, boxes, batch_idx], outputs=[roi_out], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=0, spatial_scale=1.0/float(stride))
    ])

    # Instance normalization per ROI (per-channel, across HxW) to increase contrast without learned params
    mean_hw = roi_out + '_meanhw'
    g.node.extend([
        helper.make_node('ReduceMean', inputs=[roi_out], outputs=[mean_hw], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])
    ])
    x_center = roi_out + '_center'
    g.node.extend([
        helper.make_node('Sub', inputs=[roi_out, mean_hw], outputs=[x_center], name=make_name('Sub'))
    ])
    sq_center = roi_out + '_sqcenter'
    g.node.extend([
        helper.make_node('Mul', inputs=[x_center, x_center], outputs=[sq_center], name=make_name('Square'))
    ])
    var_hw = roi_out + '_varhw'
    g.node.extend([
        helper.make_node('ReduceMean', inputs=[sq_center], outputs=[var_hw], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])
    ])
    eps_inst = 'eps_inst_norm'
    if not any(init.name == eps_inst for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=eps_inst, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-5])])
    var_eps = roi_out + '_vareps'
    g.node.extend([
        helper.make_node('Add', inputs=[var_hw, eps_inst], outputs=[var_eps], name=make_name('Add'))
    ])
    std_hw = roi_out + '_stdhw'
    g.node.extend([
        helper.make_node('Sqrt', inputs=[var_eps], outputs=[std_hw], name=make_name('Sqrt'))
    ])
    x_norm = roi_out + '_normed'
    g.node.extend([
        helper.make_node('Div', inputs=[x_center, std_hw], outputs=[x_norm], name=make_name('Div'))
    ])

    # Reduce over H and W of normalized features -> use avg+max then average them elementwise
    # Use unique suffixes to avoid clashes with part-based pooling
    avg = roi_out + '_gavg'
    mx = roi_out + '_gmax'
    g.node.extend([
        helper.make_node('ReduceMean', inputs=[x_norm], outputs=[avg], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])
    ])
    g.node.extend([
        helper.make_node('ReduceMax', inputs=[x_norm], outputs=[mx], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])
    ])
    pooled = roi_out + '_avgmax'
    half = 'const_half'
    if not any(init.name == half for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=half, data_type=TensorProto.FLOAT, dims=[1], vals=[0.5])])
    add_am = pooled + '_add'
    g.node.extend([helper.make_node('Add', inputs=[avg, mx], outputs=[add_am], name=make_name('Add'))])
    g.node.extend([helper.make_node('Mul', inputs=[add_am, half], outputs=[pooled], name=make_name('Mul'))])

    # L2 normalize along channel dim (=1)
    sq = pooled + '_sq'
    g.node.extend([helper.make_node('Mul', inputs=[pooled, pooled], outputs=[sq], name=make_name('Square'))])
    rs = pooled + '_rs'
    axes_ch = 'roi_axes_ch'
    if not any(init.name == axes_ch for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=axes_ch, data_type=TensorProto.INT64, dims=[1], vals=[1])])
    g.node.extend([helper.make_node('ReduceSum', inputs=[sq, axes_ch], outputs=[rs], name=make_name('ReduceSum'), keepdims=1)])
    eps = 'eps_roi_head'
    if not any(init.name == eps for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=eps, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
    add = pooled + '_den'
    g.node.extend([helper.make_node('Add', inputs=[rs, eps], outputs=[add], name=make_name('Add'))])
    sqrt = pooled + '_norm'
    g.node.extend([helper.make_node('Sqrt', inputs=[add], outputs=[sqrt], name=make_name('Sqrt'))])
    emb = out_name
    g.node.extend([helper.make_node('Div', inputs=[pooled, sqrt], outputs=[emb], name=make_name('Div'))])

    g.output.extend([helper.make_tensor_value_info(emb, TensorProto.FLOAT, None)])
    return model


def add_roi_head_multi_scale(
    onnx,
    helper,
    TensorProto,
    model,
    feat8: str,
    feat16: str,
    det_out_name: str,
    stride8: int = 8,
    stride16: int = 16,
    out_name: str = 'embeddings_det',
    pooled_hw: int = 14,
    gp_w: float = 0.3,
    pp_w: float = 0.7,
    color_gain_val: float = 4.0,
    pp_k: int = 7,
    pp_stripe_h: int = 2,
    pp_vertical_k: int = 7,
    pp_vertical_stripe_w: int = 2,
    use_inst_norm: bool = True,
):
    g = model.graph

    def make_name(base):
        idx = 0
        while any(n.name == f'{base}_{idx}' for n in g.node):
            idx += 1
        return f'{base}_{idx}'

    # Helper: part-based horizontal stripe pooling over H dimension
    # Splits H into K stripes and averages per stripe (ReduceMax), then averages stripes -> (N, C)
    # We use K=7 when pooled_hw=14 (each stripe height=2)
    def add_part_pool(base_input: str, base_name: str, K: int = 7, stripe_h: int = 2) -> str:
        # Common initializers for Slice along H axis
        axes_h = 'pp_axes_h'
        steps_h = 'pp_steps_h'
        init_names = {init.name for init in g.initializer}
        if axes_h not in init_names:
            g.initializer.extend([helper.make_tensor(name=axes_h, data_type=TensorProto.INT64, dims=[1], vals=[2])])
        if steps_h not in init_names:
            g.initializer.extend([helper.make_tensor(name=steps_h, data_type=TensorProto.INT64, dims=[1], vals=[1])])

        sum_name = ''
        stripe_outputs: List[str] = []
        # Ensure we never slice beyond pooled height to avoid empty tensors
        eff_K = max(1, min(int(K), int(pooled_hw // max(1, stripe_h))))
        for s in range(eff_K):
            start_val = int(s * stripe_h)
            end_val = int(min((s + 1) * stripe_h, pooled_hw))
            starts_name = f'{base_name}_pp_starts_{s}'
            ends_name = f'{base_name}_pp_ends_{s}'
            if starts_name not in init_names:
                g.initializer.extend([helper.make_tensor(name=starts_name, data_type=TensorProto.INT64, dims=[1], vals=[start_val])])
            if ends_name not in init_names:
                g.initializer.extend([helper.make_tensor(name=ends_name, data_type=TensorProto.INT64, dims=[1], vals=[end_val])])
            sl_out = f'{base_name}_stripe_{s}'
            g.node.extend([
                helper.make_node('Slice', inputs=[base_input, starts_name, ends_name, axes_h, steps_h], outputs=[sl_out], name=make_name('Slice'))
            ])
            # Pool each stripe over H and W -> (N, C)
            pooled_s = f'{sl_out}_p'
            g.node.extend([
                helper.make_node('ReduceMax', inputs=[sl_out], outputs=[pooled_s], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])
            ])
            stripe_outputs.append(pooled_s)

        # Sum stripes
        agg = f'{base_name}_sum'
        if len(stripe_outputs) == 1:
            agg = stripe_outputs[0]
        else:
            cur = stripe_outputs[0]
            for nxt in stripe_outputs[1:]:
                add_name = make_name('Add')
                out_name_add = f'{base_name}_add_{len(g.node)}'
                g.node.extend([helper.make_node('Add', inputs=[cur, nxt], outputs=[out_name_add], name=add_name)])
                cur = out_name_add
            agg = cur

        # Average across stripes by multiplying with 1/K
        invK_name = f'{base_name}_invK'
        if invK_name not in init_names:
            g.initializer.extend([helper.make_tensor(name=invK_name, data_type=TensorProto.FLOAT, dims=[1], vals=[1.0 / float(eff_K)])])
        avg_name = f'{base_name}_pp_avg'
        g.node.extend([helper.make_node('Mul', inputs=[agg, invK_name], outputs=[avg_name], name=make_name('Mul'))])
        return avg_name

    # Helper: part-based vertical stripe pooling over W dimension (axis=3)
    def add_part_pool_vertical(base_input: str, base_name: str, K: int = 7, stripe_w: int = 2) -> str:
        axes_w = 'pp_axes_w'
        steps_w = 'pp_steps_w'
        init_names = {init.name for init in g.initializer}
        if axes_w not in init_names:
            g.initializer.extend([helper.make_tensor(name=axes_w, data_type=TensorProto.INT64, dims=[1], vals=[3])])
        if steps_w not in init_names:
            g.initializer.extend([helper.make_tensor(name=steps_w, data_type=TensorProto.INT64, dims=[1], vals=[1])])

        stripe_outputs: List[str] = []
        # Ensure we never slice beyond pooled width to avoid empty tensors
        eff_K = max(1, min(int(K), int(pooled_hw // max(1, stripe_w))))
        for s in range(eff_K):
            start_val = int(s * stripe_w)
            end_val = int(min((s + 1) * stripe_w, pooled_hw))
            starts_name = f'{base_name}_ppv_starts_{s}'
            ends_name = f'{base_name}_ppv_ends_{s}'
            if starts_name not in init_names:
                g.initializer.extend([helper.make_tensor(name=starts_name, data_type=TensorProto.INT64, dims=[1], vals=[start_val])])
            if ends_name not in init_names:
                g.initializer.extend([helper.make_tensor(name=ends_name, data_type=TensorProto.INT64, dims=[1], vals=[end_val])])
            sl_out = f'{base_name}_vstripe_{s}'
            g.node.extend([
                helper.make_node('Slice', inputs=[base_input, starts_name, ends_name, axes_w, steps_w], outputs=[sl_out], name=make_name('Slice'))
            ])
            pooled_s = f'{sl_out}_p'
            g.node.extend([
                helper.make_node('ReduceMax', inputs=[sl_out], outputs=[pooled_s], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])
            ])
            stripe_outputs.append(pooled_s)

        agg = f'{base_name}_vsum'
        if len(stripe_outputs) == 1:
            agg = stripe_outputs[0]
        else:
            cur = stripe_outputs[0]
            for nxt in stripe_outputs[1:]:
                add_name = make_name('Add')
                out_name_add = f'{base_name}_vadd_{len(g.node)}'
                g.node.extend([helper.make_node('Add', inputs=[cur, nxt], outputs=[out_name_add], name=add_name)])
                cur = out_name_add
            agg = cur

        invK_name = f'{base_name}_vinvK'
        if invK_name not in init_names:
            g.initializer.extend([helper.make_tensor(name=invK_name, data_type=TensorProto.FLOAT, dims=[1], vals=[1.0 / float(eff_K)])])
        avg_name = f'{base_name}_ppv_avg'
        g.node.extend([helper.make_node('Mul', inputs=[agg, invK_name], outputs=[avg_name], name=make_name('Mul'))])
        return avg_name

    # Slice boxes [2:6] from det_out (Nx6)
    starts = helper.make_tensor(name='roi_ms_slice_starts', data_type=TensorProto.INT64, dims=[2], vals=[0, 2])
    ends = helper.make_tensor(name='roi_ms_slice_ends', data_type=TensorProto.INT64, dims=[2], vals=[9223372036854775807, 6])
    axes = helper.make_tensor(name='roi_ms_slice_axes', data_type=TensorProto.INT64, dims=[2], vals=[0, 1])
    steps = helper.make_tensor(name='roi_ms_slice_steps', data_type=TensorProto.INT64, dims=[2], vals=[1, 1])
    init_names = {init.name for init in g.initializer}
    for t in (starts, ends, axes, steps):
        if t.name not in init_names:
            g.initializer.extend([t])
    boxes = det_out_name + '_boxes_ms'
    g.node.extend([
        helper.make_node('Slice', inputs=[det_out_name, 'roi_ms_slice_starts', 'roi_ms_slice_ends', 'roi_ms_slice_axes', 'roi_ms_slice_steps'], outputs=[boxes], name=make_name('Slice'))
    ])

    # Rescale boxes from original image coords -> network input coords using scale_factor if available
    scale_inp = None
    for inp in g.input:
        n = inp.name.lower()
        if 'scale' in n:
            scale_inp = inp.name
            break
    if scale_inp is not None:
        shape2 = 'roi_ms_scale_shape2'
        if not any(init.name == shape2 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=shape2, data_type=TensorProto.INT64, dims=[1], vals=[2])])
        scale_flat = boxes + '_scale_flat'
        g.node.extend([helper.make_node('Reshape', inputs=[scale_inp, shape2], outputs=[scale_flat], name=make_name('Reshape'))])
        idx0 = 'const_idx0'
        if not any(init.name == idx0 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=idx0, data_type=TensorProto.INT64, dims=[1], vals=[0])])
        idx1 = 'const_idx1'
        if not any(init.name == idx1 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=idx1, data_type=TensorProto.INT64, dims=[1], vals=[1])])
        sy = boxes + '_sy'
        sx = boxes + '_sx'
        g.node.extend([helper.make_node('Gather', inputs=[scale_flat, idx0], outputs=[sy], name=make_name('Gather'), axis=0)])
        g.node.extend([helper.make_node('Gather', inputs=[scale_flat, idx1], outputs=[sx], name=make_name('Gather'), axis=0)])
        scale4 = boxes + '_scale4'
        g.node.extend([helper.make_node('Concat', inputs=[sx, sy, sx, sy], outputs=[scale4], name=make_name('Concat'), axis=0)])
        boxes_scaled = boxes + '_scaled'
        g.node.extend([helper.make_node('Mul', inputs=[boxes, scale4], outputs=[boxes_scaled], name=make_name('Mul'))])
        boxes = boxes_scaled

    # num_rois and batch_idx (zeros)
    shape_boxes = boxes + '_shape'
    g.node.extend([helper.make_node('Shape', inputs=[boxes], outputs=[shape_boxes], name=make_name('Shape'))])
    gather_idx0 = 'const_idx0'
    if not any(init.name == gather_idx0 for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=gather_idx0, data_type=TensorProto.INT64, dims=[1], vals=[0])])
    num_rois = boxes + '_n'
    g.node.extend([helper.make_node('Gather', inputs=[shape_boxes, gather_idx0], outputs=[num_rois], name=make_name('Gather'), axis=0)])
    batch_shape = boxes + '_batch_shape'
    g.node.extend([helper.make_node('Cast', inputs=[num_rois], outputs=[batch_shape], name=make_name('Cast'), to=TensorProto.INT64)])
    zero_i64 = 'zero_i64'
    if not any(init.name == zero_i64 for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=zero_i64, data_type=TensorProto.INT64, dims=[], vals=[0])])
    batch_idx = boxes + '_batch_idx'
    g.node.extend([helper.make_node('Expand', inputs=[zero_i64, batch_shape], outputs=[batch_idx], name=make_name('Expand'))])

    # ROIAlign on s8 and s16 (ensure unique value names even if features coincide)
    roi8 = feat8 + '_roi_ms8'
    g.node.extend([helper.make_node('RoiAlign', inputs=[feat8, boxes, batch_idx], outputs=[roi8], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=0, spatial_scale=1.0/float(stride8))])
    roi16 = feat16 + '_roi_ms16'
    g.node.extend([helper.make_node('RoiAlign', inputs=[feat16, boxes, batch_idx], outputs=[roi16], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=0, spatial_scale=1.0/float(stride16))])

    # Instance normalization per ROI for s8 (optional)
    if use_inst_norm:
        mean8 = roi8 + '_meanhw'
        g.node.extend([helper.make_node('ReduceMean', inputs=[roi8], outputs=[mean8], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])])
        cen8 = roi8 + '_center'
        g.node.extend([helper.make_node('Sub', inputs=[roi8, mean8], outputs=[cen8], name=make_name('Sub'))])
        sq8 = roi8 + '_sqcenter'
        g.node.extend([helper.make_node('Mul', inputs=[cen8, cen8], outputs=[sq8], name=make_name('Square'))])
        var8 = roi8 + '_varhw'
        g.node.extend([helper.make_node('ReduceMean', inputs=[sq8], outputs=[var8], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])])
        eps_inst = 'eps_inst_norm_ms'
        if not any(init.name == eps_inst for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=eps_inst, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-5])])
        vareps8 = roi8 + '_vareps'
        g.node.extend([helper.make_node('Add', inputs=[var8, eps_inst], outputs=[vareps8], name=make_name('Add'))])
        std8 = roi8 + '_stdhw'
        g.node.extend([helper.make_node('Sqrt', inputs=[vareps8], outputs=[std8], name=make_name('Sqrt'))])
        norm8 = roi8 + '_normed'
        g.node.extend([helper.make_node('Div', inputs=[cen8, std8], outputs=[norm8], name=make_name('Div'))])
        src8 = norm8
    else:
        src8 = roi8

    # Power-law (signed sqrt) normalization to reduce burstiness: y = sign(x)*sqrt(|x| + eps)
    eps_pl = 'eps_powerlaw'
    if not any(init.name == eps_pl for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=eps_pl, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
    abs8 = src8 + '_abs'
    g.node.extend([helper.make_node('Abs', inputs=[src8], outputs=[abs8], name=make_name('Abs'))])
    abseps8 = src8 + '_abseps'
    g.node.extend([helper.make_node('Add', inputs=[abs8, eps_pl], outputs=[abseps8], name=make_name('Add'))])
    sqrt8 = src8 + '_sqrtabs'
    g.node.extend([helper.make_node('Sqrt', inputs=[abseps8], outputs=[sqrt8], name=make_name('Sqrt'))])
    sign8 = src8 + '_sign'
    g.node.extend([helper.make_node('Sign', inputs=[src8], outputs=[sign8], name=make_name('Sign'))])
    pl8 = src8 + '_powerlaw'
    g.node.extend([helper.make_node('Mul', inputs=[sign8, sqrt8], outputs=[pl8], name=make_name('Mul'))])

    # Reduce over H and W: avg+max then average to keep dim unchanged (global pooling)
    # Use unique name suffixes to avoid colliding with part-pooling outputs
    avg8 = roi8 + '_gavg'
    max8 = roi8 + '_gmax'
    g.node.extend([helper.make_node('ReduceMean', inputs=[pl8], outputs=[avg8], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
    g.node.extend([helper.make_node('ReduceMax', inputs=[pl8], outputs=[max8], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])])
    pooled8_add = roi8 + '_avgmax_add'
    pooled8 = roi8 + '_avgmax'
    half = 'const_half'
    if not any(init.name == half for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=half, data_type=TensorProto.FLOAT, dims=[1], vals=[0.5])])
    g.node.extend([helper.make_node('Add', inputs=[avg8, max8], outputs=[pooled8_add], name=make_name('Add'))])
    g.node.extend([helper.make_node('Mul', inputs=[pooled8_add, half], outputs=[pooled8], name=make_name('Mul'))])

    # Part-based pooling over H stripes (keeps (N, C)) and blend with global pooled
    # Increase contribution of part-based pooling to improve discriminativeness
    # If pp_w <= 0, skip building part-pooling graph and use zeros to avoid NaN*0 propagation
    if pp_w is not None and float(pp_w) <= 0.0:
        # Create a zeros-like tensor by subtracting the tensor from itself
        pb8 = roi8 + '_pp_zero'
        g.node.extend([
            helper.make_node('Sub', inputs=[pooled8, pooled8], outputs=[pb8], name=make_name('Sub'))
        ])
    else:
        pb8_h = add_part_pool(pl8, roi8, K=pp_k, stripe_h=pp_stripe_h)
        # optional vertical pooling
        pb8_v: Optional[str] = None
        if pp_vertical_k and pp_vertical_k > 0:
            pb8_v = add_part_pool_vertical(pl8, roi8, K=pp_vertical_k, stripe_w=pp_vertical_stripe_w)
        # combine parts (avg if both present)
        pb8 = pb8_h
        if pb8_v is not None:
            pb8_add = roi8 + '_pp_hv_add'
            pb8 = roi8 + '_pp_hv_avg'
            g.node.extend([
                helper.make_node('Add', inputs=[pb8_h, pb8_v], outputs=[pb8_add], name=make_name('Add')),
                helper.make_node('Mul', inputs=[pb8_add, 'const_half'], outputs=[pb8], name=make_name('Mul')),
            ])
    pooled8_gp_w = 'const_w_gp'
    pooled8_pp_w = 'const_w_pp'
    init_names = {init.name for init in g.initializer}
    if pooled8_gp_w not in init_names:
        g.initializer.extend([helper.make_tensor(name=pooled8_gp_w, data_type=TensorProto.FLOAT, dims=[1], vals=[float(gp_w)])])
    if pooled8_pp_w not in init_names:
        g.initializer.extend([helper.make_tensor(name=pooled8_pp_w, data_type=TensorProto.FLOAT, dims=[1], vals=[float(pp_w)])])
    pooled8_gp = roi8 + '_gpw'
    pooled8_pp = roi8 + '_ppw'
    g.node.extend([helper.make_node('Mul', inputs=[pooled8, pooled8_gp_w], outputs=[pooled8_gp], name=make_name('Mul'))])
    g.node.extend([helper.make_node('Mul', inputs=[pb8, pooled8_pp_w], outputs=[pooled8_pp], name=make_name('Mul'))])
    pooled8_mix = roi8 + '_mix'
    g.node.extend([helper.make_node('Add', inputs=[pooled8_gp, pooled8_pp], outputs=[pooled8_mix], name=make_name('Add'))])

    # Instance normalization per ROI for s16 (optional)
    if use_inst_norm:
        mean16 = roi16 + '_meanhw'
        g.node.extend([helper.make_node('ReduceMean', inputs=[roi16], outputs=[mean16], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])])
        cen16 = roi16 + '_center'
        g.node.extend([helper.make_node('Sub', inputs=[roi16, mean16], outputs=[cen16], name=make_name('Sub'))])
        sq16 = roi16 + '_sqcenter'
        g.node.extend([helper.make_node('Mul', inputs=[cen16, cen16], outputs=[sq16], name=make_name('Square'))])
        var16 = roi16 + '_varhw'
        g.node.extend([helper.make_node('ReduceMean', inputs=[sq16], outputs=[var16], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])])
        eps_inst2 = 'eps_inst_norm_ms'
        if not any(init.name == eps_inst2 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=eps_inst2, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-5])])
        vareps16 = roi16 + '_vareps'
        g.node.extend([helper.make_node('Add', inputs=[var16, eps_inst2], outputs=[vareps16], name=make_name('Add'))])
        std16 = roi16 + '_stdhw'
        g.node.extend([helper.make_node('Sqrt', inputs=[vareps16], outputs=[std16], name=make_name('Sqrt'))])
        norm16 = roi16 + '_normed'
        g.node.extend([helper.make_node('Div', inputs=[cen16, std16], outputs=[norm16], name=make_name('Div'))])
        src16 = norm16
    else:
        src16 = roi16

    # Power-law for s16
    abs16 = src16 + '_abs'
    g.node.extend([helper.make_node('Abs', inputs=[src16], outputs=[abs16], name=make_name('Abs'))])
    abseps16 = src16 + '_abseps'
    g.node.extend([helper.make_node('Add', inputs=[abs16, eps_pl], outputs=[abseps16], name=make_name('Add'))])
    sqrt16 = src16 + '_sqrtabs'
    g.node.extend([helper.make_node('Sqrt', inputs=[abseps16], outputs=[sqrt16], name=make_name('Sqrt'))])
    sign16 = src16 + '_sign'
    g.node.extend([helper.make_node('Sign', inputs=[src16], outputs=[sign16], name=make_name('Sign'))])
    pl16 = src16 + '_powerlaw'
    g.node.extend([helper.make_node('Mul', inputs=[sign16, sqrt16], outputs=[pl16], name=make_name('Mul'))])

    # Use unique name suffixes for s16 as well
    avg16 = roi16 + '_gavg'
    max16 = roi16 + '_gmax'
    g.node.extend([helper.make_node('ReduceMean', inputs=[pl16], outputs=[avg16], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
    g.node.extend([helper.make_node('ReduceMax', inputs=[pl16], outputs=[max16], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])])
    pooled16_add = roi16 + '_avgmax_add'
    pooled16 = roi16 + '_avgmax'
    g.node.extend([helper.make_node('Add', inputs=[avg16, max16], outputs=[pooled16_add], name=make_name('Add'))])
    g.node.extend([helper.make_node('Mul', inputs=[pooled16_add, half], outputs=[pooled16], name=make_name('Mul'))])

    # Part-based pooling for s16 and blend (same weights as s8)
    if pp_w is not None and float(pp_w) <= 0.0:
        pb16 = roi16 + '_pp_zero'
        g.node.extend([
            helper.make_node('Sub', inputs=[pooled16, pooled16], outputs=[pb16], name=make_name('Sub'))
        ])
    else:
        pb16_h = add_part_pool(pl16, roi16, K=pp_k, stripe_h=pp_stripe_h)
        pb16_v: Optional[str] = None
        if pp_vertical_k and pp_vertical_k > 0:
            pb16_v = add_part_pool_vertical(pl16, roi16, K=pp_vertical_k, stripe_w=pp_vertical_stripe_w)
        pb16 = pb16_h
        if pb16_v is not None:
            pb16_add = roi16 + '_pp_hv_add'
            pb16 = roi16 + '_pp_hv_avg'
            g.node.extend([
                helper.make_node('Add', inputs=[pb16_h, pb16_v], outputs=[pb16_add], name=make_name('Add')),
                helper.make_node('Mul', inputs=[pb16_add, 'const_half'], outputs=[pb16], name=make_name('Mul')),
            ])
    pooled16_gp = roi16 + '_gpw'
    pooled16_pp = roi16 + '_ppw'
    # Reuse the same initializers for weights
    g.node.extend([helper.make_node('Mul', inputs=[pooled16, pooled8_gp_w], outputs=[pooled16_gp], name=make_name('Mul'))])
    g.node.extend([helper.make_node('Mul', inputs=[pb16, pooled8_pp_w], outputs=[pooled16_pp], name=make_name('Mul'))])
    pooled16_mix = roi16 + '_mix'
    g.node.extend([helper.make_node('Add', inputs=[pooled16_gp, pooled16_pp], outputs=[pooled16_mix], name=make_name('Add'))])

    # Optional: add a lightweight color-statistics branch using ROIAlign on the input image
    # Only enable if color_gain_val > 0 to avoid NaN*0 propagation when disabled
    color_feat_scaled: Optional[str] = None
    if color_gain_val is not None and float(color_gain_val) > 0.0:
        try:
            # Heuristic to pick the image input (first 4D input, typically NCHW with C=3)
            img_in = None
            for inp in g.input:
                tt = inp.type.tensor_type if inp.type and inp.type.tensor_type else None
                if tt and tt.shape and len(tt.shape.dim) >= 4:
                    # Prefer channel==3 when available
                    cdim = tt.shape.dim[1]
                    if not hasattr(cdim, 'dim_value') or cdim.dim_value in (0, 3):
                        img_in = inp.name
                        break
            if img_in is None and g.input:
                img_in = g.input[0].name
            if img_in is not None:
                roi_img = img_in + '_roi_ms_input'
                g.node.extend([helper.make_node('RoiAlign', inputs=[img_in, boxes, batch_idx], outputs=[roi_img], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=0, spatial_scale=1.0)])
                # Mean over H,W per channel -> (N, C)
                img_mean = roi_img + '_gmean'
                g.node.extend([helper.make_node('ReduceMean', inputs=[roi_img], outputs=[img_mean], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
                # Std over H,W per channel
                mean_hw_keep = roi_img + '_mean_keep'
                g.node.extend([helper.make_node('ReduceMean', inputs=[roi_img], outputs=[mean_hw_keep], name=make_name('ReduceMean'), keepdims=1, axes=[2, 3])])
                cen = roi_img + '_center'
                g.node.extend([helper.make_node('Sub', inputs=[roi_img, mean_hw_keep], outputs=[cen], name=make_name('Sub'))])
                sq = cen + '_sq'
                g.node.extend([helper.make_node('Mul', inputs=[cen, cen], outputs=[sq], name=make_name('Square'))])
                var = roi_img + '_var'
                g.node.extend([helper.make_node('ReduceMean', inputs=[sq], outputs=[var], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
                eps_c = 'eps_color_branch'
                if not any(init.name == eps_c for init in g.initializer):
                    g.initializer.extend([helper.make_tensor(name=eps_c, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
                var_eps = var + '_eps'
                g.node.extend([helper.make_node('Add', inputs=[var, eps_c], outputs=[var_eps], name=make_name('Add'))])
                std = roi_img + '_std'
                g.node.extend([helper.make_node('Sqrt', inputs=[var_eps], outputs=[std], name=make_name('Sqrt'))])
                color_feat = 'emb_color_feat'
                g.node.extend([helper.make_node('Concat', inputs=[img_mean, std], outputs=[color_feat], name=make_name('Concat'), axis=1)])
                # Scale color features to have meaningful impact despite low dimensionality
                color_gain = 'const_color_gain'
                if not any(init.name == color_gain for init in g.initializer):
                    # Empirical gain; default can be tuned via CLI
                    g.initializer.extend([helper.make_tensor(name=color_gain, data_type=TensorProto.FLOAT, dims=[1], vals=[float(color_gain_val)])])
                color_feat_scaled = 'emb_color_feat_scaled'
                g.node.extend([helper.make_node('Mul', inputs=[color_feat, color_gain], outputs=[color_feat_scaled], name=make_name('Mul'))])
        except Exception:
            color_feat_scaled = None

    # Concat channel-wise (s8, s16, optional color stats)
    concat = 'emb_ms_concat'
    concat_inputs = [pooled8_mix, pooled16_mix]
    if color_feat_scaled is not None:
        concat_inputs.append(color_feat_scaled)
    g.node.extend([helper.make_node('Concat', inputs=concat_inputs, outputs=[concat], name=make_name('Concat'), axis=1)])

    # Sanitize any NaNs that may arise upstream (e.g., disabled branches multiplied by 0)
    # Create zeros-like and replace NaNs with zeros before L2 normalization
    concat_zeros = concat + '_zeros'
    concat_mask = concat + '_isnan'
    concat_clean = concat + '_clean'
    g.node.extend([
        helper.make_node('Sub', inputs=[concat, concat], outputs=[concat_zeros], name=make_name('Sub')),
        helper.make_node('IsNaN', inputs=[concat], outputs=[concat_mask], name=make_name('IsNaN')),
        helper.make_node('Where', inputs=[concat_mask, concat_zeros, concat], outputs=[concat_clean], name=make_name('Where')),
    ])

    # L2 normalize across channel (dim=1)
    sq = concat + '_sq'
    g.node.extend([helper.make_node('Mul', inputs=[concat_clean, concat_clean], outputs=[sq], name=make_name('Square'))])
    rs = concat + '_rs'
    axes_ch = 'roi_axes_ch_ms'
    if not any(init.name == axes_ch for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=axes_ch, data_type=TensorProto.INT64, dims=[1], vals=[1])])
    g.node.extend([helper.make_node('ReduceSum', inputs=[sq, axes_ch], outputs=[rs], name=make_name('ReduceSum'), keepdims=1)])
    eps = 'eps_roi_head_ms'
    if not any(init.name == eps for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=eps, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
    add = concat + '_den'
    g.node.extend([helper.make_node('Add', inputs=[rs, eps], outputs=[add], name=make_name('Add'))])
    sqrt = concat + '_norm'
    g.node.extend([helper.make_node('Sqrt', inputs=[add], outputs=[sqrt], name=make_name('Sqrt'))])
    emb = out_name
    g.node.extend([helper.make_node('Div', inputs=[concat_clean, sqrt], outputs=[emb], name=make_name('Div'))])

    g.output.extend([helper.make_tensor_value_info(emb, TensorProto.FLOAT, None)])
    return model


def main():
    onnx, helper, TensorProto = import_onnx_modules()

    ap = argparse.ArgumentParser(description='Add per-detection ROI embeddings (multi-scale s8+s16) -> embeddings_det')
    ap.add_argument('--onnx_in', required=True, help='Input ONNX model path')
    ap.add_argument('--onnx_out', default=None, help='Output ONNX model path (default: *_embed_det.onnx)')
    ap.add_argument('--s8_node', default=None, help='Override tensor name for s8 feature')
    ap.add_argument('--s16_node', default=None, help='Override tensor name for s16 feature')
    ap.add_argument('--det_out', default=None, help='Detection output (Nx6) tensor name (auto if omitted)')
    ap.add_argument('--max_probe', type=int, default=60, help='Max runtime outputs to probe when auto-picking feature tensors (default: 60)')
    # Tuning knobs for part/global pooling and color features
    ap.add_argument('--gp_w', type=float, default=1.0, help='Weight for global pooled features (default: 1.0)')
    ap.add_argument('--pp_w', type=float, default=0.0, help='Weight for part-pooled features (default: 0.0)')
    ap.add_argument('--color_gain', type=float, default=1.0, help='Gain multiplier for color-statistics features (default: 1.0; set 0 to disable)')
    ap.add_argument('--pp_k', type=int, default=7, help='Number of horizontal stripes (default: 7 for pooled_hw=14)')
    ap.add_argument('--pp_stripe_h', type=int, default=2, help='Height of each horizontal stripe (default: 2 for pooled_hw=14)')
    ap.add_argument('--pp_vertical_k', type=int, default=7, help='Number of vertical stripes (default: 7 for pooled_hw=14; set 0 to disable)')
    ap.add_argument('--pp_vertical_stripe_w', type=int, default=2, help='Width of each vertical stripe (default: 2 for pooled_hw=14)')
    # InstanceNorm control: default OFF unless explicitly enabled
    ap.add_argument('--no_inst_norm', action='store_true', help='Disable per-ROI instance normalization')
    ap.add_argument('--use_inst_norm', action='store_true', help='Enable per-ROI instance normalization (overrides --no_inst_norm)')
    args = ap.parse_args()

    if not os.path.exists(args.onnx_in):
        print('[ERROR] ONNX not found:', args.onnx_in, file=sys.stderr)
        sys.exit(1)

    model = onnx.load(args.onnx_in)
    Himg, Wimg = get_input_hw(model)

    s8_name, s16_name = args.s8_node, args.s16_node
    if not s8_name or not s16_name:
        cands = [(n, s) for n, s in list_rank4_candidates(onnx, model) if s is not None]
        if not cands:
            print('[INFO] Static shape inference did not yield candidates; probing runtime shapes ...')
            cands = probe_rank4_candidates(onnx, helper, TensorProto, model, max_probe=args.max_probe)
        if not s8_name:
            s8_name = pick_stride(cands, Himg, Wimg, 8)
        if not s16_name:
            # Avoid selecting the same tensor as s8 when possible
            excl = {s8_name} if s8_name else set()
            s16_name = pick_stride(cands, Himg, Wimg, 16, exclude=excl)
    if not s8_name or not s16_name:
        print('[ERROR] Failed to auto-pick s8/s16 tensors. Consider passing --s8_node/--s16_node.', file=sys.stderr)
        sys.exit(2)

    print('Picked features:')
    print(' - s8 :', s8_name)
    print(' - s16:', s16_name)

    # Always add multi-scale per-detection ROIAlign head
    det_out = args.det_out or find_detection_output(onnx, model)
    if not det_out:
        print('[ERROR] Could not auto-detect detection output (Nx6). Provide --det_out.', file=sys.stderr)
        sys.exit(3)
    print(f'Adding multi-scale ROI head from s8 ({s8_name}) and s16 ({s16_name}), det_out={det_out}')
    # Resolve InstanceNorm usage: default OFF unless --use_inst_norm set
    use_inst = True if getattr(args, 'use_inst_norm', False) else (False if getattr(args, 'no_inst_norm', False) else False)

    model = add_roi_head_multi_scale(
        onnx,
        helper,
        TensorProto,
        model,
        s8_name,
        s16_name,
        det_out,
        stride8=8,
        stride16=16,
        out_name='embeddings_det',
        pooled_hw=14,
        gp_w=args.gp_w,
        pp_w=args.pp_w,
        color_gain_val=args.color_gain,
        pp_k=args.pp_k,
        pp_stripe_h=args.pp_stripe_h,
        pp_vertical_k=args.pp_vertical_k,
        pp_vertical_stripe_w=args.pp_vertical_stripe_w,
        use_inst_norm=use_inst,
    )

    # Save
    out_path = args.onnx_out
    if not out_path:
        base = os.path.splitext(args.onnx_in)[0]
        out_path = base + '_embed_det.onnx'
    onnx.save(model, out_path)
    print('Saved ONNX with embedding head:', out_path)
    # Print final output names to confirm presence
    try:
        outs = [o.name for o in model.graph.output]
        print('Model outputs:', outs)
    except Exception:
        pass


if __name__ == '__main__':
    main()
