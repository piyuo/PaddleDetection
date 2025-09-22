#!/usr/bin/env python3
"""
Insert a simple multi-scale embedding head into a PP-YOLOE ONNX model.

What it does:
- Auto-pick two backbone/neck feature maps around strides s8 and s16
- Apply GlobalAveragePool + Flatten on both
- Concat the pooled vectors and L2-normalize -> 'embedding' output

Notes:
- No projection layer is added (no learned weights). This is a strong
  baseline to verify BoT-SORT-style appearance features are available
  directly from the model. You can later add a learned projection if needed.
- The original model outputs are preserved.

Usage:
  python pipeline/PP-YOLOE/insert_embedding_head.py \
    --onnx_in pipeline/PP-YOLOE/backbone/ppyoloe_crn_s_36e_pphuman.onnx \
    --onnx_out pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx

Optional:
    --s8_node <tensor_name>  --s16_node <tensor_name>  (to override auto-pick)
    --pool avg|max  (default: avg)
    --roi_from {s8,s16,ms}   Add a per-detection ROI head using the selected feature map, or 'ms' for s8+s16 concat
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


def pick_stride(cands: List[Tuple[str, List[int]]], Himg: int, Wimg: int, stride: int) -> Optional[str]:
    if not cands:
        return None
    target = (Himg // stride, Wimg // stride)
    best, best_d = None, 1e9
    for name, s in cands:
        Hf, Wf = s[2], s[3]
        d = abs(Hf - target[0]) + abs(Wf - target[1])
        if d < best_d:
            best, best_d = name, d
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


def add_multi_scale_head(onnx, helper, TensorProto, model, s8_name: str, s16_name: str, pool: str, embed_out_name: str = 'embedding'):
    g = model.graph

    def make_name(base):
        idx = 0
        while any(n.name == f'{base}_{idx}' for n in g.node):
            idx += 1
        return f'{base}_{idx}'

    def pool_and_flatten(src: str) -> str:
        if pool == 'max':
            pool_op = 'GlobalMaxPool'
        else:
            pool_op = 'GlobalAveragePool'
        p_out = src + '_gp'
        g.node.extend([helper.make_node(pool_op, inputs=[src], outputs=[p_out], name=make_name(pool_op))])
        flat_out = p_out + '_flat'
        # Flatten axis=1 -> (N, C)
        g.node.extend([helper.make_node('Flatten', inputs=[p_out], outputs=[flat_out], name=make_name('Flatten'), axis=1)])
        return flat_out

    f8 = pool_and_flatten(s8_name)
    f16 = pool_and_flatten(s16_name)

    concat = 'concat_ms'
    g.node.extend([
        helper.make_node('Concat', inputs=[f8, f16], outputs=[concat], name=make_name('Concat'), axis=1)
    ])

    # L2 normalize: x / sqrt(sum(x*x, axis=1, keepdims=1) + eps)
    sq = concat + '_sq'
    g.node.extend([helper.make_node('Mul', inputs=[concat, concat], outputs=[sq], name=make_name('Square'))])

    rs = concat + '_rs'
    # For opset >=13, ReduceSum takes 'axes' as an input tensor instead of an attribute
    axes_name = 'reduce_axes_ms_head'
    if not any(init.name == axes_name for init in g.initializer):
        axes_init = helper.make_tensor(name=axes_name, data_type=TensorProto.INT64, dims=[1], vals=[1])
        g.initializer.extend([axes_init])
    g.node.extend([helper.make_node('ReduceSum', inputs=[sq, axes_name], outputs=[rs], name=make_name('ReduceSum'), keepdims=1)])

    eps_init = helper.make_tensor(name='eps_ms_head', data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])
    if not any(init.name == 'eps_ms_head' for init in g.initializer):
        g.initializer.extend([eps_init])
    add = concat + '_den'
    g.node.extend([helper.make_node('Add', inputs=[rs, 'eps_ms_head'], outputs=[add], name=make_name('Add'))])

    sqrt = concat + '_norm'
    g.node.extend([helper.make_node('Sqrt', inputs=[add], outputs=[sqrt], name=make_name('Sqrt'))])

    emb = embed_out_name
    g.node.extend([helper.make_node('Div', inputs=[concat, sqrt], outputs=[emb], name=make_name('Div'))])

    # Register output info (dtype/shape left dynamic)
    g.output.extend([helper.make_tensor_value_info(emb, TensorProto.FLOAT, None)])
    return model


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


def add_roi_head(onnx, helper, TensorProto, model, feat_name: str, det_out_name: str, stride: int, out_name: str = 'embeddings_det', pooled_hw: int = 7):
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

    # Reduce mean over H and W -> (num_rois, C)
    pooled = roi_out + '_avg'
    # For ReduceMean (opset >= 13), axes are specified as an attribute, not an input
    g.node.extend([
        helper.make_node('ReduceMean', inputs=[roi_out], outputs=[pooled], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])
    ])

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


def add_roi_head_multi_scale(onnx, helper, TensorProto, model, feat8: str, feat16: str, det_out_name: str, stride8: int = 8, stride16: int = 16, out_name: str = 'embeddings_det', pooled_hw: int = 7):
    g = model.graph

    def make_name(base):
        idx = 0
        while any(n.name == f'{base}_{idx}' for n in g.node):
            idx += 1
        return f'{base}_{idx}'

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

    # ROIAlign on s8 and s16
    roi8 = feat8 + '_roi_ms'
    g.node.extend([helper.make_node('RoiAlign', inputs=[feat8, boxes, batch_idx], outputs=[roi8], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=0, spatial_scale=1.0/float(stride8))])
    roi16 = feat16 + '_roi_ms'
    g.node.extend([helper.make_node('RoiAlign', inputs=[feat16, boxes, batch_idx], outputs=[roi16], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=0, spatial_scale=1.0/float(stride16))])

    # Reduce mean over H and W
    avg8 = roi8 + '_avg'
    avg16 = roi16 + '_avg'
    g.node.extend([helper.make_node('ReduceMean', inputs=[roi8], outputs=[avg8], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
    g.node.extend([helper.make_node('ReduceMean', inputs=[roi16], outputs=[avg16], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])

    # Concat channel-wise
    concat = 'emb_ms_concat'
    g.node.extend([helper.make_node('Concat', inputs=[avg8, avg16], outputs=[concat], name=make_name('Concat'), axis=1)])

    # L2 normalize across channel (dim=1)
    sq = concat + '_sq'
    g.node.extend([helper.make_node('Mul', inputs=[concat, concat], outputs=[sq], name=make_name('Square'))])
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
    g.node.extend([helper.make_node('Div', inputs=[concat, sqrt], outputs=[emb], name=make_name('Div'))])

    g.output.extend([helper.make_tensor_value_info(emb, TensorProto.FLOAT, None)])
    return model


def main():
    onnx, helper, TensorProto = import_onnx_modules()

    ap = argparse.ArgumentParser(description='Insert multi-scale (s8+s16) pooling+concat+L2 head into ONNX')
    ap.add_argument('--onnx_in', required=True, help='Input ONNX model path')
    ap.add_argument('--onnx_out', default=None, help='Output ONNX model path (default: *_embed.onnx)')
    ap.add_argument('--s8_node', default=None, help='Override tensor name for s8 feature')
    ap.add_argument('--s16_node', default=None, help='Override tensor name for s16 feature')
    ap.add_argument('--pool', choices=['avg', 'max'], default='avg', help='Pooling type (default: avg)')
    ap.add_argument('--roi_from', choices=['s8', 's16', 'ms'], default=None, help='Add per-detection ROI head from this feature map or multi-scale (ms)')
    ap.add_argument('--det_out', default=None, help='Detection output (Nx6) tensor name (auto if omitted)')
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
            cands = probe_rank4_candidates(onnx, helper, TensorProto, model, max_probe=150)
        if not s8_name:
            s8_name = pick_stride(cands, Himg, Wimg, 8)
        if not s16_name:
            s16_name = pick_stride(cands, Himg, Wimg, 16)
    if not s8_name or not s16_name:
        print('[ERROR] Failed to auto-pick s8/s16 tensors. Consider passing --s8_node/--s16_node.', file=sys.stderr)
        sys.exit(2)

    print('Picked features:')
    print(' - s8 :', s8_name)
    print(' - s16:', s16_name)

    model = add_multi_scale_head(onnx, helper, TensorProto, model, s8_name, s16_name, pool=args.pool, embed_out_name='embedding')

    # Optional: add per-detection ROIAlign head
    if args.roi_from is not None:
        det_out = args.det_out or find_detection_output(onnx, model)
        if not det_out:
            print('[ERROR] Could not auto-detect detection output (Nx6). Provide --det_out.', file=sys.stderr)
            sys.exit(3)
        if args.roi_from == 'ms':
            print(f'Adding multi-scale ROI head from s8 ({s8_name}) and s16 ({s16_name}), det_out={det_out}')
            model = add_roi_head_multi_scale(onnx, helper, TensorProto, model, s8_name, s16_name, det_out, stride8=8, stride16=16, out_name='embeddings_det', pooled_hw=7)
        else:
            feat = s8_name if args.roi_from == 's8' else s16_name
            stride = 8 if args.roi_from == 's8' else 16
            print(f'Adding ROI head from {args.roi_from} ({feat}), det_out={det_out}, stride={stride}')
            model = add_roi_head(onnx, helper, TensorProto, model, feat, det_out, stride=stride, out_name='embeddings_det', pooled_hw=7)

    out_path = args.onnx_out
    if not out_path:
        base = os.path.splitext(args.onnx_in)[0]
        # If ROI head is added, default to *_embed_det.onnx for clarity
        out_path = base + ('_embed_det.onnx' if args.roi_from is not None else '_embed.onnx')
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
