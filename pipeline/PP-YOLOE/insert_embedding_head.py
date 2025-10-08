#!/usr/bin/env python3
"""
Insert per-detection appearance embeddings into a PP‑YOLOE ONNX model.

What this does
- Adds a BoT‑SORT–ready output named 'embed' with shape (N, D). Each row is an L2‑normalized per‑detection embedding.
- Auto‑picks two rank‑4 features around strides s8 and s16 (multi‑scale ROIAlign). If static shapes are not enough, it can do a small
    runtime probe to confirm shapes.
- Per‑ROI pipeline (per scale): optional InstanceNorm → signed power‑law normalization → global pooling (avg/max mix) and part pooling
    (horizontal/vertical stripes) → weighted blend → concat s8/s16 (and optional color) → NaN sanitize → final L2.
- Keeps the original model outputs intact; only adds 'embed'. Embedding dim ≈ C_s8 + C_s16 (+6 if color branch enabled).

Recommended strong settings for tracking (good separation in our tests)
- Use these flags when inserting the head for robust cosine spread:
        --use_inst_norm \
        --gp_w 0.2 --pp_w 0.8 \
        --avg_w 1.0 --max_w 0.0 \
        --pp_k 9 --pp_stripe_h 2 --pp_vertical_k 2 --pp_vertical_stripe_w 2 \
        --pooled_hw 16 --sampling_ratio 2 \
        --pl_alpha 0.35 --pre_norm_scales \
        --color_gain 0.0

Expected embedding health (rule‑of‑thumb)
- After normalization, pairwise cosine on valid detections typically shows: median ~0.05–0.10, p95 < ~0.35.
- The highest cosines (>0.6) usually happen for duplicate/overlapping boxes of the same person (high IoU). Non‑overlapping pairs should
    rarely exceed ~0.45–0.50.

BoT‑SORT wiring tips
- The output name is 'embed' (float32, N×D). Make sure the tracker reads this tensor row‑aligned with detections.
- Thresholding guidance:
    * If your code uses cosine distance d = 1 − cos, set max_dist ≈ 0.40–0.45 (i.e., cos ≥ 0.55–0.60 is considered similar).
    * If it uses cosine similarity directly, use a match threshold ≈ 0.55–0.60.
- Keep IoU gating on (e.g., ≥ 0.2–0.3) to favor motion/overlap for short‑term matches and use appearance as a tiebreaker.
- Use a reasonable feature history (e.g., nn_budget 50–100) and EMA/smoothing if available for track features.

Troubleshooting / quality of life
- If auto‑picking features is slow, cap the runtime probe count: --max_probe 20 (default is 60), or specify nodes explicitly with
    --s8_node and --s16_node.
- To disable the color branch and reduce D by 6, set --color_gain 0.0 (recommended for stability across lighting changes).
- For small/skinny boxes, keep --sampling_ratio 2 and consider --pooled_hw 16–20.

Quick usage
        # Minimal (auto‑pick features and detection output)
        python pipeline/PP-YOLOE/insert_embedding_head.py \
                --onnx_in pipeline/PP-YOLOE/backbone/ppyoloe_crn_s_36e_pphuman.onnx \
                --onnx_out pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
                --use_inst_norm --gp_w 0.2 --pp_w 0.8 --avg_w 1.0 --max_w 0.0 \
                --pp_k 9 --pp_stripe_h 2 --pp_vertical_k 2 --pp_vertical_stripe_w 2 \
                --pooled_hw 16 --sampling_ratio 2 --pl_alpha 0.35 --pre_norm_scales \
                --color_gain 0.0 --max_probe 20

        # If you know the feature node names, you can bypass probing entirely
        ... --s8_node <tensor_name> --s16_node <tensor_name>

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


def pick_stride(
    cands: List[Tuple[str, List[int]]],
    Himg: int,
    Wimg: int,
    stride: int,
    exclude: Optional[set] = None,
    min_channels: Optional[int] = None,
    prefer_channels: Optional[List[int]] = None,
) -> Optional[str]:
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
        if min_channels is not None and C < int(min_channels):
            return float('inf')

        # Channel sanity: prefer mid/high channels
        ch_pen = 0.0
        if C < 32:
            ch_pen += 50.0
        elif C < 64:
            ch_pen += 10.0
        elif C > 1536:
            ch_pen += 20.0

        if prefer_channels:
            # Favor channel counts close to any preferred value
            closest = min(abs(C - pc) for pc in prefer_channels)
            ch_pen += closest * 0.1
        else:
            # Default slight preference toward larger channel counts
            ch_pen += -min(C, 1024) / 2048.0

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
        tmp_path = None
        try:
            tmp_model = onnx.load_from_string(model.SerializeToString())
            tmp_model = add_output_to_model(tmp_model, name, onnx, helper, TensorProto)
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tf:
                tmp_path = tf.name
                onnx.save(tmp_model, tf.name)
            import onnxruntime as ort  # type: ignore
            sess_options = ort.SessionOptions()
            try:
                sess_options.log_severity_level = int(os.environ.get('ORT_LOG_SEVERITY_LEVEL', '3'))
            except Exception:
                sess_options.log_severity_level = 3
            sess = ort.InferenceSession(tmp_path, sess_options)
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
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return probed


# (Removed image-level embedding head to simplify the script; we only produce per-detection embed.)


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


def add_roi_head(onnx, helper, TensorProto, model, feat_name: str, det_out_name: str, stride: int, out_name: str = 'embed', pooled_hw: int = 14):
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

    # Rescale boxes from original image coords -> network input coords using scale_factor
    # PP-YOLOE exported models with NMS output boxes in ORIGINAL image coordinates (after dividing by scale_factor).
    # For ROI alignment to work correctly, we must convert back to NETWORK coordinates (0-640 range).
    # scale_factor is a model input of shape [1,2] or [2] = [scale_y, scale_x] where scale = resized/original.
    # This multiplication converts: boxes_original * scale_factor = boxes_network (0-640 range).
    scale_inp = None
    for inp in g.input:
        n = inp.name.lower()
        if 'scale' in n:  # matches 'scale_factor'
            scale_inp = inp.name
            break
    if scale_inp is not None:
        # Validation note: During inference, verify box ranges are in network coords (typically 0-640) after this scaling.
        # If boxes still exceed network size, the model may not be using scale_factor as expected.
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
    # Note: this single-scale helper always applies InstanceNorm. It is not used by the default multi-scale path.
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

    # Reduce over H and W of normalized features -> use avg+max then average them elementwise (global pooling)
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
    max_detections: int = 16,
    stride8: int = 8,
    stride16: int = 16,
    out_name: str = 'embed',
    pooled_hw: int = 14,
    gp_w: float = 1.0,
    pp_w: float = 0.0,
    color_gain_val: float = 1.0,
    avg_w: float = 0.5,
    max_w: float = 0.5,
    pp_k: int = 7,
    pp_stripe_h: int = 2,
    pp_vertical_k: int = 7,
    pp_vertical_stripe_w: int = 2,
    use_inst_norm: bool = False,
    center_ch: bool = True,
    pl_alpha: float = 0.5,
    pre_norm_scales: bool = False,
    sampling_ratio: int = 0,
):
    g = model.graph

    # Common 0.5 constant used for averaging two tensors
    if not any(init.name == 'const_half' for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name='const_half', data_type=TensorProto.FLOAT, dims=[1], vals=[0.5])])

    def make_name(base):
        idx = 0
        while any(n.name == f'{base}_{idx}' for n in g.node):
            idx += 1
        return f'{base}_{idx}'

    # Optionally cap detections to max_detections before building ROI head
    det_src = det_out_name
    max_keep = int(max_detections)
    if max_keep > 0:
        det_shape = det_out_name + '_shape'
        g.node.extend([helper.make_node('Shape', inputs=[det_out_name], outputs=[det_shape], name=make_name('Shape'))])
        idx0 = 'const_idx0'
        if not any(init.name == idx0 for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=idx0, data_type=TensorProto.INT64, dims=[1], vals=[0])])
        det_rows = det_out_name + '_rows'
        g.node.extend([helper.make_node('Gather', inputs=[det_shape, idx0], outputs=[det_rows], name=make_name('Gather'), axis=0)])
        max_keep_name = 'embed_max_keep'
        if not any(init.name == max_keep_name for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=max_keep_name, data_type=TensorProto.INT64, dims=[1], vals=[max_keep])])
        keep_rows = det_out_name + '_keep_rows'
        g.node.extend([helper.make_node('Min', inputs=[det_rows, max_keep_name], outputs=[keep_rows], name=make_name('Min'))])
        det_cols_const = 'embed_det_cols'
        if not any(init.name == det_cols_const for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=det_cols_const, data_type=TensorProto.INT64, dims=[1], vals=[6])])
        row_slice_starts = 'embed_row_slice_starts'
        if not any(init.name == row_slice_starts for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=row_slice_starts, data_type=TensorProto.INT64, dims=[2], vals=[0, 0])])
        row_slice_axes = 'embed_row_slice_axes'
        if not any(init.name == row_slice_axes for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=row_slice_axes, data_type=TensorProto.INT64, dims=[2], vals=[0, 1])])
        row_slice_steps = 'embed_row_slice_steps'
        if not any(init.name == row_slice_steps for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=row_slice_steps, data_type=TensorProto.INT64, dims=[2], vals=[1, 1])])
        rowslice_ends = det_out_name + '_rowslice_ends'
        g.node.extend([helper.make_node('Concat', inputs=[keep_rows, det_cols_const], outputs=[rowslice_ends], name=make_name('Concat'), axis=0)])
        det_capped = det_out_name + '_topk'
        g.node.extend([
            helper.make_node(
                'Slice',
                inputs=[det_out_name, row_slice_starts, rowslice_ends, row_slice_axes, row_slice_steps],
                outputs=[det_capped],
                name=make_name('Slice'),
            )
        ])
        det_src = det_capped

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
    boxes = det_src + '_boxes_ms'
    g.node.extend([
        helper.make_node('Slice', inputs=[det_src, 'roi_ms_slice_starts', 'roi_ms_slice_ends', 'roi_ms_slice_axes', 'roi_ms_slice_steps'], outputs=[boxes], name=make_name('Slice'))
    ])

    # Rescale boxes from original image coords -> network input coords using scale_factor if available
    # See detailed explanation in add_roi_head() above for coordinate space conventions.
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
    g.node.extend([helper.make_node('RoiAlign', inputs=[feat8, boxes, batch_idx], outputs=[roi8], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=int(sampling_ratio), spatial_scale=1.0/float(stride8))])
    roi16 = feat16 + '_roi_ms16'
    g.node.extend([helper.make_node('RoiAlign', inputs=[feat16, boxes, batch_idx], outputs=[roi16], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=int(sampling_ratio), spatial_scale=1.0/float(stride16))])

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

    # Power-law (signed) normalization to reduce burstiness: y = sign(x) * (|x| + eps)^alpha
    eps_pl = 'eps_powerlaw'
    if not any(init.name == eps_pl for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=eps_pl, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
    const_pl_alpha = 'const_pl_alpha'
    if not any(init.name == const_pl_alpha for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=const_pl_alpha, data_type=TensorProto.FLOAT, dims=[1], vals=[float(pl_alpha)])])
    abs8 = src8 + '_abs'
    g.node.extend([helper.make_node('Abs', inputs=[src8], outputs=[abs8], name=make_name('Abs'))])
    abseps8 = src8 + '_abseps'
    g.node.extend([helper.make_node('Add', inputs=[abs8, eps_pl], outputs=[abseps8], name=make_name('Add'))])
    pow8 = src8 + '_powabs'
    g.node.extend([helper.make_node('Pow', inputs=[abseps8, const_pl_alpha], outputs=[pow8], name=make_name('Pow'))])
    sign8 = src8 + '_sign'
    g.node.extend([helper.make_node('Sign', inputs=[src8], outputs=[sign8], name=make_name('Sign'))])
    pl8 = src8 + '_powerlaw'
    g.node.extend([helper.make_node('Mul', inputs=[sign8, pow8], outputs=[pl8], name=make_name('Mul'))])

    # Reduce over H and W: avg+max then average to keep dim unchanged (global pooling)
    # Use unique name suffixes to avoid colliding with part-pooling outputs
    avg8 = roi8 + '_gavg'
    max8 = roi8 + '_gmax'
    g.node.extend([helper.make_node('ReduceMean', inputs=[pl8], outputs=[avg8], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
    g.node.extend([helper.make_node('ReduceMax', inputs=[pl8], outputs=[max8], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])])
    # Weighted combine: pooled8 = avg8*avg_w + max8*max_w (defaults favor avg)
    const_avg_w = 'const_avg_w'
    const_max_w = 'const_max_w'
    init_names_local = {init.name for init in g.initializer}
    if const_avg_w not in init_names_local:
        g.initializer.extend([helper.make_tensor(name=const_avg_w, data_type=TensorProto.FLOAT, dims=[1], vals=[float(avg_w)])])
    if const_max_w not in init_names_local:
        g.initializer.extend([helper.make_tensor(name=const_max_w, data_type=TensorProto.FLOAT, dims=[1], vals=[float(max_w)])])
    avg8_w = roi8 + '_gavg_w'
    max8_w = roi8 + '_gmax_w'
    pooled8 = roi8 + '_gpool'
    # Important: append nodes one-by-one so make_name() sees the updated graph
    g.node.extend([helper.make_node('Mul', inputs=[avg8, const_avg_w], outputs=[avg8_w], name=make_name('Mul'))])
    g.node.extend([helper.make_node('Mul', inputs=[max8, const_max_w], outputs=[max8_w], name=make_name('Mul'))])
    g.node.extend([helper.make_node('Add', inputs=[avg8_w, max8_w], outputs=[pooled8], name=make_name('Add'))])

    # Part-based pooling over H stripes (keeps (N, C)) and optionally blend with global pooled.
    # If pp_w <= 0 (default), skip building part-pooling graph and use zeros to avoid NaN*0 propagation.
    if pp_w is not None and float(pp_w) <= 0.0:
        # Create a zeros-like tensor by subtracting the tensor from itself
        pb8 = roi8 + '_pp_zero'
        g.node.extend([
            helper.make_node('Sub', inputs=[pooled8, pooled8], outputs=[pb8], name=make_name('Sub'))
        ])
    else:
        pb8_h = add_part_pool(pl8, roi8, K=pp_k, stripe_h=pp_stripe_h)
    # Optional vertical pooling
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
    pow16 = src16 + '_powabs'
    g.node.extend([helper.make_node('Pow', inputs=[abseps16, const_pl_alpha], outputs=[pow16], name=make_name('Pow'))])
    sign16 = src16 + '_sign'
    g.node.extend([helper.make_node('Sign', inputs=[src16], outputs=[sign16], name=make_name('Sign'))])
    pl16 = src16 + '_powerlaw'
    g.node.extend([helper.make_node('Mul', inputs=[sign16, pow16], outputs=[pl16], name=make_name('Mul'))])

    # Use unique name suffixes for s16 as well
    avg16 = roi16 + '_gavg'
    max16 = roi16 + '_gmax'
    g.node.extend([helper.make_node('ReduceMean', inputs=[pl16], outputs=[avg16], name=make_name('ReduceMean'), keepdims=0, axes=[2, 3])])
    g.node.extend([helper.make_node('ReduceMax', inputs=[pl16], outputs=[max16], name=make_name('ReduceMax'), keepdims=0, axes=[2, 3])])
    # Reuse weights
    avg16_w = roi16 + '_gavg_w'
    max16_w = roi16 + '_gmax_w'
    pooled16 = roi16 + '_gpool'
    # Important: append nodes one-by-one so make_name() sees the updated graph
    g.node.extend([helper.make_node('Mul', inputs=[avg16, 'const_avg_w'], outputs=[avg16_w], name=make_name('Mul'))])
    g.node.extend([helper.make_node('Mul', inputs=[max16, 'const_max_w'], outputs=[max16_w], name=make_name('Mul'))])
    g.node.extend([helper.make_node('Add', inputs=[avg16_w, max16_w], outputs=[pooled16], name=make_name('Add'))])

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
                g.node.extend([helper.make_node('RoiAlign', inputs=[img_in, boxes, batch_idx], outputs=[roi_img], name=make_name('RoiAlign'), mode='avg', output_height=pooled_hw, output_width=pooled_hw, sampling_ratio=int(sampling_ratio), spatial_scale=1.0)])
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

    # Optionally pre-normalize each scale vector to unit L2 before concatenation (equalize contribution)
    def l2_normalize_vec(inp: str, base: str) -> str:
        sq = base + '_sq'
        rs = base + '_rs'
        add = base + '_den'
        norm = base + '_norm'
        g.node.extend([helper.make_node('Mul', inputs=[inp, inp], outputs=[sq], name=make_name('Square'))])
        axes_ch = 'roi_axes_ch_ms_prenorm'
        if not any(init.name == axes_ch for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=axes_ch, data_type=TensorProto.INT64, dims=[1], vals=[1])])
        g.node.extend([helper.make_node('ReduceSum', inputs=[sq, axes_ch], outputs=[rs], name=make_name('ReduceSum'), keepdims=1)])
        eps = 'eps_roi_head_ms_prenorm'
        if not any(init.name == eps for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=eps, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
        g.node.extend([helper.make_node('Add', inputs=[rs, eps], outputs=[add], name=make_name('Add'))])
        g.node.extend([helper.make_node('Sqrt', inputs=[add], outputs=[norm], name=make_name('Sqrt'))])
        out = base + '_unit'
        g.node.extend([helper.make_node('Div', inputs=[inp, norm], outputs=[out], name=make_name('Div'))])
        return out

    pooled8_for_concat = pooled8_mix
    pooled16_for_concat = pooled16_mix
    if pre_norm_scales:
        pooled8_for_concat = l2_normalize_vec(pooled8_mix, roi8 + '_prenorm')
        pooled16_for_concat = l2_normalize_vec(pooled16_mix, roi16 + '_prenorm')

    # Concat channel-wise (s8, s16, optional color stats)
    concat = 'emb_ms_concat'
    concat_inputs = [pooled8_for_concat, pooled16_for_concat]
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

    # Optional channel centering (remove per-vector mean across channels) to improve angular spread
    emb_input = concat_clean
    if center_ch:
        ch_axes = 'center_ch_axes'
        if not any(init.name == ch_axes for init in g.initializer):
            g.initializer.extend([helper.make_tensor(name=ch_axes, data_type=TensorProto.INT64, dims=[1], vals=[1])])
        ch_mean = concat + '_chmean'
        emb_centered = concat + '_centered'
        g.node.extend([
            helper.make_node('ReduceMean', inputs=[concat_clean], outputs=[ch_mean], name=make_name('ReduceMean'), keepdims=1, axes=[1]),
            helper.make_node('Sub', inputs=[concat_clean, ch_mean], outputs=[emb_centered], name=make_name('Sub')),
        ])
        emb_input = emb_centered

    # L2 normalize across channel (dim=1)
    sq = (emb_input if center_ch else concat_clean) + '_sq'
    g.node.extend([helper.make_node('Mul', inputs=[emb_input, emb_input], outputs=[sq], name=make_name('Square'))])
    rs = (emb_input if center_ch else concat_clean) + '_rs'
    axes_ch = 'roi_axes_ch_ms'
    if not any(init.name == axes_ch for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=axes_ch, data_type=TensorProto.INT64, dims=[1], vals=[1])])
    g.node.extend([helper.make_node('ReduceSum', inputs=[sq, axes_ch], outputs=[rs], name=make_name('ReduceSum'), keepdims=1)])
    eps = 'eps_roi_head_ms'
    if not any(init.name == eps for init in g.initializer):
        g.initializer.extend([helper.make_tensor(name=eps, data_type=TensorProto.FLOAT, dims=[1], vals=[1e-6])])
    add = (emb_input if center_ch else concat_clean) + '_den'
    g.node.extend([helper.make_node('Add', inputs=[rs, eps], outputs=[add], name=make_name('Add'))])
    sqrt = (emb_input if center_ch else concat_clean) + '_norm'
    g.node.extend([helper.make_node('Sqrt', inputs=[add], outputs=[sqrt], name=make_name('Sqrt'))])
    emb = out_name
    g.node.extend([helper.make_node('Div', inputs=[emb_input, sqrt], outputs=[emb], name=make_name('Div'))])

    g.output.extend([helper.make_tensor_value_info(emb, TensorProto.FLOAT, None)])
    return model


def main():
    onnx, helper, TensorProto = import_onnx_modules()

    ap = argparse.ArgumentParser(description='Add per-detection ROI embeddings (multi-scale s8+s16) -> embed')
    ap.add_argument('--onnx_in', required=True, help='Input ONNX model path')
    ap.add_argument('--onnx_out', default=None, help='Output ONNX model path (default: *_embed.onnx)')
    ap.add_argument('--s8_node', default=None, help='Override tensor name for s8 feature')
    ap.add_argument('--s16_node', default=None, help='Override tensor name for s16 feature')
    ap.add_argument('--det_out', default=None, help='Detection output (Nx6) tensor name (auto if omitted)')
    ap.add_argument('--embed_topk', type=int, default=16, help='Max detections to embed (default: 16)')
    ap.add_argument('--max_probe', type=int, default=20, help='Max runtime outputs to probe when auto-picking feature tensors (default: 20)')
    # Tuning knobs for part/global pooling and color features
    ap.add_argument('--gp_w', type=float, default=0.2, help='Weight for global pooled features (default: 0.2)')
    ap.add_argument('--pp_w', type=float, default=0.8, help='Weight for part-pooled features (default: 0.8)')
    ap.add_argument('--color_gain', type=float, default=0.0, help='Gain multiplier for color-statistics features (default: 0.0; set >0 to enable)')
    ap.add_argument('--avg_w', type=float, default=1.0, help='Weight for average pooling within ROI (default: 1.0)')
    ap.add_argument('--max_w', type=float, default=0.0, help='Weight for max pooling within ROI (default: 0.0)')
    ap.add_argument('--pp_k', type=int, default=9, help='Number of horizontal stripes (default: 9 for pooled_hw=16)')
    ap.add_argument('--pp_stripe_h', type=int, default=2, help='Height of each horizontal stripe (default: 2 for pooled_hw=16)')
    ap.add_argument('--pp_vertical_k', type=int, default=2, help='Number of vertical stripes (default: 2 for pooled_hw=16; set 0 to disable)')
    ap.add_argument('--pp_vertical_stripe_w', type=int, default=2, help='Width of each vertical stripe (default: 2 for pooled_hw=16)')
    ap.add_argument('--pooled_hw', type=int, default=16, help='ROIAlign pooled size H=W (default: 16)')
    ap.add_argument('--pl_alpha', type=float, default=0.35, help='Signed power-law exponent alpha (default: 0.35)')
    # Pre-normalization control: default ON, allow disabling with --no_pre_norm_scales
    ap.add_argument('--pre_norm_scales', dest='pre_norm_scales', action='store_true', default=True, help='L2-normalize s8/s16 vectors before concatenation (default: enabled)')
    ap.add_argument('--no_pre_norm_scales', dest='pre_norm_scales', action='store_false', help='Disable L2 pre-normalization of s8/s16 vectors before concatenation')
    ap.add_argument('--sampling_ratio', type=int, default=2, help='RoiAlign sampling_ratio (0 for adaptive, >0 for fixed samples per bin; default: 2)')
    # InstanceNorm control: default ON unless explicitly disabled
    ap.add_argument('--no_inst_norm', action='store_true', help='Disable per-ROI instance normalization')
    ap.add_argument('--use_inst_norm', action='store_true', help='Enable per-ROI instance normalization (overrides --no_inst_norm)')
    ap.add_argument('--no_center_ch', action='store_true', help='Disable per-vector channel centering before L2 normalization')
    args = ap.parse_args()

    if not os.path.exists(args.onnx_in):
        print('[ERROR] ONNX not found:', args.onnx_in, file=sys.stderr)
        sys.exit(1)

    model = onnx.load(args.onnx_in)
    Himg, Wimg = get_input_hw(model)

    s8_name, s16_name = args.s8_node, args.s16_node
    cands = [(n, s) for n, s in list_rank4_candidates(onnx, model) if s is not None]
    cand_dict: Dict[str, List[int]] = {n: s for n, s in cands}

    need_autopick = (not s8_name) or (not s16_name)
    fallback_limits = [limit for limit in (80, 120, 160, 200) if limit > args.max_probe]
    probed_limits_done: set[int] = set()

    def candidate_list() -> List[Tuple[str, List[int]]]:
        return list(cand_dict.items())

    def run_probe(limit: int, *, initial: bool = False) -> None:
        if limit is None or limit <= 0 or limit in probed_limits_done:
            return
        probed_limits_done.add(limit)
        if initial:
            print(f'[INFO] Probing runtime shapes (max_probe={limit}) ...')
        else:
            print(f'[INFO] Re-probing runtime shapes with max_probe={limit} to locate stride features ...')
        new_cands = probe_rank4_candidates(onnx, helper, TensorProto, model, max_probe=limit)
        for name, shape in new_cands:
            cand_dict[name] = shape

    def try_auto_pick() -> bool:
        nonlocal s8_name, s16_name
        cand_list = candidate_list()
        if not cand_list:
            return False
        local_s8 = s8_name if s8_name else pick_stride(
            cand_list,
            Himg,
            Wimg,
            8,
            min_channels=96,
            prefer_channels=[128, 160, 192],
        )
        if not local_s8:
            return False
        local_s16 = s16_name
        if not local_s16:
            excl = {local_s8}
            local_s16 = pick_stride(
                cand_list,
                Himg,
                Wimg,
                16,
                exclude=excl,
                min_channels=192,
                prefer_channels=[256, 224, 320],
            )
        if not local_s16:
            return False
        s8_name = local_s8
        s16_name = local_s16
        return True

    if need_autopick and not cand_dict:
        run_probe(args.max_probe, initial=True)

    success = True
    if need_autopick:
        if args.max_probe > 0:
            run_probe(args.max_probe, initial=(not cand_dict))
        success = try_auto_pick()
        if not success:
            for limit in fallback_limits:
                run_probe(limit)
                if try_auto_pick():
                    success = True
                    break

    cand_shapes = dict(cand_dict)

    target_dim = 384
    target_stride_hw = {
        8: (max(1, Himg // 8), max(1, Wimg // 8)),
        16: (max(1, Himg // 16), max(1, Wimg // 16)),
    }

    if need_autopick and not cand_shapes:
        for limit in fallback_limits:
            run_probe(limit)
        cand_shapes = dict(cand_dict)

    if need_autopick:
        def has_target_combo(shapes: Dict[str, List[int]]) -> bool:
            stride8_dims = set()
            stride16_dims = set()
            th8, tw8 = target_stride_hw[8]
            th16, tw16 = target_stride_hw[16]
            for shape in shapes.values():
                if not shape or len(shape) != 4:
                    continue
                _, c, hf, wf = shape
                if abs(hf - th8) <= 4 and abs(wf - tw8) <= 4:
                    stride8_dims.add(c)
                if abs(hf - th16) <= 4 and abs(wf - tw16) <= 4:
                    stride16_dims.add(c)
            return any((c8 + c16) == target_dim for c8 in stride8_dims for c16 in stride16_dims)

        if not has_target_combo(cand_shapes):
            for limit in fallback_limits:
                run_probe(limit)
                cand_shapes = dict(cand_dict)
                if has_target_combo(cand_shapes):
                    break

    def matches_stride(shape: Optional[List[int]], stride: int) -> bool:
        if shape is None or len(shape) != 4:
            return False
        _, _, Hf, Wf = shape
        th, tw = target_stride_hw[stride]
        return abs(Hf - th) <= 4 and abs(Wf - tw) <= 4

    def best_pair_for_target() -> Optional[Tuple[str, str]]:
        best_score = float('inf')
        best_pair: Optional[Tuple[str, str]] = None
        for name8, shape8 in cand_shapes.items():
            if not matches_stride(shape8, 8):
                continue
            for name16, shape16 in cand_shapes.items():
                if name16 == name8 or not matches_stride(shape16, 16):
                    continue
                c8 = shape8[1]
                c16 = shape16[1]
                stride_pen = (
                    abs(shape8[2] - target_stride_hw[8][0])
                    + abs(shape8[3] - target_stride_hw[8][1])
                    + abs(shape16[2] - target_stride_hw[16][0])
                    + abs(shape16[3] - target_stride_hw[16][1])
                )
                channel_pen = abs(c8 - 128) + abs(c16 - 256)
                total_pen = abs((c8 + c16) - target_dim)
                score = stride_pen + 0.5 * channel_pen + 2.0 * total_pen
                if score < best_score:
                    best_score = score
                    best_pair = (name8, name16)
        return best_pair

    shape_s8 = cand_shapes.get(s8_name) if s8_name else None
    shape_s16 = cand_shapes.get(s16_name) if s16_name else None
    if shape_s8 and shape_s16:
        if shape_s8[1] + shape_s16[1] != target_dim:
            pair = best_pair_for_target()
            if pair is not None:
                if pair[0] != s8_name or pair[1] != s16_name:
                    print('[INFO] Adjusted feature selection to satisfy 384-D embedding target.')
                s8_name, s16_name = pair
                shape_s8 = cand_shapes.get(s8_name)
                shape_s16 = cand_shapes.get(s16_name)
    elif cand_shapes:
        pair = best_pair_for_target()
        if pair is not None:
            s8_name, s16_name = pair
            shape_s8 = cand_shapes.get(s8_name)
            shape_s16 = cand_shapes.get(s16_name)

    if not s8_name or not s16_name:
        print('[ERROR] Failed to auto-pick s8/s16 tensors. Consider passing --s8_node/--s16_node.', file=sys.stderr)
        sys.exit(2)

    print('Picked features:')
    print(' - s8 :', s8_name)
    print(' - s16:', s16_name)
    if args.embed_topk and args.embed_topk > 0:
        print(f'   embedding top-K limit: {args.embed_topk}')

    if not shape_s8:
        shape_s8 = cand_shapes.get(s8_name)
    if not shape_s16:
        shape_s16 = cand_shapes.get(s16_name)
    if shape_s8 and shape_s16:
        total_dim = shape_s8[1] + shape_s16[1]
        print(f'   channel summary -> s8: {shape_s8[1]}, s16: {shape_s16[1]}, total: {total_dim}')
        if total_dim != target_dim:
            print(f'[WARN] Embedding channels sum to {total_dim}, expected {target_dim}.')
    else:
        print('[WARN] Unable to resolve channel counts for selected features; embedding dim may differ from expected.')

    # Always add multi-scale per-detection ROIAlign head
    det_out = args.det_out or find_detection_output(onnx, model)
    if not det_out:
        print('[ERROR] Could not auto-detect detection output (Nx6). Provide --det_out.', file=sys.stderr)
        sys.exit(3)
    print(f'Adding multi-scale ROI head from s8 ({s8_name}) and s16 ({s16_name}), det_out={det_out}')
    # Resolve InstanceNorm usage: default ON unless --no_inst_norm set
    use_inst = False if getattr(args, 'no_inst_norm', False) else (True if getattr(args, 'use_inst_norm', False) else True)

    model = add_roi_head_multi_scale(
        onnx,
        helper,
        TensorProto,
        model,
        s8_name,
        s16_name,
        det_out,
        max_detections=int(getattr(args, 'embed_topk', 16) or 0),
        stride8=8,
        stride16=16,
        out_name='embed',
        pooled_hw=args.pooled_hw,
        gp_w=args.gp_w,
        pp_w=args.pp_w,
        color_gain_val=args.color_gain,
        avg_w=args.avg_w,
        max_w=args.max_w,
        pp_k=args.pp_k,
        pp_stripe_h=args.pp_stripe_h,
        pp_vertical_k=args.pp_vertical_k,
        pp_vertical_stripe_w=args.pp_vertical_stripe_w,
        use_inst_norm=use_inst,
        center_ch=(False if getattr(args, 'no_center_ch', False) else True),
        pl_alpha=args.pl_alpha,
        pre_norm_scales=bool(getattr(args, 'pre_norm_scales', True)),
        sampling_ratio=int(getattr(args, 'sampling_ratio', 0)),
    )

    # Save
    out_path = args.onnx_out
    if not out_path:
        base = os.path.splitext(args.onnx_in)[0]
        out_path = base + '_embed.onnx'
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
