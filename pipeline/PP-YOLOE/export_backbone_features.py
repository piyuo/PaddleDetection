#!/usr/bin/env python3
"""
Extract or export backbone feature maps from a PP-YOLOE ONNX model.

What this script can do now:
- List candidate intermediate tensors to extract (rank-4 NCHW)
- Auto-pick a good feature map (largest/s8/s16/s32) or choose explicitly
- Insert the chosen tensor as an additional ONNX model output and SAVE the
	augmented model to the output directory (recommended for deployment)
- Optionally run an inference pass to export .npy feature maps and ROI
	embeddings from detections (kept for convenience)

Quick start: create a new ONNX with an extra backbone feature output
	python pipeline/PP-YOLOE/export_backbone_features.py \
		--onnx pipeline/PP-YOLOE/backbone/ppyoloe_crn_s_36e_pphuman.onnx \
		--infer_cfg pipeline/PP-YOLOE/backbone/inference_model/ppyoloe_crn_s_36e_pphuman/infer_cfg.yml \
		--auto_pick s8 \
		--out pipeline/PP-YOLOE/models \
		--export_onnx

First, discover node/tensor names:
	python pipeline/PP-YOLOE/export_backbone_features.py --onnx <model.onnx> --list-nodes
"""

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def repo_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def default_paths() -> Tuple[str, str, str, str]:
	root = repo_root()
	model_name = 'ppyoloe_crn_s_36e_pphuman'
	onnx_path = os.path.join(root, 'pipeline', 'backbone', f'{model_name}.onnx')
	img_path = os.path.join(root, 'pipeline', 'dataset', 'demo', 'demo.jpg')
	infer_cfg = os.path.join(root, 'pipeline', 'output', 'inference_model', model_name, 'infer_cfg.yml')
	out_dir = os.path.join(root, 'pipeline', 'output', 'backbone')
	return onnx_path, img_path, infer_cfg, out_dir


def import_onnx_modules():
	try:
		import onnx  # type: ignore
		from onnx import helper, TensorProto
	except Exception:
		print('[ERROR] onnx package not installed. Install with: pip install onnx', file=sys.stderr)
		raise
	try:
		import onnxruntime as ort  # type: ignore
	except Exception:
		print('[ERROR] onnxruntime not installed. Install with: pip install onnxruntime', file=sys.stderr)
		raise
	return onnx, helper, TensorProto, ort


def load_preprocess(infer_cfg_path: str):
	# Reuse PaddleDetection ONNX preprocess Compose
	preproc_dir = os.path.join(repo_root(), 'deploy', 'third_engine', 'onnx')
	if preproc_dir not in sys.path:
		sys.path.insert(0, preproc_dir)
	import yaml
	from preprocess import Compose  # type: ignore

	with open(infer_cfg_path, 'r') as f:
		yml_conf = yaml.safe_load(f)
	preprocess_infos = yml_conf['Preprocess']
	draw_threshold = float(yml_conf.get('draw_threshold', 0.5))
	transforms = Compose(preprocess_infos)
	return transforms, draw_threshold


def list_candidate_nodes(model) -> List[Tuple[str, Optional[List[int]]]]:
	# Use ONNX shape inference to get reliable shapes
	import onnx
	try:
		model_inf = onnx.shape_inference.infer_shapes(model)
	except Exception:
		model_inf = model

	graph = model_inf.graph

	# Build sets to exclude weights and inputs
	init_names = {init.name for init in graph.initializer}
	input_names = {i.name for i in graph.input}

	# Index value_info by name
	vi_map: Dict[str, Any] = {}
	for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
		vi_map[vi.name] = vi

	candidates: List[Tuple[str, Optional[List[int]]]] = []
	for node in graph.node:
		for out in node.output:
			if out in init_names or out in input_names:
				continue
			vi = vi_map.get(out)
			shape: Optional[List[int]] = None
			if vi is not None and vi.type and vi.type.tensor_type and vi.type.tensor_type.shape:
				dims = vi.type.tensor_type.shape.dim
				if len(dims) == 4:
					s: List[int] = []
					ok = True
					for d in dims:
						if d.HasField('dim_value'):
							s.append(int(d.dim_value))
						else:
							# unknown dim
							ok = False
							break
					if ok:
						shape = s
			if shape is not None and len(shape) == 4 and shape[0] in (1,):
				# Exclude obvious params by name patterns
				bad_subs = ['.w_', '.b_', 'constant', 'full', 'scale', 'bias']
				name = out
				if any(sub in name for sub in bad_subs):
					continue
				candidates.append((name, shape))
	# Deduplicate while preserving order
	seen = set()
	uniq: List[Tuple[str, Optional[List[int]]]] = []
	for n, s in candidates:
		if n not in seen:
			seen.add(n)
			uniq.append((n, s))
	return uniq


def pick_node_auto(candidates: List[Tuple[str, List[int]]], input_hw: Tuple[int, int], mode: str) -> Optional[str]:
	# candidates: list of (name, [N,C,H,W])
	if not candidates:
		return None
	# Sort by spatial size descending (prefer highest resolution)
	c_sorted = sorted(candidates, key=lambda x: (x[1][2] * x[1][3], x[1][1]), reverse=True)
	if mode == 'largest':
		return c_sorted[0][0]

	Himg, Wimg = input_hw
	targets = {
		's8':  (Himg // 8,  Wimg // 8),
		's16': (Himg // 16, Wimg // 16),
		's32': (Himg // 32, Wimg // 32),
	}
	tgt = targets.get(mode)
	if not tgt:
		return c_sorted[0][0]
	# Choose closest by L1 distance on (H,W)
	best = None
	best_d = 1e9
	for name, shape in candidates:
		Hf, Wf = shape[2], shape[3]
		d = abs(Hf - tgt[0]) + abs(Wf - tgt[1])
		if d < best_d:
			best_d = d
			best = name
	return best


def add_output_to_model(model, value_name: str):
	graph = model.graph
	# If already an output, nothing to do
	if any(o.name == value_name for o in graph.output):
		return model
	# Try to find value info for the tensor
	vi = None
	for v in list(graph.value_info) + list(graph.output) + list(graph.input):
		if v.name == value_name:
			vi = v
			break
	if vi is not None:
		graph.output.extend([vi])
	else:
		# Create a generic float tensor type with unknown dims
		from onnx import helper, TensorProto
		out_vi = helper.make_tensor_value_info(value_name, TensorProto.FLOAT, None)
		graph.output.extend([out_vi])
	return model


def sanitize_filename_part(s: str) -> str:
	return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)


def onnx_output_shapes(model) -> Dict[str, Optional[List[int]]]:
	"""Return a mapping output_name -> static shape list (if available)."""
	try:
		import onnx
		model_inf = onnx.shape_inference.infer_shapes(model)
	except Exception:
		model_inf = model
	shapes: Dict[str, Optional[List[int]]] = {}
	vi_map: Dict[str, Any] = {vi.name: vi for vi in list(model_inf.graph.value_info) + list(model_inf.graph.output)}
	for out in model_inf.graph.output:
		shp: Optional[List[int]] = None
		vi = vi_map.get(out.name)
		if vi is not None and vi.type and vi.type.tensor_type and vi.type.tensor_type.shape:
			dims = vi.type.tensor_type.shape.dim
			s: List[int] = []
			ok = True
			for d in dims:
				if d.HasField('dim_value'):
					s.append(int(d.dim_value))
				else:
					ok = False
					break
			if ok:
				shp = s
		shapes[out.name] = shp
	return shapes


def describe_outputs_text(outputs: List[str], shapes: Dict[str, Optional[List[int]]], feat_output_name: Optional[str]) -> str:
	lines: List[str] = []
	lines.append('# Model outputs')
	lines.append('')
	lines.append(f'Total outputs: {len(outputs)}')
	lines.append('')
	for idx, name in enumerate(outputs):
		shp = shapes.get(name)
		shp_str = 'x'.join(str(d) for d in shp) if shp else 'dynamic/unknown'
		purpose = 'Original model output'
		# Heuristics for PP-YOLOE typical first output
		if idx == 0:
			purpose = 'Detections: [class_id, score, x_min, y_min, x_max, y_max] per row'
		if feat_output_name and name == feat_output_name:
			purpose = 'Backbone/neck feature map (NCHW) for downstream embedding/association'
		lines.append(f'- {idx}: {name}  shape: [{shp_str}]  — {purpose}')
	lines.append('')
	lines.append('Notes:')
	lines.append('- Shapes may be dynamic depending on opset and preprocessing; when unknown, infer at runtime.')
	lines.append('- The feature map is useful for ROI pooling or computing appearance embeddings for tracking (e.g., BoT-SORT).')
	return '\n'.join(lines)


def save_augmented_model(onnx_path: str, node_name: str, out_dir: str) -> Tuple[str, str]:
	"""
	Add the given node/tensor as an extra graph output, save to out_dir, and
	emit a small README describing outputs.
	Returns: (onnx_out_path, outputs_md_path)
	"""
	import onnx
	os.makedirs(out_dir, exist_ok=True)
	base = os.path.splitext(os.path.basename(onnx_path))[0]
	tag = sanitize_filename_part(node_name[-40:])  # keep tail, sanitize
	out_onnx = os.path.join(out_dir, f'{base}_with_{tag}.onnx')

	model = onnx.load(onnx_path)
	model = add_output_to_model(model, node_name)
	onnx.save(model, out_onnx)

	# Describe outputs
	shapes = onnx_output_shapes(model)
	outputs = [o.name for o in model.graph.output]
	md = describe_outputs_text(outputs, shapes, feat_output_name=node_name)
	out_md = os.path.join(out_dir, f'{base}_outputs.md')
	with open(out_md, 'w') as f:
		f.write(md)
	return out_onnx, out_md


def build_session(onnx_path: str, extra_output: Optional[str] = None):
	onnx, helper, TensorProto, ort = import_onnx_modules()
	if extra_output:
		model = onnx.load(onnx_path)
		model = add_output_to_model(model, extra_output)
		tmp = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
		onnx.save(model, tmp.name)
		sess = ort.InferenceSession(tmp.name)
		return sess, tmp.name
	else:
		sess = ort.InferenceSession(onnx_path)
		return sess, None


def roi_pool_average(feat_map: np.ndarray, boxes_xyxy: np.ndarray, img_hw: Tuple[int, int]) -> np.ndarray:
	"""
	Simple ROI average pooling on a feature map.
	- feat_map: (1, C, Hf, Wf)
	- boxes_xyxy: (N, 4) in original image coordinates
	- img_hw: (Himg, Wimg)
	Returns:
	  embeddings: (N, C)
	"""
	assert feat_map.ndim == 4 and feat_map.shape[0] == 1
	_, C, Hf, Wf = feat_map.shape
	Himg, Wimg = img_hw
	scale_x = Wf / float(Wimg)
	scale_y = Hf / float(Himg)

	embs = []
	for x0, y0, x1, y1 in boxes_xyxy:
		fx0 = int(max(0, np.floor(x0 * scale_x)))
		fy0 = int(max(0, np.floor(y0 * scale_y)))
		fx1 = int(min(Wf, np.ceil(x1 * scale_x)))
		fy1 = int(min(Hf, np.ceil(y1 * scale_y)))
		if fx1 <= fx0 or fy1 <= fy0:
			embs.append(np.zeros((C,), dtype=np.float32))
			continue
		region = feat_map[0, :, fy0:fy1, fx0:fx1]
		vec = region.reshape(C, -1).mean(axis=1) if region.size > 0 else np.zeros((C,), dtype=np.float32)
		embs.append(vec)
	return np.stack(embs, axis=0) if embs else np.zeros((0, C), dtype=np.float32)


def main():
	d_onnx, d_img, d_infer_cfg, d_out = default_paths()

	ap = argparse.ArgumentParser(description='Extract PP-YOLOE backbone features from ONNX model')
	ap.add_argument('--onnx', default=d_onnx, help='Path to ONNX model')
	ap.add_argument('--img', default=d_img, help='Path to input image')
	ap.add_argument('--infer_cfg', default=d_infer_cfg, help='infer_cfg.yml for preprocess')
	ap.add_argument('--node', default=None, help='Tensor name to extract as feature map (use --list-nodes to discover)')
	ap.add_argument('--list-nodes', action='store_true', help='List candidate 4D tensors and exit')
	ap.add_argument('--roi_from_det', action='store_true', help='Use detection boxes from model output[0] to ROI-pool embeddings')
	ap.add_argument('--thresh', type=float, default=0.5, help='Score threshold for ROI selection when roi_from_det')
	ap.add_argument('--out', default=d_out, help='Output directory')
	ap.add_argument('--auto_pick', choices=['largest', 's8', 's16', 's32'], default=None, help='Automatically pick a feature map by size/stride')
	args = ap.parse_args()

	for p, label in [
		(args.onnx, 'ONNX model'),
		(args.img, 'input image'),
		(args.infer_cfg, 'infer_cfg.yml'),
	]:
		if not os.path.exists(p):
			print(f'[ERROR] {label} not found: {p}', file=sys.stderr)
			sys.exit(1)

	# List candidate nodes
	onnx, helper, TensorProto, ort = import_onnx_modules()
	model = onnx.load(args.onnx)
	if args.list_nodes:
		cands = list_candidate_nodes(model)
		print('Candidate tensors (prefer rank-4 NCHW feature maps):')
		for name, shape in cands[:]:
			print('-', name, shape if shape is not None else '(shape unknown)')
		print('\nTips:')
		print('- Prefer 4D tensors with shapes like [1, C, Hf, Wf] where Hf/Wf are around input/8, /16, or /32 (neck outputs).')
		print('- For trackers, stride-8 (largest Hf/Wf) or stride-16 often works well.')
		return

	# Preprocess to get input tensor and size
	transforms, _ = load_preprocess(args.infer_cfg)
	inputs_map = transforms(args.img)
	# infer input H,W from 'image' or any tensor with shape [C,H,W]
	in_hw = None
	for k, v in inputs_map.items():
		if v.ndim == 3 and v.shape[0] in (1, 3):
			in_hw = (int(v.shape[1]), int(v.shape[2]))
			break
	if in_hw is None:
		in_hw = (640, 640)

	if not args.node and args.auto_pick:
		cands = [(n, s) for n, s in list_candidate_nodes(model) if s is not None]
		chosen = pick_node_auto(cands, (in_hw[0], in_hw[1]), args.auto_pick)
		if not chosen:
			# Fallback: probe shapes at runtime for a subset of graph outputs
			print('auto_pick: static shapes unavailable; probing intermediate tensors at runtime ...')
			# Build a list of potential outputs to probe
			graph = model.graph
			init_names = {init.name for init in graph.initializer}
			input_names = {i.name for i in graph.input}
			seen = set()
			probe_names: List[str] = []
			bad_subs = ['.w_', '.b_', 'constant', 'full', 'scale', 'bias']
			for node in graph.node:
				for out in node.output:
					if out in seen or out in init_names or out in input_names:
						continue
					if any(sub in out for sub in bad_subs):
						continue
					seen.add(out)
					probe_names.append(out)
			# Try probing up to the first 200 outputs
			probed: List[Tuple[str, List[int]]] = []
			from onnxruntime import InferenceSession
			# Prepare a base feed once we know input names (use a quick session)
			tmp_sess = InferenceSession(args.onnx)
			feed_names = [i.name for i in tmp_sess.get_inputs()]
			feed = {name: inputs_map[name][None, ] for name in feed_names}
			del tmp_sess
			for name in probe_names[:200]:
				try:
					sess, tmp_model = build_session(args.onnx, extra_output=name)
					outs = sess.run(None, feed)
					out_names = [o.name for o in sess.get_outputs()]
					name_to_out = {out_names[i]: outs[i] for i in range(len(out_names))}
					if name in name_to_out:
						arr = name_to_out[name]
						if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] in (1,):
							probed.append((name, list(arr.shape)))
					if tmp_model and os.path.exists(tmp_model):
						try: os.remove(tmp_model)
						except Exception: pass
					# Stop early if we have enough
					if len(probed) >= 20:
						break
				except Exception:
					# Ignore failures for some nodes
					try:
						if tmp_model and os.path.exists(tmp_model):
							os.remove(tmp_model)
					except Exception:
						pass
					continue
			if probed:
				chosen = pick_node_auto(probed, (in_hw[0], in_hw[1]), args.auto_pick)
		if not chosen:
			print('[ERROR] auto_pick failed to find a suitable tensor. Try --list-nodes and set --node explicitly.')
			sys.exit(2)
		print('Auto-picked feature tensor:', chosen)
		args.node = chosen
	elif not args.node:
		print('[ERROR] --node is required unless --list-nodes or --auto_pick is used. Run with --list-nodes to inspect tensor names.')
		sys.exit(2)

	# Export an augmented model with the selected feature map as an additional output
	onnx_out, md_path = save_augmented_model(args.onnx, args.node, args.out)
	print('Saved augmented ONNX:', onnx_out)
	print('Wrote outputs description:', md_path)

	# Build session with extra output (using original or temporary model)
	sess, tmp_model = build_session(args.onnx, extra_output=args.node)

	# Prepare feed for runtime
	input_names = [i.name for i in sess.get_inputs()]
	feed = {name: inputs_map[name][None, ] for name in input_names}

	# Run and capture all outputs; map by name
	output_names = [o.name for o in sess.get_outputs()]
	outputs = sess.run(None, feed)
	name_to_out = {name: outputs[i] for i, name in enumerate(output_names)}

	# Retrieve detection boxes (if present) and feature map
	# Assumption: original model first output is bboxes [N,6]
	bboxes = None
	if len(outputs) >= 1 and outputs[0].ndim == 2 and outputs[0].shape[-1] == 6:
		bboxes = outputs[0]

	if args.node not in name_to_out:
		print('[ERROR] Requested tensor is not present in session outputs:', args.node, file=sys.stderr)
		print('Available outputs:', output_names)
		sys.exit(3)

	feat_map = name_to_out[args.node]
	if feat_map.ndim != 4:
		print('[WARN] Extracted tensor is not rank-4; shape:', feat_map.shape)

	os.makedirs(args.out, exist_ok=True)
	base = os.path.splitext(os.path.basename(args.img))[0]
	np.save(os.path.join(args.out, f'{base}_feature_map.npy'), feat_map)
	print('Saved feature map:', os.path.join(args.out, f'{base}_feature_map.npy'), 'shape', feat_map.shape)

	if args.roi_from_det and bboxes is not None:
		# Filter by threshold and valid class id
		keep = [b for b in bboxes if int(b[0]) > -1 and float(b[1]) >= args.thresh]
		if keep:
			boxes = np.array([[b[2], b[3], b[4], b[5]] for b in keep], dtype=np.float32)
			# Need original image size for mapping
			import cv2
			im = cv2.imread(args.img)
			if im is None:
				print('[WARN] Could not load image to infer size; skipping ROI features')
			else:
				Himg, Wimg = im.shape[:2]
				embs = roi_pool_average(feat_map, boxes, (Himg, Wimg))
				np.save(os.path.join(args.out, f'{base}_embeddings.npy'), embs)
				print('Saved ROI embeddings:', os.path.join(args.out, f'{base}_embeddings.npy'), 'shape', embs.shape)
		else:
			print(f'No detections above threshold {args.thresh}; skipping ROI embeddings')

	# Cleanup temp model
	if tmp_model and os.path.exists(tmp_model):
		try:
			os.remove(tmp_model)
		except Exception:
			pass


if __name__ == '__main__':
	main()
