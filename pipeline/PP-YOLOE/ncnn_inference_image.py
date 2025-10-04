#!/usr/bin/env python3
"""NCNN single-image inference for PP-YOLOE Human model.

This script mirrors (conceptually) the orchestration style of the ONNX inference
scripts (`onnx_inference_image.py`) but uses the NCNN runtime. It attempts to be
robust to two common PP-YOLOE export patterns:

  1. Model already contains NMS and returns a (N,6) array: (class, score, x0, y0, x1, y1)
  2. Model returns separate raw boxes (N,4) and scores (N,) or (N,1) requiring
	 custom NMS (simple IoU-based using OpenCV's `cv2.dnn.NMSBoxes`).

Surgery / pruned NCNN models (post automatic graph surgery) remove the NMS node
and expose raw outputs plus feature maps for embedding:

	0: raw boxes  (e.g. name contains 'divide') shape (1, N, 4)
	1: raw scores (e.g. name contains 'concat') shape (1, 1, N) or (1, N)
	2: stride-8  feature map (e.g. 'batch_norm_.13') shape (1, C8=128, 80, 80)
	3: stride-16 feature map (e.g. 'batch_norm_.19') shape (1, C16=256, 40, 40)

This script now detects that pattern and:
	* Applies custom NMS on raw boxes + scores (single-class person)
	* Performs multi-scale ROI pooling (simplified but aligned with ONNX ANE path)
		to produce 384-D embeddings (128 + 256) for BoT-SORT style trackers
	* Normalizes embeddings: instance norm -> power-law (alpha=0.35) -> L2

Usage example (see shell helper):
  python3 pipeline/PP-YOLOE/ncnn_inference_image.py \
	  --img pipeline/dataset/demo/demo.jpg \
	  --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.ncnn.param \
	  --ncnn_bin   pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.ncnn.bin \
	  --out pipeline/output --thresh 0.5

Notes:
  * Requires `pip install ncnn opencv-python numpy` (ncnn Python wheels >=1.0.0)
  * Preprocessing matches ONNX path: resize (keep_ratio=False) to 640x640, RGB,
	(x/255 - mean)/std with mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
  * Attempts to auto-discover input & output blob names by parsing the .param file.
	You can override with --input-name / --output-names.
"""

from __future__ import annotations

import argparse
import os
import sys
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np

try:
	import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
	print("[ERROR] OpenCV (cv2) not installed. Install with: pip install opencv-python", file=sys.stderr)
	raise

# Attempt to import ncnn lazily later to provide clearer error if missing

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_SIZE = (640, 640)


@dataclass
class InferenceResult:
	boxes: np.ndarray       # (N,6): class, score, x0,y0,x1,y1
	embeddings: np.ndarray  # (N,D) embedding per detection (may be empty)
	benchmark: dict


# ---------------------------------------------------------------------------
# Utility helpers (kept self-contained so we do not depend on onnx utils file)
# ---------------------------------------------------------------------------

def preprocess_image(
	img_path: str,
	target_size: Tuple[int, int] = DEFAULT_SIZE,
	keep_ratio: bool = False,
	mean: Tuple[float, float, float] = MEAN,
	std: Tuple[float, float, float] = STD,
) -> np.ndarray:
	"""Load & preprocess image into CHW float32 tensor matching ONNX flow.

	Returns: np.ndarray (3, H, W)
	"""
	with open(img_path, "rb") as f:
		data = np.frombuffer(f.read(), dtype=np.uint8)
	im = cv2.imdecode(data, cv2.IMREAD_COLOR)
	if im is None:
		raise ValueError(f"Failed to decode image: {img_path}")
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	orig_h, orig_w = im.shape[:2]

	if keep_ratio:
		# (Not used for PP-YOLOE defaults, but implemented for completeness)
		min_orig, max_orig = min(orig_h, orig_w), max(orig_h, orig_w)
		min_tgt, max_tgt = min(target_size), max(target_size)
		scale = min_tgt / float(min_orig)
		if round(scale * max_orig) > max_tgt:
			scale = max_tgt / float(max_orig)
		resize_h = int(round(orig_h * scale))
		resize_w = int(round(orig_w * scale))
		im = cv2.resize(im, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
		# Pad to target size (top-left alignment)
		pad_h = target_size[0] - resize_h
		pad_w = target_size[1] - resize_w
		if pad_h < 0 or pad_w < 0:
			raise ValueError("Unexpected negative padding; check scaling logic")
		im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
	else:
		im = cv2.resize(im, target_size[::-1], interpolation=cv2.INTER_LINEAR)

	im = im.astype(np.float32) / 255.0
	mean_arr = np.array(mean, dtype=np.float32)[None, None, :]
	std_arr = np.array(std, dtype=np.float32)[None, None, :]
	im = (im - mean_arr) / std_arr
	im = im.transpose(2, 0, 1)  # CHW
	return im


def parse_param_file_for_blobs(param_path: str) -> Tuple[List[str], List[str]]:
	"""Parse NCNN .param file to guess input and output blob names.

	NCNN .param lines containing 'Input' denote inputs; 'Output' denote outputs.
	Returns: (inputs, outputs)
	"""
	inputs, outputs = [], []
	try:
		with open(param_path, "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				line = line.strip()
				# Typical pattern: 'Input  image 0 1 0'
				if line.startswith("Input"):
					parts = re.split(r"\s+", line)
					if len(parts) >= 2:
						inputs.append(parts[1])
				elif line.startswith("Output"):
					parts = re.split(r"\s+", line)
					if len(parts) >= 2:
						outputs.append(parts[1])
	except Exception:
		pass
	return inputs, outputs


def ncnn_mat_from_chw_np(chw: np.ndarray, mean: Sequence[float], std: Sequence[float]):
	"""Convert CHW numpy (float32) into ncnn.Mat using mean/std via substract_mean_normalize.

	We replicate the transform (x/255 - mean)/std in ncnn space by feeding raw
	uint8 image through Mat.from_pixels. However, here we already produced the
	normalized CHW float32. To avoid re-implementing multiple paths, we instead
	directly create an ncnn.Mat from contiguous memory if supported; if not, we
	fallback to a channel copy. For simplicity & portability we redo the
	normalization inside NCNN using the original pixel formula if we can load
	the original image again. Because we already normalized, easiest is to just
	feed as-is (bypassing ncnn's mean/normalize) using from_pixels + reorder.

	For reliability across ncnn versions, we'll choose a simpler approach:
	  1. Create an empty Mat of desired shape
	  2. Copy data channel-wise.
	"""
	import ncnn  # local import

	c, h, w = chw.shape
	# ncnn.Mat(w, h, c) layout: planar channels
	mat = ncnn.Mat(w, h, c)
	# Flatten in C-major contiguous order already: (C,H,W)
	data = chw.astype(np.float32).ravel()
	# ncnn Python binding allows buffer interface assignment via .channel(i)
	offset = 0
	size_per_ch = h * w
	for i in range(c):
		ch_view = data[offset : offset + size_per_ch]
		offset += size_per_ch
		# channel(i) returns a numpy-like wrapper supporting[:] assignment
		try:
			mat.channel(i)[:] = ch_view
		except Exception:
			# Fallback: element-wise assignment (slower)
			for j, val in enumerate(ch_view):
				mat.channel(i)[j] = float(val)
	return mat


def run_ncnn(
	param_path: str,
	bin_path: str,
	img_path: str,
	thresh: float,
	input_name: Optional[str] = None,
	output_names: Optional[List[str]] = None,
	warmup: int = 3,
	boxes_format: str = "auto",  # 'auto'|'xyxy'|'cxcywh'
	nms_threshold: float = 0.5,
	enable_embeddings: bool = True,
) -> InferenceResult:
	try:
		import ncnn  # type: ignore
	except Exception as exc:  # pragma: no cover
		print("[ERROR] ncnn Python package not installed. Install with: pip install ncnn", file=sys.stderr)
		raise exc

	if not os.path.exists(param_path):
		raise FileNotFoundError(f"Param file not found: {param_path}")
	if not os.path.exists(bin_path):
		raise FileNotFoundError(f"Model bin file not found: {bin_path}")

	auto_inputs, auto_outputs = parse_param_file_for_blobs(param_path)
	if input_name is None:
		# Prefer explicit 'image' if present, else first discovered, else default
		if "image" in auto_inputs:
			input_name = "image"
		elif auto_inputs:
			input_name = auto_inputs[0]
		else:
			input_name = "in0"
	if output_names is None:
		output_names = auto_outputs if auto_outputs else []

	net = ncnn.Net()
	# Enable light mode & opt if desired (safe defaults)
	net.opt.use_vulkan_compute = False  # ensure portability
	net.opt.use_fp16_arithmetic = True
	net.opt.use_fp16_storage = True
	net.opt.use_fp16_packed = True

	if net.load_param(param_path):
		raise RuntimeError(f"Failed to load param: {param_path}")
	if net.load_model(bin_path):
		raise RuntimeError(f"Failed to load model: {bin_path}")

	# Preprocess
	chw = preprocess_image(img_path, target_size=DEFAULT_SIZE, keep_ratio=False)
	in_mat = ncnn_mat_from_chw_np(chw, MEAN, STD)

	# Warmup runs
	for _ in range(max(0, warmup)):
		ex = net.create_extractor()
		ex.input(input_name, in_mat)
		# If we know outputs, extract them; else try typical patterns later
		if output_names:
			for name in output_names:
				ex.extract(name)
		else:
			# Do nothing; just ensure graph executes
			pass

	# Timed run
	ex = net.create_extractor()
	ex.input(input_name, in_mat)
	t0 = time.perf_counter()
	mats = {}
	extracted_names = []
	candidate_names = list(output_names)
	# If no outputs declared, we cannot enumerate graph easily; user should pass names.
	# We'll attempt some heuristics for common suffixes.
	if not candidate_names:
		candidate_names = ["output", "outputs", "detection_out", "detections", "nmsed_boxes"]
	for name in candidate_names:
		ret, m = ex.extract(name)
		if ret == 0:
			mats[name] = m
			extracted_names.append(name)
	# If still empty, brute-force try any auto_outputs discovered earlier.
	if not mats and auto_outputs:
		for name in auto_outputs:
			ret, m = ex.extract(name)
			if ret == 0:
				mats[name] = m
				extracted_names.append(name)
	# If still nothing, we fail.
	if not mats:
		raise RuntimeError(
			"Could not extract any outputs. Provide --output-names explicitly (comma-separated)."
		)
	# Convert to numpy
	outs_np = {}
	for name, m in mats.items():
		arr = mat_to_numpy(m)
		outs_np[name] = arr
	t1 = time.perf_counter()
	inference_ms = (t1 - t0) * 1000.0

	print("\n[NCNN] Outputs extracted:")
	for k, v in outs_np.items():
		print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")

	# ------------------------------------------------------------------
	# Post-process: detect pruned raw outputs vs already-NMS outputs
	# ------------------------------------------------------------------
	name_lut = {k.lower(): k for k in outs_np.keys()}
	# Heuristic keys
	raw_boxes_name = next((outs_np[k] for k in outs_np if 'divide' in k), None)
	raw_scores_name = next((outs_np[k] for k in outs_np if 'concat' in k), None)
	feat_s8 = next((outs_np[k] for k in outs_np if 'batch_norm_.13' in k or 'batch_norm_.' in k and '80' in str(outs_np[k].shape)), None)
	feat_s16 = next((outs_np[k] for k in outs_np if 'batch_norm_.19' in k or '40' in str(outs_np[k].shape) and isinstance(outs_np[k], np.ndarray)), None)

	boxes_kept: np.ndarray
	embs: np.ndarray

	if any(arr.ndim == 2 and arr.shape[1] == 6 for arr in outs_np.values()) and raw_boxes_name is None:
		# Original style with built-in NMS
		for name, arr in outs_np.items():
			if arr.ndim == 2 and arr.shape[1] == 6:
				print(f"[Post] Using '{name}' as final detections (already NMS-ed).")
				boxes_final = arr.astype(np.float32)
				break
		mask = (boxes_final[:, 0] > -1) & (boxes_final[:, 1] >= float(thresh))
		boxes_kept = boxes_final[mask]
		embs = np.zeros((boxes_kept.shape[0], 0), dtype=np.float32)
	elif raw_boxes_name is not None and raw_scores_name is not None:
		# Pruned model path: raw boxes + scores + (optional) feature maps
		raw_boxes = raw_boxes_name
		raw_scores = raw_scores_name
		# Normalize shapes
		if raw_boxes.ndim == 3:
			raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])  # (N,4)
		if raw_scores.ndim > 1:
			raw_scores = raw_scores.reshape(-1)
		print(f"[Post] Raw boxes shape: {raw_boxes.shape}; raw scores shape: {raw_scores.shape}")
		if raw_boxes.shape[1] != 4:
			raise RuntimeError(f"Expected raw boxes last dim 4, got {raw_boxes.shape}")

		# Convert centers to corners if needed
		fmt = boxes_format
		if fmt == 'auto':
			# Heuristic: if max(x0) < max(x1) when interpreted as xyxy? ambiguous. Use width/height positivity.
			# Assume xyxy by default; switch to cxcywh if any box has x2<x1 when treating as xyxy (rare) or widths negative.
			# We'll just default to xyxy; user can override.
			fmt = 'xyxy'
		if fmt == 'cxcywh':
			cx, cy, w, h = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
			x0 = cx - w / 2.0
			y0 = cy - h / 2.0
			x1 = cx + w / 2.0
			y1 = cy + h / 2.0
			boxes_xyxy = np.stack([x0, y0, x1, y1], axis=1)
		else:
			boxes_xyxy = raw_boxes.astype(np.float32)

		# Apply NMS
		print(f"[Post] Applying NMS (score >= {thresh:.2f}, IoU thresh {nms_threshold:.2f})")
		kept_struct = apply_nms(boxes_xyxy, raw_scores, score_thresh=thresh, nms_thresh=nms_threshold)
		boxes_kept = kept_struct  # already (N,6)
		# Embeddings
		if enable_embeddings and boxes_kept.shape[0] > 0 and feat_s8 is not None and feat_s16 is not None:
			feat_s8_t = feat_s8
			feat_s16_t = feat_s16
			if feat_s8_t.ndim == 3:
				feat_s8_t = feat_s8_t[None, :]
			if feat_s16_t.ndim == 3:
				feat_s16_t = feat_s16_t[None, :]
			embs = multi_scale_embeddings(
				feat_s8_t,
				feat_s16_t,
				boxes_kept[:, 2:6],  # xyxy
				img_hw=DEFAULT_SIZE,
				gp_w=0.2,
				pp_w=0.8,
				pp_k=9,
				pp_stripe_h=2,
				pp_vertical_k=2,
				pp_vertical_stripe_w=2,
				pl_alpha=0.35,
			)
			print(f"[Emb] Generated embeddings shape: {embs.shape}")
		else:
			if enable_embeddings and (feat_s8 is None or feat_s16 is None):
				print("[Emb] Feature maps missing; skipping embeddings.")
			embs = np.zeros((boxes_kept.shape[0], 0), dtype=np.float32)
	else:
		raise RuntimeError("Unable to interpret NCNN outputs (no raw or NMSed detections found).")

	# Final threshold sanity (already applied inside NMS branch for raw)
	if boxes_kept.size and boxes_kept.shape[1] == 6:
		mask2 = (boxes_kept[:, 0] > -1) & (boxes_kept[:, 1] >= float(thresh))
		boxes_kept = boxes_kept[mask2]
		if embs.shape[0] == mask2.shape[0]:
			embs = embs[mask2]

	benchmark = {
		"model_type": "ncnn-pruned" if raw_boxes_name is not None else "ncnn",
		"inference_ms": inference_ms,
		"post_ms": 0.0,
		"total_ms": inference_ms,
		"num_detections": int(boxes_kept.shape[0]),
		"outputs": list(outs_np.keys()),
		"warmup_runs": warmup,
		"emb_dim": int(embs.shape[1]) if embs.ndim == 2 else 0,
	}

	return InferenceResult(boxes=boxes_kept, embeddings=embs, benchmark=benchmark)


def multi_scale_embeddings(
	feat_s8: np.ndarray,
	feat_s16: np.ndarray,
	boxes_xyxy: np.ndarray,
	img_hw: Tuple[int, int],
	gp_w: float = 0.2,
	avg_w: float = 1.0,
	max_w: float = 0.0,
	pp_w: float = 0.8,
	pp_k: int = 9,
	pp_stripe_h: int = 2,
	pp_vertical_k: int = 2,
	pp_vertical_stripe_w: int = 2,
	use_inst_norm: bool = True,
	pl_alpha: float = 0.35,
) -> np.ndarray:
	"""Approximate reproduction of ANE multi-scale ROI pooling (simplified).

	Produces concatenated embedding from stride-8 & stride-16 feature maps.
	"""
	assert feat_s8.ndim == 4 and feat_s16.ndim == 4
	_, c8, h8, w8 = feat_s8.shape
	_, c16, h16, w16 = feat_s16.shape
	Himg, Wimg = img_hw

	def pool_features(feat, hF, wF, cF, x0, y0, x1, y1):
		# Map to feature coordinates (assuming direct scaling)
		sx = wF / float(Wimg)
		sy = hF / float(Himg)
		fx0 = int(max(0, np.floor(x0 * sx)))
		fy0 = int(max(0, np.floor(y0 * sy)))
		fx1 = int(min(wF, np.ceil(x1 * sx)))
		fy1 = int(min(hF, np.ceil(y1 * sy)))
		if fx1 <= fx0 or fy1 <= fy0:
			return np.zeros((cF,), dtype=np.float32)
		roi = feat[0, :, fy0:fy1, fx0:fx1]
		C, Hr, Wr = roi.shape
		parts: List[np.ndarray] = []
		# Global pooling mixture
		if gp_w > 0:
			g_feat = np.zeros((C,), dtype=np.float32)
			if avg_w > 0:
				g_feat += roi.mean(axis=(1, 2)) * avg_w
			if max_w > 0:
				g_feat += roi.max(axis=(1, 2)) * max_w
			parts.append(g_feat * gp_w)
		# Horizontal part pooling
		if pp_w > 0 and pp_k > 0 and pp_stripe_h > 0 and Hr >= pp_k:
			stripe_size = max(1, Hr // pp_k)
			for k in range(pp_k):
				y_start = k * stripe_size
				y_end = Hr if k == pp_k - 1 else (k + 1) * stripe_size
				chunk = roi[:, y_start:y_end, :]
				if chunk.size == 0:
					continue
				# Optional finer stripes
				if pp_stripe_h > 1:
					sub_size = max(1, (y_end - y_start) // pp_stripe_h)
					acc = np.zeros((C,), dtype=np.float32)
					cnt = 0
					for s in range(pp_stripe_h):
						ys = y_start + s * sub_size
						ye = y_end if s == pp_stripe_h - 1 else ys + sub_size
						sub = roi[:, ys:ye, :]
						if sub.size:
							acc += sub.mean(axis=(1, 2))
							cnt += 1
					if cnt > 0:
						parts.append((acc / cnt) * pp_w)
				else:
					parts.append(chunk.mean(axis=(1, 2)) * pp_w)
		# Vertical part pooling
		if pp_w > 0 and pp_vertical_k > 0 and pp_vertical_stripe_w > 0 and Wr >= pp_vertical_k:
			stripe_size_v = max(1, Wr // pp_vertical_k)
			for k in range(pp_vertical_k):
				x_start = k * stripe_size_v
				x_end = Wr if k == pp_vertical_k - 1 else (k + 1) * stripe_size_v
				chunk = roi[:, :, x_start:x_end]
				if chunk.size == 0:
					continue
				if pp_vertical_stripe_w > 1:
					sub_w = max(1, (x_end - x_start) // pp_vertical_stripe_w)
					acc = np.zeros((C,), dtype=np.float32)
					cnt = 0
					for s in range(pp_vertical_stripe_w):
						xs = x_start + s * sub_w
						xe = x_end if s == pp_vertical_stripe_w - 1 else xs + sub_w
						sub = roi[:, :, xs:xe]
						if sub.size:
							acc += sub.mean(axis=(1, 2))
							cnt += 1
					if cnt > 0:
						parts.append((acc / cnt) * pp_w)
				else:
					parts.append(chunk.mean(axis=(1, 2)) * pp_w)
		if not parts:
			return np.zeros((C,), dtype=np.float32)
		v = np.sum(parts, axis=0)
		if use_inst_norm:
			m = v.mean()
			s = v.std()
			if s > 1e-6:
				v = (v - m) / s
		if pl_alpha != 1.0:
			v = np.sign(v) * (np.abs(v) ** pl_alpha)
		return v

	embeddings: List[np.ndarray] = []
	for x0, y0, x1, y1 in boxes_xyxy:
		v8 = pool_features(feat_s8, h8, w8, c8, x0, y0, x1, y1)
		v16 = pool_features(feat_s16, h16, w16, c16, x0, y0, x1, y1)
		v = np.concatenate([v8, v16])
		# L2 normalize
		n = np.linalg.norm(v)
		if n > 1e-6:
			v = v / n
		embeddings.append(v)
	if not embeddings:
		return np.zeros((0, c8 + c16), dtype=np.float32)
	return np.stack(embeddings, axis=0).astype(np.float32)


def mat_to_numpy(mat) -> np.ndarray:
	"""Best-effort conversion of ncnn.Mat to numpy array.

	ncnn layout is (w,h,c) planar. We'll return (N,?) 2D if appropriate or (c,h,w).
	"""
	try:
		# Some bindings allow direct numpy conversion
		arr = np.array(mat)
		if arr.size:
			return arr
	except Exception:
		pass
	# Manual path
	w = getattr(mat, "w", 0)
	h = getattr(mat, "h", 0)
	c = getattr(mat, "c", 1)
	# Convert channel by channel
	chans = []
	for i in range(c):
		try:
			ch = mat.channel(i)
			# channel(i) might be 1D flattened (h*w)
			ch_np = np.array(ch, dtype=np.float32).reshape(h, w)
		except Exception:
			# Fallback: allocate zeros
			ch_np = np.zeros((h, w), dtype=np.float32)
		chans.append(ch_np)
	if c == 1:
		return chans[0]
	return np.stack(chans, axis=0)


def apply_nms(boxes_xyxy: np.ndarray, scores: np.ndarray, score_thresh: float = 0.5, nms_thresh: float = 0.5) -> np.ndarray:
	"""Apply NMS using OpenCV; return (N,6) with (class, score, x0,y0,x1,y1)."""
	if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
		raise ValueError("boxes_xyxy must be (N,4)")
	if scores.ndim != 1:
		raise ValueError("scores must be (N,)")
	# Convert to (x,y,w,h) for cv2.dnn.NMSBoxes
	xywh = np.column_stack([
		boxes_xyxy[:, 0],
		boxes_xyxy[:, 1],
		boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
		boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
	])
	indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), float(score_thresh), float(nms_thresh))
	if len(indices) == 0:
		return np.zeros((0, 6), dtype=np.float32)
	indices = np.array(indices).flatten()
	sel_boxes = boxes_xyxy[indices]
	sel_scores = scores[indices]
	cls_ids = np.zeros_like(sel_scores)
	return np.column_stack([cls_ids, sel_scores, sel_boxes]).astype(np.float32)


def draw_and_save(img_path: str, boxes: np.ndarray, thresh: float, out_path: str, label: str = "object"):
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	im = cv2.imread(img_path)
	if im is None:
		print(f"[WARN] Could not read image for drawing: {img_path}")
		return
	for b in boxes:
		cls_id, score, x0, y0, x1, y1 = b
		if score < thresh or cls_id < 0:
			continue
		p1, p2 = (int(x0), int(y0)), (int(x1), int(y1))
		color = (0, 200, 255)
		cv2.rectangle(im, p1, p2, color, 2)
		txt = f"{label}:{score:.2f}"
		cv2.putText(im, txt, (p1[0], max(0, p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
	cv2.imwrite(out_path, im)


def print_detections(boxes: np.ndarray, thresh: float):
	print("\nDetections (class score x0 y0 x1 y1):")
	kept = 0
	for b in boxes:
		cls_id, score, x0, y0, x1, y1 = b
		if cls_id > -1 and score >= float(thresh):
			kept += 1
			print(f"{int(cls_id)} {score:.4f} {x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}")
	if kept == 0:
		print(f"No boxes above threshold {thresh}.")


def main():  # noqa: C901 (complexity acceptable for orchestrator)
	parser = argparse.ArgumentParser(description="NCNN inference for PP-YOLOE Human (single image)")
	parser.add_argument("--img", required=True, help="Path to input image")
	parser.add_argument("--ncnn_param", required=True, help="Path to NCNN .param file")
	parser.add_argument("--ncnn_bin", required=True, help="Path to NCNN .bin weights file")
	parser.add_argument("--out", default="pipeline/output", help="Output directory for visualization")
	parser.add_argument("--thresh", type=float, default=0.5, help="Score threshold for display & filtering")
	parser.add_argument("--input-name", default=None, help="Override input blob name (else auto)")
	parser.add_argument(
		"--output-names",
		default=None,
		help="Comma-separated list of output blob names (else auto parse from .param)",
	)
	parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing")
	parser.add_argument("--boxes-format", default="auto", choices=["auto", "xyxy", "cxcywh"], help="Format of raw boxes if pruned model outputs (auto tries xyxy)")
	parser.add_argument("--nms-thresh", type=float, default=0.5, help="IoU threshold for NMS on pruned model")
	parser.add_argument("--no-embeddings", action="store_true", help="Disable embedding extraction even if feature maps present")
	args = parser.parse_args()

	for path, label in ((args.img, "Input image"), (args.ncnn_param, "NCNN param"), (args.ncnn_bin, "NCNN bin")):
		if not os.path.exists(path):
			print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
			sys.exit(1)

	output_names = None
	if args.output_names:
		output_names = [x.strip() for x in args.output_names.split(",") if x.strip()]

	print("[Info] Starting NCNN inference")
	print(f"  • Image: {args.img}")
	print(f"  • Param: {args.ncnn_param}")
	print(f"  • Bin:   {args.ncnn_bin}")
	print(f"  • Threshold: {args.thresh}")
	if args.input_name:
		print(f"  • Forcing input name: {args.input_name}")
	if output_names:
		print(f"  • Forcing output names: {output_names}")

	try:
		result = run_ncnn(
			param_path=args.ncnn_param,
			bin_path=args.ncnn_bin,
			img_path=args.img,
			thresh=args.thresh,
			input_name=args.input_name,
			output_names=output_names,
			warmup=args.warmup,
			boxes_format=args.boxes_format,
			nms_threshold=args.nms_thresh,
			enable_embeddings=not args.no_embeddings,
		)
	except Exception as exc:
		print(f"[ERROR] Inference failed: {exc}", file=sys.stderr)
		sys.exit(2)

	boxes = result.boxes
	print_detections(boxes, args.thresh)
	print(f"\n[Summary] detections kept: {boxes.shape[0]} (threshold {args.thresh})")
	if result.embeddings.size == 0:
		print("[Summary] embeddings: none (not exported in NCNN model)")
	else:
		print(f"[Summary] embeddings shape: {result.embeddings.shape}")
	if result.benchmark:
		print(f"[Timing] ncnn inference: {result.benchmark.get('inference_ms', 0.0):.2f} ms")
		print(f"[Timing] total pipeline: {result.benchmark.get('total_ms', 0.0):.2f} ms")

	os.makedirs(args.out, exist_ok=True)
	vis_path = os.path.join(args.out, os.path.splitext(os.path.basename(args.img))[0] + "_ncnn.jpg")
	try:
		draw_and_save(args.img, boxes, args.thresh, vis_path, label="person")
		print(f"[Output] Visualization saved to: {vis_path}")
	except Exception as exc:
		print(f"[WARN] Failed to save visualization: {exc}")

	print("\n✅ NCNN inference completed!")


if __name__ == "__main__":  # pragma: no cover
	main()
