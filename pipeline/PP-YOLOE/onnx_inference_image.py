#!/usr/bin/env python3
"""Standalone ONNX person detection inference script.

This script is completely self-contained and requires only standard Python libraries
plus NumPy, OpenCV, PIL, and ONNXRuntime. No DAMO-YOLO dependencies needed.

Required packages:
    pip install numpy opencv-python pillow onnxruntime

All preprocessing, postprocessing, NMS, and visualization are handled with
NumPy and OpenCV only. Configuration is hardcoded for person detection.

Example:
    python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
        --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
        --image pipeline/dataset/demo/demo.jpg \
        --output pipeline/output \
        --conf 0.5

If the exported ONNX uses a legacy head (person + background), the second
channel (background) is discarded so only the person channel is used.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort


def draw_bounding_boxes(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
	"""Simple visualization function to draw bounding boxes on images.

	Args:
		img: Input image (numpy array)
		boxes: Bounding boxes in format [x1, y1, x2, y2]
		scores: Confidence scores
		cls_ids: Class IDs
		conf: Confidence threshold
		class_names: List of class names

	Returns:
		Image with drawn bounding boxes
	"""
	# Ensure the image is writeable
	if not img.flags.writeable:
		img = img.copy()

	# Simple color for person detection (green)
	color = (0, 255, 0)  # BGR format for cv2
	txt_color = (255, 255, 255)  # White text
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.6
	thickness = 2

	for i in range(len(boxes)):
		box = boxes[i]
		cls_id = int(cls_ids[i])
		score = scores[i]

		if score < conf:
			continue

		x0 = int(box[0])
		y0 = int(box[1])
		x1 = int(box[2])
		y1 = int(box[3])

		# Draw bounding box
		cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

		# Prepare text
		if class_names and cls_id < len(class_names):
			text = f'{class_names[cls_id]}:{score*100:.1f}%'
		else:
			text = f'person:{score*100:.1f}%'

		# Get text size for background rectangle
		txt_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

		# Draw text background
		cv2.rectangle(img, (x0, y0 - txt_size[1] - 10),
					 (x0 + txt_size[0], y0), color, -1)

		# Draw text
		cv2.putText(img, text, (x0, y0 - 5), font, font_scale, txt_color, thickness)

	return img


class SimpleConfig:
	"""Hardcoded configuration for person detection inference."""

	def __init__(self):
		# Model head configuration
		self.num_classes = 1
		self.legacy = True
		self.nms_conf_thre = 0.01  # Internal pre-filtering threshold
		self.nms_iou_thre = 0.7    # IoU threshold for NMS

		# Dataset configuration
		self.class_names = ['person']

		# Transform configuration
		self.image_max_range = (640, 640)

		# Processing configuration
		self.fallback_max_detections = 5  # Show top 5 when no detections above threshold


def parse_args():
	parser = argparse.ArgumentParser("ONNX person inference")
	parser.add_argument('--onnx', required=True,
						help='Path to ONNX model file (.onnx)')
	parser.add_argument('--image', default='pipeline/dataset/demo/demo.jpg',
						help='Input image path')
	parser.add_argument('--output', default='pipeline/output',
						help='Directory to save annotated image')
	parser.add_argument('--infer-size', type=int, nargs=2, default=[640, 640],
						help='Inference size (h w) fed to the model')
	parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
						help='Device preference (auto picks cuda if available)')
	parser.add_argument('--conf', type=float, default=0.5,
						help='Confidence threshold for detections')
	parser.add_argument('--debug', action='store_true', help='Print extended diagnostics.')
	return parser.parse_args()


def decide_threshold(scores: np.ndarray, args) -> float:
	"""Return the confidence threshold from args."""
	return float(args.conf)


def format_results(bboxes: np.ndarray,
			   scores: np.ndarray,
			   labels: np.ndarray,
			   conf_thre: float,
			   legacy_single_class: bool = True) -> List[Tuple[int, float, List[float]]]:
	"""Return filtered & sorted detection tuples.

	Each tuple: (raw_index, score, [x1,y1,x2,y2])
	Background (label != 0) filtered if legacy_single_class.
	"""
	dets: List[Tuple[int, float, List[float]]] = []
	if bboxes.size == 0:
		return dets
	for i in range(bboxes.shape[0]):
		s = float(scores[i])
		if s < conf_thre:
			continue
		lbl = int(labels[i])
		if legacy_single_class and lbl != 0:
			continue
		x1, y1, x2, y2 = [float(v) for v in bboxes[i].tolist()]
		dets.append((i, s, [x1, y1, x2, y2]))
	# Sort by descending score
	dets.sort(key=lambda t: t[1], reverse=True)
	return dets



def debug_print_model_outputs(session, outputs, raw_scores, raw_boxes, args):
	"""Print comprehensive debug information about model outputs for C++ porting."""
	print("\n" + "="*80)
	print("🔍 DETAILED MODEL OUTPUT DEBUG INFO (for C++ porting)")
	print("="*80)

	# 1. Model Information
	print(f"\n📋 MODEL INFO:")
	print(f"  Model path: {args.onnx}")
	print(f"  Input size: {args.infer_size[0]}x{args.infer_size[1]}")

	# 2. ONNX Session Info
	inputs = session.get_inputs()
	outputs_meta = session.get_outputs()
	print(f"\n📥 MODEL INPUTS:")
	for i, input_meta in enumerate(inputs):
		print(f"  Input {i}: '{input_meta.name}' shape={input_meta.shape} type={input_meta.type}")

	print(f"\n📤 MODEL OUTPUTS:")
	for i, output_meta in enumerate(outputs_meta):
		print(f"  Output {i}: '{output_meta.name}' shape={output_meta.shape} type={output_meta.type}")

	# 3. Raw Output Analysis with Smart Detection
	print(f"\n🔢 RAW MODEL OUTPUTS:")
	print(f"  Total outputs: {len(outputs)}")
	for i, output in enumerate(outputs):
		print(f"  Output[{i}]: shape={output.shape} dtype={output.dtype}")
		print(f"    Min: {output.min():.6f}, Max: {output.max():.6f}, Mean: {output.mean():.6f}")

	# 4. Smart Output Interpretation
	print(f"\n🧠 OUTPUT INTERPRETATION:")
	for i, output in enumerate(outputs):
		shape = output.shape
		if len(shape) == 3 and shape[1] > 100:  # Detection format: (batch, anchors, features)
			if shape[2] <= 2:
				print(f"  📊 Output[{i}]: CONFIDENCE SCORES")
				print(f"      Shape: {shape} → (batch_size={shape[0]}, num_detections={shape[1]}, num_classes={shape[2]})")
				print(f"      Usage: Person detection confidence scores")
			elif shape[2] == 4:
				print(f"  📦 Output[{i}]: BOUNDING BOXES")
				print(f"      Shape: {shape} → (batch_size={shape[0]}, num_detections={shape[1]}, coordinates=4)")
				print(f"      Usage: Box coordinates in format (x1, y1, x2, y2)")
			else:
				print(f"  🎯 Output[{i}]: DETECTION HEAD")
				print(f"      Shape: {shape} → (batch_size, anchors, classes+coords)")
				print(f"      Usage: Combined detection output (scores + boxes)")
		elif len(shape) == 4:  # Feature map format: (batch, channels, height, width)
			batch, channels, height, width = shape
			# Determine feature level based on spatial resolution
			if height >= 80:
				level = "P3 (high resolution)"
				purpose = "Fine-grained features for small objects"
			elif height >= 40:
				level = "P4 (medium resolution)"
				purpose = "Mid-level features for medium objects"
			elif height >= 20:
				level = "P5 (low resolution)"
				purpose = "Coarse features for large objects"
			else:
				level = "P6+ (very low resolution)"
				purpose = "Global context features"

			print(f"  🗺️  Output[{i}]: FEATURE MAP - {level}")
			print(f"      Shape: {shape} → (batch={batch}, channels={channels}, H={height}, W={width})")
			print(f"      Usage: {purpose}")
			print(f"      BOT-SORT: Used for re-identification feature extraction")
			print(f"      Total features per detection: {channels} dimensions")
		else:
			print(f"  ❓ Output[{i}]: UNKNOWN FORMAT")
			print(f"      Shape: {shape}")
			print(f"      Usage: Please check model documentation")	# 4. Detailed Score Analysis (Output 0)
	print(f"\n📊 SCORES ANALYSIS (Output[0] - typically confidence scores):")
	print(f"  Shape: {raw_scores.shape} -> (batch_size, num_detections, num_classes)")
	batch_size, num_detections, num_classes = raw_scores.shape
	print(f"    Batch size: {batch_size}")
	print(f"    Number of detections: {num_detections}")
	print(f"    Number of classes: {num_classes}")

	# Sample first few raw scores
	print(f"  Raw score samples (first 5 detections, all classes):")
	for i in range(min(5, num_detections)):
		score_str = ", ".join([f"{raw_scores[0,i,c]:.6f}" for c in range(num_classes)])
		print(f"    Detection[{i}]: [{score_str}]")

	# Score statistics per class
	for c in range(num_classes):
		class_scores = raw_scores[0, :, c]
		print(f"  Class {c} scores: min={class_scores.min():.6f} max={class_scores.max():.6f} "
			  f"mean={class_scores.mean():.6f} >0.1: {(class_scores > 0.1).sum()}")

	# 5. Detailed Box Analysis (Output 1)
	print(f"\n📦 BOUNDING BOXES ANALYSIS (Output[1] - typically bounding boxes):")
	print(f"  Shape: {raw_boxes.shape} -> (batch_size, num_detections, 4)")
	batch_size, num_detections, coords = raw_boxes.shape
	print(f"    Batch size: {batch_size}")
	print(f"    Number of detections: {num_detections}")
	print(f"    Coordinates per box: {coords} (typically x1,y1,x2,y2)")

	# Sample first few raw boxes
	print(f"  Raw box samples (first 5 detections):")
	for i in range(min(5, num_detections)):
		box = raw_boxes[0, i, :]
		print(f"    Detection[{i}]: x1={box[0]:.2f} y1={box[1]:.2f} x2={box[2]:.2f} y2={box[3]:.2f}")

	# Box coordinate statistics
	box_flat = raw_boxes[0].reshape(-1, 4)
	print(f"  Box coordinate ranges:")
	print(f"    X1: min={box_flat[:, 0].min():.2f} max={box_flat[:, 0].max():.2f}")
	print(f"    Y1: min={box_flat[:, 1].min():.2f} max={box_flat[:, 1].max():.2f}")
	print(f"    X2: min={box_flat[:, 2].min():.2f} max={box_flat[:, 2].max():.2f}")
	print(f"    Y2: min={box_flat[:, 3].min():.2f} max={box_flat[:, 3].max():.2f}")

	# 6. Post-processing Logic Explanation
	print(f"\n⚙️  POST-PROCESSING LOGIC:")
	print(f"  1. Extract scores from Output[0] shape {raw_scores.shape}")
	if num_classes > 1:
		print(f"  2. Legacy head detected: Taking only class 0 (person) from {num_classes} classes")
		print(f"     - Class 0: person scores")
		print(f"     - Class 1+: background/other (discarded)")
	else:
		print(f"  2. Single class head: Using all scores")
	print(f"  3. Extract boxes from Output[1] shape {raw_boxes.shape}")
	print(f"  4. Apply NMS with IoU threshold = 0.7, score threshold = 0.01")
	print(f"  5. Scale boxes back to original image size")
	print(f"  6. Filter by final confidence threshold = {args.conf}")

	# Add comprehensive summary for C++ team
	print(f"\n🎯 SUMMARY FOR C++ IMPLEMENTATION:")
	feature_outputs = []
	for i, output in enumerate(outputs):
		if len(output.shape) == 4:  # Feature map
			feature_outputs.append(f"Output[{i}] ({output.shape[2]}×{output.shape[3]})")

	if len(outputs) == 2:
		print(f"  📋 Model Type: DETECTION ONLY")
		print(f"  📊 Output[0]: Confidence scores for person detection")
		print(f"  📦 Output[1]: Bounding box coordinates (x1,y1,x2,y2)")
		print(f"  🚫 Feature maps: None (basic detection model)")
	elif len(outputs) == 5:
		print(f"  📋 Model Type: DETECTION + FEATURE EXTRACTION")
		print(f"  📊 Output[0]: Confidence scores for person detection")
		print(f"  📦 Output[1]: Bounding box coordinates (x1,y1,x2,y2)")
		print(f"  🗺️  Output[2-4]: Feature maps for BOT-SORT tracking:")
		for i, output in enumerate(outputs[2:], 2):
			h, w = output.shape[2], output.shape[3]
			channels = output.shape[1]
			print(f"      Output[{i}]: {channels}D features at {h}×{w} resolution")
		print(f"  🎯 BOT-SORT Usage: Extract features from Output[2-4] for re-identification")
	else:
		print(f"  📋 Model Type: CUSTOM ({len(outputs)} outputs)")
		print(f"  ⚠️  Requires manual interpretation of outputs")

	print("="*80)


def debug_print(scores: np.ndarray, labels: np.ndarray, conf: float):
	if scores.size == 0:
		print('\n🚫 [POST-NMS DEBUG] No detections survived NMS filtering.')
		return

	print(f"\n" + "="*60)
	print(f"📊 POST-NMS ANALYSIS (After NMS but before final threshold)")
	print(f"="*60)

	max_score = float(scores.max())
	min_score = float(scores.min())
	above_threshold = int((scores >= conf).sum())
	total_detections = scores.size

	print(f"📈 CONFIDENCE STATISTICS:")
	print(f"  • Total detections after NMS: {total_detections}")
	print(f"  • Score range: {min_score:.4f} → {max_score:.4f}")
	print(f"  • Above final threshold ({conf:.3f}): {above_threshold} detections")
	print(f"  • Will be filtered out: {total_detections - above_threshold} detections")

	unique_labels = sorted(set(labels.tolist()))
	print(f"\n🏷️  DETECTION CLASSES:")
	print(f"  • Unique labels found: {unique_labels}")
	print(f"  • Class 0 = Person detections")

	print(f"\n🥇 TOP 10 HIGHEST CONFIDENCE DETECTIONS:")
	topk = min(10, scores.size)
	idxs = np.argsort(-scores)[:topk]
	for rank, idx in enumerate(idxs):
		score = scores[idx]
		status = "✅ KEPT" if score >= conf else "❌ FILTERED"
		print(f"  #{rank+1:2d}. Confidence: {score:.4f} | Class: {int(labels[idx])} (person) | {status}")

	print(f"="*60)


# ------------------------
# Minimal NumPy NMS helpers
# ------------------------
def _nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
		inds = np.where(ovr <= iou_thr)[0]
		order = order[inds + 1]
	return keep


def multiclass_nms_np(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, score_thr: float) -> np.ndarray | None:
	"""Return dets [K,6] -> x1,y1,x2,y2,score,cls (NumPy)."""
	final = []
	num_classes = scores.shape[1]
	for c in range(num_classes):
		cls_scores = scores[:, c]
		mask = cls_scores > score_thr
		if not mask.any():
			continue
		cls_boxes = boxes[mask]
		cls_scores_sel = cls_scores[mask]
		keep = _nms_single_class(cls_boxes, cls_scores_sel, iou_thr)
		if keep:
			kept_boxes = cls_boxes[keep]
			kept_scores = cls_scores_sel[keep]
			cls_ids = np.full((len(keep), 1), c)
			dets = np.concatenate([kept_boxes, kept_scores[:, None], cls_ids], axis=1)
			final.append(dets)
	if not final:
		return None
	return np.concatenate(final, axis=0)


def run_inference(args):
	if not os.path.isfile(args.onnx):
		raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
	if not os.path.isfile(args.image):
		raise FileNotFoundError(f"Image not found: {args.image}")
	os.makedirs(args.output, exist_ok=True)

	# Use hardcoded configuration instead of loading from file
	cfg = SimpleConfig()

	# Determine providers (cuda preference > ORT defaults)
	provider_list = None
	if args.device == 'cuda':
		available = ort.get_available_providers()
		if 'CUDAExecutionProvider' in available:
			provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
	session = ort.InferenceSession(args.onnx, providers=provider_list or ort.get_available_providers())
	input_meta = session.get_inputs()[0]
	input_name = input_meta.name

	# Load & preprocess image
	origin_img = np.asarray(Image.open(args.image).convert('RGB'))
	oh, ow, _ = origin_img.shape
	target_h, target_w = args.infer_size
	resized = cv2.resize(origin_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # (w,h)
	# Normalize (mean=0, std=1) -> no-op but keep structure if future changes
	img_chw = resized.transpose(2, 0, 1).astype(np.float32)
	batch = img_chw[None, ...]  # (1,C,H,W)

	# Run session
	outputs = session.run(None, {input_name: batch})
	# Expect [cls_scores, bboxes] shapes: (1,N,C) and (1,N,4)
	raw_scores = outputs[0]
	raw_boxes = outputs[1]

	# Add comprehensive debug output for C++ porting team (only when debug enabled)
	if args.debug:
		debug_print_model_outputs(session, outputs, raw_scores, raw_boxes, args)

	if raw_scores.ndim != 3 or raw_boxes.ndim != 3:
		raise ValueError(f"Unexpected ONNX output shapes: {raw_scores.shape}, {raw_boxes.shape}")
	_, N, C = raw_scores.shape
	# Handle legacy background channel (keep only first class when single-class scenario)
	if C > 1:
		person_scores = raw_scores[0, :, 0:1]  # keep only person class
	else:
		person_scores = raw_scores[0]
	boxes = raw_boxes[0]  # (N,4)

	# Internal NMS / filtering (use hardcoded values)
	score_thr_internal = 0.01  # Low threshold for internal pre-filtering
	iou_thr = cfg.nms_iou_thre  # Use config value (0.7)
	dets = multiclass_nms_np(boxes, person_scores, iou_thr=iou_thr, score_thr=score_thr_internal)
	if dets is None:
		final_boxes = np.zeros((0, 4), dtype=np.float32)
		final_scores = np.zeros((0,), dtype=np.float32)
		final_labels = np.zeros((0,), dtype=np.int32)
	else:
		# dets: x1,y1,x2,y2,score,cls  (coordinates in resized image space)
		# Scale back to original image size (linear scaling since keep_ratio=False)
		scale_x = ow / float(target_w)
		scale_y = oh / float(target_h)
		final_boxes = dets[:, :4].copy()
		final_boxes[:, 0] *= scale_x
		final_boxes[:, 2] *= scale_x
		final_boxes[:, 1] *= scale_y
		final_boxes[:, 3] *= scale_y
		final_scores = dets[:, 4].astype(np.float32)
		final_labels = dets[:, 5].astype(np.int32)

	# Decide final display/printing threshold based on resulting scores
	thr = decide_threshold(final_scores, args)
	if args.debug:
		debug_print(final_scores, final_labels, thr)

	formatted = format_results(final_boxes, final_scores, final_labels, thr, legacy_single_class=True)

	# Fallback if nothing above thr - show top detections
	if not formatted and final_scores.size > 0:
		topk = min(cfg.fallback_max_detections, final_scores.size)
		idxs = np.argsort(-final_scores)[:topk]
		for idx in idxs:
			if final_labels[idx] != 0:
				continue
			x1, y1, x2, y2 = final_boxes[idx].tolist()
			formatted.append((int(idx), float(final_scores[idx]), [x1, y1, x2, y2]))
		if formatted:
			print(f"[INFO] No boxes above threshold {thr:.4f}; using top-{len(formatted)} fallback detections.")

	if not formatted:
		print(f"No person detections (threshold={thr:.4f}).")
	else:
		print(f"Detections (threshold={thr:.4f}, kept={len(formatted)}):")
		for rank, (raw_idx, score, (x1, y1, x2, y2)) in enumerate(formatted, 1):
			print(f"  {rank:02d}. person (raw_idx={raw_idx}) score={score:.4f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

	# Visualization with threshold thr
	vis_img = draw_bounding_boxes(origin_img.copy(), final_boxes, final_scores, final_labels, conf=thr, class_names=cfg.class_names)
	save_name = os.path.basename(args.image)
	out_path = os.path.join(args.output, save_name)
	cv2.imwrite(out_path, vis_img[:, :, ::-1])
	return thr, formatted


def main():
    args = parse_args()
    thr, dets = run_inference(args)
    if args.debug:
        print(f"\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"   Final confidence threshold: {thr:.3f}")
        print(f"   Total persons detected: {len(dets)}")
        print(f"   Output image saved to: {args.output}")
        print(f"   Debug mode: ON (detailed analysis shown above)")
    else:
        print(f"✅ Inference completed. Found {len(dets)} person(s) with confidence ≥ {thr:.3f}")


if __name__ == '__main__':
    main()
