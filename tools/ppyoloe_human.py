import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO
import logging

from olive.data.registry import Registry

# Configure logging for progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------------------
# Preprocessing (match PaddleDetection infer_cfg for PP-YOLOE)
# -------------------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image_pil(pil_image, target_size=(640, 640)):
    """Preprocess PIL image to match PP-YOLOE infer_cfg: Resize(keep_ratio=False, 640x640) -> Normalize -> CHW.

    Returns:
      image: float32 CHW
      im_shape: float32 [h, w] of resized image
      scale_factor: float32 [scale_y, scale_x] from original to resized
    """
    # Convert PIL to RGB numpy (HWC)
    img = np.array(pil_image.convert("RGB"))

    orig_h, orig_w = img.shape[:2]
    resize_h, resize_w = target_size

    # keep_ratio = False -> direct resize to target
    image_resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    # Normalize: to float [0,1], then (x - mean) / std (channel-wise)
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = (image_resized - MEAN) / STD

    # HWC -> CHW
    image_chw = image_resized.transpose(2, 0, 1).astype(np.float32)

    # im_shape is resized size, scale_factor is resize/ori
    im_shape = np.array([resize_h, resize_w], dtype=np.float32)
    scale_factor = np.array([resize_h / float(orig_h), resize_w / float(orig_w)], dtype=np.float32)

    return image_chw, im_shape, scale_factor


def cxcywh_to_xyxy(boxes):
    """Convert (cx,cy,w,h) -> (x1,y1,x2,y2)."""
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# -------------------------------
# Dataset
# -------------------------------
class CocoDetection(Dataset):
    def __init__(self, data_dir, annotation_file):
        logging.info(f"Loading COCO dataset from {data_dir} with annotations {annotation_file}")
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.data_dir = data_dir
        logging.info(f"Loaded {len(self.ids)} images for calibration/evaluation")

    def __getitem__(self, idx):
        if idx % 100 == 0:  # Log progress every 100 samples
            logging.info(f"Processing sample {idx}/{len(self.ids)} ({idx/len(self.ids)*100:.1f}%)")

        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.data_dir, path)).convert("RGB")

        # Preprocess to PP-YOLOE input tensors
        image_tensor, im_shape_tensor, scale_factor_tensor = preprocess_image_pil(img)

        # Prepare ground truth boxes and labels (for evaluation)
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            # Convert to normalized cxcywh format (relative to resized 640 for simplicity)
            # Use original image size for normalization to keep [0,1]
            cx = (x + w / 2) / img.width
            cy = (y + h / 2) / img.height
            nw = w / img.width
            nh = h / img.height
            boxes.append([cx, cy, nw, nh])
            labels.append(ann.get("category_id", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Olive will add batch dimension
        return {
            "im_shape": im_shape_tensor.astype(np.float32),          # [2]
            "image": image_tensor.astype(np.float32),                # [3, 640, 640]
            "scale_factor": scale_factor_tensor.astype(np.float32),  # [2]
        }, {
            "boxes": boxes.numpy(),
            "labels": labels.numpy(),
        }

    def __len__(self):
        return len(self.ids)


@Registry.register_dataset()
def dataset_load(data_dir, annotation_file, **kwargs):
    logging.info("Creating COCO dataset for Olive optimization (PP-YOLOE)")
    return CocoDetection(data_dir, annotation_file)


# -------------------------------
# Postprocess (match PaddleDetection ONNX output for YOLO)
# -------------------------------
def decode_outputs(outputs, conf_threshold=0.1):
    """Handle [N,6] or [B,N,C] style outputs where each det is [class_id, score, x1, y1, x2, y2]."""
    detections = []
    for out in outputs:
        out = np.array(out)
        if out.size == 0:
            continue
        # Case 1: [N,6]
        if out.ndim == 2 and out.shape[1] >= 6:
            for det in out:
                cls_id, conf, x1, y1, x2, y2 = det[:6]
                if conf >= conf_threshold:
                    detections.append((int(cls_id), float(conf), [x1, y1, x2, y2]))
        # Case 2: [B,N,C]
        elif out.ndim == 3 and out.shape[2] >= 6:
            for det in out.reshape(-1, out.shape[-1]):
                cls_id, conf, x1, y1, x2, y2 = det[:6]
                if conf >= conf_threshold:
                    detections.append((int(cls_id), float(conf), [x1, y1, x2, y2]))
    return detections


@Registry.register_post_process()
def dataset_post_process(outputs):
    """Turn model raw outputs into usable predictions for Olive metrics."""
    logging.debug("Post-processing model outputs for quantization")

    # Try common YOLO ONNX output names; fallback to the first tensor
    preferred_keys = [
        "multiclass_nms3_0.tmp_0",  # common in PaddleDetection exports
        "multiclass_nms3_1.tmp_0",
        "fetch_name_0",
    ]
    detections_output = None
    for k in preferred_keys:
        if k in outputs:
            detections_output = outputs[k]
            break
    if detections_output is None:
        detections_output = list(outputs.values())[0]

    detections = decode_outputs([detections_output], conf_threshold=0.1)  # Low threshold for calibration
    logging.debug(f"Decoded {len(detections)} detections")

    if len(detections) == 0:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "scores": np.empty((0,), dtype=np.float32),
        }

    boxes = []
    labels = []
    scores = []

    # Convert from absolute xyxy (pixel, on 640x640) to normalized cxcywh
    for cls_id, conf, bbox in detections:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / 640.0
        cy = (y1 + y2) / 2.0 / 640.0
        w = (x2 - x1) / 640.0
        h = (y2 - y1) / 640.0
        boxes.append([cx, cy, w, h])
        labels.append(cls_id)
        scores.append(conf)

    return {
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "scores": np.array(scores, dtype=np.float32),
    }


# -------------------------------
# Evaluation (simple IoU/F1)
# -------------------------------
def calculate_iou_f1(preds, targets, iou_threshold=0.5):
    if len(preds.get("boxes", [])) == 0 or len(targets.get("boxes", [])) == 0:
        return 0.0, 0.0

    pred_xyxy = cxcywh_to_xyxy(torch.from_numpy(preds["boxes"]))
    tgt_xyxy = cxcywh_to_xyxy(torch.from_numpy(targets["boxes"]))

    ious = []
    tp, fp = 0, 0
    for pb in pred_xyxy:
        iou = box_iou(pb.unsqueeze(0), tgt_xyxy).max().item()
        ious.append(iou)
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    fn = len(tgt_xyxy) - tp

    mean_iou = float(np.mean(ious)) if ious else 0.0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return mean_iou, f1


def box_iou(box1, box2):
    """IoU between sets of boxes (xyxy)."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def evaluate(outputs, targets):
    mean_iou, f1 = calculate_iou_f1(outputs, targets)
    return {"mean_of_max_iou": mean_iou, "f1_score": f1}
