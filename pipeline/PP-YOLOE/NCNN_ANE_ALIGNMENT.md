# NCNN and ANE Model Alignment

## Overview
This document describes the alignment between `ncnn_inference_image.py` and `onnx_inference_ane_model.py` to ensure consistent behavior between Python NCNN implementation and the C++ port.

## Code Alignment Status: ✅ Complete

The NCNN implementation now uses **identical logic** to the ANE model for:
1. NMS (Non-Maximum Suppression)
2. Multi-scale feature embedding extraction
3. Post-processing pipeline

This alignment enables the C++ team to port the same codebase to NCNN models with confidence.

---

## 1. NMS Algorithm Alignment

### Shared Logic
Both implementations use the **same NMS approach**:

```python
# Extract box coordinates
x0, y0, x1, y1 = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
w = x1 - x0
h = y1 - y0
nms_boxes = np.column_stack([x0, y0, w, h]).tolist()

# Apply NMS with cv2.dnn.NMSBoxes
score_threshold = float(score_thresh)
nms_threshold = float(nms_thresh)
selected_indices = cv2.dnn.NMSBoxes(nms_boxes, person_scores.tolist(), score_threshold, nms_threshold)

# Format output as (class_id, score, x0, y0, x1, y1)
if len(selected_indices) > 0:
    selected_indices = np.array(selected_indices).flatten()
    boxes_nms = boxes_raw[selected_indices]
    scores_nms = person_scores[selected_indices]
    class_ids = np.zeros_like(scores_nms)
    return np.column_stack([class_ids, scores_nms, boxes_nms]).astype(np.float32)
```

### Key Points
- Uses `cv2.dnn.NMSBoxes` (same as ANE model)
- Converts boxes from xyxy → xywh format for NMS
- Returns boxes in format: `(class_id, score, x0, y0, x1, y1)`
- Default NMS IoU threshold: **0.5**

---

## 2. Multi-Scale Embedding Extraction

### Function Signature
```python
def roi_align_pool_multi_scale(
    feat_s8: np.ndarray,      # Stride-8 feature map (128, 80, 80)
    feat_s16: np.ndarray,     # Stride-16 feature map (256, 40, 40)
    boxes_xyxy: np.ndarray,   # Detection boxes in xyxy format
    img_hw: tuple,            # Original image size (height, width)
    input_size_hw: tuple = (640, 640),  # Model input size
    gp_w: float = 0.2,        # Global pooling weight
    avg_w: float = 1.0,       # Average pooling weight
    max_w: float = 0.0,       # Max pooling weight
    pp_w: float = 0.8,        # Part-based pooling weight
    pp_k: int = 9,            # Horizontal stripes
    pp_stripe_h: int = 2,     # Sub-stripes per horizontal stripe
    pp_vertical_k: int = 2,   # Vertical stripes
    pp_vertical_stripe_w: int = 2,  # Sub-stripes per vertical stripe
    use_inst_norm: bool = True,     # Instance normalization
    pl_alpha: float = 0.35,   # Power-law transformation
) -> np.ndarray:
```

### Feature Extraction Strategy

#### 1. **Global Pooling** (gp_w = 0.2)
- Average pooling over entire ROI
- Captures overall appearance

#### 2. **Part-Based Pooling - Horizontal** (pp_w = 0.8)
- Divides ROI into 9 horizontal stripes
- Each stripe is further divided into 2 sub-stripes
- Total: 18 horizontal features
- Captures vertical body structure (head, torso, legs)

#### 3. **Part-Based Pooling - Vertical** (pp_w = 0.8)
- Divides ROI into 2 vertical stripes
- Each stripe is further divided into 2 sub-stripes
- Total: 4 vertical features
- Captures left-right symmetry

### Post-Processing
1. **Instance Normalization**: `(x - mean) / std`
2. **Power-Law Transformation**: `sign(x) * |x|^0.35`
3. **L2 Normalization**: `x / ||x||`

### Output Format
- **Embedding dimension**: 384 (128 from stride-8 + 256 from stride-16)
- **L2 normalized**: All embeddings have norm = 1.0
- **Shape**: `(num_detections, 384)`

---

## 3. Model I/O Specification

### NCNN Model Inputs
| Name  | Shape         | Description             |
| ----- | ------------- | ----------------------- |
| `in0` | (2,)          | Scale factor [1.0, 1.0] |
| `in1` | (3, 640, 640) | Normalized image (RGB)  |

### NCNN Model Outputs
| Name   | Shape         | Description               |
| ------ | ------------- | ------------------------- |
| `out0` | (8400, 4)     | Raw boxes in xyxy format  |
| `out1` | (1, 8400)     | Raw scores (person class) |
| `out2` | (128, 80, 80) | Stride-8 feature map      |
| `out3` | (256, 40, 40) | Stride-16 feature map     |

### Preprocessing
```python
# Image normalization (same as ANE model)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Formula: (pixel / 255.0 - MEAN) / STD
mean_vals = [m * 255.0 for m in MEAN]
norm_vals = [1.0 / (s * 255.0) for s in STD]
mat.substract_mean_normalize(mean_vals, norm_vals)
```

---

## 4. Usage Examples

### Basic Detection (No Embeddings)
```bash
python3 pipeline/PP-YOLOE/ncnn_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --out pipeline/output \
    --thresh 0.5
```

### Detection + Embeddings
```bash
python3 pipeline/PP-YOLOE/ncnn_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --out pipeline/output \
    --thresh 0.5 \
    --save-embeddings  # Saves embeddings to .npy file
```

### Custom NMS Parameters
```bash
python3 pipeline/PP-YOLOE/ncnn_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --out pipeline/output \
    --thresh 0.3 \
    --nms-thresh 0.6
```

---

## 5. Output Files

### Visualization
- **File**: `{image_basename}_ncnn.jpg`
- **Format**: JPEG with bounding boxes and scores
- **Box scaling**: Automatically scaled from 640x640 to original image size

### Embeddings (Optional)
- **File**: `{image_basename}_embeddings.npy`
- **Format**: NumPy array, shape `(N, 384)`, dtype `float32`
- **Properties**: L2 normalized (all norms = 1.0)
- **Usage**: Load with `embeddings = np.load('demo_embeddings.npy')`

---

## 6. Benchmark Results

### Test Configuration
- **Image**: 1280x720 (demo.jpg)
- **Model**: PP-YOLOE CRN-S 36e (pruned + surgery)
- **Hardware**: CPU (Apple Silicon)
- **Warmup**: 3 runs

### Performance Metrics
| Metric           | Value      |
| ---------------- | ---------- |
| Inference time   | ~83ms      |
| Detections found | 16 persons |
| Embedding dim    | 384        |
| Total pipeline   | <150ms     |

### Output Statistics
```
Detections: 16
Embedding norms: min=1.0000, max=1.0000, mean=1.0000
Score range: 0.519 - 0.959
```

---

## 7. C++ Porting Guide

### Key Functions to Port

#### 1. NMS Function
```cpp
// Port from apply_nms()
std::vector<Detection> apply_nms(
    const std::vector<Box>& boxes_raw,
    const std::vector<float>& scores_raw,
    float score_thresh = 0.5f,
    float nms_thresh = 0.5f
);
```

#### 2. ROI Pooling Function
```cpp
// Port from roi_align_pool_multi_scale()
std::vector<std::vector<float>> extract_embeddings(
    const ncnn::Mat& feat_s8,        // (128, 80, 80)
    const ncnn::Mat& feat_s16,       // (256, 40, 40)
    const std::vector<Box>& boxes,
    int img_h, int img_w,
    int input_h = 640, int input_w = 640
);
```

### Parameter Values (Must Match Python)
```cpp
// Global pooling
constexpr float GP_W = 0.2f;
constexpr float AVG_W = 1.0f;
constexpr float MAX_W = 0.0f;

// Part-based pooling
constexpr float PP_W = 0.8f;
constexpr int PP_K = 9;              // Horizontal stripes
constexpr int PP_STRIPE_H = 2;       // Sub-stripes (horizontal)
constexpr int PP_VERTICAL_K = 2;     // Vertical stripes
constexpr int PP_VERTICAL_STRIPE_W = 2;  // Sub-stripes (vertical)

// Normalization
constexpr bool USE_INST_NORM = true;
constexpr float PL_ALPHA = 0.35f;
```

### Expected Output Dimensions
```cpp
// Embedding dimension calculation
int s8_channels = 128;
int s16_channels = 256;
int embedding_dim = s8_channels + s16_channels;  // 384

// Output shape: (num_detections, 384)
```

---

## 8. Validation Checklist

To verify C++ implementation matches Python:

### ✅ NMS Validation
- [ ] Same number of detections (16 for demo.jpg at thresh=0.5)
- [ ] Same bounding box coordinates (tolerance: ±1 pixel)
- [ ] Same confidence scores (tolerance: ±0.001)

### ✅ Embedding Validation
- [ ] Embedding shape: (16, 384)
- [ ] All L2 norms = 1.0 (tolerance: ±0.0001)
- [ ] Cosine similarity between Python and C++ embeddings > 0.999

### ✅ Performance Validation
- [ ] Inference time within 20% of Python version
- [ ] Memory usage comparable to Python version
- [ ] No memory leaks (valgrind or similar tools)

---

## 9. Common Issues and Solutions

### Issue 1: Embedding Dimension Mismatch
**Symptom**: C++ embeddings have different dimension than 384
**Solution**: Ensure stride-8 (128 channels) and stride-16 (256 channels) are correctly identified

### Issue 2: L2 Norms Not Equal to 1.0
**Symptom**: Embedding norms are not normalized
**Solution**: Apply L2 normalization after concatenating stride-8 and stride-16 features:
```cpp
float norm = std::sqrt(std::inner_product(emb.begin(), emb.end(), emb.begin(), 0.0f));
if (norm > 1e-6f) {
    for (float& val : emb) val /= norm;
}
```

### Issue 3: Different NMS Results
**Symptom**: C++ finds different number of detections
**Solution**: Verify NMS uses same IoU calculation and threshold (0.5)

### Issue 4: Feature Map Scaling
**Symptom**: Embeddings are very different
**Solution**: Ensure scale factors are correctly computed:
```cpp
float scale_x_s8 = (input_w / (float)img_w) * (feat_w_s8 / (float)input_w);
float scale_y_s8 = (input_h / (float)img_h) * (feat_h_s8 / (float)input_h);
```

---

## 10. References

### Source Files
- **Python NCNN**: `pipeline/PP-YOLOE/ncnn_inference_image.py`
- **Python ANE**: `pipeline/PP-YOLOE/onnx_inference_ane_model.py`
- **Test Script**: `pipeline/PP-YOLOE/simple_ncnn_test.py`

### Related Documentation
- [NCNN Fix Summary](NCNN_FIX_SUMMARY.md)
- [ONNX Export Guide](../../deploy/EXPORT_ONNX_MODEL.md)

---

## Summary

The NCNN implementation (`ncnn_inference_image.py`) now uses **identical algorithms** to the ANE model (`onnx_inference_ane_model.py`) for:

1. ✅ **NMS**: Same cv2.dnn.NMSBoxes approach
2. ✅ **Embedding Extraction**: Multi-scale ROI pooling with same parameters
3. ✅ **Post-processing**: Instance normalization + power-law + L2 normalization

This alignment ensures that the C++ port of the NCNN model will produce **identical results** to the ONNX/ANE version, enabling seamless integration with existing C++ tracking pipelines.

---
**Status**: ✅ Alignment Complete
**Date**: October 4, 2025
**Tested**: Python NCNN implementation verified with 16 detections, 384-dim embeddings
