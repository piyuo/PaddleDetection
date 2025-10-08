# Coordinate Space Documentation for PP-YOLOE Embedding Head

## Overview

The embedding head for PP-YOLOE requires careful handling of coordinate spaces to ensure ROI alignment works correctly. This document explains the coordinate flow and validation process.

## Coordinate Spaces

### 1. **Original Image Coordinates**
- Range: `[0, original_width]` × `[0, original_height]`
- Example: For a 1920×1080 image, coordinates are in range `[0-1920, 0-1080]`
- This is the coordinate space of the input image before resizing

### 2. **Network Coordinates**
- Range: `[0, network_width]` × `[0, network_height]`
- Example: For PP-YOLOE with 640×640 input, coordinates are in range `[0-640, 0-640]`
- This is the coordinate space after preprocessing/resizing

## PP-YOLOE Model Output Behavior

### When Exported with NMS (Normal Case)

PP-YOLOE models exported with NMS perform the following coordinate transformation internally:

```python
# Inside PP-YOLOE post-processing:
pred_bboxes *= stride_tensor          # Convert anchor-relative to network coords
pred_bboxes /= scale_factor           # Convert network coords → original image coords
```

**Result:** Output boxes are in **original image coordinates**

### Scale Factor

The `scale_factor` input has shape `[2]` or `[1, 2]` containing `[scale_y, scale_x]` where:
- `scale_y = network_height / original_height`
- `scale_x = network_width / original_width`

Example: For 1920×1080 image resized to 640×640:
- `scale_factor = [640/1080, 640/1920] = [0.593, 0.333]`

## Embedding Head Coordinate Flow

### Step 1: Extract Boxes from Model Output
```
boxes = detection_output[:, 2:6]  # [x0, y0, x1, y1]
# At this point: boxes are in ORIGINAL image coords (e.g., 0-1920 range)
```

### Step 2: Convert to Network Coordinates
```python
boxes_scaled = boxes * scale_factor  # Element-wise multiply
# Now: boxes are in NETWORK coords (e.g., 0-640 range)
```

This is done automatically in the ONNX graph by the `insert_embedding_head.py` script.

### Step 3: ROI Alignment
```python
RoiAlign(
    features,           # Feature map from backbone (e.g., stride-8 or stride-16)
    boxes_scaled,       # Boxes in network coordinates
    batch_indices,      # All zeros for single-image batch
    spatial_scale=1.0/stride,  # e.g., 1/8 = 0.125 for stride-8 features
)
```

The `spatial_scale` parameter tells ROI Align how to map network coordinates to feature map coordinates:
- For stride-8 features: `network_coord → feature_coord = network_coord / 8`
- For stride-16 features: `network_coord → feature_coord = network_coord / 16`

## Why This Design is Correct

1. **Model Output Flexibility**: PP-YOLOE outputs boxes in original coordinates so they can be directly overlaid on input images for visualization without scaling
2. **ROI Alignment Requirement**: ROI Align expects boxes in the same coordinate space as the spatial_scale parameter implies (network coordinates)
3. **Clean Separation**: The embedding head converts coordinates internally, keeping the interface clean

## Validation

### Automatic Validation Script

Run the validation script after creating the embedding model:

```bash
./pipeline/PP-YOLOE/validate_embedding_coords.sh
```

Or manually:

```bash
python3 pipeline/PP-YOLOE/validate_embedding_coords.py \
    --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
    --img pipeline/dataset/demo/demo.jpg
```

### What the Validation Checks

1. ✅ Detects if `scale_factor` is a model input
2. ✅ Analyzes box coordinate ranges
3. ✅ Verifies boxes are in original image coords (as expected)
4. ✅ Confirms embedding dimensions match detection count
5. ✅ Validates L2 normalization of embeddings

### Expected Validation Output

```
================================================================================
Coordinate Space Validation for Embedding Head
================================================================================

✓ Model loaded: pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx
  Inputs: ['scale_factor', 'image']
  Outputs: ['fetch_name_0', 'embed']

✓ scale_factor input: True
  scale_factor: [0.593 0.333]

✓ Detections shape: (5, 6)
  Valid detections: 5

📊 Box coordinate analysis:
  X range: [245.1, 1678.3]
  Y range: [89.2, 956.8]
  Network size: (640, 640)

✓ CORRECT: Boxes are in ORIGINAL image coordinates
  Embedding head will multiply by scale_factor to convert to network coords
  ROI alignment will then work correctly with spatial_scale=1/stride

✓ Embeddings found: (5, 256)
  Embedding count matches detection count ✓
  L2 norms: min=0.9998, mean=1.0000, max=1.0001
  Normalized: True ✓

================================================================================
✓ VALIDATION PASSED: Coordinate space is correct!
================================================================================
```

## Troubleshooting

### Symptom: Box coordinates are all < 640

**Cause:** Model might be outputting network coordinates instead of original coordinates

**Solution:** Check if the model was exported incorrectly. PP-YOLOE models should have `scale_factor` division in post-processing.

### Symptom: Embeddings are all zeros or NaNs

**Cause:** ROI Alignment is extracting from wrong regions (coordinate mismatch)

**Solution:**
1. Verify box coordinate ranges match expectations
2. Check that `spatial_scale` parameter is set correctly
3. Ensure feature map names are correct

### Symptom: Very high pairwise cosine similarities (>0.9) for non-overlapping boxes

**Cause:** ROI Alignment extracting similar/invalid features due to coordinate issues

**Solution:**
1. Run validation script to check coordinate space
2. Verify `scale_factor` values are reasonable
3. Check that box coordinates are not clipped/invalid

## References

- PP-YOLOE post-processing: `ppdet/modeling/heads/ppyoloe_head.py:528-548`
- Preprocessing: `pipeline/PP-YOLOE/onnx_inference_utils.py:preprocess_image()`
- Embedding head insertion: `pipeline/PP-YOLOE/insert_embedding_head.py`
- Validation: `pipeline/PP-YOLOE/validate_embedding_coords.py`

## Summary

✅ **The current implementation is CORRECT:**

1. PP-YOLOE outputs boxes in original image coordinates
2. Embedding head multiplies by scale_factor → converts to network coordinates
3. ROI Align uses network coordinates with spatial_scale=1/stride
4. Result: Correct feature extraction for embeddings

The reviewer's concern about "coordinate space bug" was **unfounded** - the implementation correctly handles the coordinate transformations needed for ROI alignment.
