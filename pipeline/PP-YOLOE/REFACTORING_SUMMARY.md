# PP-YOLOE ONNX Optimization Workflow - Refactoring Summary

## Overview

The `ane_graph_surgery.py` script has been refactored into three focused scripts, each handling a specific part of the model optimization pipeline:

1. **onnx_customize.py** - Model customization (NMS removal, feature output selection)
2. **ane_graph_surgery.py** - CoreML ANE-specific optimizations (graph rewrites)
3. **onnx_cleanup.py** - Final cleanup and simplification (optimization passes)

## New Workflow

```
export_to_onnx.sh
    ↓
ppyoloe_crn_s_36e_pphuman.onnx (base model)
    ↓
onnx_customize.py --auto-discover
    ↓
ppyoloe_crn_s_36e_pphuman_cust.onnx (customized: NMS removed, features added)
    ↓
ane_graph_surgery.py (ANE optimizations)
    ↓
ppyoloe_crn_s_36e_pphuman_cust_ane.onnx (ANE-optimized)
    ↓
onnx_cleanup.py (final cleanup)
    ↓
ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx (final model)
```

## Script Details

### 1. onnx_customize.py

**Purpose**: Transform base model to customized version with selected outputs

**Features**:
- Automatically discovers NMS nodes and their inputs (boxes, scores)
- Finds optimal stride-8 feature map (fine-grained, ~80×80)
- Finds optimal stride-16 feature map (semantic, ~40×40)
- Prunes model to keep only selected outputs (removes NMS)
- Prints comprehensive usage guide for outputs

**Usage**:
```bash
python3 pipeline/PP-YOLOE/onnx_customize.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
    --output pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
    --input-shape 1,3,640,640 \
    --auto-discover
```

**Manual Output Selection** (instead of auto-discover):
```bash
python3 pipeline/PP-YOLOE/onnx_customize.py \
    --model model.onnx \
    --output model_cust.onnx \
    --keep-outputs "boxes_output,scores_output,feature_map_s8,feature_map_s16"
```

### 2. ane_graph_surgery.py (Refactored)

**Purpose**: Apply CoreML ANE-specific graph optimizations

**Features** (ANE optimizations only):
- Fix input shapes to static (required for ANE)
- Split large Concat nodes
- Fold static shape computation chains
- Rewrite Div → Mul with reciprocal
- Rewrite Pow patterns (x², √x, 1/x, x¹ → Identity)
- Rewrite HardSigmoid → Mul+Add+Clip
- Rewrite Slice → Gather
- Rewrite dynamic Resize → static sizes
- Rewrite Reduce → GlobalPool
- Remove no-op Slice operations
- Optional FP16 casting
- ANE compatibility analysis and profiling

**What was REMOVED** (moved to other scripts):
- ❌ NMS discovery and removal → onnx_customize.py
- ❌ Feature map discovery → onnx_customize.py
- ❌ Graph pruning → onnx_customize.py
- ❌ onnxoptimizer passes → onnx_cleanup.py
- ❌ onnx-simplifier → onnx_cleanup.py
- ❌ Tensor cleanup → onnx_cleanup.py

**Usage**:
```bash
python3 pipeline/PP-YOLOE/ane_graph_surgery.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
    --input-shape 1,3,640,640 \
    --warmup 20 --runs 80 \
    --img pipeline/dataset/demo/demo.jpg \
    --outdir pipeline/PP-YOLOE/models/surgery \
    --fix-input-shapes \
    --fold-iterations 15 \
    --split-concat 4 \
    --fold-static-shapes \
    --rewrite-div \
    --rewrite-pow \
    --rewrite-hardsigmoid \
    --rewrite-slice-to-gather \
    --rewrite-slice-range-to-gather \
    --rewrite-resize-to-static \
    --remove-noop-slice \
    --rewrite-reduce-to-globalpool \
    --output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx
```

### 3. onnx_cleanup.py

**Purpose**: Final model cleanup and optimization

**Features**:
- Shape inference
- onnxoptimizer passes (eliminate dead nodes, fuse operations)
- onnx-simplifier (constant folding, simplification)
- Clean unused tensors and initializers

**Usage**:
```bash
python3 pipeline/PP-YOLOE/onnx_cleanup.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx \
    --output pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx
```

**Skip certain steps**:
```bash
python3 pipeline/PP-YOLOE/onnx_cleanup.py \
    --model model.onnx \
    --output model_clean.onnx \
    --skip-optimizer \      # Skip onnxoptimizer
    --skip-simplifier \     # Skip onnx-simplifier
    --skip-cleanup \        # Skip tensor cleanup
    --shape-inference       # Add shape inference
```

## Complete Workflow Script

Use `ane_graph_surgery.sh` to run all three steps automatically:

```bash
./pipeline/PP-YOLOE/ane_graph_surgery.sh
```

This script:
1. Checks for base model existence
2. Runs onnx_customize.py to create customized model
3. Runs ane_graph_surgery.py for ANE optimizations
4. Runs onnx_cleanup.py for final cleanup
5. Reports success and generated models

## Benefits of Refactoring

### Separation of Concerns
- **onnx_customize.py**: Domain-specific customization (PP-YOLOE specific)
- **ane_graph_surgery.py**: Platform-specific optimization (CoreML ANE)
- **onnx_cleanup.py**: Generic ONNX optimization (reusable)

### Reusability
- `onnx_customize.py` can be adapted for other detection models
- `onnx_cleanup.py` can be used for any ONNX model
- `ane_graph_surgery.py` is focused on ANE compatibility

### Maintainability
- Each script has a single, clear purpose
- Easier to debug and modify
- Functions are organized logically

### Flexibility
- Can run scripts independently
- Can skip certain optimizations
- Can customize each step's parameters

## Migration from Old Workflow

**Old workflow** (single script):
```bash
./pipeline/PP-YOLOE/ane_graph_surgery.sh  # Did everything in one script
```

**New workflow** (three scripts):
```bash
./pipeline/PP-YOLOE/ane_graph_surgery.sh  # Now orchestrates three scripts
```

The shell script interface remains the same, but now it's modular!

## File Naming Convention

| Stage       | Input                     | Output                       | Description                              |
| ----------- | ------------------------- | ---------------------------- | ---------------------------------------- |
| Export      | -                         | `*_pphuman.onnx`             | Base model from PaddleDetection          |
| Customize   | `*_pphuman.onnx`          | `*_pphuman_cust.onnx`        | Customized (NMS removed, features added) |
| ANE Surgery | `*_pphuman_cust.onnx`     | `*_pphuman_cust_ane.onnx`    | ANE-optimized                            |
| Cleanup     | `*_pphuman_cust_ane.onnx` | `*_pphuman_cust_ane_cu.onnx` | Final cleaned model                      |

## Testing the Refactored Scripts

1. Ensure base model exists:
```bash
ls pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx
```

2. Test customization:
```bash
python3 pipeline/PP-YOLOE/onnx_customize.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
    --output /tmp/test_cust.onnx \
    --input-shape 1,3,640,640 \
    --auto-discover
```

3. Test ANE surgery:
```bash
python3 pipeline/PP-YOLOE/ane_graph_surgery.py \
    --model /tmp/test_cust.onnx \
    --input-shape 1,3,640,640 \
    --img pipeline/dataset/demo/demo.jpg \
    --outdir /tmp/surgery \
    --fix-input-shapes \
    --fold-static-shapes \
    --output-model /tmp/test_ane.onnx
```

4. Test cleanup:
```bash
python3 pipeline/PP-YOLOE/onnx_cleanup.py \
    --model /tmp/test_ane.onnx \
    --output /tmp/test_final.onnx
```

## Backup Files

Original files have been backed up:
- `ane_graph_surgery.py.backup` - Original monolithic script
- `ane_graph_surgery.sh.backup` - Original shell script
- `ane_graph_surgery_old.py` - Intermediate version

You can restore these if needed.

## Summary

The refactoring successfully separates the model optimization pipeline into three focused, reusable scripts while maintaining the same high-level workflow through the shell script. Each script can now be used independently or as part of the complete pipeline.
