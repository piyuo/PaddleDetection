#!/bin/bash
# pipeline/RT-DETRv3/export_backbone.sh
#
# RT-DETRv3 Backbone Feature Exporter
# ===================================
#
# This script extracts backbone feature maps from the original RT-DETRv3 ONNX model
# and creates a new model with accessible backbone outputs for Re-ID embedding generation.
#
# Process:
# 1. Analyzes the original model structure to identify feature map candidates
# 2. Automatically selects the optimal backbone level (C4 recommended)
# 3. Exports a modified model with backbone feature outputs accessible
# 4. Validates the exported features meet quality requirements
#
# Input:  pipeline/input/rtdetrv3_r18vd_6x.onnx  (Original RT-DETRv3 model)
# Output: pipeline/backbone/rtdetrv3_r18vd_6x.onnx (Model with backbone features)
#
# Usage Examples:
#   ./pipeline/RT-DETRv3/export_backbone.sh                    # Export C4 features (recommended)
#   ./pipeline/RT-DETRv3/export_backbone.sh --list-only        # List available feature candidates
#   ./pipeline/RT-DETRv3/export_backbone.sh --level C3         # Export C3 features (higher resolution)
#

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/env/bin/activate

# Verify environment activation
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "🚀 Starting backbone feature extraction..."
echo "   Input model: pipeline/RT-DETRv3/input/rtdetrv3_r18vd_6x.onnx"
echo "   Output model: pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Feature level: C3 (optimal for Re-ID)"

# Run the backbone export script with optimized parameters
python3 pipeline/RT-DETRv3/export_backbone.py \
    --input pipeline/RT-DETRv3/input/rtdetrv3_r18vd_6x.onnx \
    --output pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx \
    --level C3

# Optional: List available feature candidates without exporting
# Add --list-only flag to see all available backbone feature options

# Check if export was successful
if [ $? -eq 0 ]; then
    echo "✅ Backbone feature extraction completed successfully"
    echo "   Next step: Run validate_reid_pipeline.sh to validate the exported features"
else
    echo "❌ Backbone feature extraction failed"
    exit 1
fi
