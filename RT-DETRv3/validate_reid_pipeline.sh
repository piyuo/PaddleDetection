#!/bin/bash
# pipeline/RT-DETRv3/validate_reid_pipeline.sh
#
# Pedestrian Re-ID Pipeline Validator for BoT-SORT Tracking
# ========================================================
#
# This script performs specialized validation of the Re-ID embedding pipeline with focus
# on pedestrian feature embeddings for BoT-SORT multi-object tracking. The validation is
# designed to minimize ID switching by ensuring high-quality, discriminative embeddings.
#
# Pedestrian-Focused Validation Components:
# 1. Person detection quality and reliability
# 2. Pedestrian feature embedding discriminability
# 3. BoT-SORT tracking algorithm compatibility
# 4. Temporal consistency for tracking stability
# 5. Anti-ID-switch metrics and recommendations
#
# Prerequisites:
# - Backbone feature model must exist (run export_backbone.sh first)
# - Demo image with pedestrians must be available for testing
#
# Output: Comprehensive pedestrian tracking validation report with BoT-SORT metrics
#
# Important: Check export_backbone.sh output for the correct feature map name
# Example output: "Exported feature: Concat.3" -> use --feature-map-name Concat.3
#

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/env/bin/activate

# Verify environment activation
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Verify required files exist
if [ ! -f "pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx" ]; then
    echo "❌ Backbone model not found. Please run export_backbone.sh first."
    exit 1
fi

if [ ! -f "pipeline/dataset/demo/demo.jpg" ]; then
    echo "❌ Demo image not found at pipeline/dataset/demo/demo.jpg"
    exit 1
fi

echo "� Starting pedestrian-focused Re-ID pipeline validation for BoT-SORT..."
echo "   Model: pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Test image: pipeline/dataset/demo/demo.jpg"
echo "   Feature map: Concat.3 (C4 layer for RT-DETRv3-R18vd-6x)"
echo "   Focus: Pedestrian tracking with minimal ID switching"
echo "   Output: pipeline/output/validation/pedestrian_reid_report.json"

# Create validation output directory if it doesn't exist
mkdir -p pipeline/output/validation

# Run pedestrian-focused validation with BoT-SORT compatibility testing
python3 pipeline/RT-DETRv3/validate_reid_pipeline.py \
    --model pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx \
    --image pipeline/dataset/demo/demo.jpg \
    --feature-map-name Concat.5 \
    --pedestrian-focused \
    --output pipeline/output/validation/pedestrian_reid_report.json

# Check validation results
if [ $? -eq 0 ]; then
    echo "✅ Pedestrian Re-ID pipeline validation completed successfully"
    echo "   Validation report saved to: pipeline/validation/pedestrian_reid_report.json"
    echo "   Next step: Run botsort_integration_test.sh to test actual tracking"
else
    echo "❌ Pedestrian Re-ID pipeline validation failed"
    echo "   Check the output above for specific error details"
    echo "   Focus on person detection quality and embedding discriminability"
    exit 1
fi

# Display validation summary if report exists
if [ -f "pipeline/output/validation/pedestrian_reid_report.json" ]; then
    echo ""
    echo "📊 Pedestrian Tracking Summary:"
    echo "   🚶 Person Detection Quality: Check detection confidence and bbox quality"
    echo "   🎯 Embedding Discriminability: Inter-person similarity should be < 0.3"
    echo "   🤖 BoT-SORT Compatibility: Overall score should be > 0.7"
    echo "   ⏱️ Temporal Consistency: Should be > 0.95 for stable tracking"
    echo "   📋 Full metrics available in the JSON report"
fi
