#!/bin/bash
# onnx/validate_pedestrian.sh
#
# Pedestrian-Specific Re-ID Validation
# ===================================
#
# This script performs specialized validation for pedestrian Re-ID applications,
# focusing on person-to-person distinguishability and counting accuracy. It's
# particularly useful for crowd analysis, surveillance, and pedestrian tracking.
#
# Validation Features:
# 1. Person detection accuracy assessment
# 2. Pedestrian Re-ID embedding quality
# 3. Person-to-person distinguishability metrics
# 4. Duplicate detection and filtering
# 5. Pedestrian counting accuracy
# 6. Real-world scenario testing
#
# Use Cases:
# - Pedestrian counting systems
# - Crowd monitoring applications
# - Person tracking in surveillance
# - Re-identification across camera views
#
# Prerequisites:
# - Backbone model with exported features
# - Test images containing pedestrians
#
# Important: Uses the same feature map as pipeline validation (Concat.3 for C4)
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
if [ ! -f "onnx/backbone/rtdetrv3_r18vd_6x.onnx" ]; then
    echo "❌ Backbone model not found. Please run export_backbone.sh first."
    exit 1
fi

if [ ! -f "onnx/demo/demo.jpg" ]; then
    echo "❌ Demo image not found at onnx/demo/demo.jpg"
    exit 1
fi

echo "🚶 Starting pedestrian-specific Re-ID validation..."
echo "   Model: onnx/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Test image: onnx/demo/demo.jpg"
echo "   Feature map: Concat.3 (C4 layer optimized for person Re-ID)"
echo "   Focus: Person detection and distinguishability"

# Run pedestrian-specific validation
python3 onnx/validate_pedestrian.py \
    --model onnx/backbone/rtdetrv3_r18vd_6x.onnx \
    --image onnx/demo/demo.jpg \
    --feature-map-name Concat.3

# Check validation results
if [ $? -eq 0 ]; then
    echo "✅ Pedestrian Re-ID validation completed successfully"
    echo "   Person detection and Re-ID capabilities validated"
    echo "   Ready for pedestrian tracking applications"
else
    echo "❌ Pedestrian Re-ID validation failed"
    echo "   Check the output above for specific issues"
    exit 1
fi

echo ""
echo "🚶‍♂️ Pedestrian Validation Summary:"
echo "   • Person detection accuracy: Evaluated"
echo "   • Re-ID embedding quality: Tested"
echo "   • Person distinguishability: Validated"
echo "   • Suitable for pedestrian counting and tracking applications"
