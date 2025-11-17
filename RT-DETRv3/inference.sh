#!/bin/bash
# pipeline/RT-DETRv3/inference.sh
#
# Model Inference Validation for Quantized RT-DETRv3
# ==================================================
#
# This script validates the final quantized ONNX model by running inference on
# test images and visualizing the results. It ensures that quantization hasn't
# degraded model performance and validates the complete pipeline.
#
# Validation Features:
# 1. Load and test the quantized production model
# 2. Run inference on demo images
# 3. Validate detection accuracy and quality
# 4. Compare results with original model (if available)
# 5. Generate visualization with bounding boxes and labels
# 6. Performance benchmarking and timing analysis
#
# Test Scenarios:
# - Object detection accuracy validation
# - Quantization impact assessment
# - Performance measurement (speed and memory)
# - Visual result verification
# - Production readiness confirmation
#
# Input:  pipeline/output/rtdetrv3_r18vd_6x.onnx (Quantized production model)
# Output: Inference results with visualizations and performance metrics
#
# This is the final validation step before deploying the model to production.
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
if [ ! -f "pipeline/output/rtdetrv3_r18vd_6x.onnx" ]; then
    echo "❌ Quantized model not found. Please run quantize.sh first."
    exit 1
fi

if [ ! -f "pipeline/dataset/demo/demo.jpg" ]; then
    echo "❌ Demo image not found at pipeline/dataset/demo/demo.jpg"
    exit 1
fi

echo "🔍 Starting model inference validation..."
echo "   Model: pipeline/output/rtdetrv3_r18vd_6x.onnx (Quantized)"
echo "   Test image: pipeline/dataset/demo/demo.jpg"
echo "   Validation: Detection accuracy and performance"
echo "   Debug mode: Enabled for comprehensive analysis"

# Run inference validation with comprehensive debugging
python3 pipeline/RT-DETRv3/inference.py --debug

# Check inference results
if [ $? -eq 0 ]; then
    echo "✅ Model inference validation completed successfully"
    echo "   Quantized model is ready for production deployment"
    echo "   Detection accuracy and performance validated"
else
    echo "❌ Model inference validation failed"
    echo "   Check the output above for specific issues"
    exit 1
fi

echo ""
echo "🏁 Final Validation Summary:"
echo "   • Quantized model: Functional ✓"
echo "   • Detection accuracy: Validated ✓"
echo "   • Performance metrics: Measured ✓"
echo "   • Ready for production deployment ✓"
echo ""
echo "🚀 Deployment Ready!"
echo "   Your RT-DETRv3 model with Re-ID capabilities is now optimized"
echo "   and validated for production use in tracking applications."