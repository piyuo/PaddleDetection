#!/bin/bash
# pipeline/RT-DETRv3/reid_embeddings.sh
#
# Re-ID Embeddings Generator for RT-DETRv3
# ========================================
#
# This script generates Re-ID embeddings from input images using the RT-DETRv3 model
# with extracted backbone features. It provides a complete pipeline for embedding
# generation with comprehensive debugging and validation output.
#
# Features:
# 1. Robust image preprocessing with letterbox support
# 2. Object detection with confidence filtering
# 3. Backbone feature extraction for detected objects
# 4. L2-normalized embedding generation (2048-dimensional: 512 channels x 2x2 adaptive pooling)
# 5. Quality validation and metrics
# 6. Comprehensive debugging output
# 7. Visualization and analysis tools
#
# Processing Pipeline:
# 1. Load and preprocess input image (letterbox or resize)
# 2. Run RT-DETRv3 inference for object detection
# 3. Extract backbone features for each detected object
# 4. Generate normalized Re-ID embeddings
# 5. Validate embedding quality and consistency
# 6. Save results with detailed metadata
#
# Output: Re-ID embeddings saved in  pipeline/output/reid/ with visualization
#
# Use Cases:
# - Multi-object tracking systems
# - Object re-identification across cameras
# - Surveillance and monitoring applications
# - Research and development of tracking algorithms
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

echo "🎯 Starting Re-ID embedding generation..."
echo "   Model: pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx"
echo "   Input image: pipeline/dataset/demo/demo.jpg"
echo "   Output directory:  pipeline/output/reid/"
echo "   Preprocessing: Letterbox (preserves aspect ratio)"
echo "   Debug mode: Enabled (comprehensive output)"

# Create output directory if it doesn't exist
mkdir -p pipeline/output/reid

# Generate Re-ID embeddings with letterbox preprocessing and comprehensive debugging
python3 pipeline/RT-DETRv3/reid_embeddings.py \
    --model pipeline/RT-DETRv3/backbone/rtdetrv3_r18vd_6x.onnx \
    --image pipeline/dataset/demo/demo.jpg \
    --feature-map-name Concat.5 \
    --output  pipeline/output/reid \
    --use-letterbox \
    --debug

# Check generation results
if [ $? -eq 0 ]; then
    echo "✅ Re-ID embedding generation completed successfully"
    echo "   Embeddings and metadata saved to:  pipeline/output/reid/"
    echo "   Check the debug output above for quality metrics"
else
    echo "❌ Re-ID embedding generation failed"
    echo "   Check the output above for specific errors"
    exit 1
fi

echo ""
echo "📊 Generation Summary:"
echo "   • Object detection: Completed"
echo "   • Feature extraction: Successful"
echo "   • Embedding generation: Validated"
echo "   • Quality metrics: Calculated"
echo "   • Ready for tracking applications"
