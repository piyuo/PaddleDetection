# pipeline/PP-YOLOE/analyze_best_model.sh
#!/bin/bash
# Analyze the best-performing model from auto-tuning to identify CoreML bottlenecks

# From your tuning results, the best model was:
# trial_000 with --rewrite-hardsigmoid --rewrite-slice-to-gather --fold-static-shapes --rewrite-div --fp16

BEST_MODEL="pipeline/output/auto_tune/ppyoloe_crn_s_36e_pphuman_embed/stage11_fp16_1/ppyoloe_crn_s_36e_pphuman_embed_final.onnx"
TEST_IMAGE="pipeline/dataset/demo/demo.jpg"
OUTPUT_JSON="pipeline/output/coreml_partition_analysis.json"

echo "==================================="
echo "CoreML Partition Analysis"
echo "==================================="
echo ""
echo "Analyzing best model: $BEST_MODEL"
echo "Performance: 45.08ms (47.5% improvement)"
echo "CoreML nodes: 7 out of 1222 total"
echo ""

if [ ! -f "$BEST_MODEL" ]; then
    echo "Error: Best model not found at $BEST_MODEL"
    echo "Please run your auto-tuner first or adjust the path"
    exit 1
fi

if [ ! -f "$TEST_IMAGE" ]; then
    echo "Error: Test image not found at $TEST_IMAGE"
    exit 1
fi

# Run the analyzer
python3 pipeline/PP-YOLOE/analyze_coreml_partitions.py \
    --model "$BEST_MODEL" \
    --img "$TEST_IMAGE" \
    --output "$OUTPUT_JSON" \
    --verbose

echo ""
echo "==================================="
echo "Analysis complete!"
echo "Detailed JSON saved to: $OUTPUT_JSON"
echo ""
echo "Next steps to reach 29ms target:"
echo "  1. Identify the 7 CoreML ops - are they Conv/BatchNorm?"
echo "  2. Find what CPU ops immediately follow them"
echo "  3. Consider splitting model at partition boundary"
echo "  4. Focus optimization on the CPU-bound sections"
echo "==================================="