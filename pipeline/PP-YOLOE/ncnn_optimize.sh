# pipeline/PP-YOLOE/ncnn_optimize.sh
#!/usr/bin/env bash

echo "🔧 Optimizing NCNN model with feature map preservation..."
echo "   Preserving stride-8 (blob 72, 128ch) and stride-16 (blob 114, 256ch) features"
echo "   Applying safe optimizations (dropout, noop, split, orphaned memorydata)"
echo ""

python3 pipeline/PP-YOLOE/ncnn_optimize.py \
    --input-param pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --input-bin pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --output-param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param \
    --output-bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin \
    --keep "72,114" \
    --verbose
