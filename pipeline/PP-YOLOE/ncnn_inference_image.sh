# pipeline/PP-YOLOE/ncnn_inference_image.sh
#!/usr/bin/env bash

echo "🔧 Running NCNN inference..."

# Use python from the (possibly) activated venv
python3 pipeline/PP-YOLOE/ncnn_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --out pipeline/output \
    --thresh 0.5 \
    --save-embeddings \
    --nms-thresh 0.5
