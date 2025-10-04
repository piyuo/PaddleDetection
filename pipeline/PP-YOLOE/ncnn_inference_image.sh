# pipeline/PP-YOLOE/ncnn_inference_image.sh
#!/usr/bin/env bash

echo "🔧 Running NCNN inference..."

# Use python from the (possibly) activated venv
python3 pipeline/PP-YOLOE/ncnn_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.ncnn.bin \
    --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.ncnn.param \
    --out pipeline/output \
    --thresh 0.5