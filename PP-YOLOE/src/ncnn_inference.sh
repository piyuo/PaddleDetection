# PP-YOLOE/src/ncnn_inference.sh
#!/usr/bin/env bash

echo "🔧 Running NCNN inference..."

# Use python from the (possibly) activated venv
python3 PP-YOLOE/src/ncnn_inference.py \
    --img PP-YOLOE/build/dataset/demo/demo.jpg \
    --ncnn_param PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_ncnn.param \
    --ncnn_bin PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin \
    --out PP-YOLOE/build/output \
    --thresh 0.5 \
    --save-embeddings \
    --nms-thresh 0.5 \
    --threads 4 \
    --warmup 3
