# PP-YOLOE/src/reid/ncnn_inference_reid.sh
#!/usr/bin/env bash

echo "🔧 Running NCNN Detection + NCNN ReID inference..."

# Ensure we are in the root directory
cd "$(dirname "$0")/../../.."

# Paths
DET_PARAM="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_ncnn.param"
DET_BIN="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin"
REID_PARAM="PP-YOLOE/build/models/human_reid.param"
REID_BIN="PP-YOLOE/build/models/human_reid.bin"
IMG="PP-YOLOE/build/dataset/demo/demo.jpg"
OUT="PP-YOLOE/build/output_ncnn_full"

# Run
python3 PP-YOLOE/src/reid/ncnn_inference_reid.py \
    --img "$IMG" \
    --det_param "$DET_PARAM" \
    --det_bin "$DET_BIN" \
    --reid_param "$REID_PARAM" \
    --reid_bin "$REID_BIN" \
    --out "$OUT" \
    --thresh 0.5 \
    --nms-thresh 0.5 \
    --threads 4
