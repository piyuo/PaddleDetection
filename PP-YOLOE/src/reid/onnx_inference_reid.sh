# PP-YOLOE/src/reid/onnx_inference_reid.sh
#!/usr/bin/env bash

echo "🔧 Running ONNX Object Detection + ReID inference..."

# Use python from the (possibly) activated venv
python3 PP-YOLOE/src/reid/onnx_inference_reid.py \
    --img PP-YOLOE/build/dataset/demo/demo.jpg \
    --det_onnx PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx \
    --reid_onnx PP-YOLOE/build/models/human_reid.onnx \
    --out PP-YOLOE/build/output \
    --thresh 0.5
