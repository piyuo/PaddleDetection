#!/bin/bash
# Simple test of automatic output discovery

cd /Users/cc/Dropbox/PaddleDetection

python3 pipeline/PP-YOLOE/coreml_graph_surgery.py \
  --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
  --input-shape 1,3,640,640 \
  --ep coreml \
  --warmup 5 \
  --runs 10 \
  --img pipeline/dataset/demo/demo.jpg \
  --outdir /tmp/surgery_test \
  --fix-input-shapes \
  --fp16 \
  --output-model /tmp/ppyoloe_auto_ane.onnx

echo ""
echo "=== Test Complete ==="
echo "Check output above for 'Automatic Output Discovery' section"
