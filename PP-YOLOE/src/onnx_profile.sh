# PP-YOLOE/src/onnx_profile.sh
#!/usr/bin/env bash

python3 PP-YOLOE/src/onnx_profile.py \
  --model PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
  --img PP-YOLOE/build/dataset/demo/demo.jpg \
  --input-shape 1,3,640,640 \
  --ep coreml \
  --warmup 3 --runs 10 \
  --use-demo