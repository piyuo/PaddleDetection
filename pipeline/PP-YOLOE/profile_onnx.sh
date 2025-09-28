# pipeline/PP-YOLOE/profile_onnx.sh
#!/usr/bin/env bash

python3 pipeline/PP-YOLOE/profile_onnx.py \
  --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
  --input-shape 1,3,640,640 \
  --ep coreml \
  --warmup 3 --runs 10