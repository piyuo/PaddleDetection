# pipeline/PP-YOLOE/auto_tune_graph_surgery.sh
#!/usr/bin/env bash

  python3 pipeline/PP-YOLOE/auto_tune_graph_surgery.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
    --input-shape 1,3,640,640 --ep coreml --runs 10 --warmup 3 \
    --split-candidates 10,8,6 --keep-outputs ppyoloe_output1,ppyoloe_output2
