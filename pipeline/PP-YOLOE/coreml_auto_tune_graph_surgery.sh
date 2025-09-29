# pipeline/PP-YOLOE/coreml_auto_tune_graph_surgery.sh
#!/usr/bin/env bash

  python3 pipeline/PP-YOLOE/coreml_auto_tune_graph_surgery.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
    --input-shape 1,3,640,640 --ep coreml --runs 20 --warmup 10 \
    --img pipeline/dataset/demo/demo.jpg \
    --split-candidates 10,8,6 --keep-outputs ppyoloe_output1,ppyoloe_output2
