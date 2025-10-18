# pipeline/PP-YOLOE/ane_graph_surgery.sh
#!/usr/bin/env bash

# Pipeline PP-YOLOE Apple Neural Engine Graph Surgery

python3 pipeline/PP-YOLOE/ane_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
--input-shape 1,3,640,640 --warmup 20 --runs 80 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx

rm -rf pipeline/PP-YOLOE/models/surgery
