# pipeline/PP-YOLOE/ncnn_graph_surgery.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate


python3 pipeline/PP-YOLOE/coreml_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
--input-shape 1,3,640,640 --warmup 20 --runs 80 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--output-model pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.onnx


mkdir -p pipeline/PP-YOLOE/models/ncnn

pnnx pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.onnx

cp pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman.ncnn.bin pipeline/PP-YOLOE/models/
cp pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman.ncnn.param pipeline/PP-YOLOE/models/
