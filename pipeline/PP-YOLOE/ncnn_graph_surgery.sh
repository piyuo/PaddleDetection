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
--fix-input-shapes \
--fold-iterations 15 \
--split-concat 4 \
--fold-static-shapes \
--rewrite-div \
--rewrite-pow \
--rewrite-hardsigmoid \
--rewrite-slice-to-gather \
--rewrite-slice-range-to-gather \
--rewrite-resize-to-static \
--remove-noop-slice \
--rewrite-reduce-to-globalpool \
--fp16 \
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.onnx


mkdir -p pipeline/PP-YOLOE/models/ncnn

cp pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx pipeline/PP-YOLOE/models/ncnn/

cd pipeline/PP-YOLOE/models/ncnn

pnnx ppyoloe_crn_s_36e_pphuman.onnx

cp ppyoloe_crn_s_36e_pphuman.ncnn.bin ../
cp ppyoloe_crn_s_36e_pphuman.ncnn.param ../
