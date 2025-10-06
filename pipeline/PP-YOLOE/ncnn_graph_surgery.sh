# pipeline/PP-YOLOE/ncnn_graph_surgery.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

mkdir -p pipeline/PP-YOLOE/models/ncnn

python3 pipeline/PP-YOLOE/ncnn_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
--input-shape 1,3,640,640 --warmup 20 --runs 80 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--rewrite-div \
--rewrite-pow \
--rewrite-slice-to-gather \
--rewrite-slice-range-to-gather \
--rewrite-resize-to-static \
--remove-noop-slice \
--rewrite-reduce-to-globalpool \
--output-model pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.onnx

pnnx pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.onnx fp16=1 optlevel=2 device=gpu

# mark out2 and out3 as outputs in the .param file before ncnnoptimize. it'll prevent them from being pruned away.

FILE_PATH="/Users/cc/Dropbox/PaddleDetection/pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param"

LINE1="Output                   out2                     1 1 backbone_output_2"
LINE2="Output                   out3                     1 1 backbone_output_3"


if ! grep -qF "$LINE1" "$FILE_PATH"; then
    echo "$LINE1" >> "$FILE_PATH"
    echo "Appended: $LINE1"
else
    echo "Line already exists: $LINE1"
fi

if ! grep -qF "$LINE2" "$FILE_PATH"; then
    echo "$LINE2" >> "$FILE_PATH"
    echo "Appended: $LINE2"
else
    echo "Line already exists: $LINE2"
fi

python3 pipeline/PP-YOLOE/ncnn_optimize.py \
    --input-param pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --input-bin pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --output-param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param \
    --output-bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin \
    --keep p2o.pd_op.batch_norm_.13.0,p2o.pd_op.batch_norm_.19.0

ncnnoptimize pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
             pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
             pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param \
             pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin \
             1
./pipeline/PP-YOLOE/ncnn_inference_image.sh


cp pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin
cp pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param
