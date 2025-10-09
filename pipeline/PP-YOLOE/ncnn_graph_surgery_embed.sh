# pipeline/PP-YOLOE/ncnn_graph_surgery_embed.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

mkdir -p pipeline/PP-YOLOE/models/ncnn

python3 pipeline/PP-YOLOE/ncnn_graph_surgery_embed.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
--input-shape 1,3,640,640 \
--outdir pipeline/PP-YOLOE/models/surgery \
--rewrite-div \
--rewrite-pow \
--rewrite-resize-to-static \
--remove-noop-slice \
--rewrite-reduce-to-globalpool \
--output-model pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_embed_ncnn.onnx

# Note: Disabled --rewrite-slice-to-gather and --rewrite-slice-range-to-gather
# because they interfere with proper graph pruning for the embedding head

# convert onnx to ncnn
pnnx pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_embed_ncnn.onnx fp16=1 optlevel=2 device=gpu

# customized ncnnoptimize to keep out2 and out3
# please reference vision_sdk project to get the customized ncnnoptimize
pipeline/ncnnoptimize pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_embed_ncnn.ncnn.param \
             pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_embed_ncnn.ncnn.bin \
             pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ncnn.param \
             pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ncnn.bin \
             1 keep=embed

./pipeline/PP-YOLOE/ncnn_inference_embed.sh
