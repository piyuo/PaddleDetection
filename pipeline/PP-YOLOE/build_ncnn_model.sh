# pipeline/PP-YOLOE/build_ncnn_model.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

mkdir -p pipeline/PP-YOLOE/models/ncnn

python3 pipeline/PP-YOLOE/build_ncnn_model.py \
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

# convert onnx to ncnn
pnnx pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.onnx fp16=1 optlevel=2 device=gpu

# customized ncnnoptimize to keep out2 and out3
# please reference vision_sdk project to get the customized ncnnoptimize
pipeline/ncnnoptimize pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
             pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
             pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param \
             pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin \
             1 keep=out2,out3
#./pipeline/PP-YOLOE/ncnn_inference_image.sh


#cp pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin
#cp pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param
