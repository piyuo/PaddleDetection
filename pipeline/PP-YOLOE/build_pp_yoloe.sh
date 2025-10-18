# pipeline/PP-YOLOE/build_pp_yoloe.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

# build all PP-YOLOE models, including:
# ane model (pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ane.onnx)
# ncnn model (pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param + .bin)
# onnx model (pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx)


# export to onnx will create base onnx model from PP-YOLOE Paddle model weights
# output: pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx
pipeline/PP-YOLOE/export_to_onnx.sh \
    --config configs/pphuman/ppyoloe_crn_s_36e_pphuman.yml \
	--weights pipeline/PP-YOLOE/weights/ppyoloe_crn_s_36e_pphuman.pdparams

# Run customized
python3 pipeline/PP-YOLOE/onnx_customize.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
    --output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx

# ane graph surgery to create ANE optimized onnx model
pipeline/PP-YOLOE/ane_graph_surgery.sh \
    --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
    --output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx \
    --img pipeline/dataset/demo/demo.jpg


# Run inference
python3 pipeline/PP-YOLOE/onnx_inference_cust.py \
    --img pipeline/dataset/demo/demo.jpg \
    --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx


# output: pipeline/PP-YOLOE/models/mot_ppyoloe_s_36e_ppvehicle.onnx
pipeline/PP-YOLOE/export_to_onnx.sh \
    --config configs/ppvehicle/mot_ppyoloe_s_36e_ppvehicle.yml \
	--weights pipeline/PP-YOLOE/weights/mot_ppyoloe_s_36e_ppvehicle.pdparams


# build ncnn model
# output: pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param + .bin
./pipeline/PP-YOLOE/build_ncnn_model.sh

# create embedding head on onnx model may slow down the model inference on edge device, cause RoIAlign operator is not well supported on Edge devices,
# event onnx with tensorrt backend may well support RoIAlign operator, we decide not to include embedding head into the base onnx model for now.
# create embeded onnx model for BOT-SORT
# output: pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx
#python pipeline/PP-YOLOE/insert_embedding_head.py \
#                --onnx_in pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
#                --onnx_out pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx
