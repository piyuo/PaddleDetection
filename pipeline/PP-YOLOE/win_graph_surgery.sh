# pipeline/PP-YOLOE/win_graph_surgery.sh
#!/usr/bin/env bash

# Pipeline PP-YOLOE Apple Neural Engine Graph Surgery
# This script automatically discovers and removes NMS (NonMaxSuppression) from the model
# to enable full ANE acceleration. It also identifies optimal stride-8 and stride-16
# feature maps for multi-scale embedding extraction.
#
# The script will automatically:
# - Find NMS inputs (boxes, scores)
# - Discover stride-8 feature map for fine-grained embeddings
# - Discover stride-16 feature map for semantic embeddings
# - Print comprehensive output guide for inference script integration
#
# NMS and embedding extraction will be performed in the inference script.

python3 pipeline/PP-YOLOE/ane_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
--input-shape 1,3,640,640 --warmup 20 --runs 80 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--fold-static-shapes \
--fold-iterations 5 \
--rewrite-div \
--remove-noop-slice \
--rewrite-pow \
--rewrite-reduce-to-globalpool \
--rewrite-hardsigmoid \
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_win.onnx

rm -rf pipeline/PP-YOLOE/models/surgery

# --fp16 \
# To enable profiling, add this flag (profiling is automatic when specified):
# --ort-profile-dir pipeline/PP-YOLOE/models/profile \


#Potential Side Effects and Considerations
#The script doesn't have obvious bugs, but the optimizations involve trade-offs you should be aware of.

#1. Numerical Precision and Accuracy

#FP16 Casting (--fp16): The script casts the model's weights and intermediate operations to 16-bit floating-point numbers.

#Side Effect: While FP16 significantly improves performance and reduces model size, it can introduce minor numerical differences. For most models, this has a negligible impact on accuracy, but it's not guaranteed.

#Recommendation: You must validate the accuracy of the final optimized model (ppyoloe_crn_s_36e_pphuman_ane.onnx) against your original model using a representative validation dataset. Check metrics like mean Average Precision (mAP) for detection and embedding retrieval performance to ensure there hasn't been an unacceptable regression.

#2. Loss of Dynamic Shapes

#Static Input Shape (--fix-input-shapes): The script locks the model's input to a fixed size (e.g., 1x3x640x640). This is a requirement for achieving maximum performance on the ANE, as specified by the RequireStaticInputShapes option in the Core ML provider.

#Side Effect: The model can no longer accept images of varying sizes directly. All input images must be pre-processed (e.g., resized and padded) to the exact static shape the model was optimized for.

#Recommendation: This is a standard and necessary trade-off for on-device performance. Just ensure your application's pre-processing pipeline is consistent with the fixed shape.

#3. Model-Specific Heuristics

#auto_discover_outputs: The function that finds the feature maps uses heuristics that are very effective for this PP-YOLOE model but might not generalize perfectly to other architectures without modification.

#Side Effect: If you use this script on a different model, the hardcoded ranges for feature map size (10 <= Hf <= 80), channels (64 <= C <= 512), and stride might not identify the correct layers.

#Recommendation: This is not an issue for your current use case. If you adapt the script for a new model, be prepared to inspect the model architecture (e.g., with Netron) and adjust the filtering logic in auto_discover_outputs.
