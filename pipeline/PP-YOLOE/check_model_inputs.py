import onnx
model = onnx.load("pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx")
input_names = [inp.name for inp in model.graph.input]
print("Model inputs:", input_names)