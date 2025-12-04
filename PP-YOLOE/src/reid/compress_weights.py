import onnx
from onnx import helper, TensorProto
import numpy as np
import sys

def compress_weights_to_fp16(model_path, output_path):
    print(f"Compressing weights to FP16 for model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    new_initializers = []
    new_nodes = []

    # Track names of initializers we've processed
    processed_inits = set()

    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Get data
            if init.raw_data:
                data = np.frombuffer(init.raw_data, dtype=np.float32)
            else:
                data = np.array(init.float_data, dtype=np.float32)

            # Convert to FP16
            data_fp16 = data.astype(np.float16)

            # Create new initializer name
            fp16_name = init.name + "_fp16_compressed"

            # Create new FP16 initializer
            new_init = onnx.helper.make_tensor(
                name=fp16_name,
                data_type=TensorProto.FLOAT16,
                dims=init.dims,
                vals=data_fp16.tobytes(),
                raw=True
            )
            new_initializers.append(new_init)

            # Create Cast node (FP16 -> FP32)
            # Input: fp16_name
            # Output: init.name (so existing nodes consume this)
            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[fp16_name],
                outputs=[init.name],
                to=TensorProto.FLOAT
            )
            new_nodes.append(cast_node)

            processed_inits.add(init.name)
        else:
            # Keep non-float initializers as is
            new_initializers.append(init)

    # Replace initializers
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    # Prepend Cast nodes to the graph
    # We insert them at the beginning to ensure they are available
    # (Topological sort usually handles this, but prepending is safe)
    for node in reversed(new_nodes):
        graph.node.insert(0, node)

    onnx.save(model, output_path)
    print(f"Saved compressed model to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 compress_weights.py <input> <output>")
        sys.exit(1)

    compress_weights_to_fp16(sys.argv[1], sys.argv[2])
