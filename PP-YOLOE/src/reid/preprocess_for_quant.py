import onnx
import sys
import os

def constants_to_initializers(model_path, output_path):
    print(f"Preprocessing model: {model_path}")
    model = onnx.load(model_path)

    # Create a mapping of constant node outputs to the constant node itself
    constant_nodes = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for output in node.output:
                constant_nodes[output] = node

    new_nodes = []
    new_initializers = []

    # Iterate through nodes and keep non-Constant nodes
    for node in model.graph.node:
        if node.op_type == "Constant":
            # Convert Constant to Initializer
            tensor = node.attribute[0].t
            tensor.name = node.output[0] # Ensure name matches output
            new_initializers.append(tensor)
        else:
            new_nodes.append(node)

    # Update graph
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)

    onnx.save(model, output_path)
    print(f"Saved preprocessed model to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 preprocess_for_quant.py <input_model> <output_model>")
        sys.exit(1)

    constants_to_initializers(sys.argv[1], sys.argv[2])
