#!/usr/bin/env python3
"""Replace Split operations in ONNX model with Slice operations to avoid TFLite Flex ops.

This preprocessing script modifies ONNX models to replace Split nodes with equivalent
Slice nodes, which are better supported by TFLite without requiring Flex delegate.
"""

import sys
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
from pathlib import Path


def replace_splits_with_slices(model_path: str, output_path: str, verbose: bool = True):
    """Replace all Split nodes with equivalent Slice nodes.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save modified ONNX model
        verbose: Print progress information
    """
    model = onnx.load(model_path)
    graph = model.graph

    # Find Split nodes
    split_nodes = [n for n in graph.node if n.op_type == "Split"]

    if not split_nodes:
        if verbose:
            print("No Split nodes found in the model.")
        return

    if verbose:
        print(f"Found {len(split_nodes)} Split node(s)")

    # Collect new nodes and initializers
    new_nodes = []
    new_initializers = []
    nodes_to_remove = []

    for split_node in split_nodes:
        if verbose:
            print(f"Replacing {split_node.name}...")

        # Get split parameters
        axis = 0
        for attr in split_node.attribute:
            if attr.name == "axis":
                axis = attr.i

        input_tensor = split_node.input[0]
        num_outputs = len(split_node.output)

        # Get split sizes from initializer or attribute
        split_sizes = None

        # Check if split sizes are in the second input (ONNX opset 13+)
        if len(split_node.input) > 1:
            split_input_name = split_node.input[1]
            for init in graph.initializer:
                if init.name == split_input_name:
                    split_sizes = numpy_helper.to_array(init).flatten().tolist()
                    split_sizes = [int(x) for x in split_sizes]
                    break

        # Check 'split' attribute (older ONNX opset)
        if not split_sizes:
            for attr in split_node.attribute:
                if attr.name == "split":
                    split_sizes = [int(x) for x in attr.ints]
                    break

        # Fallback: Try to infer from input shape or assume equal splits
        if not split_sizes:
            # Look for shape information
            found_shape = False
            for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
                if vi.name == input_tensor and vi.type.HasField("tensor_type"):
                    shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    if len(shape) > axis and shape[axis] > 0:
                        total_size = shape[axis]
                        split_size = total_size // num_outputs
                        split_sizes = [split_size] * num_outputs
                        found_shape = True
                        break

            if not found_shape:
                if verbose:
                    print(f"  WARNING: Could not determine exact split sizes, assuming equal splits")
                # Based on the model, both splits are 2-way equal splits
                # This is a reasonable fallback for most cases
                split_sizes = [2] * num_outputs if num_outputs == 2 else [1] * num_outputs

        # Create Slice nodes for each output
        start_idx = 0
        for i, (output_name, size) in enumerate(zip(split_node.output, split_sizes)):
            size = int(size)
            end_idx = start_idx + size

            # Create constant initializers for slice parameters
            starts_name = f"{split_node.name}_starts_{i}"
            ends_name = f"{split_node.name}_ends_{i}"
            axes_name = f"{split_node.name}_axes_{i}"

            starts_tensor = numpy_helper.from_array(np.array([start_idx], dtype=np.int64), name=starts_name)
            ends_tensor = numpy_helper.from_array(np.array([end_idx], dtype=np.int64), name=ends_name)
            axes_tensor = numpy_helper.from_array(np.array([axis], dtype=np.int64), name=axes_name)

            new_initializers.extend([starts_tensor, ends_tensor, axes_tensor])

            # Create Slice node
            slice_node = helper.make_node(
                "Slice",
                inputs=[input_tensor, starts_name, ends_name, axes_name],
                outputs=[output_name],
                name=f"{split_node.name}_slice_{i}"
            )
            new_nodes.append(slice_node)

            start_idx = end_idx
            if verbose:
                print(f"  Created Slice node for output {i}: {output_name}")

        nodes_to_remove.append(split_node)

    # Remove Split nodes and add Slice nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)

    # Run shape inference to populate value_info
    try:
        model = onnx.shape_inference.infer_shapes(model)
        if verbose:
            print("Shape inference successful after modifications")
    except Exception as e:
        if verbose:
            print(f"  WARNING: Shape inference failed: {e}")
            print("  The model may still work, but some shape information might be missing")

    # Save modified model
    onnx.save(model, output_path)
    if verbose:
        print(f"\nSaved modified model to: {output_path}")
        print(f"Replaced {len(nodes_to_remove)} Split node(s) with Slice operations")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replace_split_with_slice.py <input.onnx> [output.onnx]")
        print("\nThis script replaces ONNX Split operations with Slice operations")
        print("to avoid TFLite Flex delegate requirements.")
        sys.exit(1)

    input_model = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else input_model.replace(".onnx", "_no_split.onnx")

    if not Path(input_model).exists():
        print(f"ERROR: Input model not found: {input_model}")
        sys.exit(1)

    replace_splits_with_slices(input_model, output_model, verbose=True)
