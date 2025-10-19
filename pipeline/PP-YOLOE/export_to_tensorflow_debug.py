#!/usr/bin/env python3
"""Export with selectively disabled transpose heuristics for debugging."""

import sys
from pathlib import Path

# Modify the export module temporarily
import export_to_tensorflow as etf

# Store original function
original_build_param = etf._build_param_replacement_file


def patched_build_param(onnx_path: Path, verbose: bool):
    """Modified version that lets you disable specific heuristics."""

    # Define which fixes to DISABLE (set to True to disable)
    DISABLE_FLAGS = {
        "reshape_bias_fixes": False,         # Reshape [1,C,1,1] fixes
        "softmax_layout_fixes": False,       # Softmax [N,4,17,*] fixes
        "conv_input_fixes": False,           # Conv input pre-transpose
        "connectivity_fixes": False,         # Softmax→Conv connectivity
        "squeeze_fixes": False,              # Squeeze axis fixes
        "add_const_transpose": False,        # Add 2D constant transpose
        "mul_const_transpose": False,        # Mul 2D constant transpose
        "div_transpose": False,               # ⚠️ DISABLE DIV FIXES (likely culprit)
        "name_based_fixes": False,            # ⚠️ DISABLE NAME-BASED FIXES (Div.0, etc)
    }

    print("[INFO] Custom export with disabled heuristics:")
    for name, disabled in DISABLE_FLAGS.items():
        status = "DISABLED" if disabled else "enabled"
        print(f"  - {name}: {status}")

    # Call original but intercept the result
    param_file = original_build_param(onnx_path, verbose)

    if param_file is None:
        return None

    # Read the generated operations
    import json
    with open(param_file, 'r') as f:
        data = json.load(f)

    operations = data.get("operations", [])
    original_count = len(operations)

    # Filter operations based on disable flags
    filtered_ops = []
    removed_by_category = {}

    for op in operations:
        op_name = op.get("op_name", "")
        param_name = op.get("param_name", "")

        # Categorize and potentially skip
        skip = False
        category = None

        # Check against disable flags
        if DISABLE_FLAGS.get("div_transpose", False):
            if "Div" in op_name:
                skip = True
                category = "div_transpose"

        if DISABLE_FLAGS.get("name_based_fixes", False):
            # These are the name-specific rules at the end
            if op_name in ["Softmax.0", "Softmax.1", "Softmax.2",
                          "Conv.71", "Conv.78", "Conv.85",
                          "Squeeze.0", "Squeeze.1", "Squeeze.2",
                          "Div.0"]:
                skip = True
                category = "name_based_fixes"

        if DISABLE_FLAGS.get("add_const_transpose", False):
            if "Add" in op_name and "pre_process_transpose_perm" in op:
                perm = op["pre_process_transpose_perm"]
                if perm == [1, 0]:  # 2D transpose
                    skip = True
                    category = "add_const_transpose"

        if DISABLE_FLAGS.get("mul_const_transpose", False):
            if "Mul" in op_name and "pre_process_transpose_perm" in op:
                perm = op["pre_process_transpose_perm"]
                if perm == [1, 0]:  # 2D transpose
                    skip = True
                    category = "mul_const_transpose"

        if DISABLE_FLAGS.get("softmax_layout_fixes", False):
            if "Softmax" in op_name and "post_process_transpose_perm" in op:
                perm = op["post_process_transpose_perm"]
                if perm == [0, 1, 3, 2]:  # Softmax layout swap
                    skip = True
                    category = "softmax_layout_fixes"

        if DISABLE_FLAGS.get("conv_input_fixes", False):
            if "Conv" in op_name and "pre_process_transpose_perm" in op:
                perm = op["pre_process_transpose_perm"]
                if perm == [0, 1, 3, 2]:
                    skip = True
                    category = "conv_input_fixes"

        if DISABLE_FLAGS.get("squeeze_fixes", False):
            if "Squeeze" in op_name and "pre_process_transpose_perm" in op:
                skip = True
                category = "squeeze_fixes"

        if skip:
            removed_by_category[category] = removed_by_category.get(category, 0) + 1
        else:
            filtered_ops.append(op)

    # Report what was removed
    if removed_by_category:
        print(f"[INFO] Removed {original_count - len(filtered_ops)} operations:")
        for cat, count in removed_by_category.items():
            print(f"  - {cat}: {count}")

    # Write filtered operations back
    if not filtered_ops:
        print("[INFO] No operations remaining after filtering - skipping param file")
        param_file.unlink()
        return None

    with open(param_file, 'w') as f:
        json.dump({"operations": filtered_ops}, f, indent=2)

    print(f"[INFO] Using {len(filtered_ops)} filtered operations")
    return param_file


# Monkey patch
etf._build_param_replacement_file = patched_build_param


if __name__ == "__main__":
    # Run the normal export with our patches
    etf.main()
