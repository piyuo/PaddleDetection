import re
from pathlib import Path

def clean_pnnx_expression(param_path, output_path=None, remove_only=True):
    """
    Remove or patch all 'pnnx.Expression' layers in an NCNN .param file.

    Args:
        param_path (str | Path): Path to the .param file.
        output_path (str | Path | None): Output path. If None, overwrite original.
        remove_only (bool):
            - True = remove Expression layers and disconnect their outputs.
            - False = replace them with MemoryData constants (value=1.0).
    """
    param_path = Path(param_path)
    output_path = Path(output_path) if output_path else param_path

    lines = param_path.read_text().splitlines()

    # Identify lines containing pnnx.Expression
    expr_lines = [i for i, line in enumerate(lines) if "pnnx.Expression" in line]

    if not expr_lines:
        print("✅ No pnnx.Expression layers found.")
        return

    print(f"🧠 Found {len(expr_lines)} pnnx.Expression layers — cleaning...")

    # Extract their output tensor IDs
    expr_outputs = []
    for i in expr_lines:
        parts = lines[i].strip().split()
        if len(parts) >= 5:
            expr_outputs.append(parts[-1])  # last token is usually output blob name

    cleaned_lines = []

    for line in lines:
        if "pnnx.Expression" in line:
            if remove_only:
                # Skip this layer entirely
                continue
            else:
                # Replace with a MemoryData constant
                parts = line.split()
                name = parts[1] if len(parts) > 1 else "const_expr"
                output = parts[-1]
                new_line = f"MemoryData {name}_const 0 1 {output} 0=1.0"
                cleaned_lines.append(new_line)
                continue

        # Rewire downstream ops to skip deleted outputs if possible
        for blob in expr_outputs:
            # If an op references expr output as input, just remove it
            line = re.sub(rf"\b{blob}\b", "", line)
        # Clean double spaces after replacements
        line = re.sub(r"\s{2,}", " ", line).strip()
        cleaned_lines.append(line)

    # Write new param
    output_path.write_text("\n".join(cleaned_lines) + "\n")
    print(f"✅ Cleaned file saved to: {output_path}")
    print(f"🧹 Removed layers: {expr_lines}")
    print(f"🔗 Removed blobs: {expr_outputs}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean pnnx.Expression layers from NCNN .param")
    parser.add_argument("param", help="Path to .param file")
    parser.add_argument("--output", help="Path to save cleaned .param", default=None)
    parser.add_argument("--replace", action="store_true", help="Replace with MemoryData instead of removing")
    args = parser.parse_args()

    clean_pnnx_expression(args.param, args.output, remove_only=not args.replace)
