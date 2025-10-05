#!/usr/bin/env python3
"""Test NCNN param file parsing"""

def parse_param_file_for_blobs(param_path):
    """Parse NCNN .param file to guess input and output blob names."""
    inputs, outputs = [], []
    try:
        with open(param_path, 'r') as f:
            lines = f.readlines()

        # Track all produced and consumed blobs
        produced = set()
        consumed = set()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            layer_type = parts[0]

            # Input layers: format "Input <name> 0 1 <blob_name>"
            if layer_type == 'Input':
                if len(parts) >= 5:
                    inputs.append(parts[4])
                    produced.add(parts[4])
            else:
                # Regular layers: format "<type> <name> <num_inputs> <num_outputs> <input_blobs...> <output_blobs...>"
                if len(parts) >= 4:
                    try:
                        num_inputs = int(parts[2])
                        num_outputs = int(parts[3])
                        blob_start = 4

                        # Input blobs
                        for i in range(num_inputs):
                            if blob_start + i < len(parts):
                                blob = parts[blob_start + i]
                                if not blob.startswith('-') and not blob.startswith('0='):
                                    consumed.add(blob)

                        # Output blobs
                        for i in range(num_outputs):
                            idx = blob_start + num_inputs + i
                            if idx < len(parts):
                                blob = parts[idx]
                                if not blob.startswith('-') and not blob.startswith('0='):
                                    produced.add(blob)
                    except (ValueError, IndexError):
                        continue

        # Outputs are blobs that are produced but never consumed
        outputs = sorted(list(produced - consumed))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return inputs, outputs

if __name__ == '__main__':
    param_path = 'pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param'
    print(f"Parsing: {param_path}")
    inputs, outputs = parse_param_file_for_blobs(param_path)
    print(f"\nInputs ({len(inputs)}):", inputs)
    print(f"\nOutputs ({len(outputs)}):", outputs)
