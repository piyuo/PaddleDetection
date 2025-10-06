import argparse
import re
import shutil

def parse_param_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    # Handle old or new NCNN header format
    if len(lines[0].split()) == 3:
        magic, num_layers, num_blobs = map(int, lines[0].split())
        start_idx = 1
    else:
        # sometimes the first two lines are split like:
        # 7767517
        # 302 357
        magic = int(lines[0])
        num_layers, num_blobs = map(int, lines[1].split())
        start_idx = 2

    layers = []
    for line in lines[start_idx:]:
        parts = re.split(r'\s+', line)
        if len(parts) < 4:
            continue
        layer_type, layer_name = parts[0], parts[1]
        layers.append({'type': layer_type, 'name': layer_name, 'raw': line})
    return magic, num_layers, num_blobs, layers


def add_preserve_outputs(layers, blob_names):
    for blob in blob_names:
        lname = f'Output_{blob}'
        line = f'Output {lname} 1 1 {blob}'
        layers.append({'type': 'Output', 'name': lname, 'raw': line})
    return layers


def write_param_file(path, magic, layers):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{magic} {len(layers)} {len(layers)*2}\n")
        for l in layers:
            f.write(l['raw'] + '\n')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-param", required=True)
    ap.add_argument("--input-bin", required=True)
    ap.add_argument("--output-param", required=True)
    ap.add_argument("--output-bin", required=True)
    ap.add_argument("--keep", required=False, default="")
    args = ap.parse_args()

    keep_blobs = []
    if args.keep:
        # support both comma and space separated names
        keep_blobs = re.split(r'[,\s]+', args.keep.strip())

    magic, num_layers, num_blobs, layers = parse_param_file(args.input_param)
    if keep_blobs:
        layers = add_preserve_outputs(layers, keep_blobs)
    write_param_file(args.output_param, magic, layers)

    shutil.copyfile(args.input_bin, args.output_bin)
    print(f"✅ Done! {len(keep_blobs)} outputs preserved → {args.output_param}")
