# pipeline/PP-YOLOE/ncnn_graph_surgery.sh
#!/usr/bin/env bash

# Pipeline PP-YOLOE Apple Neural Engine Graph Surgery

MODEL="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx"
OUTPUT_MODEL="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ncnn.onnx"
IMG="pipeline/dataset/demo/demo.jpg"
OUTDIR="pipeline/PP-YOLOE/models/surgery"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--model)
			if [[ -n "$2" ]]; then
				MODEL="$2"
				shift 2
			else
				echo "Error: --model requires a path" >&2
				exit 1
			fi
			;;
		--output-model)
			if [[ -n "$2" ]]; then
				OUTPUT_MODEL="$2"
				shift 2
			else
				echo "Error: --output-model requires a path" >&2
				exit 1
			fi
			;;
		--img)
			if [[ -n "$2" ]]; then
				IMG="$2"
				shift 2
			else
				echo "Error: --img requires a path" >&2
				exit 1
			fi
			;;
		--outdir)
			if [[ -n "$2" ]]; then
				OUTDIR="$2"
				shift 2
			else
				echo "Error: --outdir requires a path" >&2
				exit 1
			fi
			;;
		--)
			shift
			break
			;;
		*)
			echo "Warning: Unknown option '$1'" >&2
			shift
			;;
	esac
done

python3 pipeline/PP-YOLOE/ncnn_graph_surgery.py \
	--model "$MODEL" \
	--input-shape 1,3,640,640 --warmup 20 --runs 80 \
	--img "$IMG" \
	--outdir "$OUTDIR" \
	--fix-input-shapes \
	--fold-iterations 15 \
	--split-concat 4 \
	--fold-static-shapes \
	--rewrite-div \
	--rewrite-pow \
	--rewrite-slice-to-gather \
	--rewrite-slice-range-to-gather \
	--rewrite-resize-to-static \
	--remove-noop-slice \
	--rewrite-reduce-to-globalpool \
	--output-model "$OUTPUT_MODEL"

rm -rf "$OUTDIR"
