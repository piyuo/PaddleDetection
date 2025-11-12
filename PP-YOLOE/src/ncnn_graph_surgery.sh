# PP-YOLOE/src/ncnn_graph_surgery.sh
#!/usr/bin/env bash

MODEL="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust.onnx"
OUTPUT_MODEL="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust_ncnn.onnx"
IMG="PP-YOLOE/build/dataset/demo/demo.jpg"
OUTDIR="PP-YOLOE/build/models/surgery"
WARMUP=20
RUNS=80
RUN_BENCHMARK=0
ORT_PROVIDER="cpu"

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
		--warmup)
			if [[ -n "$2" ]]; then
				WARMUP="$2"
				shift 2
			else
				echo "Error: --warmup requires a value" >&2
				exit 1
			fi
			;;
		--runs)
			if [[ -n "$2" ]]; then
				RUNS="$2"
				shift 2
			else
				echo "Error: --runs requires a value" >&2
				exit 1
			fi
			;;
		--ort-provider)
			if [[ -n "$2" ]]; then
				ORT_PROVIDER="$2"
				shift 2
			else
				echo "Error: --ort-provider requires a provider id" >&2
				exit 1
			fi
			;;
		--run-benchmark)
			RUN_BENCHMARK=1
			shift
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

CMD=(python3 PP-YOLOE/src/ncnn_graph_surgery.py \
	--model "$MODEL" \
	--input-shape 1,3,640,640 \
	--warmup "$WARMUP" --runs "$RUNS" \
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
	--remove-identity \
	--rewrite-reduce-to-globalpool \
	--output-model "$OUTPUT_MODEL")

if [[ $RUN_BENCHMARK -eq 1 ]]; then
	CMD+=(--run-benchmark --ort-provider "$ORT_PROVIDER" --img "$IMG")
fi

"${CMD[@]}"

rm -rf "$OUTDIR"
