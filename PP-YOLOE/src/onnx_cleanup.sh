# PP-YOLOE/src/onnx_cleanup.sh

#!/usr/bin/env bash


# simplify PP-YOLOE ONNX model

MODEL="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx"
OUTPUT_MODEL="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx"

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

python3 PP-YOLOE/src/onnx_cleanup.py \
    --model "$MODEL" \
    --output-model "$OUTPUT_MODEL"
