#!/usr/bin/env bash
# bash apps/main/configs/modular_arithmetic/create_and_submit_tokenizers.sh
ROOT_DIR="apps/main/configs/modular_arithmetic"
BASE_CONFIG="${ROOT_DIR}/base.yaml"

tokenizers=(
	flexitok/mod-tokenizers-individual
	flexitok/mod-tokenizers-ltr_3digit
	flexitok/mod-tokenizers-ltr_2digit
	flexitok/mod-tokenizers-ltr_4digit
	flexitok/mod-tokenizers-ltr_5digit
	flexitok/mod-tokenizers-rtl_2digit
	flexitok/mod-tokenizers-rtl_3digit
	flexitok/mod-tokenizers-rtl_4digit
	flexitok/mod-tokenizers-rtl_5digit
)
# Set DRY_RUN=1 to only print commands without submitting jobs
DRY_RUN="${DRY_RUN:-0}"

to_safe_num() {
	python - "$1" <<'PY'
from decimal import Decimal
import sys

raw = sys.argv[1]
val = Decimal(raw)
txt = format(val, "f")
if "." in txt:
		txt = txt.rstrip("0").rstrip(".")
txt = txt.replace("-", "m").replace(".", "p")
print(txt)
PY
}

count=0
for tokenizer in "${tokenizers[@]}"; do
	safe_tokenizer="$(basename "${tokenizer}")"
	safe_name="${safe_tokenizer}"
	cfg_path="${ROOT_DIR}/${safe_name}.yaml"

	cat >"${cfg_path}" <<EOF
name: mod_arith_${safe_name}
dump_dir: /fsx/craffel/lingua_logs/modular_arithmetic/${safe_name}

data:
  tokenizer:
    name: huggingface
    path: ${tokenizer}

logging:
  wandb:
    group: modular-arithmetic-ablations-corrected
    name: ${safe_name}

EOF
	gputype="l40s"
	if [[ "${tokenizer}" == *"5digit"* ]]; then
		gputype="h100"
	fi
	cmd=(
		python -m lingua.stool_ccdb
		ngpu=1
		ncpu=4
		mem=32G
		time=480
		nodes=1
		gpu_type=${gputype}
		launcher=sbatch
		script=apps.main.train_answer_only
		"config=[\"${BASE_CONFIG}\",\"${cfg_path}\"]"
	)

	if [[ "${DRY_RUN}" == "1" ]]; then
		echo "[DRY_RUN] TRACK_ACCURACY=True ${cmd[*]}"
	else
		echo "Submitting ${safe_name}"
		TRACK_ACCURACY=True "${cmd[@]}"
	fi

	count=$((count + 1))
done

echo "Prepared ${count} ablation configs under ${ROOT_DIR}"
