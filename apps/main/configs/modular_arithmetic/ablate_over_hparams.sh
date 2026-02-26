#!/usr/bin/env bash
set -euo pipefail

# bash apps/main/configs/modular_arithmetic/ablate_over_hparams.sh
ROOT_DIR="apps/main/configs/modular_arithmetic"
BASE_CONFIG="${ROOT_DIR}/base.yaml"
ABLATION_DIR="${ROOT_DIR}/ablations"

mkdir -p "${ABLATION_DIR}"

weight_decays=(0.5 1.0 2.0)
lrs=(3e-4 1e-3 3e-3)
batch_sizes=(128 512)

model_variants=(base small)

# Set DRY_RUN=1 to only print commands without submitting jobs
DRY_RUN="${DRY_RUN:-0}"

count=0
for wd in "${weight_decays[@]}"; do
	for lr in "${lrs[@]}"; do
		for bs in "${batch_sizes[@]}"; do
			for model_variant in "${model_variants[@]}"; do
				if [[ "${model_variant}" == "base" ]]; then
					model_dim=256
					model_layers=4
					model_heads=4
				else
					model_dim=128
					model_layers=2
					model_heads=4
				fi

				exp_name="wd${wd}_lr${lr}_bs${bs}_${model_variant}_d${model_dim}_l${model_layers}_h${model_heads}"
				safe_name="${exp_name//./p}"
				safe_name="${safe_name//-/m}"
				cfg_path="${ABLATION_DIR}/${safe_name}.yaml"

				cat > "${cfg_path}" <<EOF
name: mod_arith_${safe_name}
dump_dir: /fsx/craffel/lingua_logs/modular_arithmetic_ablations/${safe_name}

optim:
  weight_decay: ${wd}
  lr: ${lr}

data:
  batch_size: ${bs}

model:
  dim: ${model_dim}
  n_layers: ${model_layers}
  n_heads: ${model_heads}
  weight_tying: false

logging:
  wandb:
    group: modular-arithmetic-ablations
    name: ${safe_name}

EOF

				cmd=(
					python -m lingua.stool_ccdb
					ngpu=1
					ncpu=4
					mem=32G
					time=480
					nodes=1
					gpu_type=l40s
					launcher=sbatch
					script=apps.main.train
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
		done
	done
done

echo "Prepared ${count} ablation configs under ${ABLATION_DIR}"
