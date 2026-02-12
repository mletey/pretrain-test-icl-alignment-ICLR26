#!/bin/bash
#SBATCH --job-name=change_train_power_d32
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_output/%x_%A_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_output/%x_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_grads
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --array=0-142%15

set -euo pipefail

module purge
module load python/3.10.12-fasrc01
source activate try4

# --- (Optional) XLA memory settings ---
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}

# --- Results save path (NOT dependent on kappa) ---
# One directory per array master job id
SAVE_DIR_BASE=${SAVE_DIR_BASE:-/n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_results}
SAVE_DIR="${SAVE_DIR_BASE}/${SLURM_JOB_NAME}"
mkdir -p "${SAVE_DIR}"

RESULTS_CSV="${SAVE_DIR}/results_csv"
echo "Saving results to: ${RESULTS_CSV}"

# --------- build parameter grids (exact decimal steps) ----------
# train_power: 0.5, 0.6, ..., 2.9   (step 0.1)
# kappa:       0.1, 0.2, ..., 2.1   (step 0.1)
mapfile -t TRAIN_POWERS < <(python - <<'PY'
vals = [0.5 + 0.2*i for i in range(int(round((2.9-0.5)/0.2))+1)]
print("\n".join(f"{v:.1f}" for v in vals))
PY
)

mapfile -t KAPPAS < <(python - <<'PY'
vals = [0.2 + 0.2*i for i in range(int(round((2.2-0.2)/0.2))+1)]
print("\n".join(f"{v:.1f}" for v in vals))
PY
)

N_TP=${#TRAIN_POWERS[@]}   # 25
N_K=${#KAPPAS[@]}          # 21
N_SEEDS=1

TOTAL=$((N_TP * N_K * N_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Error: TASK_ID=${TASK_ID} out of range [0, $((TOTAL-1))]"
  exit 1
fi

# --------- map task id -> (train_power, kappa, seed) ----------
SEED_IDX=$(( TASK_ID % N_SEEDS + 1))          # 1..3
COMBO_IDX=$(( TASK_ID / N_SEEDS ))         # 0..(N_TP*N_K - 1)

K_IDX=$(( COMBO_IDX % N_K ))               # 0..N_K-1
TP_IDX=$(( COMBO_IDX / N_K ))              # 0..N_TP-1

TRAIN_POWER=${TRAIN_POWERS[$TP_IDX]}
KAPPA=${KAPPAS[$K_IDX]}

# Choose whatever seed values you want; here we just use 0,1,2:
SEED=$SEED_IDX

echo "TASK_ID=${TASK_ID}/${TOTAL} -> train_power=${TRAIN_POWER}, kappa=${KAPPA}, seed=${SEED}"

python -u figure4.py \
  --task_power "${TRAIN_POWER}" \
  --seed "${SEED}" \
  --kappa "${KAPPA}" \
  --savedirectory "${RESULTS_CSV}" \
  --d 32 \
  --no_input_projection \
  --num_epochs 10000 \
  --batch_size 1024 \
  --test_power 0.9 \
  --tau 4.0 \
