#!/bin/bash
# train_array_kappa.sbatch

#SBATCH --job-name=linearattention_power0p9_d16
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_output/%x_%A_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_output/%x_%A_%a.err
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_grads
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --array=0-10

set -euo pipefail

echo "Job: ${SLURM_JOB_ID} ArrayJob: ${SLURM_ARRAY_JOB_ID} ArrayTask: ${SLURM_ARRAY_TASK_ID} Host: $(hostname)"
echo "Start time: $(date)"

module purge
module load python/3.10.12-fasrc01
source activate try4

# --- (Optional) XLA memory settings ---
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}

# --- kappa grid: np.linspace(0.1, 2.1, 11) ---
KAPPAS=(0.10 0.30 0.50 0.70 0.90 1.10 1.30 1.50 1.70 1.90 2.10)
KAPPA="${KAPPAS[${SLURM_ARRAY_TASK_ID}]}"
echo "Using KAPPA=${KAPPA}"

# --- Results save path (NOT dependent on kappa) ---
# One directory per array master job id
SAVE_DIR_BASE=${SAVE_DIR_BASE:-/n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_results}
SAVE_DIR="${SAVE_DIR_BASE}/${SLURM_JOB_NAME}"
mkdir -p "${SAVE_DIR}"

# Interpretable filename (no kappa)
RESULTS_CSV="${SAVE_DIR}/results_csv"
echo "Saving results to: ${RESULTS_CSV}"

# --- Weights & Biases setup ---
export WANDB_PROJECT=${WANDB_PROJECT:-lin-att-verification}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$PWD/wandb}
mkdir -p "$WANDB_DIR"

# W&B run name includes kappa (fine)
KAPPA_TAG="$(printf "%0.2f" "${KAPPA}" | sed 's/\./p/g')"
RUN_NAME="${SLURM_JOB_NAME}_kappa${KAPPA_TAG}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

export WANDB_NOTES="${WANDB_NOTES:-slurm_array_job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} kappa=${KAPPA} partition=${SLURM_JOB_PARTITION}}"
export WANDB_TAGS="${WANDB_TAGS:-slurm,h100,kappa=${KAPPA}}"

# --- Seeding ---
SEED_BASE=${SEED_BASE:-1234}
SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID))
echo "Using SEED=${SEED}"

python -u linear_attention.py \
  --seed "${SEED}" \
  --kappa "${KAPPA}" \
  --savedirectory "${RESULTS_CSV}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --d 16 \
  --pure_linear_self_att \
  --no_input_projection \
  --num_epochs 10000 \
  --batch_size 256 \
  --task_power 0.9

echo "End time: $(date)"