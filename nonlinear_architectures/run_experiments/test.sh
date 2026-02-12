#!/bin/bash
# train.sbatch

#SBATCH --job-name=icl_linearattention_test
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_output/%x_%A.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICLR26_finalruns/test_output/%x_%A.err
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH -p kempner_h100
#SBATCH --account=kempner_grads
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

set -euo pipefail

echo "Job: ${SLURM_JOB_ID} Host: $(hostname)"
echo "Start time: $(date)"

module purge
module load python/3.10.12-fasrc01
source activate try4

# --- (Optional) XLA memory settings ---
# These often help avoid OOM from XLA preallocating most GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}

# --- Weights & Biases setup ---
# Recommended: export WANDB_API_KEY in your shell (or ~/.bashrc) before sbatch.
# e.g. `export WANDB_API_KEY=...`
export WANDB_PROJECT=${WANDB_PROJECT:-icl-task-alignment}
export WANDB_ENTITY=${WANDB_ENTITY:-}          # set if your org requires it
export WANDB_MODE=${WANDB_MODE:-online}        # set to "offline" if needed
export WANDB_DIR=${WANDB_DIR:-$PWD/wandb}      # local dir for wandb files
mkdir -p "$WANDB_DIR"

# Make W&B runs unique and informative (job + array task)
RUN_NAME="${SLURM_JOB_NAME}_job${SLURM_JOB_ID}"
export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_ID}}"

# Useful metadata in W&B
export WANDB_NOTES="${WANDB_NOTES:-slurm_job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION}}"
export WANDB_TAGS="${WANDB_TAGS:-slurm,h100}"

# --- Seeding ---
# A common pattern: use array task id to vary seeds
SEED_BASE=${SEED_BASE:-1234}
SEED=$((SEED_BASE))
echo "Using SEED=${SEED}"

python -u test.py \
  --seed "${SEED}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --d 32 \
  --no_input_projection \
  --n_layers 1 \
  --plot_name "only_softmax_d32_alltasks" \
  --kappa -1

echo "End time: $(date)"