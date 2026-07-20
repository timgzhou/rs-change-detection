#!/bin/bash
#SBATCH --job-name=manyup
#SBATCH --account=aip-gpleiss
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/manyup_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Train a mAnyUp low-res -> high-res feature upsampler (with projector head). Configure via
# --export env vars:
#   LR_CFG      low-res INPUT feature dir  (default oe_base_s2_ps4_tile4)
#   HR_CFG      high-res TARGET feature dir(default oe_base_s2_ps1_tile64)
#   EPOCHS      (default 10)
#   BATCH_SIZE  (default 16)
#   LR          learning rate (default 1e-3)
#   DOWN_REG    anyup_down weight (default 0.1)
#   EXTRA       any extra args passed verbatim to train_manyup.py (e.g. "--no-proj_head")
# Features are staged to $SLURM_TMPDIR automatically. Emails at start and finish.
#
# Examples:
#   sbatch train_manyup.sh                                             # defaults: ps4_tile4 -> ps1_tile64
#   sbatch --export=ALL,LR_CFG=oe_base_s2_ps4_tile32,HR_CFG=oe_base_s2_ps1_tile1 train_manyup.sh
#   sbatch --export=ALL,LR_CFG=oe_base_s2_ps2_tile8,HR_CFG=oe_base_s2_ps1_tile64,EPOCHS=20 train_manyup.sh
#   sbatch --export=ALL,EXTRA="--no-proj_head --no-linear_baseline" train_manyup.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1

# Configurable knobs (env vars from --export override these defaults).
LR_CFG="${LR_CFG:-oe_base_s2_ps4_tile4}"
HR_CFG="${HR_CFG:-oe_base_s2_ps1_tile8}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
DOWN_REG="${DOWN_REG:-0.1}"
EXTRA="${EXTRA:-}"

cd "$SLURM_SUBMIT_DIR"
source env_olmo.sh

# Per-run output dir keyed by the LR->HR pair so parallel jobs don't clobber each other.
OUT_DIR="checkpoints/manyup/${LR_CFG}__to__${HR_CFG}"

ARGS="--lr_cfg $LR_CFG --hr_cfg $HR_CFG --epochs $EPOCHS --batch_size $BATCH_SIZE \
--lr $LR --down_reg $DOWN_REG --stage_to_tmpdir --out_dir $OUT_DIR $EXTRA"
TAG="${LR_CFG} -> ${HR_CFG}"

LOG="logs/manyup_${SLURM_JOB_ID}.out"
python -u train_manyup.py $ARGS
STATUS=$?

# Email the outcome: config + last epoch summaries (the "== epoch N done" lines) + log tail.
{
    echo "args: $ARGS"
    echo "exit status: $STATUS"
    echo "out_dir: $OUT_DIR"
    echo "--- epoch summaries ---"
    grep -aE "^== epoch|^\[baseline\]|beats|WORSE" "$LOG" | tail -20 || echo "(no summaries; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] mAnyUp $TAG" "$EMAIL"
