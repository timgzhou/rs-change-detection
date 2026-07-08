#!/bin/bash
#SBATCH --job-name=oe_extract
#SBATCH --account=aip-gpleiss
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/oe_extract_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Cache OlmoEarth per-timestep features for PASTIS. Configure via --export env vars:
#   MODEL_SIZE  (default base)
#   MODALITIES  (default sentinel2_l2a; comma-separated, e.g. sentinel2_l2a,sentinel1)
#   PATCH_SIZE  (default 1)
#   TILE_SIZE   (default 1)
# Emails at start and finish.
#
# Examples:
#   sbatch extract_olmoearth_features.sh                                   # defaults: ps1 tile1, s2
#   sbatch --export=ALL,PATCH_SIZE=4,TILE_SIZE=64 extract_olmoearth_features.sh
#   sbatch --export=ALL,MODALITIES=sentinel2_l2a,sentinel1,PATCH_SIZE=4,TILE_SIZE=64 extract_olmoearth_features.sh
#   sbatch --export=ALL,MODALITIES=sentinel2_l2a,PATCH_SIZE=4,TILE_SIZE=64 extract_olmoearth_features.sh

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1   # silence tqdm progress bars in the batch log

# Configurable knobs (env vars from --export override these defaults).
MODEL_SIZE="${MODEL_SIZE:-base}"
MODALITIES="${MODALITIES:-sentinel2_l2a}"
PATCH_SIZE="${PATCH_SIZE:-1}"
TILE_SIZE="${TILE_SIZE:-1}"
# Write features to project space (scratch is near quota); override with OUT_ROOT.
OUT_ROOT="${OUT_ROOT:-$HOME/projects/aip-gpleiss/timz/features}"

cd "$SLURM_SUBMIT_DIR"
source env_olmo.sh

ARGS="--model_size $MODEL_SIZE --modalities $MODALITIES --patch_size $PATCH_SIZE --tile_size $TILE_SIZE --out_root $OUT_ROOT"
TAG="${MODEL_SIZE} ${MODALITIES} ps${PATCH_SIZE} tile${TILE_SIZE}"

# Email at start.
echo "extract_olmoearth_features.py $ARGS" \
    | mail -s "[START job $SLURM_JOB_ID] extract $TAG" "$EMAIL"

LOG="logs/oe_extract_${SLURM_JOB_ID}.out"
python -u extract_olmoearth_features.py $ARGS
STATUS=$?

# Email the outcome: config line + log tail.
{
    echo "args: $ARGS"
    echo "exit status: $STATUS"
    echo "---"
    grep -aE "^Extraction config:|^Wrote |^Total size" "$LOG" || echo "(no summary found; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] extract $TAG" "$EMAIL"
