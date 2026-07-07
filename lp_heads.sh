#!/bin/bash
#SBATCH --job-name=oe_lp
#SBATCH --account=aip-gpleiss
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/oe_lp_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

# Train every LP head type on ONE cached feature set. GPU is requested because the AnyUp
# heads load torch.hub AnyUp; the lp_* heads will simply use it too. Emails at start/finish.
# The feature set is passed via the FEATURES env var (use submit_lp_heads.sh to fan out one
# job per feature set), e.g.:
#   sbatch --export=ALL,FEATURES=oe_base_s2_ps1_tile1 lp_heads.sh
#
# Configurable knobs (env vars from --export override these defaults):
#   BATCH_SIZE        batch for the lp_* heads (default 32)
#   ANYUP_BATCH_SIZE  batch for the anyup* heads (default 4). AnyUp's depthwise conv builds a
#                     tensor whose element count must stay under 2^31 or CUDA's 32-bit indexed
#                     kernels error ("canUse32BitIndexMath ... false"); finer token grids
#                     (e.g. ps2_tile32) overflow at batch 32, so anyup runs with a small batch.
#   NUM_WORKERS       DataLoader workers (default 0). >0 ships batches via /dev/shm, which is
#                     tightly capped on Compute Canada and overflows to "Bus error" for the
#                     large 64x64-grid feature sets; 0 loads in-process and avoids shm.
#   MAX_RAM_GB        cap for preloading features into RAM (default 32). The big grids
#                     (ps1_tile8) can OOM; lower this to force disk streaming instead.

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1   # silence tqdm progress bars in the batch log

if [ -z "$FEATURES" ]; then
    echo "ERROR: set FEATURES, e.g. sbatch --export=ALL,FEATURES=oe_base_s2_ps1_tile1 lp_heads.sh" >&2
    exit 1
fi

# Configurable knobs (env vars override these defaults).
BATCH_SIZE="${BATCH_SIZE:-32}"
ANYUP_BATCH_SIZE="${ANYUP_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_RAM_GB="${MAX_RAM_GB:-32}"

cd "$SLURM_SUBMIT_DIR"
source env_olmo.sh

HEADS=(lp_pa2pa_bu lp_pa2px anyup anyup_t2 anyup_t2_ens anyup_t1 anyup_t1_ens)

LOG="logs/oe_lp_${SLURM_JOB_ID}.out"

# Email at start.
echo "features: $FEATURES | head modes: ${HEADS[*]} | batch $BATCH_SIZE (anyup $ANYUP_BATCH_SIZE) | workers $NUM_WORKERS | max_ram ${MAX_RAM_GB}G" \
    | mail -s "[START job $SLURM_JOB_ID] LP heads $FEATURES" "$EMAIL"

for head in "${HEADS[@]}"; do
    # AnyUp heads use a smaller batch to stay under AnyUp's 32-bit indexing limit.
    case "$head" in
        anyup*) bs="$ANYUP_BATCH_SIZE" ;;
        *)      bs="$BATCH_SIZE" ;;
    esac
    echo "=== RUN features=$FEATURES head_mode=$head batch_size=$bs ==="
    python -u lp_on_cached_features.py --features "$FEATURES" --head_mode "$head" \
        --batch_size "$bs" --num_workers "$NUM_WORKERS" --max_ram_gb "$MAX_RAM_GB"
    echo "=== EXIT status=$? features=$FEATURES head_mode=$head ==="
done

# Email the outcome: per-head BEST/TEST metrics + log tail.
{
    echo "features: $FEATURES"
    echo "---"
    grep -aE "^=== RUN|^BEST|^TEST|^=== EXIT" "$LOG" || echo "(no metrics found; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID] LP heads $FEATURES" "$EMAIL"
