#!/bin/bash
#SBATCH --job-name=oe_pastis_ft
#SBATCH --account=aip-gpleiss
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/oe_pastis_ft_%j.out

# Full OlmoEarth + AnyUp finetune on PASTIS (~3 hr at 64 epochs on an L40S).
#   sbatch finetune_olmoearth_pastis.sh                          # uses CONFIG below
#   sbatch finetune_olmoearth_pastis.sh configs/base_s2s1_lp.yaml  # override config
# Emails the resolved config at start and the final metrics at end.

CONFIG="${1:-configs/base_s2s1_anyup.yaml}"
EMAIL="tiange.zhou@outlook.com"

cd "$SLURM_SUBMIT_DIR"
module load python/3.12 scipy-stack opencv libspatialindex proj hdf5/1.14.6
source env_olmo/bin/activate

# Email the resolved config at start.
python -c "from config import load_config, to_dict; import json; print(json.dumps(to_dict(load_config('$CONFIG')), indent=2))" \
    | mail -s "[START job $SLURM_JOB_ID] $CONFIG" "$EMAIL"

LOG="logs/oe_pastis_ft_${SLURM_JOB_ID}.out"
python -u finetune_olmoearth_pastis.py --config "$CONFIG"
STATUS=$?

# Email the outcome: final BEST/TEST metrics (or a failure notice) + log tail.
{
    echo "config: $CONFIG"
    echo "exit status: $STATUS"
    echo "---"
    grep -aE "^Run:|^BEST|^TEST" "$LOG" || echo "(no metrics found; see log)"
    echo "--- last 20 log lines ---"
    tail -n 20 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] $CONFIG" "$EMAIL"
