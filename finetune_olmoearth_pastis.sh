#!/bin/bash
#SBATCH --job-name=oe_pastis_ft
#SBATCH --account=aip-gpleiss
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/oe_pastis_ft_%j.out

# OlmoEarth + AnyUp finetune on PASTIS. Pass config fields as --set key=value (all args
# are forwarded). Architecture fields are REQUIRED; tuning knobs default from
# configs/defaults.yaml. See runs.txt for ready-to-run commands.
#   sbatch finetune_olmoearth_pastis.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t1 freeze_backbone=true
# Emails the resolved config at start and the final metrics at end.

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1   # silence tqdm progress bars in the batch log

cd "$SLURM_SUBMIT_DIR"
source env_olmo.sh
# module load python/3.12 scipy-stack opencv libspatialindex proj hdf5/1.14.6
# source env_olmo/bin/activate

# Email the resolved config at start (also validates required fields before training).
python -c "import sys,json; from config import load_config, to_dict; ov=[a for a in sys.argv[1:] if '=' in a]; print(json.dumps(to_dict(load_config(None, ov)), indent=2))" "$@" \
    | mail -s "[START job $SLURM_JOB_ID] $*" "$EMAIL"

LOG="logs/oe_pastis_ft_${SLURM_JOB_ID}.out"
python -u finetune_olmoearth_pastis.py "$@"
STATUS=$?

# Email the outcome: final BEST/TEST metrics (or a failure notice) + log tail.
{
    echo "args: $*"
    echo "exit status: $STATUS"
    echo "---"
    grep -aE "^Run:|^BEST|^TEST" "$LOG" || echo "(no metrics found; see log)"
    echo "--- last 5 log lines ---"
    tail -n 5 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] $*" "$EMAIL"

# Examples (see runs.txt):
# sbatch finetune_olmoearth_pastis.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=lp freeze_backbone=false
# sbatch finetune_olmoearth_pastis.sh --set model_size=base modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t2 freeze_backbone=true