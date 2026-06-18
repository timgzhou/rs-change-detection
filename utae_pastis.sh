#!/bin/bash
#SBATCH --job-name=utae_pastis
#SBATCH --account=aip-gpleiss
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/utae_pastis_%j.out

# UTAE baseline on PASTIS (uni/multimodal). Uses env/ (torch 2.12), NOT env_olmo.
# Pass config fields as --set key=value (all args forwarded). modalities is REQUIRED;
# fusion required when multimodal. Tuning knobs default from configs/utae_defaults.yaml.
#   sbatch utae_pastis.sh --set modalities=S2,S1A fusion=late
# Emails the resolved config at start and the final metrics at end.

EMAIL="tiange.zhou@outlook.com"
export TQDM_DISABLE=1   # silence tqdm progress bars in the batch log

cd "$SLURM_SUBMIT_DIR"
module load python/3.12 scipy-stack opencv libspatialindex proj
source env/bin/activate

# Email the resolved config at start (also validates required fields before training).
python -c "import sys,json; from dataclasses import asdict; from utae_pastis import load_config; ov=[a for a in sys.argv[1:] if '=' in a]; print(json.dumps(asdict(load_config(None, ov)), indent=2))" "$@" \
    | mail -s "[START job $SLURM_JOB_ID] $*" "$EMAIL"

LOG="logs/utae_pastis_${SLURM_JOB_ID}.out"
python -u utae_pastis.py "$@"
STATUS=$?

# Email the outcome: final BEST/TEST metrics (or a failure notice) + log tail.
{
    echo "args: $*"
    echo "exit status: $STATUS"
    echo "---"
    grep -aE "^Run:|^BEST|^TEST" "$LOG" || echo "(no metrics found; see log)"
    echo "--- last 20 log lines ---"
    tail -n 20 "$LOG"
} | mail -s "[DONE job $SLURM_JOB_ID status=$STATUS] $*" "$EMAIL"

# Examples (see runs.txt):
# sbatch utae_pastis.sh --set modalities=S2
# sbatch utae_pastis.sh --set modalities=S2,S1A fusion=early
