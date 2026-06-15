#!/bin/bash
#SBATCH --job-name=pastis_prep
#SBATCH --account=aip-gpleiss
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=pastis_prep_%j.out

# One-time PASTIS-R -> OlmoEarth .pt prep. Needs lots of RAM: PASTISRProcessor
# accumulates all folds in memory before saving, which OOMs at the 64G interactive
# allocation. Run as a batch job:  sbatch prepare_pastis_olmoearth.sbatch

cd "$SLURM_SUBMIT_DIR"
module load python/3.12 scipy-stack opencv libspatialindex proj hdf5/1.14.6
source env_olmo/bin/activate
python -u prepare_pastis_olmoearth.py
