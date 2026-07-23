#!/bin/bash
#SBATCH --job-name=usf_lp_sweep
#SBATCH --account=aip-gpleiss
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/usf_lp_sweep_%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL

#
# UrbanSARFloods LP sweep over (tile_size, patch_size) x head.
# Assumes the dataset is already downloaded+extracted (data/urban_sar_floods/).
# Results -> results/urbansarfloods_lp.csv ; per-run viz -> viz/<features>_<head>_wce.png
#
# Grid (patch_size must divide tile_size):
#   tile 16 -> patch 2, 4
#   tile 32 -> patch 4, 8
#   tile 64 -> patch 4, 8
# Each feature config is probed with BOTH heads: concat, diff.
#
#   sbatch run_urbansarfloods_sweep.sh          # batch (recommended)
#   bash   run_urbansarfloods_sweep.sh          # or interactively inside a GPU salloc

set -e

cd "${SLURM_SUBMIT_DIR:-.}"      # SLURM starts in $HOME; no-op when run interactively
export TQDM_DISABLE=1            # silence tqdm progress bars in the batch log
source env_olmo.sh

CSV=results/urbansarfloods_lp.csv

# (tile_size, space-separated patch sizes)
declare -A PATCHES=( [16]="2 4" [32]="2 4 8" [64]="4 8" )

# 1. Tile once per tile size (writes data/urbansarfloods_tiles_t<tile>/)
for TILE in 16 32 64; do
  echo "=== TILE $TILE: prep ==="
  python -u prep_urbansarfloods_tiles.py --splits train,valid --tile_size "$TILE"
done

# 2. Extract features for each (tile, patch); 3. LP with both heads per feature config
for TILE in 16 32 64; do
  for PS in ${PATCHES[$TILE]}; do
    FEAT="usf_base_s1_ps${PS}_res20_t${TILE}"
    echo "=== EXTRACT tile=$TILE patch=$PS -> $FEAT ==="
    python -u extract_urbansarfloods_features.py --splits train,valid --tile_size "$TILE" --patch_size "$PS"

    for HEAD in concat diff; do
      echo "=== LP $FEAT head=$HEAD ==="
      python -u lp_urbansarfloods.py --features "$FEAT" --head "$HEAD" \
        --weighted_ce --results_csv "$CSV"
    done
  done
done

echo "=== DONE. Results: $CSV ; visualizations: viz/ ==="

# sbatch run_urbansarfloods_sweep.sh