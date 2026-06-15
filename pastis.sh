#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/pastis/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
source env.sh
python -u pastis.py
# sbatch pastis.sh