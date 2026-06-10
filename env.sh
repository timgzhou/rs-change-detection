module load python/3.12 scipy-stack opencv libspatialindex proj
export PROJ_DATA=$EBROOTPROJ/share/proj
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install torchgeo==0.9
pip install wandb==0.27