module load python/3.12 scipy-stack opencv libspatialindex proj
export PROJ_DATA=$EBROOTPROJ/share/proj
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade -q pip 
pip install --no-index -q -r requirements.txt
pip install -q torchgeo==0.9
pip install -q wandb olmoearth-pretrain-minimal  ai2-olmo-core class-registry rioxarray