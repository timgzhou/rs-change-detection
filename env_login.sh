module load python/3.12 scipy-stack opencv libspatialindex proj
export PROJ_DATA=$EBROOTPROJ/share/proj
virtualenv --no-download /tmp/env
source /tmp/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
