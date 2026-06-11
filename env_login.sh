module load python/3.12 scipy-stack opencv libspatialindex proj
export PROJ_DATA=$EBROOTPROJ/share/proj
virtualenv --no-download --system-site-packages env
source env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install torchgeo==0.9
pip install --no-index --ignore-installed numpy matplotlib scipy # this is necessary for python interpreter
# source env_login.sh
