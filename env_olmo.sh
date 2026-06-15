# Separate venv for OlmoEarth finetuning (prepare_pastis_olmoearth.py,
# finetune_olmoearth_pastis.py). Kept apart from ./env because olmoearth-pretrain
# pins torch<2.8, which conflicts with ./env's torch 2.12 (pastis.py / torchgeo).
#
# hdf5/1.14.6 is required at RUN TIME too: the cluster h5py wheel needs HDF5 1.14.6
# (symbol H5T_IEEE_F16BE_g); without this module it resolves an older libhdf5 and
# fails to import. The finetune scripts also import olmo_shims/olmo_bootstrap first,
# which stubs hdf5plugin and loads h5py before rasterio to dodge the same ABI clash.
module load python/3.12 scipy-stack opencv libspatialindex proj hdf5/1.14.6
export PROJ_DATA=$EBROOTPROJ/share/proj
virtualenv --no-download --system-site-packages env_olmo
source env_olmo/bin/activate
pip install --no-index --upgrade pip

# Base scientific stack from cluster wheels (also makes the interpreter resolve them).
pip install --no-index --ignore-installed numpy scipy matplotlib scikit-learn cartopy

# OlmoEarth wants torch 2.7.x (olmoearth-pretrain requires torch<2.8,>=2.7).
pip install --no-index --ignore-installed torch==2.7.1 torchvision==0.22.1

# OlmoEarth + eval-pipeline deps. olmo-core is the package `ai2-olmo-core`.
pip install olmoearth-pretrain-minimal olmoearth-pretrain ai2-olmo-core
pip install class-registry rioxarray

# Install h5py from the cluster wheel LAST so the olmoearth deps don't pull a
# PyPI build with a mismatched HDF5/numpy ABI. --no-deps so it leaves numpy/torch alone.
pip install --no-index --ignore-installed --no-deps h5py
pip install torchmetrics
# source env_olmo.sh
