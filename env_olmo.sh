# Separate venv for OlmoEarth finetuning (prepare_pastis_olmoearth.py,
# finetune_olmoearth_pastis.py). Kept apart from ./env because olmoearth-pretrain
# pins torch<2.8, which conflicts with ./env's torch 2.12 (pastis.py / torchgeo).
#
# The venv is (re)built fresh PER JOB into $SLURM_TMPDIR (node-local temp) when running under
# Slurm: building into a shared scratch path lets several sbatch jobs starting at once
# pip-install into the SAME location simultaneously, truncating each other's files (seen as
# e.g. "libtorch_cuda.so: file too short"), and a scratch venv also gets corrupted over time.
# $SLURM_TMPDIR is private to each job, so every job gets a clean, isolated install.
# Outside Slurm ($SLURM_TMPDIR unset, e.g. a login/interactive shell) it builds in ./env_olmo
# as before.
#
# hdf5/1.14.6 is required at RUN TIME too: the cluster h5py wheel needs HDF5 1.14.6
# (symbol H5T_IEEE_F16BE_g); without this module it resolves an older libhdf5 and
# fails to import. The finetune scripts also import olmo_shims/olmo_bootstrap first,
# which stubs hdf5plugin and loads h5py before rasterio to dodge the same ABI clash.
module load python/3.12 scipy-stack opencv libspatialindex proj hdf5/1.14.6
export PROJ_DATA=$EBROOTPROJ/share/proj

# Per-job node-local venv under Slurm; ./env_olmo otherwise.
VENV_DIR="${SLURM_TMPDIR:+$SLURM_TMPDIR/}env_olmo"

# -q on every pip and virtualenv keeps the build quiet: suppresses the "Requirement already
# satisfied ..." spam and progress bars, while still printing real warnings/errors.
# --no-warn-conflicts additionally hides the inherited-system-package conflict notice
# (e.g. "blosc2 ... requires msgpack/requests"), which is about cluster packages we don't use.
PIP_Q="-q --no-warn-conflicts"
virtualenv -q --no-download --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install $PIP_Q --no-index --upgrade pip

# Base scientific stack from cluster wheels (also makes the interpreter resolve them).
pip install $PIP_Q --no-index --ignore-installed numpy scipy matplotlib scikit-learn cartopy

# OlmoEarth wants torch 2.7.x (olmoearth-pretrain requires torch<2.8,>=2.7).
pip install $PIP_Q --no-index --ignore-installed torch==2.7.1 torchvision==0.22.1

# OlmoEarth + eval-pipeline deps. olmo-core is the package `ai2-olmo-core`.
# PIN to the known-good versions: olmoearth-pretrain 0.1.1+ eagerly imports new eval dataset
# modules (fifty_cities, geobench_v2, ...) that drag in undeclared deps (pydantic, pandas,
# tacoreader->pyarrow), breaking `import lp_on_cached_features`. 0.1.0 is what we validated.
pip install $PIP_Q olmoearth-pretrain-minimal==0.0.5 olmoearth-pretrain==0.1.0 ai2-olmo-core==2.4.0
pip install $PIP_Q class-registry rioxarray

# Install h5py from the cluster wheel LAST so the olmoearth deps don't pull a
# PyPI build with a mismatched HDF5/numpy ABI. --no-deps so it leaves numpy/torch alone.
pip install $PIP_Q --no-index --ignore-installed --no-deps h5py
pip install $PIP_Q torchmetrics
# source env_olmo.sh
