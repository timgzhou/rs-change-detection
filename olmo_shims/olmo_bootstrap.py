"""Import bootstrap for using OlmoEarth's PASTIS eval pipeline on this cluster.

Two problems this works around, both from OlmoEarth's eval package eagerly importing
every dataset/model wrapper (geobench, anysat, clay, ...) we never use for PASTIS:

1. h5py/hdf5plugin/rasterio ABI clash on the cluster wheels:
   - the real `hdf5plugin` (HDF5 1.14.2) breaks the cluster `h5py` (needs 1.14.6);
   - importing `rasterio` before `h5py` loads an older libhdf5 that also breaks h5py.
   Fix: shadow `hdf5plugin` with a no-op stub (PASTIS never uses HDF5) and import
   `h5py` early, before anything pulls rasterio.

2. uninstallable / unused deps (geobench, and many competitor-model wrappers).
   Fix: pre-register lightweight stub modules in sys.modules so the eager package
   __init__ sweeps succeed. get_eval_wrapper does isinstance() against the model
   wrapper classes, so each stub exposes dummy classes (our model is matched by
   FlexiVitBase/STBase, which are NOT stubbed, so dispatch still works).

Call apply() FIRST, before any olmoearth_pretrain import:
    import olmo_bootstrap
    olmo_bootstrap.apply()
"""
import sys
import types


def _stub_module(name: str, classes: list[str]) -> None:
    mod = types.ModuleType(name)
    for cls in classes:
        setattr(mod, cls, type(cls, (), {}))
    sys.modules[name] = mod


def apply() -> None:
    """Install the import shims. Idempotent; call before importing olmoearth_pretrain."""
    # --- (1) stub hdf5plugin, load h5py early ---
    # The real hdf5plugin (HDF5 1.14.2) breaks the cluster h5py (needs 1.14.6); and
    # importing rasterio before h5py loads an older libhdf5 that also breaks it.
    # PASTIS never uses HDF5, so a no-op hdf5plugin stub + early h5py import is safe.
    stub = types.ModuleType("hdf5plugin")
    stub.FILTERS = {}
    sys.modules.setdefault("hdf5plugin", stub)
    import h5py  # noqa: F401  # load cluster HDF5 1.14.6 before rasterio can shadow it

    # --- (2) stub unused eval model wrappers + datasets (heavy/uninstallable deps) ---
    # get_eval_wrapper does isinstance() against these wrapper classes; stubbing them as
    # unique dummies means our OlmoEarth encoder (FlexiVitBase, NOT stubbed) still matches.
    _stub_module(
        "olmoearth_pretrain.evals.models",
        ["AnySat", "Clay", "Croma", "DINOv3", "GalileoWrapper", "Panopticon",
         "PrestoWrapper", "PrithviV2", "Satlas", "Terramind", "Tessera"],
    )
    # evals/datasets/__init__ eagerly imports every dataset sibling; stub the unused ones
    # (they pull geobench / extra deps). pastis_dataset + normalize load for real.
    _stub_module("olmoearth_pretrain.evals.datasets.breizhcrops", ["BreizhCropsDataset"])
    _stub_module("olmoearth_pretrain.evals.datasets.floods_dataset", ["Sen1Floods11Dataset"])
    _stub_module("olmoearth_pretrain.evals.datasets.geobench_dataset", ["GeobenchDataset"])
    _stub_module("olmoearth_pretrain.evals.datasets.mados_dataset", ["MADOSDataset"])
    _stub_module("olmoearth_pretrain.evals.datasets.rslearn_dataset", ["RslearnToOlmoEarthDataset"])
