"""One-time prep: convert raw PASTIS-R into the .pt format OlmoEarth's eval expects.

Uses OlmoEarth's own PASTISRProcessor, which:
  - imputes the 10 PASTIS S2 bands up to OlmoEarth's 13->12 band layout (by duplication),
  - aggregates the irregular time series into <=12 monthly averages,
  - splits each 128x128 tile into 4x 64x64,
  - maps PASTIS void class 19 -> ignore label,
  - writes pastis_r_{train,valid,test}/{s2_images,s1_images}/, months.pt, targets.pt.

Run once (uses env_olmo, NOT env):
    source env_olmo.sh
    python -u prepare_pastis_olmoearth.py
"""
import os
import sys
# Bootstrap MUST run before any olmoearth_pretrain import (see olmo_shims/olmo_bootstrap.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()  # MUST run before any olmoearth_pretrain import

from olmoearth_pretrain.evals.datasets.pastis_processor import PASTISRProcessor  # noqa: E402

DATA_DIR = "data/PASTIS-R"            # raw PASTIS-R (DATA_S2/, DATA_S1A/, ANNOTATIONS/, metadata.geojson)
OUTPUT_DIR = "data/pastis_olmoearth"  # consumed by PASTISRDataset(path_to_splits=...)

if __name__ == "__main__":
    processor = PASTISRProcessor(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        resize_to_64=True,  # 64x64 tiles -> use the "pastis" config (not "pastis128")
    )
    processor.process()
    print(f"Done. Processed PASTIS-R written to {OUTPUT_DIR}/")
