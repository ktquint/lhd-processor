"""Regionalized Manning's n lookup, sourced from the Lynker/OWP channel-geometry
hydrofabric that underlies the National Water Model (NWM).

feature_id in this table is the NHDPlus V2 COMID (NWM CONUS is built directly on
the NHDPlus V2 medium-resolution flowline network), so dams whose reach was
resolved against NHDPlus in Step 1 (site_data['reach_id']) can be looked up
directly with no additional crosswalk.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import s3fs

_PARQUET_KEY = "lynker-spatial/tabular/riverml_channel_geometry_with_ahg.parquet"


def _cache_path() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    cache_dir = package_root / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'riverml_channel_geometry_with_ahg.parquet'


def _download_geometry_table(status_callback=None, force: bool = False) -> Path:
    path = _cache_path()
    if path.exists() and not force:
        return path
    if status_callback:
        status_callback("Downloading NWM channel-geometry table (one-time, ~190MB)...")
    fs = s3fs.S3FileSystem(anon=True)
    fs.get(_PARQUET_KEY, str(path))
    return path


def get_regional_manning_n(reach_ids, status_callback=None) -> dict:
    """Look up NWM-calibrated Manning's n for a set of NHDPlus COMIDs (NWM feature_ids).

    Returns {reach_id: n}. Reaches missing from the table, or with no valid
    roughness value, are simply absent from the result -- callers should fall
    back to a default n for those.
    """
    reach_ids = {int(r) for r in reach_ids if pd.notna(r)}
    if not reach_ids:
        return {}

    parquet_path = _download_geometry_table(status_callback=status_callback)

    table = pq.read_table(
        parquet_path,
        columns=["feature_id", "owp_roughness_bathy", "owp_roughness_no_bathy"],
        filters=[("feature_id", "in", reach_ids)],
    )
    df = table.to_pandas()

    result = {}
    for _, row in df.iterrows():
        n = row["owp_roughness_bathy"]
        if pd.isna(n) or n <= 0:
            n = row["owp_roughness_no_bathy"]
        if pd.notna(n) and n > 0:
            result[int(row["feature_id"])] = round(float(n), 4)
    return result
