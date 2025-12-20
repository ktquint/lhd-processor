import sys
import os
import io
import requests
import math
import zipfile
import pandas as pd
import geopandas as gpd
import numpy as np
import s3fs
import fiona
import pyproj
from shapely.geometry import Point
from typing import Optional, Any, List, Tuple
from math import radians, sin, cos, sqrt, atan2

# --- CRITICAL FIX FOR PROJ/GDAL CONFLICTS (Updated for robustness) ---
try:
    conda_prefix = sys.prefix
    proj_dir = os.path.join(conda_prefix, 'share', 'proj')
    pyproj.datadir.set_data_dir(proj_dir)
    os.environ['PROJ_LIB'] = proj_dir
    os.environ['PROJ_DATA'] = proj_dir
    print(f"--- Environment Configuration ---")
    print(f"Forcing PROJ data directory to: {pyproj.datadir.get_data_dir()}")
except Exception as e:
    print(f"Warning: Could not set PROJ paths automatically: {e}")


# --------------------------------------------------------------------

# =====================================================================
# --- CORE GEOGRAPHIC UTILITY FUNCTIONS ---
# =====================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes approximate distance (km) between two points."""
    R = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def make_bbox(latitude: float, longitude: float, distance_deg: float = 0.5) -> List[float]:
    """Creates a bounding box around a point (lat, lon) ±distance_deg degrees."""
    lat_min = latitude - distance_deg
    lat_max = latitude + distance_deg
    lon_dist_adj = distance_deg / math.cos(radians(latitude))
    lon_min = longitude - lon_dist_adj
    lon_max = longitude + lon_dist_adj
    return [lon_min, lat_min, lon_max, lat_max]


def sanitize_filename(filename):
    """Sanitizes filename from invalid characters like '/'. (lhd_processor/prep/download_dem.py)"""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "-")
    return filename


# =====================================================================
# --- GEOGRAPHIC IDENTIFICATION FUNCTIONS ---
# =====================================================================

def find_huc4(latitude: float, longitude: float) -> Optional[str]:
    """Finds the HUC4 code by querying the USGS water services."""
    bbox = make_bbox(latitude, longitude, 0.1)
    bbox_url = (f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
                f"&bBox={bbox[0]:.7f},{bbox[1]:.7f},{bbox[2]:.7f},{bbox[3]:.7f}"
                f"&siteType=ST")

    try:
        response = requests.get(bbox_url, timeout=10)
        response.raise_for_status()
        response_df = pd.read_csv(io.StringIO(response.text), sep="\t", comment="#", skip_blank_lines=True)

        stream_df = response_df[response_df['site_no'].astype(str).str.len() <= 8].copy()
        if stream_df.empty: return None

        stream_df['dec_lat_va'] = pd.to_numeric(stream_df['dec_lat_va'], errors='coerce')
        stream_df['dec_long_va'] = pd.to_numeric(stream_df['dec_long_va'], errors='coerce')
        stream_df = stream_df.dropna(subset=['dec_lat_va', 'dec_long_va'])

        stream_df['distance_km'] = stream_df.apply(
            lambda row: haversine(latitude, longitude, row['dec_lat_va'], row['dec_long_va']),
            axis=1
        )
        nearest_site = stream_df.loc[stream_df['distance_km'].idxmin()]

        if 'huc_cd' in nearest_site and pd.notna(nearest_site['huc_cd']):
            return str(nearest_site['huc_cd'])[:4]
        return None
    except Exception as e:
        print(f"USGS query failed: {e}")
        return None


def find_vpu(latitude: float, longitude: float, vpu_map_path: str) -> Optional[str]:
    """Finds the VPU code by querying the VPU boundary map."""
    if not os.path.exists(vpu_map_path):
        print(f"CRITICAL ERROR: Could not find VPU boundaries file at {vpu_map_path}")
        return None
    try:
        vpu_gdf = gpd.read_file(vpu_map_path)

        # --- ROBUST CRS HANDLING ---
        try:
            # 1. Attempt standard conversion if CRS is not WGS84
            if vpu_gdf.crs is None or vpu_gdf.crs.to_string() != 'EPSG:4326':
                vpu_gdf = vpu_gdf.to_crs(epsg=4326)
        except Exception:
            # 2. If conversion fails (due to PROJ/CRS definition error), force-set WGS84
            vpu_gdf = vpu_gdf.set_crs(epsg=4326, allow_override=True)
            print("  > VPU Warning: Forced CRS to EPSG:4326 due to projection instability.")
        # --- END CRS HANDLING ---

        dam_point = Point(longitude, latitude)
        vpu_polygon = vpu_gdf[vpu_gdf.geometry.contains(dam_point)]
        if vpu_polygon.empty: return None

        cols_lower = {col.lower(): col for col in vpu_polygon.columns}
        vpu_col_name = cols_lower.get('vpu') or cols_lower.get('vpucode')
        return str(vpu_polygon.iloc[0][vpu_col_name]) if vpu_col_name else None
    except Exception as e:
        print(f"VPU map query failed: {e}")
        return None


# =====================================================================
# --- DOWNLOAD FUNCTIONS ---
# =====================================================================

def download_geoglows_flowline(vpu_code: str, flowline_dir: str) -> Optional[str]:
    """Downloads the TDXHYDRO (GEOGLOWS) VPU GeoPackage from S3."""
    gpkg_loc = os.path.join(flowline_dir, f"streams_{vpu_code}.gpkg")
    if os.path.exists(gpkg_loc):
        print(f"  > GEOGLOWS: Local file {gpkg_loc} already exists. Using it.")
        return gpkg_loc

    print(f"  > GEOGLOWS: Downloading VPU {vpu_code} from S3...")
    base_url = 'geoglows-v2/hydrography/'
    gpkg_name = f"streams_{vpu_code}.gpkg"
    gpkg_url = f"{base_url}vpu={vpu_code}/{gpkg_name}"

    try:
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(gpkg_url, 'rb') as f_in:
            with open(gpkg_loc, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"  > GEOGLOWS: Successfully downloaded {os.path.basename(gpkg_loc)}")
        return gpkg_loc
    except Exception as e:
        print(f"  > GEOGLOWS: Failed to download {gpkg_url}: {e}")
        return None


def download_nhdplus_flowline(huc4: str, latitude: float, longitude: float, flowline_dir: str) -> Optional[str]:
    """Downloads, unzips, and merges NHDPlus HR data for the HUC4. (Modified to check against user's naming convention)"""

    # 1. NEW LOCAL CHECK: Find files that match the HUC4 and the expected VAA_RI suffix
    # Accounts for leading zero in HUC4 if necessary
    huc4_padded = huc4 if len(huc4) == 4 else f"0{huc4}"
    local_files = [f for f in os.listdir(flowline_dir) if
                   f.startswith(f"NHDPLUS_H_{huc4_padded}") and f.endswith(".gpkg")]

    if local_files:
        # Use the first match found.
        processed_loc = os.path.join(flowline_dir, local_files[0])
        print(f"  > NHDPlus: Found locally existing file {local_files[0]}. Using it.")
        return processed_loc

    # 2. EXTERNAL DOWNLOAD (If local file not found)
    processed_filename = f"NHDPlus_HR_{huc4}_VAA_RI.gpkg"
    processed_loc = os.path.join(flowline_dir, processed_filename)

    # TNM Access Query setup
    bbox_query = make_bbox(latitude, longitude, 0.01)
    product = "National Hydrography Dataset Plus High Resolution (NHDPlus HR)"
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params = {"bbox": f"{bbox_query[0]},{bbox_query[1]},{bbox_query[2]},{bbox_query[3]}",
              "datasets": product, "max": 10, "format": "GeoPackage", "outputFormat": "JSON", }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        huc_results = [item for item in results if
                       f'_{huc4}_' in item.get("downloadURL", "").lower() and 'gpkg' in item.get("downloadURL",
                                                                                                 "").lower()]

        if not huc_results:
            print(f"  > NHDPlus: No NHDPlus HR GeoPackage found for HUC4: {huc4}")
            return None

        final_gpkg = huc_results[0]
        title = final_gpkg.get("title", f"NHD_{huc4}")
        download_url = final_gpkg.get("downloadURL", "")

        gpkg_zip_loc = os.path.join(flowline_dir, sanitize_filename(title) + ".zip")
        gpkg_extract_loc = os.path.join(flowline_dir, download_url.rsplit('/', 1)[-1].replace('.zip', '.gpkg'))

        # 1. Download, 2. Unzip, 3. Merge VAA, 4. Save Processed
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(gpkg_zip_loc, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        with zipfile.ZipFile(gpkg_zip_loc, 'r') as zip_ref:
            zip_ref.extractall(flowline_dir)
        os.remove(gpkg_zip_loc)

        flowlines_gdf = gpd.read_file(filename=gpkg_extract_loc, layer='NHDFlowline', engine='fiona')
        metadata_gdf = gpd.read_file(filename=gpkg_extract_loc, layer='NHDPlusFlowlineVAA', engine='fiona')
        merged_gdf = flowlines_gdf.merge(metadata_gdf, on=['nhdplusid', 'reachcode', 'vpuid'])

        # Use the standard name for the downloaded file
        merged_gdf.to_file(filename=processed_loc, layer='NHDFlowline', driver='GPKG')
        os.remove(gpkg_extract_loc)

        return processed_loc
    except Exception as e:
        print(f"  > NHDPlus: An unexpected error occurred: {e}")
        return None


# =====================================================================
# --- SPATIAL QUERY FUNCTIONS (for finding IDs AND DISTANCE) ---
# =====================================================================

def find_nearest_flowline_data(
        latitude: float,
        longitude: float,
        flowline_gpkg_path: str,
        layer_name: str,
        id_col: str,
        order_col: str,
        extra_col: Optional[str] = None
) -> Tuple[Optional[Any], Optional[Any], Optional[float]]:
    """
    Finds the primary ID, an optional secondary ID (like hydroseq or None), and distance (in meters).
    """
    # Define Conus Albers Projected CRS (meters)
    PROJECTED_CRS = "EPSG:5070"

    if not os.path.exists(flowline_gpkg_path):
        return None, None, None

    try:
        with fiona.open(flowline_gpkg_path, layer=layer_name) as src:
            # We don't use src.crs as the target_crs for distance, but to project the point
            target_crs_for_read = src.crs
            cols_lower = {col.lower(): col for col in src.schema['properties']}

            if id_col.lower() not in cols_lower: return None, None, None
            id_col_case = cols_lower[id_col.lower()]
            order_col_case = cols_lower.get(order_col.lower())
            extra_col_case = cols_lower.get(extra_col.lower()) if extra_col else None
    except Exception:
        return None, None, None

        # 1. Prepare Data for Calculation (Projected CRS)
    dam_point_latlon = Point(longitude, latitude)
    dam_point_gdf = gpd.GeoSeries([dam_point_latlon], crs="EPSG:4326")

    # Read GDF and set/convert CRS to the metric one
    try:
        gdf = gpd.read_file(flowline_gpkg_path, layer=layer_name)
    except Exception as e:
        print(f"Error reading local GPKG for spatial query: {e}")
        return None, None, None

    if gdf.empty: return None, None, None

    # A. Project GDF to the metric CRS (this suppresses the UserWarning)
    try:
        gdf = gdf.to_crs(PROJECTED_CRS)
    except Exception:
        # If the GeoPackage CRS definition is broken, this conversion might fail.
        # In this case, we rely on the internal error handling in the conversion logic.
        print("Warning: Failed to convert GeoDataFrame to EPSG:5070. Skipping site.")
        return None, None, None

    # B. Project Point to the metric CRS
    dam_point_projected = dam_point_gdf.to_crs(PROJECTED_CRS).iloc[0]

    # 2. Read GeoDataFrame with Buffer of 500 meters
    search_buffer = dam_point_projected.buffer(500)
    # Use the bounding box of the projected buffer to filter the already loaded (and projected) GDF
    streams_inside_buffer = gdf.cx[search_buffer.bounds[0]:search_buffer.bounds[2],
    search_buffer.bounds[1]:search_buffer.bounds[3]].copy()

    if streams_inside_buffer.empty: return None, None, None

    # 3. Calculate Distance and Find Nearest (Prioritizing Stream Order)
    streams_inside_buffer["distance"] = streams_inside_buffer.geometry.distance(dam_point_projected)

    nearest = None
    if order_col_case and order_col_case in streams_inside_buffer.columns:
        try:
            valid_order_streams = streams_inside_buffer.dropna(subset=[order_col_case])
            if not valid_order_streams.empty:
                max_strm_order = valid_order_streams[order_col_case].max()
                highest_order_streams = streams_inside_buffer[streams_inside_buffer[order_col_case] == max_strm_order]
                if not highest_order_streams.empty:
                    nearest = highest_order_streams.loc[highest_order_streams["distance"].idxmin()]
        except Exception:
            pass

            # Fallback to simple nearest neighbor
    if nearest is None:
        nearest = streams_inside_buffer.loc[streams_inside_buffer["distance"].idxmin()]

    # 4. Extract Results
    primary_id = int(nearest[id_col_case]) if pd.notna(nearest[id_col_case]) else None
    secondary_id = int(nearest[extra_col_case]) if extra_col_case and pd.notna(nearest[extra_col_case]) else None
    distance = float(nearest['distance'])

    return primary_id, secondary_id, distance


# =====================================================================
# --- MAIN PROCESSING FUNCTION (DE-DUPLICATED) ---
# =====================================================================
import concurrent.futures


# --- HELPER WORKER FUNCTIONS (Must be top-level for pickling) ---

def _worker_identify_metadata(args):
    """Phase 1 Worker: Just finds the HUC4 and VPU for a site."""
    index, row, vpu_map_path = args
    lat = row.get("latitude")
    lon = row.get("longitude")
    site_id = row.get("site_id", f"Row_{index}")

    if pd.isna(lat) or pd.isna(lon):
        return None

    huc4 = find_huc4(lat, lon)
    vpu_code = find_vpu(lat, lon, vpu_map_path)

    return {
        "index": index,
        "site_id": site_id,
        "latitude": lat,
        "longitude": lon,
        "HUC4": huc4,
        "VPU_Code": vpu_code
    }


def _worker_extract_data(args):
    """Phase 3 Worker: Performs the heavy spatial extraction."""
    meta, strm_dir, unique_huc4_paths, unique_vpu_paths = args

    lat = meta['latitude']
    lon = meta['longitude']
    huc4 = meta['HUC4']
    vpu_code = meta['VPU_Code']

    # Resolve local paths
    nhd_gpkg_path = unique_huc4_paths.get(huc4)
    geoglows_gpkg_path = unique_vpu_paths.get(vpu_code)

    # NHDPlus Extraction
    nhd_id, hydroseq, dist_nhd = None, None, None
    if nhd_gpkg_path:
        nhd_id, hydroseq, dist_nhd = find_nearest_flowline_data(
            lat, lon, nhd_gpkg_path, 'NHDFlowline',
            id_col='NHDPlusID', order_col='StreamOrder', extra_col='hydroseq'
        )

    # GEOGLOWS Extraction
    geoglows_id, dist_tdx = None, None
    if geoglows_gpkg_path:
        geoglows_id, _, dist_tdx = find_nearest_flowline_data(
            lat, lon, geoglows_gpkg_path, f'streams_{vpu_code}',
            id_col='LINKNO', order_col='strmOrder'
        )

    # Return combined result
    result = meta.copy()
    result.update({
        "NHD_Reach_ID": nhd_id,
        "NHD_Hydroseq_ID": hydroseq,
        "GEOGLOWS_Link_No": geoglows_id,
        "dist_to_nhd": dist_nhd,
        "dist_to_tdx": dist_tdx
    })
    return result


# --- MAIN FUNCTION REPLACEMENT ---

def extract_flowline_data_deduped(database_xlsx, output_csv, vpu_map_path, strm_dir, status_callback=None):
    if not os.path.exists(database_xlsx):
        print(f"Error: Database not found: {database_xlsx}")
        return 0

    # 1. Load Data
    try:
        df_original = pd.read_excel(database_xlsx)
    except Exception:
        try:
            df_original = pd.read_csv(database_xlsx, encoding='latin-1')
        except Exception:
            print("Error: Could not read file.")
            return 0

    # Normalize columns
    if 'Latitude' in df_original.columns: df_original.rename(columns={'Latitude': 'latitude'}, inplace=True)
    if 'Longitude' in df_original.columns: df_original.rename(columns={'Longitude': 'longitude'}, inplace=True)

    dedupe_cols = ['site_id', 'latitude', 'longitude']
    df_unique = df_original.drop_duplicates(subset=dedupe_cols).reset_index(drop=True).copy()
    total = len(df_unique)

    print(f"--- Processing {total} Unique Sites ---")

    # =================================================================
    # PHASE 1: Identify HUC4 and VPU (Parallel IO/CPU)
    # =================================================================
    print(f"Phase 1: Identifying Hydrologic Regions (Parallel)...")
    site_metadata = []

    # Prepare arguments for workers
    phase1_args = [(i, row, vpu_map_path) for i, row in df_unique.iterrows()]

    # Use ProcessPoolExecutor for mixed IO/CPU work
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_worker_identify_metadata, phase1_args))

    # Filter out failures
    site_metadata = [r for r in results if r is not None]

    # =================================================================
    # PHASE 2: Download Missing Files (Sequential to avoid corruption)
    # =================================================================
    print(f"Phase 2: Verifying & Downloading Data Files...")

    unique_huc4s = set(m['HUC4'] for m in site_metadata if m['HUC4'])
    unique_vpus = set(m['VPU_Code'] for m in site_metadata if m['VPU_Code'])

    unique_huc4_paths = {}
    unique_vpu_paths = {}

    # Download NHD Files
    for huc4 in unique_huc4s:
        # We need a representative lat/lon to query the map service if downloading
        rep_site = next(m for m in site_metadata if m['HUC4'] == huc4)
        path = download_nhdplus_flowline(huc4, rep_site['latitude'], rep_site['longitude'], strm_dir)
        if path: unique_huc4_paths[huc4] = path

    # Download GEOGLOWS Files
    for vpu in unique_vpus:
        path = download_geoglows_flowline(vpu, strm_dir)
        if path: unique_vpu_paths[vpu] = path

    # =================================================================
    # PHASE 3: Spatial Extraction (Parallel CPU)
    # =================================================================
    print(f"Phase 3: Extracting Flowline IDs (Parallel)...")

    phase3_args = [(m, strm_dir, unique_huc4_paths, unique_vpu_paths) for m in site_metadata]

    final_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # We use map to keep order, or as_completed for progress bar
        futures = {executor.submit(_worker_extract_data, arg): arg for arg in phase3_args}

        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            final_results.append(res)
            completed_count += 1
            if status_callback and completed_count % 5 == 0:
                status_callback(f"Phase 3 Progress: {completed_count}/{len(phase3_args)}")
            elif completed_count % 10 == 0:
                print(f"  > Processed {completed_count}/{len(phase3_args)} sites")

    # =================================================================
    # MERGE & SAVE
    # =================================================================
    if not final_results:
        print("No results generated.")
        return 0

    df_results = pd.DataFrame(final_results)

    # Merge back to original (handling duplicates)
    df_final = pd.merge(df_original, df_results[['site_id', 'latitude', 'longitude',
                                                 'HUC4', 'VPU_Code', 'NHD_Reach_ID',
                                                 'NHD_Hydroseq_ID', 'GEOGLOWS_Link_No',
                                                 'dist_to_nhd', 'dist_to_tdx']],
                        on=dedupe_cols, how='left')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    print(f"--- Complete. Saved {len(df_final)} rows to {output_csv} ---")
    return len(df_final)


def main():
    # --- USER INPUT SECTION ---
    # !! REPLACE THESE WITH YOUR ACTUAL PATHS !!
    # NOTE: Changed input file extension to .xlsx based on user's previous context
    database_xlsx = '/Users/kennyquintana/Developer/lhd-processor/lhd_processor/data/LHD Sites and Fatalities.xlsx'
    output_csv = '/Users/kennyquintana/Developer/lhd-processor/lhd_processor/data/nhd_geoglows_flowline_data_final.csv'
    # Path to the vpu-boundaries.gpkg file (download this once)
    vpu_map_path = '/Users/kennyquintana/Developer/lhd-processor/lhd_processor/data/vpu-boundaries.gpkg'
    # This directory must exist and is where the flowlines will be stored.
    strm_dir = '/Volumes/KenDrive/LHD_Project/STRM'

    # --------------------------

    def update_status(msg):
        print(msg)

    print("--- Starting DEDUPLICATED Flowline Extraction ---")
    count = extract_flowline_data_deduped(database_xlsx, output_csv, vpu_map_path, strm_dir, update_status)
    print(f"--- Finished. Processed {count} total rows. ---")
    print(f"Output file: {output_csv}")


if __name__ == "__main__":
    main()