import os
import re
import s3fs
import laspy
import requests
import shutil
import tempfile
import zipfile
import rasterio
import numpy as np
import pandas as pd
import laspy.errors
import pynhd as nhd
from pynhd import NLDI
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import Point
from datetime import datetime, timedelta
import time
import traceback
from typing import Optional, Union, Tuple, List

try:
    import gdal
except ImportError:
    from osgeo import gdal

# Direct NLDI endpoint for the initial COMID lookup. We bypass pynhd's
# comid_byloc here because its pygeoutils JSON parser crashes (surfaces as
# "CRS attribute" AttributeError) on certain server responses, which would
# make ~70% of NID-sourced dam coordinates fail before the navigation step.
_NLDI_BASE = "https://labs-beta.waterdata.usgs.gov/api/nldi/linked-data"
_NLDI_COMID_URL = f"{_NLDI_BASE}/comid/position"


def _fetch_seed_flowline(comid: int, timeout: float = 30) -> Optional[gpd.GeoDataFrame]:
    """Return a 1-row GeoDataFrame with the geometry of `comid`, or None.

    Used as a fallback when neither upstream nor downstream navigation
    returns anything (e.g. headwater + terminal reach, coastline,
    disconnected canal). Lets us at least produce a flowline gpkg
    containing the seed reach so the dam isn't fully dropped from the
    pipeline.
    """
    try:
        r = requests.get(f"{_NLDI_BASE}/comid/{int(comid)}", timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except ValueError:
        return None
    feats = payload.get("features") or []
    if not feats:
        return None
    try:
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    except Exception:
        return None
    if gdf.empty:
        return None
    if "comid" not in gdf.columns and "identifier" in gdf.columns:
        gdf = gdf.rename(columns={"identifier": "comid"})
    return gdf


def _lookup_comid_direct(lon: float, lat: float, timeout: float = 30) -> Optional[int]:
    """Return the NHDPlus V2 COMID nearest (lon, lat), or None if NLDI 404s.

    No retries (caller decides). 404 means NLDI has no flowline at that point.
    """
    try:
        r = requests.get(_NLDI_COMID_URL, params={"coords": f"POINT({lon} {lat})"}, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except ValueError:
        return None
    feats = payload.get("features") or []
    if not feats:
        return None
    props = feats[0].get("properties") or {}
    val = props.get("comid") or props.get("identifier")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def sanitize_filename(filename):
    """
        some files have '/' in their name like "1/3 arc-second," so we'll fix it
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "-")  # Replace invalid characters with '_'
    return filename


# =================================================================
# 1. FLOWLINE FUNCTIONS (NHDPlus & TDX-Hydro)
# =================================================================

def download_nhd_flowline(lat: float, lon: float, flowline_dir: str, distance_km: Union[float, Tuple[float, float], List[float]] = (1, 2), site_id=None, vaa_df=None):
    """
    Uses HyRiver NLDI to fetch NHDPlus flowlines, merges VAAs (hydroseq, dnhydroseq),
    and standardizes ID columns.

    vaa_df : pre-fetched result of nhd.nhdplus_vaa(). When provided, the function
             skips the download entirely — pass this when calling from a parallel
             context to avoid concurrent writes to the shared cache file.
    """
    try:
        if distance_km is None:
            distance_km = (1, 2)

        # Retry NLDI initialization as it fetches metadata from the web and can fail
        nldi = None
        for attempt in range(3):
            try:
                nldi = NLDI()
                # Explicitly check if nldi was actually created
                if nldi is None:
                    raise ValueError("NLDI service returned None")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)  # Increase sleep time
                else:
                    print(f"Failed to initialize NLDI after retries: {e}")
                    return None, None

        # Direct HTTP call to NLDI for COMID resolution — see _lookup_comid_direct
        # for why we bypass pynhd here.
        comid_val = _lookup_comid_direct(lon, lat)
        if comid_val is None:
            return None, None

        # Create site-specific subdirectory if site_id is provided
        if site_id:
            site_flowline_dir = os.path.join(flowline_dir, str(site_id))
        else:
            site_flowline_dir = flowline_dir
            
        os.makedirs(site_flowline_dir, exist_ok=True)
        output_path = os.path.join(site_flowline_dir, f"nhd_flowline_{comid_val}.gpkg")
        
        if os.path.exists(output_path):
            try:
                # print(f"Found existing flowline file: {output_path}") # User asked to limit prints
                gdf = gpd.read_file(output_path)
                return output_path, gdf
            except Exception as e:
                print(f"Error reading existing file, redownloading: {e}")

        all_reaches = []
        # upstreamTributaries walks every branch (catches braided / canal /
        # side-channel networks that main-stem navigation skips); downstreamMain
        # stays on the main path because tributaries downstream rarely make
        # sense from a screening perspective.
        nav_modes = [("upstreamTributaries", "up"), ("downstreamMain", "down")]

        for mode, direction in nav_modes:
            try:
                # Handle distance logic safely
                dist = distance_km
                if isinstance(distance_km, (tuple, list)):
                    if len(distance_km) == 2:
                        dist = distance_km[0] if direction == "up" else distance_km[1]
                    elif len(distance_km) > 0:
                        dist = distance_km[0]
                    else:
                        dist = 1 # Default fallback

                reach = nldi.navigate_byid(
                    fsource="comid",
                    fid=str(comid_val),
                    navigation=mode,
                    source="flowlines",
                    distance=dist
                )
                if reach is not None and not reach.empty:
                    all_reaches.append(reach)
            except Exception as e:
                print(f"NLDI Navigation Error ({mode}): {e}")

        if not all_reaches:
            # Fallback: both nav directions came back empty (headwater +
            # terminal reach, coastline, disconnected canal). Save just the
            # seed COMID's own geometry so the dam still gets a flowline file.
            seed = _fetch_seed_flowline(comid_val)
            if seed is None or seed.empty:
                print(f"  Both navigations empty AND seed fetch failed for COMID {comid_val} @ ({lat:.4f},{lon:.4f})")
                return None, None
            print(f"  Both navigations empty — using seed-only flowline for COMID {comid_val}")
            all_reaches.append(seed)

        combined_df = pd.concat(all_reaches)
        combined_df.columns = combined_df.columns.str.lower()

        # Standardization: Ensure 'nhdplusid' exists
        id_map = ['nhdplus_comid', 'comid', 'nhdplusid']
        for col in id_map:
            if col in combined_df.columns:
                combined_df = combined_df.rename(columns={col: 'nhdplusid'})
                break

        # --- FETCH AND MERGE VAAs ---
        # Convert ID to numeric for the join
        if 'nhdplusid' in combined_df.columns:
            combined_df['nhdplusid'] = pd.to_numeric(combined_df['nhdplusid'])

            # Fetch the VAA table (hydroseq and dnhydroseq are in the 'vaa' service)
            # Note: nhdplus_vaa() fetches the national parquet file
            try:
                if vaa_df is None:
                    vaa_df = nhd.nhdplus_vaa()
                if vaa_df is not None and not vaa_df.empty:
                    vaa_subset = vaa_df[['comid', 'hydroseq', 'dnhydroseq']].rename(columns={'comid': 'nhdplusid'})

                    # Merge VAAs into the flowline dataframe
                    combined_df = combined_df.merge(vaa_subset, on='nhdplusid', how='left')
            except Exception as e:
                print(f"Warning: Could not fetch/merge VAAs: {e}")
        # ----------------------------

        combined_gdf = gpd.GeoDataFrame(combined_df, crs=all_reaches[0].crs)
        if 'nhdplusid' in combined_gdf.columns:
            combined_gdf = combined_gdf.drop_duplicates(subset='nhdplusid')

        # output_path is already defined above
        if not os.path.exists(output_path):
            combined_gdf.to_file(output_path, driver="GPKG", layer="NHDFlowline")
            print(f"Flowlines saved to: {output_path}")

        return output_path, combined_gdf

    except Exception as e:
        print(f"Unexpected error in download_nhd_flowline: {e}")
        traceback.print_exc()
        return None, None


def navigate_tdx_network(dam_point: Point, gpkg_path: str, distance_km: Union[float, Tuple[float, float], List[float]] = (1, 2), site_id=None):
    """
        Navigates GEOGLOWS (TDX-Hydro) network using LengthGeodesicMeters.
        At junctions, it follows the main stem (highest strmOrder).
    """
    # 1. Determine CRS and Buffer Size
    try:
        # Read metadata to check CRS (reading 1 row is usually sufficient and fast)
        meta = gpd.read_file(gpkg_path, rows=1)
        file_crs = meta.crs
    except Exception as e:
        print(f"Error reading VPU file {gpkg_path} for Site {site_id}: {e}")
        return None, -1

    # Default settings (assuming EPSG:4326)
    search_point = dam_point
    buffer_size = 0.005  # ~500m in degrees

    if file_crs:
        # Adjust buffer size based on CRS type
        if file_crs.is_projected:
            buffer_size = 500.0  # 500m in projected units
        
        # Reproject dam_point (assumed EPSG:4326) to file CRS if needed
        try:
            transformer = Transformer.from_crs("EPSG:4326", file_crs, always_xy=True)
            search_point = transform(transformer.transform, dam_point)
        except Exception as e:
            print(f"Warning: CRS transformation failed ({e}). Using original point.")

    # Create search area
    search_area = search_point.buffer(buffer_size)

    try:
        streams = gpd.read_file(gpkg_path, bbox=search_area)
    except Exception as e:
        print(f"Error reading VPU file {gpkg_path} for Site {site_id}: {e}")
        return None, -1

    if streams.empty:
        print(f"No streams found in VPU file near dam location. File: {gpkg_path}, Site: {site_id}")
        return None, -1

    # Check for active geometry to avoid "active geometry column to use has not been set" error
    try:
        if streams.geometry is None:
             raise AttributeError
    except AttributeError:
        print(f"No active geometry column in streams file. File: {gpkg_path}, Site: {site_id}")
        return None, -1

    # 2. Find the starting reach
    try:
        # Use search_point (which matches streams CRS) for nearest neighbor
        nearest_idx = streams.sindex.nearest(search_point, return_all=False)[1]
        
        # Handle potential Series return from nearest() if multiple indices are returned
        if isinstance(nearest_idx, pd.Series):
             nearest_idx = nearest_idx.iloc[0]
        elif isinstance(nearest_idx, (np.ndarray, list)):
             nearest_idx = nearest_idx[0]

        start_reach = streams.iloc[nearest_idx]
        start_id = start_reach['LINKNO']
    except Exception as e:
        print(f"Error finding nearest stream: {e}")
        return None, -1

    # Handle tuple distance (upstream, downstream)
    if isinstance(distance_km, (tuple, list)):
        if len(distance_km) == 2:
            threshold_m_us = distance_km[0] * 1000.0
            threshold_m_ds = distance_km[1] * 1000.0
        elif len(distance_km) > 0:
            threshold_m_us = distance_km[0] * 1000.0
            threshold_m_ds = distance_km[0] * 1000.0
        else:
            threshold_m_us = 1000.0
            threshold_m_ds = 2000.0
    else:
        threshold_m_us = distance_km * 1000.0
        threshold_m_ds = distance_km * 1000.0

    def trace_network(current_id, direction='downstream'):
        found_reaches = []
        visited = {current_id}
        queue = [(current_id, 0.0)]

        threshold_m = threshold_m_ds if direction == 'downstream' else threshold_m_us

        while queue:
            cid, accumulated_m = queue.pop(0)
            reach_data = streams[streams['LINKNO'] == cid]
            if reach_data.empty: continue

            found_reaches.append(reach_data)

            seg_len_m = reach_data['LengthGeodesicMeters'].values[0]
            total_m = accumulated_m + seg_len_m

            if total_m < threshold_m:
                if direction == 'downstream':
                    next_ids = reach_data['DSLINKNO'].values
                else:
                    # UPSTREAM JUNCTION LOGIC
                    # Find all reaches that flow into this one
                    upstream_candidates = streams[streams['DSLINKNO'] == cid]

                    if upstream_candidates.empty:
                        next_ids = []
                    elif len(upstream_candidates) > 1:
                        # Pick the "Main Stem" based on highest stream order
                        main_stem = upstream_candidates.sort_values('strmOrder', ascending=False).iloc[0]
                        next_ids = [main_stem['LINKNO']]
                        # print(f"Junction at LINKNO {cid}: Following main stem (Order {main_stem['strmOrder']})")
                    else:
                        next_ids = upstream_candidates['LINKNO'].values

                for nid in next_ids:
                    if nid not in visited and nid != -1:
                        visited.add(nid)
                        queue.append((nid, total_m))

        return found_reaches

    # 3. Execute and Combine
    ds = trace_network(start_id, direction='downstream')
    us = trace_network(start_id, direction='upstream')

    combined_list = ds + us
    if not combined_list:
        return None, -1

    combined = pd.concat(combined_list).drop_duplicates(subset='LINKNO')
    return gpd.GeoDataFrame(combined, crs=streams.crs), start_id


def download_tdx_flowline(latitude: float, longitude: float, flowline_dir: str, vpu_map_path: str, distance_km: Union[float, Tuple[float, float], List[float]] = (1, 2), site_id=None):
    """
        Downloads GEOGLOWS/TDX-Hydro flowlines based on VPU boundaries.
    """
    # 1. read in the vpu map to figure out which vpu flowlines to download
    try:
        vpu_gdf = gpd.read_file(vpu_map_path)
    except Exception as e:
        print(f"Error reading VPU map {vpu_map_path}: {e}")
        return None, None

    if vpu_gdf.crs.to_epsg() != 4326:
        vpu_gdf = vpu_gdf.to_crs(epsg=4326)

    dam_point = Point(longitude, latitude)
    vpu_polygon = vpu_gdf[vpu_gdf.contains(dam_point)]

    if vpu_polygon.empty:
        return None, None
    # 2. extract the vpu to download
    vpu_col = [c for c in vpu_polygon.columns if c.lower() in ['vpu', 'vpucode']][0]
    vpu_code = str(vpu_polygon.iloc[0][vpu_col])
    
    # 3. download the streams_vpu.gpkg (Keep this in the main flowline_dir to share across sites)
    raw_tdx_dir = os.path.join(flowline_dir, "raw_tdx")
    os.makedirs(raw_tdx_dir, exist_ok=True)
    flowline_vpu = os.path.join(raw_tdx_dir, f"streams_{vpu_code}.gpkg")
    
    if not os.path.exists(flowline_vpu):
        try:
            fs = s3fs.S3FileSystem(anon=True)
            s3_path = f"geoglows-v2/hydrography/vpu={vpu_code}/streams_{vpu_code}.gpkg"
            with fs.open(s3_path, 'rb') as f_in, open(flowline_vpu, 'wb') as f_out:
                f_out.write(f_in.read())
        except Exception as e:
            print(f"Error downloading VPU {vpu_code}: {e}")
            return None, None

    # Pass the distance_km argument
    flowline_gdf, linkno = navigate_tdx_network(dam_point, flowline_vpu, distance_km=distance_km, site_id=site_id)
    
    if flowline_gdf is None or flowline_gdf.empty:
        return None, None

    # Save the clipped flowline in a site-specific subdirectory
    if site_id:
        site_flowline_dir = os.path.join(flowline_dir, str(site_id))
    else:
        site_flowline_dir = flowline_dir
        
    os.makedirs(site_flowline_dir, exist_ok=True)
    output_path = os.path.join(site_flowline_dir, f"tdx_flowline_{linkno}.gpkg")

    if not os.path.exists(output_path):
        try:
            flowline_gdf.to_file(output_path, driver="GPKG", layer="TDXFlowline")
            print(f"Flowlines saved to: {output_path}")
        except Exception as e:
            print(f"Error saving flowline to {output_path}: {e}")
            return None, None

    return output_path, flowline_gdf


# =================================================================
# 2. DEM FUNCTIONS (3DEP & METADATA)
# =================================================================

_RASTER_EXTS = (".tif", ".tiff", ".img")


def _extract_zipped_raster(zip_path: str, raw_dem_dir: str) -> Union[str, None]:
    """
    Extract the first raster (.tif/.tiff/.img) member from `zip_path` into
    `raw_dem_dir` (flattening any internal directory structure), delete the
    zip, and return the extracted file path. Returns None on failure.

    USGS TNM serves 1/9 arc-second NED tiles as .zip archives wrapping an
    .img raster; gdal.Warp can't merge a bare .zip path directly.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            raster_members = [
                m for m in zf.namelist()
                if not m.endswith("/")
                and os.path.basename(m).lower().endswith(_RASTER_EXTS)
            ]
            if not raster_members:
                print(f"Error: no raster (.tif/.img) inside {zip_path}")
                return None
            member = raster_members[0]
            out_name = sanitize_filename(os.path.basename(member))
            out_path = os.path.join(raw_dem_dir, out_name)
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(zip_path)
        return out_path
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return None


def download_dem(lhd_id, flowline_gdf, dem_dir, resolution=None):
    """
    Directly queries the TNM API to find all 1m tiles along the flowline,
    downloads them, and merges them into a rectangular grid for ARC.
    
    resolution: "1" for 1m, "10" for 1/3 arc-second (approx 10m).
    """
    try:
        if flowline_gdf is None or flowline_gdf.empty:
            print(f"Dam {lhd_id}: Flowline GDF is None or empty.")
            return None, None, "No Flowlines"

        # 1. Project to WGS84 for API query
        if flowline_gdf.crs and flowline_gdf.crs.to_epsg() != 4326:
            flowline_gdf = flowline_gdf.to_crs(epsg=4326)

        b = flowline_gdf.total_bounds
        buffer_deg = 0.002
        bbox = (float(b[0] - buffer_deg), float(b[1] - buffer_deg),
                float(b[2] + buffer_deg), float(b[3] + buffer_deg))
        
        print(f"Dam {lhd_id}: Bounding box for query: {bbox}")

        # 2. Query TNM API
        tnm_url = "https://tnmaccess.nationalmap.gov/api/v1/products"
        
        # Determine datasets to query based on resolution
        datasets_to_try = []
        
        # If resolution is explicitly 10m, prioritize 1/3 arc-second
        if str(resolution) == "10":
            datasets_to_try.append("National Elevation Dataset (NED) 1/3 arc-second Current")
            # Optionally try 1m as fallback? Or just stick to 10m?
            # Let's stick to 10m if requested, but maybe fallback to 1m if 10m fails?
            # Usually 10m covers more area.
        else:
            # Default behavior: Try 1m first, then 1/9 arc-second, then 1/3 arc-second
            datasets_to_try.append("Digital Elevation Model (DEM) 1 meter")
            datasets_to_try.append("National Elevation Dataset (NED) 1/9 arc-second")
            datasets_to_try.append("National Elevation Dataset (NED) 1/3 arc-second Current")

        items = []
        found_dataset = ""
        
        for dataset_name in datasets_to_try:
            params = {
                "datasets": dataset_name,
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "outputFormat": "JSON"
            }

            print(f"Dam {lhd_id}: Searching for {dataset_name} tiles...")
            print(f"Dam {lhd_id}: Querying TNM API at {tnm_url} with params: {params}")
            
            response_obj = requests.get(tnm_url, params=params)
            print(f"Dam {lhd_id}: TNM API Response Status: {response_obj.status_code}")
            
            try:
                response = response_obj.json()
            except Exception as json_err:
                print(f"Dam {lhd_id}: Failed to parse JSON response: {response_obj.text[:500]}")
                # Don't raise here, try next dataset
                continue
                
            items = response.get("items", [])
            print(f"Dam {lhd_id}: Found {len(items)} items for {dataset_name}.")
            
            if items:
                found_dataset = dataset_name
                break

        if not items:
            print(f"Dam {lhd_id}: No DEM tiles found in TNM API for requested resolutions.")
            return None, None, {"status": "No DEM Found"}

        # 3. Download tiles to a regional folder
        raw_dem_dir = os.path.join(dem_dir, 'raw_3dep')
        os.makedirs(raw_dem_dir, exist_ok=True)
        downloaded_tiles = []
        tile_projects = set()
        tile_pub_dates = set()

        def _record_tile_meta(source_item):
            # TNM tile URLs look like:
            #   .../Elevation/1m/Projects/<project_name>/TIFF/<tile>.tif
            url = source_item.get("downloadURL") or ""
            match = re.search(r"/Projects/([^/]+)/", url)
            if match:
                tile_projects.add(match.group(1))
            pub_date = source_item.get("publicationDate")
            if pub_date:
                tile_pub_dates.add(str(pub_date))

        for item in items:
            tile_url = item.get("downloadURL")
            if not tile_url:
                print(f"Dam {lhd_id}: Item missing downloadURL: {item}")
                continue

            tile_name = sanitize_filename(os.path.basename(tile_url))  # Using your sanitize logic
            local_path = os.path.join(raw_dem_dir, tile_name)

            if os.path.exists(local_path):
                # Cache hit: a prior run may have left either the raster or an
                # unextracted zip.
                if local_path.lower().endswith(".zip"):
                    extracted = _extract_zipped_raster(local_path, raw_dem_dir)
                    if extracted:
                        downloaded_tiles.append(extracted)
                        _record_tile_meta(item)
                    continue
                downloaded_tiles.append(local_path)
                _record_tile_meta(item)
                continue

            print(f"Downloading: {tile_name} from {tile_url}")
            try:
                r = requests.get(tile_url, stream=True)
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as dl_err:
                print(f"Dam {lhd_id}: Error downloading {tile_url}: {dl_err}")
                continue

            # USGS TNM serves some tiers (e.g. 1/9 arc-second NED) as a .zip
            # wrapping an .img raster; gdal.Warp needs the bare raster.
            if local_path.lower().endswith(".zip"):
                extracted = _extract_zipped_raster(local_path, raw_dem_dir)
                if extracted:
                    downloaded_tiles.append(extracted)
                    _record_tile_meta(item)
                continue

            downloaded_tiles.append(local_path)
            _record_tile_meta(item)

        print(f"Dam {lhd_id}: Downloaded tiles: {downloaded_tiles}")

        if not downloaded_tiles:
            print(f"Dam {lhd_id}: No tiles were successfully downloaded.")
            return None, None, {"status": "Download Failed"}

        # 4. Warp & Merge (Produces the Rectangular DEM for ARC)
        site_dem_dir = os.path.join(dem_dir, str(lhd_id))
        os.makedirs(site_dem_dir, exist_ok=True)
        out_path = os.path.join(site_dem_dir, f"LHD_{lhd_id}_3DEP_DEM.tif")

        print(f"Dam {lhd_id}: Merging {len(downloaded_tiles)} tiles into rectangular DEM at {out_path}...")
        try:
            gdal.Warp(
                out_path,
                downloaded_tiles,
                outputBounds=bbox,
                outputBoundsSRS='EPSG:4326',
                resampleAlg=gdal.GRA_Bilinear,
                format='GTiff',
                creationOptions=['COMPRESS=LZW']
            )
        except Exception as warp_err:
             print(f"Dam {lhd_id}: gdal.Warp failed: {warp_err}")
             raise warp_err

        if os.path.exists(out_path):
             print(f"Dam {lhd_id}: Output DEM created successfully at {out_path}")
        else:
             print(f"Dam {lhd_id}: Output DEM NOT created at {out_path} (gdal.Warp returned but file missing)")

        # Return resolution in meters (approx)
        if "1 meter" in found_dataset:
            res_val = 1
        elif "1/9 arc-second" in found_dataset:
            res_val = 3
        else:
            res_val = 10

        dem_meta = {
            "status": "OK",
            "dataset": found_dataset,
            "tile_count": len(downloaded_tiles),
            "tile_files": [os.path.basename(t) for t in downloaded_tiles],
            "projects": sorted(tile_projects),
            "publication_dates": sorted(tile_pub_dates),
        }
        return out_path, res_val, dem_meta

    except Exception as e:
        print(f"Dam {lhd_id}: Critical Error in download_dem: {e}")
        traceback.print_exc()
        return None, None, {"status": "Error", "error": str(e)}


def prune_raw_dem_tiles(dem_dir: str) -> Tuple[int, int]:
    """Deletes the shared raw_3dep/ tile cache under dem_dir.

    Raw 3DEP tiles (downloaded/extracted by download_dem) are only needed
    transiently to build each dam's merged/warped DEM; they're shared across
    all dams in a batch, so this should only be called once the whole batch
    is done. Mirrors lhd-screening's end-of-run raw tile prune.

    Returns (files_deleted, bytes_freed). Safe to call if nothing to prune.
    """
    raw_dem_dir = os.path.join(dem_dir, 'raw_3dep')
    if not os.path.isdir(raw_dem_dir):
        return 0, 0

    count = 0
    total_bytes = 0
    for root, _, files in os.walk(raw_dem_dir):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                total_bytes += os.path.getsize(fpath)
                count += 1
            except OSError:
                pass

    shutil.rmtree(raw_dem_dir)
    return count, total_bytes


# =================================================================
# 3. BASEFLOW & LIDAR DATE ESTIMATION
# =================================================================

def find_water_gpstime(lat, lon):
    """
    Queries USGS TNM for raw LiDAR points to find the collection date.
    """
    bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
              "datasets": "Lidar Point Cloud (LPC)", "max": 1, "outputFormat": "JSON"}

    try:
        r = requests.get("https://tnmaccess.nationalmap.gov/api/v1/products", params=params)
        items = r.json().get("items", [])
        if not items: return None

        dl_url = items[0].get("downloadURL")
        with tempfile.TemporaryDirectory() as tmp:
            las_data = requests.get(dl_url).content
            las_path = os.path.join(tmp, "data.las")
            with open(las_path, 'wb') as f: f.write(las_data)

            las = laspy.read(las_path)
            # Standard conversion from GPS seconds to Date
            gps_time = las.gps_time[0] + 1_000_000_000
            return gps_time
    except Exception:
        return None


def est_dem_baseflow(stream_reach, source, method, project_hint=None):
    lat, lon = stream_reach.latitude, stream_reach.longitude
    found_date = None

    if "lidar date" in method.lower():
        # Step 1: Try the high-precision LiDAR file method
        lidar_time = find_water_gpstime(lat, lon)

        if lidar_time:
            found_date = (datetime(1980, 1, 6) + timedelta(seconds=lidar_time)).strftime('%Y-%m-%d')

        # Step 2: Fallback to the project_hint if the .las download failed
        elif project_hint:
            # Example: extract '2019' from 'Iowa_Lidar_2019'
            import re
            match = re.search(r'20\d{2}', project_hint)
            if match:
                found_date = f"{match.group(0)}-06-15"  # Assume mid-summer for the year
                print(f"Using estimated date from project name: {found_date}")

        # Step 3: Get the flow
        if found_date:
            flow = stream_reach.get_flow_on_date(found_date, source)
        else:
            flow = stream_reach.get_median_flow(source)

        return flow, found_date


# =================================================================
# 4. LAND USE (constant placeholder raster)
# =================================================================
# ARC requires a landcover raster + a Manning's n lookup table keyed to it,
# but this app assigns a single Manning's n per dam (uniform or NWM-
# regionalized -- see ui/arc_tab.py's create_mannings_table) rather than
# varying roughness by landcover class. A real classified raster (previously
# ESA WorldCover) therefore has no effect on the Manning's n ARC samples, so
# we use a constant placeholder instead -- this also drops FindBanksBasedOnLandCover's
# only landcover-derived input, now that lhd_arc.py disables that flag in favor
# of ARC's DEM-only ("flat water") bank-finding.
CONSTANT_LC_CODE = 1


def make_constant_land_raster(lhd_id: int, dem_path: str, land_dir: str, lc_code: int = CONSTANT_LC_CODE):
    """Writes a constant-value land cover raster aligned to the DEM grid."""
    site_land_dir = os.path.join(land_dir, str(lhd_id))
    os.makedirs(site_land_dir, exist_ok=True)

    final_path = os.path.join(site_land_dir, f"{lhd_id}_LAND_Raster.tif")
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    if os.path.exists(final_path):
        # A cached raster is only valid if it still lines up with the DEM's
        # grid -- if the DEM was ever regenerated with a different extent
        # (e.g. re-fetched tiles, different bbox), ARC's "Rows do not Match!"
        # check will reject a stale land raster of the wrong shape.
        with rasterio.open(final_path) as existing:
            if existing.width == profile["width"] and existing.height == profile["height"]:
                return final_path

    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    data = np.full((profile["height"], profile["width"]), lc_code, dtype=np.uint8)

    with rasterio.open(final_path, "w", **profile) as dst:
        dst.write(data, 1)

    return final_path
