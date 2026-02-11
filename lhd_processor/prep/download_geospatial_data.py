import os
import s3fs
import laspy
import requests
import tempfile
import rasterio
import numpy as np
import pandas as pd
import laspy.errors
import pynhd as nhd
from pynhd import NLDI
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import Point, box
from datetime import datetime, timedelta
import time
import traceback
from typing import Union, Tuple, List

try:
    import gdal
except ImportError:
    from osgeo import gdal


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

def download_nhd_flowline(lat: float, lon: float, flowline_dir: str, distance_km: Union[float, Tuple[float, float], List[float]] = (1, 2), site_id=None):
    """
    Uses HyRiver NLDI to fetch NHDPlus flowlines, merges VAAs (hydroseq, dnhydroseq),
    and standardizes ID columns.
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

        try:
            comid_df = nldi.comid_byloc((lon, lat))
        except Exception as e:
            print(f"NLDI Error: {e}")
            return None, None

        if comid_df is None or comid_df.empty:
            print(f"No NHD COMID found for location: {lat}, {lon}")
            return None, None

        if 'comid' not in comid_df.columns:
             print(f"No 'comid' column in NLDI result. Columns: {comid_df.columns}")
             return None, None

        comid_val = comid_df.comid.values[0]
        if pd.isna(comid_val):
             print("Found None/NaN for COMID")
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
        nav_modes = ["upstreamMain", "downstreamMain"]

        for mode in nav_modes:
            try:
                # Handle distance logic safely
                dist = distance_km
                if isinstance(distance_km, (tuple, list)):
                    if len(distance_km) == 2:
                        dist = distance_km[0] if mode == "upstreamMain" else distance_km[1]
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
            return None, None

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
            # Default behavior: Try 1m first, then 1/3 arc-second
            datasets_to_try.append("Digital Elevation Model (DEM) 1 meter")
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
            return None, None, "No DEM Found"

        # 3. Download tiles to a regional folder
        raw_dem_dir = os.path.join(dem_dir, 'raw_3dep')
        os.makedirs(raw_dem_dir, exist_ok=True)
        downloaded_tiles = []

        for item in items:
            tile_url = item.get("downloadURL")
            if not tile_url:
                print(f"Dam {lhd_id}: Item missing downloadURL: {item}")
                continue
                
            tile_name = sanitize_filename(os.path.basename(tile_url))  # Using your sanitize logic
            local_path = os.path.join(raw_dem_dir, tile_name)

            if not os.path.exists(local_path):
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

            downloaded_tiles.append(local_path)
            
        print(f"Dam {lhd_id}: Downloaded tiles: {downloaded_tiles}")

        if not downloaded_tiles:
            print(f"Dam {lhd_id}: No tiles were successfully downloaded.")
            return None, None, "Download Failed"

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
        res_val = 1 if "1 meter" in found_dataset else 10
        return out_path, res_val, "USGS TNM API"

    except Exception as e:
        print(f"Dam {lhd_id}: Critical Error in download_dem: {e}")
        traceback.print_exc()
        return None, None, "Error"


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
# 4. LAND USE FUNCTIONS (ESA WorldCover Alignment)
# =================================================================

def download_land_raster(lhd_id: int, dem_path: str, land_dir: str):
    """
    Downloads ESA WorldCover and aligns it exactly to the grid of the downloaded DEM.
    """
    # Create site-specific subdirectory for the final raster
    site_land_dir = os.path.join(land_dir, str(lhd_id))
    os.makedirs(site_land_dir, exist_ok=True)
    
    final_path = os.path.join(site_land_dir, f"{lhd_id}_LAND_Raster.tif")
    if os.path.exists(final_path): return final_path

    # Get target DEM specs
    with rasterio.open(dem_path) as src:
        bounds = [src.bounds.left, src.bounds.top, src.bounds.right, src.bounds.bottom]
        ncols, nrows = src.width, src.height
        dem_crs = src.crs
        # Create polygon for ESA intersection
        geom = box(*src.bounds)

    # Keep raw ESA tiles in the main LAND folder to share across sites
    raw_esa_dir = os.path.join(land_dir, 'raw_esa')
    raw_lc_tif = _download_esa_land_cover(geom, raw_esa_dir, dem_crs)

    if raw_lc_tif:
        # Align using GDAL Warp to ensure exact grid match
        options = gdal.WarpOptions(
            format='GTiff',
            outputBounds=bounds,
            width=ncols,
            height=nrows,
            dstSRS=dem_crs.to_wkt(),
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=['COMPRESS=LZW']
        )
        gdal.Warp(final_path, raw_lc_tif, options=options)
        return final_path
    return None


def _download_esa_land_cover(geom, output_dir, dem_crs, year=2021):
    # Reproject geometry to WGS84 for ESA S3 query
    transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    geom_wgs84 = transform(transformer.transform, geom)

    grid = gpd.read_file("https://esa-worldcover.s3.eu-central-1.amazonaws.com/esa_worldcover_grid.geojson")
    tiles = grid[grid.intersects(geom_wgs84)]

    os.makedirs(output_dir, exist_ok=True)
    downloaded = []
    for tile in tiles.ll_tile:
        url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/{year}/map/ESA_WorldCover_10m_{year}_v200_{tile}_Map.tif"
        path = os.path.join(output_dir, os.path.basename(url))
        
        # Validate existing file
        if os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    pass
            except Exception:
                print(f"Removing corrupted file: {path}")
                os.remove(path)

        if not os.path.exists(path):
            print(f"Downloading {url}...")
            try:
                resp = requests.get(url, stream=True)
                if resp.status_code == 200:
                    with open(path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"Failed to download {url}: Status {resp.status_code}")
            except Exception as e:
                print(f"Download error for {url}: {e}")

        if os.path.exists(path):
            downloaded.append(path)

    if not downloaded:
        return None

    merged_path = os.path.join(output_dir, "merged_esa.tif")
    try:
        gdal.Warp(merged_path, downloaded, options=gdal.WarpOptions(format='GTiff', creationOptions=['COMPRESS=LZW']))
        return merged_path
    except Exception as e:
        print(f"Error merging ESA tiles: {e}")
        return None
