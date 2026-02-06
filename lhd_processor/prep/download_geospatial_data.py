import os
import s3fs
import laspy
import py3dep
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
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
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
    os.makedirs(flowline_dir, exist_ok=True)
    flowline_vpu = os.path.join(flowline_dir, f"streams_{vpu_code}.gpkg")
    
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

def download_dem(lhd_id, flowline_gdf, dem_dir, res_meters=1):
    """
        Fetches 3DEP DEM using flowline extent with 10m fallback and metadata tracking.
    """
    try:
        if flowline_gdf is None or flowline_gdf.empty:
            print(f"Dam {lhd_id}: Flowline GDF is empty.")
            return None, None, "No Flowlines"

        # Ensure CRS is EPSG:4326
        if flowline_gdf.crs and flowline_gdf.crs.to_epsg() != 4326:
            try:
                # print(f"Dam {lhd_id}: Reprojecting flowlines from {flowline_gdf.crs} to EPSG:4326")
                flowline_gdf = flowline_gdf.to_crs(epsg=4326)
            except Exception as e:
                print(f"Dam {lhd_id}: Reprojection failed: {e}")
                return None, None, "Reprojection Error"

        # Remove empty geometries
        flowline_gdf = flowline_gdf[~flowline_gdf.is_empty]
        if flowline_gdf.empty:
             print(f"Dam {lhd_id}: Flowline GDF contains only empty geometries.")
             return None, None, "Empty Geometries"

        # 1. Setup Bounding Box (tuple of length 4) and buffered Geometry
        b = flowline_gdf.total_bounds
        
        if np.any(np.isnan(b)):
             print(f"Dam {lhd_id}: Flowline bounds contain NaNs.")
             return None, None, "Invalid Bounds"

        # Ensure bbox has non-zero dimensions to avoid degenerate polygons
        minx, miny, maxx, maxy = b
        if maxx - minx < 1e-5:
            minx -= 0.0005
            maxx += 0.0005
        if maxy - miny < 1e-5:
            miny -= 0.0005
            maxy += 0.0005

        # Apply buffer to the bbox coordinates directly
        # This effectively creates an "envelope" around the flowline extent
        buffer_deg = 0.002
        minx -= buffer_deg
        miny -= buffer_deg
        maxx += buffer_deg
        maxy += buffer_deg

        bbox = (float(minx), float(miny), float(maxx), float(maxy))
        
        # 2. Extract Lidar Project Metadata
        try:
            sources_gdf = py3dep.query_3dep_sources(bbox)
        except Exception as e:
            print(f"Warning: Could not query 3DEP sources: {e}")
            sources_gdf = gpd.GeoDataFrame()

        project_info = "Unknown"
        if not sources_gdf.empty:
            res_str = f"{res_meters}m" if res_meters in [1, 3, 10] else "1m"
            relevant_sources = sources_gdf[sources_gdf['dem_res'] == res_str]

            if not relevant_sources.empty:
                # Define a list of possible column names for the project name
                possible_cols = ['workunit_name', 'workunit', 'project_id', 'workunitname', 'dataset_id']

                # Find the first one that actually exists in the dataframe
                actual_col = next((c for c in possible_cols if c in relevant_sources.columns), None)

                if actual_col:
                    names = relevant_sources[actual_col].unique().tolist()
                    project_info = "; ".join(map(str, names))
                else:
                    project_info = "Metadata columns mismatch"

        # 3. Availability and Download Loop
        try:
            availability = py3dep.check_3dep_availability(bbox)
        except Exception as e:
            # DEBUG: Print why availability check failed
            # print(f"Dam {lhd_id}: check_3dep_availability failed. BBox: {bbox}")
            # print(f"Error: {e}")
            availability = {}
            
        res_to_try = [1, 10] if (res_meters == 1 and availability.get('1m')) else [10]

        for res in res_to_try:
            # Note: crs="EPSG:4326" tells Py3DEP our 'geom' is WGS84
            dem = None
            last_error = None
            for attempt in range(3):
                try:
                    # CHANGED: Pass bbox tuple directly to get_dem
                    dem = py3dep.get_dem(bbox, resolution=res, crs="EPSG:4326")
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(1 * (attempt + 1))
            
            if dem is None and last_error:
                print(f"Dam {lhd_id}: py3dep.get_dem failed for res {res}m after retries. Error: {last_error}")
                continue

            if dem is not None and not np.all(np.isnan(dem.values)):
                clean_name = f"LHD_{lhd_id}_3DEP_{res}m_NAVD88.tif"
                out_path = os.path.join(dem_dir, str(lhd_id), clean_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                dem.rio.to_raster(out_path)
                return out_path, res, project_info
        
        # If we exit the loop without returning, it means no DEM was found
        return None, None, "No DEM Found"

    except Exception as e:
        print(f"Error prepping Dam {lhd_id}: {e}")
        if 'b' in locals():
            print(f"  - Bounds used: {b}")
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
