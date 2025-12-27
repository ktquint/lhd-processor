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

try:
    import gdal
except ImportError:
    from osgeo import gdal


# =================================================================
# 1. FLOWLINE FUNCTIONS (NHDPlus & TDX-Hydro)
# =================================================================

def download_nhd_flowline(lat: float, lon: float, flowline_dir: str, distance_km=2):
    """
    Uses HyRiver NLDI to fetch NHDPlus flowlines, merges VAAs (hydroseq, dnhydroseq),
    and standardizes ID columns.
    """
    nldi = NLDI()
    try:
        comid_df = nldi.comid_byloc((lon, lat))
    except Exception as e:
        print(f"NLDI Error: {e}")
        return None, None

    if comid_df.empty:
        print(f"No NHD COMID found for location: {lat}, {lon}")
        return None, None

    comid_val = comid_df.comid.values[0]
    
    # OPTIMIZATION: Check if file already exists
    os.makedirs(flowline_dir, exist_ok=True)
    output_path = os.path.join(flowline_dir, f"nhd_flowline_{comid_val}.gpkg")
    
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
            reach = nldi.navigate_byid(
                fsource="comid",
                fid=str(comid_val),
                navigation=mode,
                source="flowlines",
                distance=distance_km
            )
            if not reach.empty:
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
    combined_df['nhdplusid'] = pd.to_numeric(combined_df['nhdplusid'])

    # Fetch the VAA table (hydroseq and dnhydroseq are in the 'vaa' service)
    # Note: nhdplus_vaa() fetches the national parquet file
    try:
        vaa_df = nhd.nhdplus_vaa()
        vaa_subset = vaa_df[['comid', 'hydroseq', 'dnhydroseq']].rename(columns={'comid': 'nhdplusid'})

        # Merge VAAs into the flowline dataframe
        combined_df = combined_df.merge(vaa_subset, on='nhdplusid', how='left')
    except Exception as e:
        print(f"Warning: Could not fetch/merge VAAs: {e}")
    # ----------------------------

    combined_gdf = gpd.GeoDataFrame(combined_df, crs=all_reaches[0].crs)
    combined_gdf = combined_gdf.drop_duplicates(subset='nhdplusid')

    # output_path is already defined above
    if not os.path.exists(output_path):
        combined_gdf.to_file(output_path, driver="GPKG", layer="NHDFlowline")
        print(f"Flowlines saved to: {output_path}")

    return output_path, combined_gdf


def navigate_tdx_network(dam_point: Point, gpkg_path: str, distance_km: float = 2.0):
    """
        Navigates GEOGLOWS (TDX-Hydro) network using LengthGeodesicMeters.
        At junctions, it follows the main stem (highest strmOrder).
    """
    streams = gpd.read_file(gpkg_path, bbox=dam_point.buffer(0.1))

    if streams.empty:
        print("No streams found in VPU file near dam location.")
        return gpd.GeoDataFrame(), -1

    # 1. Find the starting reach
    nearest_idx = streams.sindex.nearest(dam_point, return_all=False)[1]
    start_reach = streams.iloc[nearest_idx]
    start_id = start_reach['LINKNO']

    threshold_m = distance_km * 1000.0

    def trace_network(current_id, direction='downstream'):
        found_reaches = []
        visited = {current_id}
        queue = [(current_id, 0.0)]

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

    # 2. Execute and Combine
    ds = trace_network(start_id, direction='downstream')
    us = trace_network(start_id, direction='upstream')

    combined = pd.concat(ds + us).drop_duplicates(subset='LINKNO')
    return gpd.GeoDataFrame(combined, crs=streams.crs), start_id


def download_tdx_flowline(latitude: float, longitude: float, flowline_dir: str, vpu_map_path: str):
    """
        Downloads GEOGLOWS/TDX-Hydro flowlines based on VPU boundaries.
    """
    # 1. read in the vpu map to figure out which vpu flowlines to download
    vpu_gdf = gpd.read_file(vpu_map_path)
    if vpu_gdf.crs.to_epsg() != 4326:
        vpu_gdf = vpu_gdf.to_crs(epsg=4326)

    dam_point = Point(longitude, latitude)
    vpu_polygon = vpu_gdf[vpu_gdf.contains(dam_point)]

    if vpu_polygon.empty:
        return None, None
    # 2. extract the vpu to download
    vpu_col = [c for c in vpu_polygon.columns if c.lower() in ['vpu', 'vpucode']][0]
    vpu_code = str(vpu_polygon.iloc[0][vpu_col])
    # 3. download the streams_vpu.gpkg
    flowline_vpu = os.path.join(flowline_dir, f"streams_{vpu_code}.gpkg")
    if not os.path.exists(flowline_vpu):
        fs = s3fs.S3FileSystem(anon=True)
        s3_path = f"geoglows-v2/hydrography/vpu={vpu_code}/streams_{vpu_code}.gpkg"
        with fs.open(s3_path, 'rb') as f_in, open(flowline_vpu, 'wb') as f_out:
            f_out.write(f_in.read())

    flowline_gdf, linkno = navigate_tdx_network(dam_point, flowline_vpu)
    
    output_path = os.path.join(flowline_dir, f"tdx_flowline_{linkno}.gpkg")

    if not os.path.exists(output_path):
        flowline_gdf.to_file(output_path, driver="GPKG", layer="TDXFlowline")
        print(f"Flowlines saved to: {output_path}")

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

        bbox = (float(minx), float(miny), float(maxx), float(maxy))
        
        # CHANGED: Added .envelope to ensure a rectangular geometry (no curved edges)
        geom = box(*bbox).buffer(0.001).envelope

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
        except Exception:
            availability = {}

        res_to_try = [1, 10] if (res_meters == 1 and availability.get('1m')) else [10]

        for res in res_to_try:
            # Note: crs="EPSG:4326" tells Py3DEP our 'geom' is WGS84
            dem = py3dep.get_dem(geom, resolution=res, crs="EPSG:4326")

            if dem is not None and not np.all(np.isnan(dem.values)):
                clean_name = f"LHD_{lhd_id}_3DEP_{res}m_NAVD88.tif"
                out_path = os.path.join(dem_dir, str(lhd_id), clean_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                dem.rio.to_raster(out_path)
                return out_path, res, project_info

    except Exception as e:
        print(f"Error prepping Dam {lhd_id}: {e}")
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
    final_path = os.path.join(land_dir, f"{lhd_id}_LAND_Raster.tif")
    if os.path.exists(final_path): return final_path

    # Get target DEM specs
    with rasterio.open(dem_path) as src:
        bounds = [src.bounds.left, src.bounds.top, src.bounds.right, src.bounds.bottom]
        ncols, nrows = src.width, src.height
        dem_crs = src.crs
        # Create polygon for ESA intersection
        geom = box(*src.bounds)

    raw_esa_dir = os.path.join(land_dir, 'raw_esa')
    raw_lc_tif = _download_esa_land_cover(geom, raw_esa_dir, dem_crs)

    if raw_lc_tif:
        # Align using GDAL Translate
        options = gdal.TranslateOptions(projWin=bounds, width=ncols, height=nrows, creationOptions=['COMPRESS=LZW'])
        gdal.Translate(final_path, raw_lc_tif, options=options)
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
        if not os.path.exists(path):
            with open(path, 'wb') as f: f.write(requests.get(url).content)
        downloaded.append(path)

    merged_path = os.path.join(output_dir, "merged_esa.tif")
    gdal.Warp(merged_path, downloaded, options=gdal.WarpOptions(format='GTiff', creationOptions=['COMPRESS=LZW']))
    return merged_path
