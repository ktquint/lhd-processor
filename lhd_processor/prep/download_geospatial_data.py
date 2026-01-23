import os
import re
import math
import time
import requests
import geopandas as gpd
from dateutil.parser import parse

# GIS Imports
try:
    from osgeo import gdal
except ImportError:
    import gdal

# Try importing py7zr, warn if missing
try:
    import py7zr
except ImportError:
    py7zr = None


# =================================================================
# 1. HELPER FUNCTIONS (USGS API & UTILS)
# =================================================================

def meters_to_latlon(lat0, lon0, dx, dy):
    """Estimate lat/lon offsets from meters."""
    R = 6378137  # Earth radius
    d_lat = dy / R
    d_lon = dx / (R * math.cos(math.radians(lat0)))
    return lat0 + math.degrees(d_lat), lon0 + math.degrees(d_lon)


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "-", filename)


def download_file_stream(url, path):
    """Standard stream downloader with retry."""
    if os.path.exists(path): return True

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"  Downloading: {os.path.basename(path)}...")
    try:
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                print(f"  HTTP Error {r.status_code} for {url}")
                return False
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(path): os.remove(path)
        return False


# =================================================================
# 2. RAW DEM ACQUISITION (Same as before)
# =================================================================

def query_tnm_api(lat, lon, buffer_m, dataset_str):
    """Queries USGS TNM for products covering the buffered point."""
    bbox_min_lat, bbox_min_lon = meters_to_latlon(lat, lon, -buffer_m, -buffer_m)
    bbox_max_lat, bbox_max_lon = meters_to_latlon(lat, lon, buffer_m, buffer_m)

    bbox_str = f"{bbox_min_lon},{bbox_min_lat},{bbox_max_lon},{bbox_max_lat}"

    params = {
        "bbox": bbox_str,
        "datasets": dataset_str,
        "prodFormats": "GeoTIFF",
        "outputFormat": "JSON"
    }

    for _ in range(3):
        try:
            r = requests.get("https://tnmaccess.nationalmap.gov/api/v1/products", params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("items", [])
        except Exception as e:
            print(f"  API Error ({dataset_str}): {e}")
            time.sleep(1)
    return []


def filter_latest_tiles(items):
    """Deduplicates tiles by ID, keeping only the most recent."""
    tiles = {}
    for item in items:
        title = item.get("title", "unknown")
        # Extract generic grid ID (e.g. n40w112)
        grid_match = re.search(r'[ns]\d+[ew]\d+', title, re.IGNORECASE)
        tile_id = grid_match.group(0).lower() if grid_match else title

        pub_date = item.get("publicationDate") or item.get("dateCreated")
        try:
            dt = parse(pub_date) if pub_date else parse("1900-01-01")
        except:
            dt = parse("1900-01-01")

        if tile_id not in tiles or dt > tiles[tile_id]['dt']:
            tiles[tile_id] = {'dt': dt, 'item': item}

    return [v['item'] for v in tiles.values()]


def ensure_dem_tiles(lat, lon, weir_length, cache_dir):
    """
    Downloads raw 1m (or best available) DEM tiles to the cache.
    Returns: (list of file paths, resolution_meters, project_name)
    """
    print(f"Fetching Raw DEM Tiles for {lat}, {lon}...")

    buffer_m = max(weir_length * 10, 2000)

    datasets = [
        ("Digital Elevation Model (DEM) 1 meter", 1.0),
        ("National Elevation Dataset (NED) 1/3 arc-second Current", 10.0)
    ]

    local_paths = []
    res_meters = None
    project_info = "Unknown"

    for ds_name, res in datasets:
        items = query_tnm_api(lat, lon, buffer_m, ds_name)
        if items:
            items = filter_latest_tiles(items)

            # Download all required tiles
            success = True
            current_paths = []
            for item in items:
                fname = sanitize_filename(item.get("title", "dem")) + ".tif"
                out_path = os.path.join(cache_dir, fname)

                if download_file_stream(item.get("downloadURL"), out_path):
                    current_paths.append(out_path)
                else:
                    success = False
                    break

            if success and current_paths:
                local_paths = current_paths
                res_meters = res
                project_info = ds_name
                break  # Stop after finding the best resolution

    if not local_paths:
        print("  No DEM tiles found.")
        return [], None, None

    return local_paths, res_meters, project_info


# =================================================================
# 3. NHDPLUS V2 ACQUISITION (EPA Archive)
# =================================================================

def find_huc4_wbd(lat, lon):
    """Queries USGS WBD to get the HUC4 code."""
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/2/query"
    params = {
        'geometry': f"{lon},{lat}",
        'geometryType': 'esriGeometryPoint',
        'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': 'huc4',
        'f': 'json',
        'returnGeometry': 'false'
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if 'features' in data and data['features']:
            attrs = data['features'][0]['attributes']
            return attrs.get('huc4') or attrs.get('HUC4')
    except Exception:
        pass
    return None


def get_v2_vpu(huc4):
    """
    Maps HUC4 to NHDPlus V2 VPU (e.g. 01, 10L, 03S).
    Most are simple HUC2s, but some are split.
    """
    if not huc4: return None
    huc2 = huc4[:2]

    # Common splits
    if huc2 == '03':
        # 03 is split into N/S/W. Rough simplification based on HUC4:
        # Real logic is complex, but this covers major basins
        if huc4.startswith('0308') or huc4.startswith('0309') or huc4.startswith('031'):
            return '03S'  # Florida/South
        elif huc4.startswith('0315') or huc4.startswith('0316') or huc4.startswith('0317'):
            return '03W'  # Gulf
        else:
            return '03N'  # Atlantic

    if huc2 == '10':
        # 10 is split into Lower (10L) and Upper (10U)
        # Upper: 1018, 1019, 1025. Lower is the rest? Roughly.
        # This is an approximation. For rigorous use, we'd check a VPU map.
        # Let's try 10L as default for "Lower Missouri" if unknown, but NWM uses both.
        # Check specific known subregions:
        if int(huc4) >= 1018 and int(huc4) <= 1025:
            return '10U'
        return '10L'

    if huc2 == '08': return '08'  # Mississippi Delta

    # Default is just the HUC2 code
    return huc2


def ensure_nhd_v2_local(lat, lon, cache_dir):
    """
    Downloads NHDPlus V2 for the VPU, merges Geometry + VAA, saves as GPKG.
    """
    if py7zr is None:
        print("  CRITICAL: 'py7zr' is not installed. Run 'pip install py7zr' to process NHDPlus V2.")
        return None

    # 1. Identify VPU
    huc4 = find_huc4_wbd(lat, lon)
    vpu = get_v2_vpu(huc4)

    if not vpu:
        print("  Could not identify VPU.")
        return None

    final_gpkg = os.path.join(cache_dir, f"NHDPlusV2_{vpu}.gpkg")
    if os.path.exists(final_gpkg):
        return final_gpkg

    print(f"Preparing NHDPlus V2 for VPU {vpu} (this happens once)...")
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Download Geometry (Snapshot) and Attributes (Attributes)
    # EPA/AWS Bucket structure
    base_url = "https://s3.amazonaws.com/ard-bucket/NHDPlusV21/Compress"
    files_to_get = {
        "geo": f"NHDPlusV21_{vpu}_NHDSnapshot_05.7z",
        "attr": f"NHDPlusV21_{vpu}_NHDPlusAttributes_09.7z"
    }

    extracted_paths = {}

    for key, fname in files_to_get.items():
        local_7z = os.path.join(cache_dir, fname)
        if not download_file_stream(f"{base_url}/{fname}", local_7z):
            print(f"  Failed to download {fname}")
            return None

        # Extract
        print(f"  Extracting {fname}...")
        try:
            with py7zr.SevenZipFile(local_7z, mode='r') as z:
                z.extractall(path=cache_dir)
            extracted_paths[key] = True
            # os.remove(local_7z) # Optional: keep zip for backup?
        except Exception as e:
            print(f"  Extraction error: {e}")
            return None

    # 3. Find and Merge Data
    # The extraction creates a folder like "NHDPlus{VPU}"
    search_root = os.path.join(cache_dir, f"NHDPlus{vpu}")
    if not os.path.exists(search_root):
        # Sometimes VPU folders have different names like "NHDPlus03S"
        search_root = [p for p in os.listdir(cache_dir) if
                       f"NHDPlus{vpu}" in p and os.path.isdir(os.path.join(cache_dir, p))]
        if search_root: search_root = os.path.join(cache_dir, search_root[0])

    if not search_root or not os.path.exists(search_root):
        print("  Could not find extracted folder.")
        return None

    print("  Merging Geometry and Attributes...")
    try:
        # Paths inside the standard NHD folder structure
        # Geometry: ./NHDPlusXX/NHDSnapshot/Hydrography/NHDFlowline.shp
        # Attributes: ./NHDPlusXX/NHDPlusAttributes/PlusFlowlineVAA.dbf

        # We search recursively because structure varies slightly
        flow_path = None
        vaa_path = None

        for root, dirs, files in os.walk(search_root):
            for f in files:
                if f.lower() == "nhdflowline.shp":
                    flow_path = os.path.join(root, f)
                elif f.lower() == "plusflowlinevaa.dbf":
                    vaa_path = os.path.join(root, f)

        if not flow_path or not vaa_path:
            print("  Missing NHDFlowline.shp or PlusFlowlineVAA.dbf")
            return None

        # Read using Geopandas
        gdf = gpd.read_file(flow_path)
        vaa = gpd.read_file(vaa_path)

        # Standardize columns (V2 uses 'COMID', 'Hydroseq')
        # We ensure they are uppercase for consistency with NWM tools
        gdf.columns = [c.upper() for c in gdf.columns]
        vaa.columns = [c.upper() for c in vaa.columns]

        # Merge
        merged = gdf.merge(vaa, on='COMID', how='left')

        # Save compressed GPKG
        merged.to_file(final_gpkg, driver="GPKG", layer='NHDFlowline')
        print(f"  Saved NHDPlus V2 cache: {final_gpkg}")

        return final_gpkg

    except Exception as e:
        print(f"  Merge failed: {e}")
        return None


# =================================================================
# 4. RAW LAND USE & OTHERS (Existing)
# =================================================================

def ensure_esa_tiles(lat, lon, cache_dir, year=2021):
    """Downloads ESA WorldCover tiles."""
    os.makedirs(cache_dir, exist_ok=True)
    lat_floor = math.floor(lat / 3) * 3
    lon_floor = math.floor(lon / 3) * 3

    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"

    tile_name = f"{lat_prefix}{abs(lat_floor):02d}{lon_prefix}{abs(lon_floor):03d}"
    filename = f"ESA_WorldCover_10m_{year}_v200_{tile_name}_Map.tif"
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        return [local_path]

    print(f"Downloading ESA Tile: {tile_name}...")
    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/{year}/map/{filename}"
    if download_file_stream(url, local_path):
        return [local_path]
    return []


def get_flowlines_local(lat, lon, nhd_cache_dir, site_id=None):
    """
    Public function to get V2 flowlines for a site.
    """
    # 1. Ensure Big File (V2)
    big_gpkg_path = ensure_nhd_v2_local(lat, lon, nhd_cache_dir)

    if not big_gpkg_path: return None, None

    # 2. Extract Specific Site
    # V2 is usually decimal degrees (NAD83), effectively EPSG:4269 or 4326.
    # We treat it as 4326 for simplicity in bounds check.
    buffer = 0.02
    bbox = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)

    try:
        gdf = gpd.read_file(big_gpkg_path, bbox=bbox)
        if gdf.empty: return None, None

        # Ensure we return a temp path
        import tempfile
        temp_flow = tempfile.NamedTemporaryFile(suffix=".gpkg", prefix=f"lhd_{site_id}_nhd_", delete=False)
        temp_flow.close()
        gdf.to_file(temp_flow.name, driver="GPKG")

        return temp_flow.name, gdf
    except Exception as e:
        print(f"Error reading V2 cache: {e}")
        return None, None