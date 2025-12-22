import os
import s3fs
import zipfile
import requests
import geopandas as gpd
from shapely.geometry import Point
from .download_dem import sanitize_filename
from math import radians, sin, cos, sqrt, atan2


def make_bbox(latitude: float, longitude: float, distance_deg: float=0.5) -> list[float]:
    """
        creates a bounding box around a point (lat, lon) ±distance_deg degrees.
    """
    lat_min = latitude - distance_deg
    lat_max = latitude + distance_deg
    lon_min = longitude - distance_deg / cos(radians(latitude))  # adjust for longitude convergence
    lon_max = longitude + distance_deg / cos(radians(latitude))
    return [lon_min, lat_min, lon_max, lat_max]


def haversine(lat1, lon1, lat2, lon2):
    """
        computes approximate distance (km) between two points.
    """
    R = 6371.0  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# noinspection PyTypeHints
def find_huc(latitude: float, longitude: float) -> str or None:
    """
    Queries the USGS WBD Map Service to find the HUC4 code for a point.
    """
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/2/query"

    params = {
        'geometry': f"{longitude},{latitude}",
        'geometryType': 'esriGeometryPoint',
        'inSR': '4326',  # <--- Tells server these are Lat/Lon coordinates
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': 'huc4,name',
        'f': 'json',
        'returnGeometry': 'false'
    }

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        if 'features' in data and len(data['features']) > 0:
            # Handle potential case sensitivity (huc4 vs HUC4)
            attrs = data['features'][0]['attributes']
            # Try lowercase keys first, then uppercase, or standard HUC4
            huc4 = attrs.get('huc4') or attrs.get('HUC4')
            # name = attrs.get('name') or attrs.get('NAME')

            if huc4:
                # print(f"Found {huc4} ({name})") # Uncomment if you want verbose logs
                return huc4

        print(f"Point {latitude},{longitude} is not inside a US HUC4 boundary.")
        return None

    except Exception as e:
        print(f"Error looking up HUC4: {e}")
        return None


def download_nhdplus(latitude: float, longitude: float, flowline_dir: str) -> str | None:
    # 1. Get the HUC
    hu4 = find_huc(latitude, longitude)
    if hu4 is None:
        return None

    hu4 = f"{int(hu4):04d}"  # Ensure 4-digit string

    # 2. Check local file (prevent re-download)
    already_processed = [f for f in os.listdir(flowline_dir) if hu4 in f and 'VAA_RI' in f]
    if already_processed:
        print(f"    – HUC {hu4} is already downloaded and processed.")
        return os.path.join(flowline_dir, already_processed[0])

    # 3. Setup TNM API Query
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    # We remove 'bbox' here to be safer and just rely on the HUC search
    params = {
        "datasets": "National Hydrography Dataset Plus High Resolution (NHDPlus HR)",
        "q": hu4,
        "prodFormats": "GeoPackage",
        "max": 10
    }

    print(f"    – Searching USGS database for HUC {hu4}...")

    try:
        r = requests.get(base_url, params=params)
        r.raise_for_status()
        items = r.json().get('items', [])

        if not items:
            print("    – No NHDPlus HR products found for this HUC.")
            return None

        # Sort by date (Newest First)
        items.sort(key=lambda x: x.get('publicationDate', ''), reverse=True)

        final_gpkg_loc = None

        for item in items:
            download_url = item['downloadURL']
            file_title = item['title']
            pub_date = item['publicationDate']

            print(f"    – Attempting version: {pub_date}...")

            try:
                # Setup download path
                sanitized_title = sanitize_filename(file_title)
                local_zip = os.path.join(flowline_dir, f"{sanitized_title}.zip")

                # Stream Download
                with requests.get(download_url, stream=True) as stream:
                    # Check if the link is actually valid before downloading
                    if stream.status_code == 404:
                        print(f"      [404 Error] This USGS link is broken. Trying next version...")
                        continue  # Skip to next item in loop

                    stream.raise_for_status()  # Raise other errors (500, etc.)

                    # If we get here, the link is good! Download it.
                    with open(local_zip, "wb") as f:
                        for chunk in stream.iter_content(chunk_size=8192):
                            f.write(chunk)

                # Unzip
                print("      Extracting...")
                gpkg_name = None
                with zipfile.ZipFile(local_zip, 'r') as z:
                    z.extractall(flowline_dir)
                    extracted_files = z.namelist()
                    gpkg_name = next((f for f in extracted_files if f.endswith('.gpkg')), None)

                os.remove(local_zip)

                if not gpkg_name:
                    print("      [Error] No .gpkg found in zip. Trying next version...")
                    continue

                gpkg_loc = os.path.join(flowline_dir, gpkg_name)

                # --- Processing (Merge/Reindex) ---
                print("      Processing Flowlines and VAAs...")
                try:
                    flowlines_gdf = gpd.read_file(gpkg_loc, layer='NHDFlowline', engine='fiona')
                    metadata_gdf = gpd.read_file(gpkg_loc, layer='NHDPlusFlowlineVAA', engine='fiona')
                except Exception as e:
                    print(f"      [Error] Could not read layers ({e}). Trying next version...")
                    continue

                # Normalizing & Merging
                flowlines_gdf.columns = flowlines_gdf.columns.str.lower()
                metadata_gdf.columns = metadata_gdf.columns.str.lower()
                common_cols = ['nhdplusid', 'reachcode', 'vpuid']

                if not all(col in flowlines_gdf.columns for col in common_cols):
                    print("      [Error] Missing join columns. Trying next version...")
                    continue

                merged_gdf = flowlines_gdf.merge(metadata_gdf, on=common_cols)

                # Re-Indexing
                reindexed_cols = {'hydroseq', 'uphydroseq', 'dnhydroseq'}
                if reindexed_cols.issubset(merged_gdf.columns) and not merged_gdf.empty:
                    min_val = merged_gdf['hydroseq'].min()
                    if min_val > 1e9:
                        offset = min_val - 1
                        print(f"      Re-indexing huge Hydroseq (Min: {min_val:.0f}). Offset: {offset:.0f}")
                        for col in reindexed_cols:
                            merged_gdf[col] = merged_gdf[col] - offset

                # Save Final
                final_path = gpkg_loc.replace('.gpkg', '_VAA_RI.gpkg')
                merged_gdf.to_file(final_path, layer='NHDFlowline', driver='GPKG')

                # Cleanup
                for ext in ['.gpkg', '.xml', '.jpg']:
                    f_path = gpkg_loc.replace('.gpkg', ext)
                    if os.path.exists(f_path):
                        os.remove(f_path)

                # SUCCESS! Break the loop and return
                final_gpkg_loc = final_path
                print(f"    – Success! Processed: {os.path.basename(final_path)}")
                break

            except Exception as e:
                print(f"      [Error] Failed processing {pub_date}: {e}")
                continue

        if final_gpkg_loc:
            return final_gpkg_loc
        else:
            print("    – CRITICAL: All available versions failed to download/process.")
            return None

    except Exception as e:
        print(f"    – API Error: {e}")
        return None

# noinspection PyTypeChecker
def download_tdx_hydro(latitude: float, longitude: float, flowline_dir: str, vpu_map_path: str) -> str:
    """
    Finds the correct VPU code for a lat/lon by reading the vpu_map_path file,
    then downloads the individual VPU .gpkg file from S3 if it doesn't
    already exist in flowline_dir.

    This functions just like the NHDPlus download workflow.
    """
    print(f"Identifying VPU for dam at {latitude}, {longitude}...")

    # 1. Read the VPU Boundaries map file (vpu_boundaries.gpkg)
    try:
        vpu_gdf = gpd.read_file(vpu_map_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read the VPU boundaries file at {vpu_map_path}")
        print(f"Error: {e}")
        return None  # This will be caught by prep/dam.py

    dam_point = Point(longitude, latitude)  # Note: (lon, lat) for shapely

    # 2. Find which VPU polygon contains the dam point
    # Ensure VPU map is in the same CRS (EPSG:4326) as the point
    if vpu_gdf.crs.to_epsg() != 4326:
        vpu_gdf = vpu_gdf.to_crs(epsg=4326)

    vpu_polygon = vpu_gdf[vpu_gdf.contains(dam_point)]

    if vpu_polygon.empty:
        print(f"No VPU boundary polygon found for point {longitude}, {latitude}")
        print("Dam is likely outside the VPU boundaries.")
        return None  # This will be caught by prep/dam.py

    # 3. Get the VPU code from that polygon (NEW, MORE ROBUST LOGIC)
    try:
        # Create a lowercase-to-original-case mapping of columns
        cols_lower = {col.lower(): col for col in vpu_polygon.columns}

        if 'vpu' in cols_lower:
            vpu_col_name = cols_lower['vpu']
        elif 'vpucode' in cols_lower:
            vpu_col_name = cols_lower['vpucode']
        else:
            # If we can't find a match, raise an error
            raise KeyError("Could not find a VPU column ('VPU', 'vpu', or 'VPUCode')")

        vpu_code = str(vpu_polygon.iloc[0][vpu_col_name])
        print(f"VPU identified: {vpu_code}")

    except KeyError as e:
        print(f"ERROR: {e}")
        print(f"Could not find a VPU ID column in {vpu_map_path}")
        print(f"Available columns are: {vpu_gdf.columns.to_list()}")
        return None  # This will be caught by prep/dam.py

    # 4. Create the path to where the individual VPU file *should* be
    os.makedirs(flowline_dir, exist_ok=True)  # Make sure the LHD_STRMs dir exists
    gpkg_loc = os.path.join(flowline_dir, f"streams_{vpu_code}.gpkg")

    # 5. Check if we already downloaded it. If so, we're done.
    if os.path.exists(gpkg_loc):
        print(f"Local file {gpkg_loc} already exists. Using it.")
        return gpkg_loc

    # 6. If it doesn't exist, download it from S3
    print(f"Local file not found. Downloading {gpkg_loc}...")
    base_url = 'geoglows-v2/hydrography/'
    gpkg_name = f"streams_{vpu_code}.gpkg"
    gpkg_url = f"{base_url}vpu={vpu_code}/{gpkg_name}"

    try:
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(gpkg_url, 'rb') as f_in:
            with open(gpkg_loc, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"Successfully downloaded {gpkg_loc}")
    except Exception as e:
        print(f"Failed to download {gpkg_url}: {e}")
        return None  # This will be caught by prep/dam.py

    # 7. Return the path to the newly downloaded file
    return gpkg_loc
