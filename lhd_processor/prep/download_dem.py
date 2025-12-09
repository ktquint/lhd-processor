import os
import re
import math
import requests
import rasterio
import numpy as np
from rasterio.crs import CRS
from dateutil.parser import parse
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling


def extract_date(item):
    return item.get("dateCreated") or item.get("publishedDate") or ""


def extract_tile_id(title):
    match = re.search(r'x\d+y\d+', title)
    return match.group(0) if match else None


def sanitize_filename(filename):
    """
        some files have '/' in their name like "1/3 arc-second," so we'll fix it
    """
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, "-")  # Replace invalid characters with '_'
    return filename


def meters_to_latlon(lat0, lon0, dx, dy):
    R = 6378137  # radius of Earth in meters (WGS84)
    d_lat = int(dy) / R
    d_lon = int(dx) / (R * math.cos(math.radians(lat0)))

    new_lat = lat0 + math.degrees(d_lat)
    new_lon = lon0 + math.degrees(d_lon)
    return float(new_lat), float(new_lon)


def merge_dems(dem_files, output_filename):
    dem_dir = os.path.dirname(dem_files[0])
    output_loc = os.path.join(dem_dir, output_filename)

    mosaic_inputs = []
    mem_files = []
    target_crs = None

    for fp in dem_files:
        src = rasterio.open(fp)
        if target_crs is None:
            target_crs = src.crs

        if src.crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            mem_file = MemoryFile()
            mem_files.append(mem_file)  # Keep reference alive
            dst = mem_file.open(**kwargs)

            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)

            dst.close()
            dst = mem_file.open()
            mosaic_inputs.append(dst)
        else:
            mosaic_inputs.append(src)

    mosaic, out_transform = merge(mosaic_inputs)

    out_meta = mosaic_inputs[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": target_crs
    })

    with rasterio.open(output_loc, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Clean up
    for src in mosaic_inputs:
        src.close()
    for mem_file in mem_files:
        mem_file.close()

    # Delete original files
    if os.path.exists(output_loc):
        for fp in dem_files:
            try:
                os.remove(fp)
                print(f"Deleted {fp}")
            except Exception as e:
                print(f"Could not delete {fp}: {e}")
    else:
        print("Merged DEM not found. Original files not deleted.")


# noinspection PyArgumentList
def check_bbox_coverage(dem_path, bbox):
    """
    Check if all four corners of the bounding box are contained within the DEM
    and have valid elevation values. Also returns the approximate resolution.

    Args:
        dem_path: Path to the merged DEM file
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) in EPSG:4326

    Returns:
        tuple: (bool: coverage_good, resolution_meters: float or None)
               Returns resolution in meters if determinable, otherwise None.
    """
    try:
        with rasterio.open(dem_path) as dem:
            dem_bounds = dem.bounds
            nodata = dem.nodata
            crs_needs_transform = dem.crs.to_string() != 'EPSG:4326'

            # --- Get Resolution ---
            pixel_width = dem.res[0]
            pixel_height = abs(dem.res[1]) # Resolution is often negative for y
            resolution_meters = None

            # Check if CRS units are meters
            if dem.crs.is_projected and 'metre' in dem.crs.linear_units.lower():
                 # Average pixel dimensions if they differ slightly
                resolution_meters = (pixel_width + pixel_height) / 2
            elif dem.crs.is_geographic:
                # Approximate conversion from degrees to meters at the DEM center
                # This is less precise, prefer projected CRS when possible
                # center_lon = (dem_bounds.left + dem_bounds.right) / 2
                center_lat = (dem_bounds.bottom + dem_bounds.top) / 2
                m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
                m_per_deg_lat = 110574
                res_m_x = pixel_width * m_per_deg_lon
                res_m_y = pixel_height * m_per_deg_lat
                resolution_meters = (res_m_x + res_m_y) / 2
            else:
                 print(f"Warning: Could not determine units for CRS {dem.crs}. Resolution check skipped.")
            # --- End Resolution Check ---


            # Define the four corners (in EPSG:4326)
            corners_4326 = [
                (bbox[0], bbox[1]),  # bottom-left
                (bbox[2], bbox[1]),  # bottom-right
                (bbox[2], bbox[3]),  # top-right
                (bbox[0], bbox[3])   # top-left
            ]

            corners_native = corners_4326
            # Transform corners to raster CRS if needed
            if crs_needs_transform:
                xs, ys = zip(*corners_4326)
                src_crs = CRS.from_epsg(4326)
                try:
                    xs_native, ys_native = rasterio.warp.transform(src_crs, dem.crs, xs, ys)
                    corners_native = list(zip(xs_native, ys_native))
                except Exception as transform_error:
                    print(f"Error transforming coordinates for coverage check: {transform_error}")
                    return False, resolution_meters # Return False for coverage if transform fails

            # Check each corner
            for x, y in corners_native:
                # Bounds check
                if not (dem_bounds.left <= x <= dem_bounds.right and
                        dem_bounds.bottom <= y <= dem_bounds.top):
                    print(f"Corner ({x:.2f}, {y:.2f}) outside DEM bounds.")
                    return False, resolution_meters

                # Convert to pixel indices safely
                try:
                    row, col = dem.index(x, y)
                except ValueError:
                     print(f"Corner ({x:.2f}, {y:.2f}) likely outside raster grid.")
                     return False, resolution_meters


                # Check if indices are within valid range
                if not (0 <= row < dem.height and 0 <= col < dem.width):
                    print(f"Pixel index ({row}, {col}) out of range.")
                    return False, resolution_meters

                # Read elevation value
                try:
                    # Read only the single pixel needed
                    value = dem.read(1, window=((row, row + 1), (col, col + 1)))[0][0]
                except IndexError:
                    print(f"IndexError reading pixel at ({row}, {col}).")
                    return False, resolution_meters

                # Check for nodata or NaN
                # Explicitly check against nodata first
                if nodata is not None and value == nodata:
                    print(f"Corner ({x:.2f}, {y:.2f}) has nodata value.")
                    return False, resolution_meters
                # Then check for NaN if it's a float type
                elif isinstance(value, float) and math.isnan(value):
                     print(f"Corner ({x:.2f}, {y:.2f}) has NaN value.")
                     return False, resolution_meters
                # Handle potential non-numeric types if necessary (though less likely for DEMs)
                elif not isinstance(value, (int, float, np.number)):
                     print(f"Warning: Unexpected data type {type(value)} at ({row}, {col}). Treating as invalid.")
                     return False, resolution_meters


            # If all corners passed
            return True, resolution_meters

    except rasterio.RasterioIOError as e:
        print(f"Error opening DEM file {dem_path}: {e}")
        return False, None
    except Exception as e:
        print(f"Unexpected error checking bbox coverage for {dem_path}: {e}")
        return False, None


def query_tnm_api(location: list[float], dataset: str, buffer: float):
    """
        location: [latitude, longitude]
        resolution: meters
    """
    lat, lon = location[0], location[1]
    bounding_dist = 2 * buffer
    upper_lat, upper_lon = meters_to_latlon(lat, lon, bounding_dist, bounding_dist)
    lower_lat, lower_lon = meters_to_latlon(lat, lon, -1 * bounding_dist, -1 * bounding_dist)

    bbox = (lower_lon, lower_lat, upper_lon, upper_lat)

    tnm_url = "https://tnmaccess.nationalmap.gov/api/v1/products"  # all products are found here

    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
              "datasets": dataset,
              # "max": 10,
              "prodFormats": ["GeoTIFF"],
              "outputFormat": "JSON", }
    try:
        response = requests.get(tnm_url, params=params)
        response.raise_for_status()
        results = response.json().get("items", [])
        return results

    except requests.RequestException as e:
        print(f"Error occurred: {e}")


def filter_latest_tiles(api_results):
    filtered = []
    for item in api_results:
        title = item.get("title", "")
        if extract_tile_id(title) and extract_date(item):
            filtered.append(item)

    # Build dict of most recent item per tile
    tile_to_item = {}
    for item in filtered:
        title = item.get("title", "")
        tile_id = extract_tile_id(title)
        date_str = extract_date(item)
        try:
            date_obj = parse(date_str)
        except ValueError:
            continue  # skip if date format is wrong

        if tile_id not in tile_to_item or date_obj > tile_to_item[tile_id][0]:
            tile_to_item[tile_id] = (date_obj, item)

    # Extract only the most recent item per tile
    return [entry[1] for entry in tile_to_item.values()]


def download_dem(lhd_id: int, lat: float, lon: float, weir_length: float, dem_dir: str, resolution: str):
    print("Starting DEM Download Process...")

    all_datasets = ["Digital Elevation Model (DEM) 1 meter",
                    "National Elevation Dataset (NED) 1/9 arc-second",
                    "National Elevation Dataset (NED) 1/3 arc-second Current"]

    if resolution == "1 m":
        datasets = all_datasets
    elif resolution == "1/9 arc-second (~3 m)":
        datasets = all_datasets[1:]
    else:
        datasets = all_datasets[2:]

    # Check if DEM already exists
    dem_subdir = os.path.join(dem_dir, str(lhd_id))
    os.makedirs(dem_subdir, exist_ok=True)
    dem_path = os.path.join(dem_subdir, f"{lhd_id}_MERGED_DEM.tif")

    # Calculate bbox for coverage checking
    # ... (bbox calculation remains the same) ...
    bounding_dist = 2 * weir_length
    upper_lat, upper_lon = meters_to_latlon(lat, lon, bounding_dist, bounding_dist)
    lower_lat, lower_lon = meters_to_latlon(lat, lon, -1 * bounding_dist, -1 * bounding_dist)
    bbox = (lower_lon, lower_lat, upper_lon, upper_lat)

    if os.path.isfile(dem_path):
        print(f"DEM already exists at {dem_path}, checking coverage and resolution...")
        coverage_good, existing_res_meters = check_bbox_coverage(dem_path, bbox)  # Capture resolution

        if coverage_good:
            print(f"Existing DEM has good coverage with resolution ~{existing_res_meters:.2f}m")
            # Return subdir, empty list for titles (not downloaded), and the detected resolution
            return dem_subdir, [], existing_res_meters
        else:
            print(f"Existing DEM has poor coverage, will re-download.")
            try:
                os.remove(dem_path)
                print(f"Removed existing DEM with poor coverage: {dem_path}")
            except Exception as e:
                print(f"Could not remove existing DEM: {e}")

    # --- Downloading logic ---
    print(f"Getting DEM info for {lhd_id}")
    # ... (API query loop remains the same) ...

    final_titles = []
    final_dataset = None
    final_resolution_meters = None  # Store the resolution here
    coverage_achieved = False

    for dataset in datasets:
        print(f"\nTrying dataset: {dataset}")
        # final_dataset = dataset # Keep track of the *attempted* dataset

        try:
            results = query_tnm_api([lat, lon], dataset, weir_length)
            # ... (filtering results logic remains the same) ...
            if len(results) == 0:
                print(f"No results for {dataset}... Trying next resolution")
                continue

            # (Logic for handling single vs multiple results and setting final_results, final_titles, final_download_urls)
            # Determine approximate resolution based on dataset name *before* download
            current_res_meters = None
            if dataset == "Digital Elevation Model (DEM) 1 meter":
                current_res_meters = 1.0
            elif dataset == "National Elevation Dataset (NED) 1/9 arc-second":
                current_res_meters = 3.0  # Approximate
            elif dataset == "National Elevation Dataset (NED) 1/3 arc-second Current":
                current_res_meters = 10.0  # Approximate

            # Handle cases with multiple results (filtering latest tiles for 1m, etc.)
            if len(results) > 1 and dataset == "Digital Elevation Model (DEM) 1 meter":
                final_results = filter_latest_tiles(results)
            else:
                final_results = results  # Use all results for other datasets or single results

            # Check if filtering left any results
            if not final_results:
                print(f"No valid/latest tiles found for {dataset} after filtering.")
                continue

            final_titles = [sanitize_filename(dem.get("title", "")) for dem in final_results]
            final_download_urls = [dem.get("downloadURL") for dem in final_results]

            # Download the DEM(s) for this dataset
            print(f"Downloading DEM for {lhd_id} using {dataset} (~{current_res_meters}m)")
            # ... (downloading and merging logic remains the same) ...
            if len(final_results) > 1:
                # Multiple DEMs - download and merge
                temp_paths = [os.path.join(dem_subdir, f"{title}.tif") for title in final_titles]
                print(f"Downloading {len(final_results)} tiles...")

                for i in range(len(final_results)):
                    url = final_download_urls[i]
                    path = temp_paths[i]
                    if not os.path.exists(path):  # Avoid re-downloading if merging failed previously
                        print(f"  Downloading tile {i + 1}/{len(final_results)}: {final_titles[i]}")
                        # (Download logic)
                        with requests.get(url, stream=True) as r:
                            r.raise_for_status()
                            with open(path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                    else:
                        print(f" Tile {i + 1} already exists: {path}")

                print("Merging tiles...")
                try:
                    merge_dems(temp_paths, os.path.basename(dem_path))  # Pass only filename to merge_dems
                except Exception as merge_error:
                    print(f"Error merging DEMs for {dataset}: {merge_error}")
                    # Clean up temporary files if merge fails
                    for fp in temp_paths:
                        if os.path.exists(fp):
                            os.remove(fp)
                    continue  # Try next dataset

            else:
                # Single DEM - download directly
                print(f"Downloading single DEM: {final_titles[0]}")
                # (Download logic)
                with requests.get(final_download_urls[0], stream=True) as r:
                    r.raise_for_status()
                    with open(dem_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

            # Check coverage after download
            print("Checking DEM coverage...")
            coverage_good, downloaded_res_meters = check_bbox_coverage(dem_path, bbox)  # Get resolution

            if coverage_good:
                print(f"✓ Good coverage achieved with {dataset} (Actual res: ~{downloaded_res_meters:.2f}m)")
                coverage_achieved = True
                final_dataset = dataset  # Confirm this dataset was successful
                final_resolution_meters = downloaded_res_meters  # Store the *actual* resolution
                break  # Success! Exit the dataset loop
            else:
                print(f"✗ Poor coverage with {dataset}, trying next resolution...")
                # (Remove failed DEM logic remains the same) ...
                try:
                    if os.path.exists(dem_path):
                        os.remove(dem_path)
                        print(f"Removed DEM with poor coverage")
                except Exception as e:
                    print(f"Could not remove failed DEM: {e}")
                continue  # Try next dataset


        except requests.RequestException as e:
            print(f"Error querying/downloading {dataset}: {e}")
            continue

    # Final results
    if coverage_achieved:
        print(f"\n✓ Successfully processed DEM for {lhd_id} using {final_dataset}")
        # Return subdir, titles, and the *measured* resolution
        return dem_subdir, final_titles, final_resolution_meters
    else:
        print(f"\n✗ No DEM dataset provided adequate coverage for {lhd_id}")
        return None, [], None
