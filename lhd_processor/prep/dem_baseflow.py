import os
import laspy
import requests
import tempfile
import numpy as np
import laspy.errors
from pyproj import Transformer
from scipy.spatial import cKDTree
from datetime import datetime, timedelta

# global variables
base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"


def gpstime_to_date(gps_time: float) -> str:
    """
    Converts GPS time (seconds since 1980-01-06) to a date string 'yyyy-MM-DD'.

    Parameters:
        gps_time (float): GPS time in seconds.

    Returns:
        str: Date string in 'yyyy-MM-DD' format.
    """
    gps_epoch = datetime(1980, 1, 6)
    utc_time = gps_epoch + timedelta(seconds=gps_time)

    return utc_time.strftime('%Y-%m-%d')


def find_water_gpstime(lat, lon):
    bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)

    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
              "datasets": "Lidar Point Cloud (LPC)",
              "max": 1, "outputFormat": "JSON"}

    las_path = "Unknown"  # Define las_path here for error logging
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])
        if not items:
            print("No Lidar data found.")
            return None

        download_url = items[0].get("downloadURL")
        if not download_url:
            print("No download URL found.")
            return None

        print("Downloading LiDAR file...")
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "lidar_data")

            lidar_response = requests.get(download_url)
            lidar_response.raise_for_status()

            # Determine extension
            content_type = lidar_response.headers.get("Content-Type", "").lower()
            if "zip" in content_type or download_url.endswith(".zip"):
                file_path += ".zip"
            elif download_url.endswith(".laz"):
                file_path += ".laz"
            elif download_url.endswith(".las"):
                file_path += ".las"
            else:
                print("Unknown file type.")
                return None

            with open(file_path, 'wb') as f:
                f.write(lidar_response.content)

            if file_path.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                laz_files = [f for f in os.listdir(tmpdir) if f.lower().endswith((".las", ".laz"))]
                if not laz_files:
                    print("No LAS/LAZ files found in ZIP.")
                    return None
                las_path = os.path.join(tmpdir, laz_files[0])
            else:
                las_path = file_path  # raw .las or .laz file

            print(f"Processing {os.path.basename(las_path)}...")
            las = laspy.read(las_path)
            crs_obj = las.header.parse_crs()
            if crs_obj is None:
                print("CRS not found in LAS header.")
                # Option: Assume a default if you know it, e.g.,
                # known_epsg = 26912 # Replace with the correct code
                # print(f"Assuming known EPSG:{known_epsg}")
                # transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{known_epsg}", always_xy=True)
                return None  # Or handle assumption

            # Try getting EPSG code first
            crs_auth = crs_obj.sub_crs_list[0].to_authority()
            if crs_auth and crs_auth[0] == 'EPSG':
                epsg_code = crs_auth[1]
                print(f"Found EPSG code: {epsg_code}")
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
            else:
                # If no EPSG, try using the CRS object directly (pyproj might handle it)
                print("No standard EPSG code found. Trying to create Transformer from CRS object.")
                try:
                    # Pass the CRS object itself instead of an EPSG string
                    transformer = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
                    print("Successfully created transformer from CRS object.")
                except Exception as e:
                    print(f"Could not create transformer from CRS object: {e}")
                    # Option: Assume a default if you know it
                    # known_epsg = 26912 # Replace with the correct code
                    # print(f"Assuming known EPSG:{known_epsg}")
                    # try:
                    #      transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{known_epsg}", always_xy=True)
                    # except Exception as e2:
                    #      print(f"Failed even with assumed EPSG: {e2}")
                    #      return None
                    return None  # Give up if transformation isn't possible

            easting, northing = transformer.transform(lon, lat)

            # Get all points and times
            x = las.x
            y = las.y
            gpstimes = las.gps_time

            # --- MODIFICATION START ---

            # First, try to find a water point (classification 9)
            water_mask = las.classification == 9
            if np.any(water_mask):
                print("Water-classified points found. Searching for nearest water point...")
                water_points = np.vstack((x[water_mask], y[water_mask])).T
                water_tree = cKDTree(water_points)
                dist, idx = water_tree.query([easting, northing])

                if dist <= 100:  # Found a nearby water point
                    print(f"Found nearby water point (dist: {dist:.2f}m).")
                    water_gpstimes = gpstimes[water_mask]
                    adjusted_time = water_gpstimes[idx]
                    full_gps_time = adjusted_time + 1_000_000_000
                    return full_gps_time
                else:
                    print(f"Nearest water point is too far (dist: {dist:.2f}m).")
            else:
                print("No water-classified points found.")

            # Fallback: If no water points were found, or they were too far
            print("Fallback: Searching for the absolute nearest point of ANY classification...")

            all_points = np.vstack((x, y)).T
            all_tree = cKDTree(all_points)
            dist, idx = all_tree.query([easting, northing])

            # You might still want a reasonable distance threshold
            if dist > 100:  # If even the *nearest* point is > 100m away, something is wrong
                print(f"No nearby points of any kind found within 100m (nearest dist: {dist:.2f}m).")
                return None

            print(f"Found absolute nearest point (dist: {dist:.2f}m) with classification: {las.classification[idx]}")
            adjusted_time = gpstimes[idx]  # Use the index on the *full* gpstimes array
            full_gps_time = adjusted_time + 1_000_000_000  # adjust to standard GPS time

            return full_gps_time

    except (ImportError, laspy.errors.LaspyException) as e:
        print("=" * 50)
        print("ERROR: FAILED TO READ LIDAR FILE")
        print(f"File: {las_path}")
        print(f"Error: {e}")
        print("\nThis means your environment is missing a library to read compressed")
        print("LiDAR files (.laz). To fix this, please ensure your environment is active")
        print("(e.g., 'conda activate lhd-environment') and run:")
        print("\n    pip install lazrs\n")
        print("=" * 50)
        return None  # Continue to fallback (median flow)
    except Exception as e:
        print(f"An unexpected error occurred in find_water_gpstime: {e}")
        print(f"File: {las_path if 'las_path' in locals() else 'Unknown'}")
        return None


def est_dem_baseflow(stream_reach, source, method):
    """
    Finds baseflow for a dem along a stream reach.
    Returns a tuple: (dem_baseflow, found_date)
    """
    # extract the lat and lon
    lat = stream_reach.latitude
    lon = stream_reach.longitude

    found_date = None  # Variable to store the date we find from LiDAR
    dem_baseflow = None

    # Robust check for method string
    if "lidar date" in method.lower():
        print("Searching for Lidar Point Cloud to find date...")
        lidar_gpstime = find_water_gpstime(lat, lon)

        if lidar_gpstime:
            found_date = gpstime_to_date(lidar_gpstime)
            print(f"LiDAR Date found: {found_date}")

        # Use the date to estimate baseflow
        if not found_date:
            print("No LiDAR date found. Assuming median flow.")
            dem_baseflow = stream_reach.get_median_flow(source)
        else:
            print(f"Estimating flow for date: {found_date}...")
            dem_baseflow = stream_reach.get_flow_on_date(found_date, source)

        print(f'The baseflow estimate is {dem_baseflow} cms')

    elif method == "WSE and Median Daily Flow":
        dem_baseflow = stream_reach.get_median_flow(source)

    else:  # method == 2-yr Q and banks
        dem_baseflow = stream_reach.get_2yr_return_period(source)

    # Return BOTH the flow and the date
    return dem_baseflow, found_date
