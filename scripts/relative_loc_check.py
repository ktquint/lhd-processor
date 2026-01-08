import os
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, LineString
from shapely.ops import linemerge

# --- FIX FOR PROJ DATABASE ERROR ---
import pyproj

try:
    # 1. Attempt to find the proj directory in the current environment
    # In mamba/conda, it is usually at {env_root}/share/proj
    proj_lib_path = Path(sys.prefix) / "share" / "proj"

    if proj_lib_path.exists():
        # Set the environment variable and the internal pyproj directory
        os.environ["PROJ_LIB"] = str(proj_lib_path)
        pyproj.datadir.set_data_dir(str(proj_lib_path))
        print(f"✅ Successfully set PROJ database path to: {proj_lib_path}")
    else:
        print(f"⚠️  Warning: Could not find PROJ directory at {proj_lib_path}")
        print("    If you see projection errors, try: export PROJ_LIB=/path/to/share/proj")

except Exception as e:
    print(f"⚠️  Error setting PROJ path: {e}")


# -----------------------------------


# --- 1. GEOMETRIC LOGIC ---

def get_hydraulic_target_points(dam_row):
    """
    Recreates the logic to find Upstream/Downstream points based on Weir Length.
    Returns: (Dictionary of points {'Upstream': Point...}, CRS used for calculation)
    """
    # 1. Parse Inputs
    weir_length = dam_row.get('weir_length', 30)
    lat = dam_row.get('latitude')
    lon = dam_row.get('longitude')
    source = dam_row.get('flowline_source', 'NHDPlus')

    if source == 'TDX-Hydro':
        flow_path = dam_row.get('flowline_path_tdx')
    else:
        flow_path = dam_row.get('flowline_path_nhd')

    # Basic validation
    if not flow_path or not os.path.exists(flow_path):
        return None, None

    # 2. Load Flowline
    try:
        flowline_gdf = gpd.read_file(flow_path)
    except Exception:
        return None, None

    if flowline_gdf.empty:
        return None, None

    # 3. Project to UTM (or planar) for accurate distance calc
    try:
        projected_crs = flowline_gdf.estimate_utm_crs()
    except:
        # If estimation fails, this is likely where your error was happening
        # Now that PROJ is fixed, this default should work.
        projected_crs = "EPSG:3857"

    flowline_gdf = flowline_gdf.to_crs(projected_crs)

    # Project Dam Point
    dam_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]

    # 4. Merge Flowlines
    merged_geom = linemerge(flowline_gdf.geometry.tolist())

    if merged_geom.geom_type == 'MultiLineString':
        merged_geom = min(merged_geom.geoms, key=lambda g: g.distance(dam_point))
    elif merged_geom.geom_type != 'LineString':
        return None, None

    # 5. Handle Directionality (TDX vs NHD)
    if source == 'TDX-Hydro':
        merged_geom = LineString(list(merged_geom.coords)[::-1])

    # 6. Linear Referencing
    start_dist = merged_geom.project(dam_point)
    total_len = merged_geom.length

    target_points = {}

    # Upstream
    up_dist = max(start_dist - weir_length, 0)
    target_points['Upstream'] = merged_geom.interpolate(up_dist)

    # Downstream 1-4
    for i in range(1, 5):
        down_dist = min(start_dist + (i * weir_length), total_len)
        target_points[f'Downstream{i}'] = merged_geom.interpolate(down_dist)

    return target_points, projected_crs


# --- 2. FILE UPDATE LOGIC ---

def fix_relative_loc(xs_path, dam_row):
    """
    Calculates the nearest target point for every XS line and assigns Relative_Loc.
    """
    # 1. Get the Target Points
    targets_dict, calc_crs = get_hydraulic_target_points(dam_row)

    if not targets_dict:
        print(f"    ⚠️ Could not calculate flowline targets for {xs_path.name}")
        return False

    # 2. Load the XS File
    try:
        xs_gdf = gpd.read_file(xs_path)
    except Exception as e:
        print(f"    ❌ Error reading XS GPKG: {e}")
        return False

    if xs_gdf.empty:
        return False

    # 3. Work in the same CRS
    try:
        xs_working = xs_gdf.to_crs(calc_crs).copy()
    except Exception as e:
        print(f"    ❌ CRS Error during transform: {e}")
        return False

    # 4. Calculate Relative_Loc
    new_locs = []
    for geom in xs_working.geometry:
        center = geom.centroid
        best_label = min(targets_dict.keys(), key=lambda k: center.distance(targets_dict[k]))
        new_locs.append(best_label)

    # 5. Apply and Save
    xs_gdf['Relative_Loc'] = new_locs

    try:
        xs_gdf.to_file(xs_path, driver="GPKG")
        return True
    except Exception as e:
        print(f"    ❌ Failed to save: {e}")
        return False


# --- 3. MAIN SCRIPT ---

def main():
    # --- CONFIGURATION ---
    # Update these paths if they changed
    search_path = Path("/Volumes/KenDrive/lhd_testing/Results_tdx_geo")
    master_db_path = "/Volumes/KenDrive/lhd_testing/merged_sites.xlsx"
    # ---------------------

    print(f"Loading Master Database from {master_db_path}...")
    if str(master_db_path).endswith('.xlsx'):
        df_master = pd.read_excel(master_db_path)
    else:
        df_master = pd.read_csv(master_db_path)

    # Ensure ID column is string to match filenames
    id_col = df_master.columns[0]
    df_master[id_col] = df_master[id_col].astype(str)

    fixed_files_list = []

    print(f"Scanning directory: {search_path}...")

    for file_path in search_path.rglob("*_Local_XS.gpkg"):
        try:
            # Quick check: does column exist?
            gdf_check = gpd.read_file(file_path, rows=0)

            if 'Relative_Loc' not in gdf_check.columns:
                print(f"Processing: {file_path.name}")

                dam_id_str = file_path.name.split('_')[0]
                row_lookup = df_master[df_master[id_col] == dam_id_str]

                if row_lookup.empty:
                    print(f"    ⚠️ Dam ID {dam_id_str} not found in inventory. Skipping.")
                    continue

                dam_row = row_lookup.iloc[0].to_dict()

                success = fix_relative_loc(file_path, dam_row)

                if success:
                    fixed_files_list.append(file_path.name)
                    print(f"    ✅ Fixed.")

        except Exception as e:
            # Now that PROJ is fixed, we shouldn't see 'Internal Proj Error' here
            print(f"Error inspecting {file_path}: {e}")

    print("\n" + "=" * 30)
    print("FILES UPDATED WITH Relative_Loc")
    print("=" * 30)
    print(fixed_files_list)


if __name__ == "__main__":
    main()