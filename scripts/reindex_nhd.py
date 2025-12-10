import os
import pyproj
import glob

# Force pyproj to use the database inside your specific environment
# We construct the path relative to the environment you are running in
env_path = "/opt/anaconda3/envs/lhd-environment"
proj_lib_path = os.path.join(env_path, "share", "proj")

# Set the environment variable explicitly inside the script
os.environ["PROJ_LIB"] = proj_lib_path
pyproj.datadir.set_data_dir(proj_lib_path)

# NOW import geopandas
import geopandas as gpd


def fix_reindexing_existing_files(flowline_dir: str):
    """
    Targets existing '_VAA_RI.gpkg' files for specific HUCs where re-indexing
    was missed. Reads the file, re-indexes hydroseq values, and overwrites
    the file in place.
    """

    # HUCs identified as having abnormally high values.
    # Padded to 4 digits to match standard NHD naming (e.g., '103' -> '0103').
    abnormal_hucs = [103, 105, 206, 702, 1020, 1106, 1506, 1602, 1702]
    target_hucs_str = [f"{huc:04d}" for huc in abnormal_hucs]

    print(f"Targeting existing VAA_RI files for HUCs: {target_hucs_str}")

    for huc in target_hucs_str:
        # Search for EXISTING processed files (*_VAA_RI.gpkg) containing the HUC
        search_pattern = os.path.join(flowline_dir, f"*{huc}*_VAA_RI.gpkg")
        found_files = glob.glob(search_pattern)

        if not found_files:
            print(f"  [Info] No '_VAA_RI.gpkg' files found for HUC {huc}")
            continue

        for gpkg_loc in found_files:
            print(f"  Processing {os.path.basename(gpkg_loc)}...")

            try:
                # 1. Read the file (assuming single layer, so we don't specify layer name)
                gdf = gpd.read_file(gpkg_loc)

                # Normalize columns to lowercase just to be safe
                gdf.columns = gdf.columns.str.lower()

                # 2. Check for Hydroseq columns
                reindexed_cols = {'hydroseq', 'uphydroseq', 'dnhydroseq'}

                if not reindexed_cols.issubset(gdf.columns):
                    print(f"    [Error] Missing hydroseq columns in {os.path.basename(gpkg_loc)}. Skipping.")
                    continue

                # 3. Re-Index Logic
                min_val = gdf['hydroseq'].min()

                # Safety Check: Only re-index if values are actually huge (e.g., > 1 billion)
                # This prevents corrupting the data if you accidentally run the script twice.
                if min_val < 1e9:
                    print(f"    [Skip] Values appear normal (Min: {min_val}). Skipping re-index.")
                    continue

                offset = min_val - 1

                print(f"    - Original Min Hydroseq: {min_val:.0f}")
                print(f"    - Applying Offset: {offset:.0f}")

                for col in reindexed_cols:
                    # Ensure we handle NaN values safely if present (though hydroseq shouldn't be NaN)
                    gdf[col] = gdf[col] - offset

                print(f"    - New Min Hydroseq: {gdf['hydroseq'].min():.0f}")

                # 4. Overwrite the file
                #    We use the same filename. Note: 'driver=GPKG' is standard.
                gdf.to_file(gpkg_loc, driver='GPKG', layer='NHDFlowline')
                print(f"    - Successfully overwritten {os.path.basename(gpkg_loc)}")

            except Exception as e:
                print(f"    [Error] Failed to process {gpkg_loc}: {e}")

# Example usage:
fix_reindexing_existing_files('/Volumes/KenDrive/LHD_Project/STRM')