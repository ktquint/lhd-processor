import os
import glob
import geopandas as gpd
import pandas as pd


def batch_fix_reindexing(directory):
    """
    Loops through all .gpkg files in a directory.
    Skips GEOGLOWS files (starting with 'streams_').
    Checks if hydroseq/up/dn columns are massive (> 1 billion).
    If so, calculates the offset and reindexes them.
    """
    files = glob.glob(os.path.join(directory, "*.gpkg"))
    print(f"Found {len(files)} GeoPackages in {directory}...")

    for gpkg_path in files:
        filename = os.path.basename(gpkg_path)

        # Skip GEOGLOWS files
        if filename.startswith("streams_"):
            print(f"[SKIP] {filename} (GEOGLOWS file)")
            continue

        try:
            # 1. Read the file
            gdf = gpd.read_file(gpkg_path, layer='NHDFlowline')

            # Normalize columns to lowercase
            gdf.columns = gdf.columns.str.lower()

            target_cols = ['hydroseq', 'uphydroseq', 'dnhydroseq']
            existing_cols = [c for c in target_cols if c in gdf.columns]

            if 'hydroseq' not in existing_cols:
                print(f"[SKIP] {filename} (No hydroseq column)")
                continue

            # 3. Analyze values
            min_hydroseq = gdf['hydroseq'].min()

            # Threshold: 1 billion
            if min_hydroseq > 1e9:
                print(f"[FIXING] {filename}")
                print(f"  - Current Min Hydroseq: {min_hydroseq:,.0f}")

                offset = min_hydroseq - 1
                print(f"  - Applying Offset:      -{offset:,.0f}")

                for col in existing_cols:
                    gdf[col] = gdf[col] - offset

                # 4. Save/Overwrite
                gdf.to_file(gpkg_path, layer='NHDFlowline', driver='GPKG')
                print(f"  - Done. Saved {filename}")

            else:
                # Edge Case: Partial Reindex Check
                # max_up = gdf['uphydroseq'].max() if 'uphydroseq' in gdf.columns else 0
                max_dn = gdf['dnhydroseq'].max() if 'dnhydroseq' in gdf.columns else 0

                if max_dn > 1e9:
                    print(f"[WARNING] {filename}")
                    print("  - Hydroseq is small, but Dn Hydroseq are still massive!")
                    print("  - Recommend re-downloading this file.")
                else:
                    print(f"[OK] {filename} (Already reindexed)")

        except Exception as e:
            print(f"[ERROR] Could not process {filename}: {e}")


# --- USAGE ---
my_flowline_dir = '/Volumes/KenDrive/LHD_Project/STRM'
batch_fix_reindexing(my_flowline_dir)