import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def main():
    # ---------------- Configuration ----------------
    # Path to your CSV file
    csv_path = '../lhd_processor/data/nhd_geoglows_flowline_data_final.csv'

    # Directory containing your .gpkg files
    # UPDATE THIS PATH to the folder containing your HUC/VPU .gpkg files
    gpkg_directory = '/Volumes/KenDrive/LHD_Project/STRM'

    # Output filename
    output_gpkg = '../lhd_processor/data/combined_output.gpkg'

    # Column names in the GPKG files to filter by
    # (Adjust these if your GPKG column names differ, e.g., 'HydroSeq', 'COMID')
    nhd_id_col = 'hydroseq'  # Column in HUC .gpkg
    geoglows_id_col = 'LINKNO'  # Column in VPU .gpkg
    # -----------------------------------------------

    print("Reading CSV...")
    # Read CSV, ensuring codes are read as strings to preserve leading zeros (e.g. '0508')
    df = pd.read_csv(csv_path, dtype={'HUC4': str, 'VPU_Code': str})

    # Filter out rows with missing IDs and ensure numeric types for IDs
    df_clean = df.dropna(subset=['NHD_Hydroseq_ID', 'GEOGLOWS_Link_No', 'HUC4', 'VPU_Code']).copy()

    # Convert IDs to integers for accurate matching
    df_clean['NHD_Hydroseq_ID'] = df_clean['NHD_Hydroseq_ID'].astype(float).astype(int)
    df_clean['GEOGLOWS_Link_No'] = df_clean['GEOGLOWS_Link_No'].astype(float).astype(int)

    # Lists to store collected features
    nhd_features = []
    geoglows_features = []

    # Get list of all files in directory for searching
    if not os.path.exists(gpkg_directory):
        print(f"Error: Directory not found: {gpkg_directory}")
        return

    all_files = [f for f in os.listdir(gpkg_directory) if f.endswith('.gpkg')]

    # Group by HUC and VPU to minimize file opening/closing
    grouped = df_clean.groupby(['HUC4', 'VPU_Code'])

    print(f"Processing {len(grouped)} unique HUC/VPU combinations...")

    for (huc, vpu), group in grouped:
        # Identify the filenames
        # Logic: find first file that contains the code string
        huc_filename = next((f for f in all_files if huc in f), None)
        vpu_filename = next((f for f in all_files if vpu in f), None)

        target_hydroseqs = group['NHD_Hydroseq_ID'].tolist()
        target_linknos = group['GEOGLOWS_Link_No'].tolist()

        # --- Process NHD (HUC file) ---
        if huc_filename:
            path = os.path.join(gpkg_directory, huc_filename)
            try:
                gdf = gpd.read_file(path)

                # Case-insensitive column search
                cols_upper = {c.upper(): c for c in gdf.columns}
                target_col = cols_upper.get(nhd_id_col.upper())

                if target_col:
                    # Filter for matching features
                    subset = gdf[gdf[target_col].isin(target_hydroseqs)]
                    if not subset.empty:
                        nhd_features.append(subset)
                else:
                    print(f"Warning: Column '{nhd_id_col}' not found in {huc_filename}")
            except Exception as e:
                print(f"Error reading {huc_filename}: {e}")
        else:
            print(f"Warning: No file found for HUC {huc}")

        # --- Process GEOGLOWS (VPU file) ---
        if vpu_filename:
            path = os.path.join(gpkg_directory, vpu_filename)
            try:
                gdf = gpd.read_file(path)

                # Case-insensitive column search
                cols_upper = {c.upper(): c for c in gdf.columns}
                target_col = cols_upper.get(geoglows_id_col.upper())

                if target_col:
                    # Filter for matching features
                    subset = gdf[gdf[target_col].isin(target_linknos)]
                    if not subset.empty:
                        geoglows_features.append(subset)
                else:
                    print(f"Warning: Column '{geoglows_id_col}' not found in {vpu_filename}")
            except Exception as e:
                print(f"Error reading {vpu_filename}: {e}")
        else:
            print(f"Warning: No file found for VPU {vpu}")

    # ---------------- Compile and Save ----------------
    print("Compiling layers...")

    # 1. NHD Layer
    if nhd_features:
        final_nhd = pd.concat(nhd_features, ignore_index=True)
        # Ensure it's a GeoDataFrame (handling potential CRS loss in concat)
        if isinstance(final_nhd, pd.DataFrame):
            final_nhd = gpd.GeoDataFrame(final_nhd, geometry=final_nhd.geometry, crs=nhd_features[0].crs)
    else:
        final_nhd = gpd.GeoDataFrame()
        print("Warning: No NHD features found.")

    # 2. GEOGLOWS Layer
    if geoglows_features:
        final_geoglows = pd.concat(geoglows_features, ignore_index=True)
        if isinstance(final_geoglows, pd.DataFrame):
            final_geoglows = gpd.GeoDataFrame(final_geoglows, geometry=final_geoglows.geometry,
                                              crs=geoglows_features[0].crs)
    else:
        final_geoglows = gpd.GeoDataFrame()
        print("Warning: No GEOGLOWS features found.")

    # 3. LHD Layer (Points from CSV lat/lon)
    # Create geometry from longitude and latitude
    geometry = [Point(xy) for xy in zip(df_clean.longitude, df_clean.latitude)]
    # Create GeoDataFrame (Standard WGS84 CRS: EPSG:4326)
    final_lhd = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")

    # Export to GPKG
    print(f"Saving to {output_gpkg}...")

    # We use 'w' mode (write) for the first layer, but geopandas handles layers gracefully.
    # Note: to_file writes a new file. To add layers, we rely on the driver's capability or separate calls.
    # For GPKG, simply writing to the same filename with different 'layer' arguments works in recent geopandas versions.

    try:
        final_nhd.to_file(output_gpkg, layer='NHD', driver='GPKG')
        final_geoglows.to_file(output_gpkg, layer='GEOGLOWS', driver='GPKG')
        final_lhd.to_file(output_gpkg, layer='LHD', driver='GPKG')
        print("Success! Output saved.")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
