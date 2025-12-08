import xarray as xr
import numpy as np
import os
import pandas as pd  # Added pandas
from dask.diagnostics import ProgressBar

# --- User-defined paths ---

# 1. SET THE PATH TO YOUR EXCEL FILE HERE
# This file must contain 'latitude' and 'longitude' columns
excel_file_path = "/lhd_processor/data/unique_site_coordinates.xlsx"

# 2. SET YOUR DESIRED OUTPUT PARQUET FILE PATH
output_path = '../lhd_processor/data/nwm_v3_daily_retrospective.parquet'

# --- Load coordinates from Excel ---
print(f"Reading coordinates from: {excel_file_path}")
try:
    if excel_file_path.endswith('.csv'):
        df_coords = pd.read_csv(excel_file_path)
    else:
        df_coords = pd.read_excel(excel_file_path)

    hf_coords = df_coords[['latitude', 'longitude']].values.tolist()
    print(f"Found {len(hf_coords)} coordinates to process.")
except FileNotFoundError:
    print(f"ERROR: File not found at {excel_file_path}. Please check the path.")
    exit()
except KeyError:
    print(f"ERROR: File must contain 'latitude' and 'longitude' columns.")
    exit()
except Exception as e:
    print(f"An error occurred reading the Excel/CSV file: {e}")
    exit()

# --- Load dataset ---
print("Opening NWM dataset...")
ds = xr.open_zarr(
    's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr',
    storage_options={'anon': True}
)

# --- Load coordinate arrays ---
print("Loading coordinate data...")
lats = ds['latitude'].values
lons = ds['longitude'].values
feature_ids_all = ds['feature_id'].values

# --- Find nearest feature for each coordinate ---
def find_nearest_feature(lat, lon, lats, lons, feature_ids_all):
    """Return the feature_id of the grid cell closest to (lat, lon)."""
    distance = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    idx = np.argmin(distance)
    return feature_ids_all[idx], lats[idx], lons[idx]

nearest_features = [find_nearest_feature(lat, lon, lats, lons, feature_ids_all) for lat, lon in hf_coords]

# Unpack results and get unique feature_ids
nearest_feature_ids = [f[0] for f in nearest_features]
unique_feature_ids = list(np.unique(nearest_feature_ids))

# Create a mapping for lats/lons for the unique IDs
id_to_lat = {fid: lat for fid, lat, lon in nearest_features}
id_to_lon = {fid: lon for fid, lat, lon in nearest_features}

matched_lats_unique = [id_to_lat[fid] for fid in unique_feature_ids]
matched_lons_unique = [id_to_lon[fid] for fid in unique_feature_ids]

print("\nNearest features found:")
print(f"  Processed {len(nearest_features)} coordinates.")
print(f"  Found {len(unique_feature_ids)} unique NWM feature_ids.")


# --- Select streamflow data ---
print(f"\nSelecting streamflow data for {len(unique_feature_ids)} unique feature(s)...")
streamflow_subset = ds['streamflow'].sel(feature_id=unique_feature_ids)

# --- Resample to daily average ---
print("Resampling from hourly to daily average...")
daily_streamflow = streamflow_subset.resample(time='1D').mean()

# --- Build final dataset (KEEPING 'feature_id') ---
final_ds = xr.Dataset({'streamflow': daily_streamflow})
final_ds = final_ds.assign_coords({
    'latitude': ('feature_id', matched_lats_unique),
    'longitude': ('feature_id', matched_lons_unique)
})

# --- Add metadata ---
# (Metadata attributes remain the same as previous step)
final_ds.attrs = {
    'title': 'Selected Coordinates Daily Streamflow (cms)',
    'source': 'NOAA National Water Model v3.0 Retrospective',
    'units': 'cubic meters per second (cms)',
    'temporal_resolution': 'daily average',
    'original_resolution': 'hourly',
    'created_by': 'K.T. Quintana',
    'selected_points': len(unique_feature_ids)
}
final_ds['streamflow'].attrs = {
    'long_name': 'Daily Average Streamflow',
    'units': 'cubic meters per second',
    'standard_name': 'water_volume_transport_in_river_channel'
}
final_ds['feature_id'].attrs = {
    'long_name': 'River Reach ID (NWM feature_id)',
    'description': 'Unique identifier for each river reach selected by nearest lat/lon'
}


# --- Save output to Parquet ---
print(f"\nSaving to: {output_path}")

# Convert the xarray Dataset to a pandas DataFrame
# This creates a DataFrame with a MultiIndex (time, feature_id)
print("Converting xarray Dataset to pandas DataFrame...")
final_df = final_ds.to_dataframe()

# Ensure output directory exists
output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

print(f"Writing to Parquet file... (this may take a moment)")
with ProgressBar():
    # Save to Parquet, preserving the multi-index
    final_df.to_parquet(output_path, index=True, engine='pyarrow')

print("\n✅ Successfully saved daily retrospective data to Parquet!")
print(f"Final DataFrame has {len(final_df)} rows.")
# Print time and points from the index
print(f"Time range: {final_df.index.get_level_values('time').min()} to {final_df.index.get_level_values('time').max()}")
print(f"Number of points: {final_df.index.get_level_values('feature_id').nunique()}")
print("Units: cms (cubic meters per second)")
print("Temporal resolution: Daily average")
