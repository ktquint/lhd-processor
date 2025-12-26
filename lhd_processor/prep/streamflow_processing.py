import os
import s3fs
import shutil
import pandas as pd
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

def create_reanalysis_file(comid_list, output_folder, source, package_root, comid_date_map=None):
    """
    Generates a reanalysis CSV with flow statistics for the given COMIDs.
    """
    if not comid_list:
        print("No COMIDs provided for reanalysis.")
        return
    
    reanalysis_path = os.path.join(output_folder, "reanalysis.csv")
    if os.path.exists(reanalysis_path):
        print(f"Reanalysis file already exists at {reanalysis_path}. Skipping generation.")
        return

    stats_list = []

    if source == 'National Water Model':
        zarr_path = os.path.join(package_root, 'data', 'nwm_v3_daily_retrospective.zarr')
        if not os.path.exists(zarr_path):
            print(f"Zarr store not found at {zarr_path}")
            return

        print(f"Reading NWM data for {len(comid_list)} COMIDs from Zarr...")
        try:
            # Ensure comid_list contains integers
            comid_list = [int(x) for x in comid_list]
            
            try:
                ds = xr.open_zarr(zarr_path, consolidated=True)
            except (KeyError, FileNotFoundError):
                print("Warning: Consolidated metadata not found. Retrying with consolidated=False...")
                ds = xr.open_zarr(zarr_path, consolidated=False)
            
            # Filter IDs that actually exist in the dataset
            available_ids = ds['feature_id'].values
            valid_ids = [fid for fid in comid_list if fid in available_ids]
            
            if not valid_ids:
                print("No valid IDs found in Zarr dataset.")
                print(f"Requested (first 5): {comid_list[:5]}")
                print(f"Available in Zarr (first 5): {available_ids[:5] if len(available_ids) > 0 else 'None'}")
                return

            ds_subset = ds.sel(feature_id=valid_ids)
            # Convert to DataFrame
            df = ds_subset.to_dataframe().reset_index()
            
        except Exception as e:
            print(f"Error reading Zarr: {e}")
            return

        if df.empty:
            print("No data found in Zarr for the provided COMIDs.")
            return

        # Ensure 'time' column is available
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['year'] = df['time'].dt.year
        else:
            print("Time column missing in NWM data.")
            return

        # Process each COMID
        unique_ids = df['feature_id'].unique()
        for comid in unique_ids:
            site_df = df[df['feature_id'] == comid]
            if site_df.empty: continue

            # Flow stats
            q_median = site_df['streamflow'].median()
            q_max = site_df['streamflow'].max()

            # Known Baseflow Logic
            known_baseflow = q_median
            if comid_date_map:
                target_date = comid_date_map.get(int(comid))
                if target_date:
                     day_match = site_df[site_df['time'].dt.strftime('%Y-%m-%d') == target_date]
                     if not day_match.empty:
                         known_baseflow = day_match['streamflow'].iloc[0]

            # Annual Maxima for Return Periods
            annual_max = site_df.groupby('year')['streamflow'].max()
            
            if len(annual_max) < 2:
                continue

            # Weibull Plotting Position
            sorted_max = np.sort(annual_max.values)[::-1]
            n = len(sorted_max)
            ranks = np.arange(1, n + 1)
            exceedance_probs = ranks / (n + 1)
            return_periods = 1 / exceedance_probs

            # Interpolation
            # x (RP) must be increasing for np.interp
            x_interp = return_periods[::-1]
            y_interp = sorted_max[::-1]

            target_rps = [2, 5, 10, 25, 50, 100]
            rp_flows = {}

            for rp in target_rps:
                if rp > x_interp[-1]:
                    val = y_interp[-1] # Cap at max observed
                elif rp < x_interp[0]:
                    val = y_interp[0]
                else:
                    val = np.interp(rp, x_interp, y_interp)
                rp_flows[f'rp{rp}'] = val

            row = {
                'comid': int(comid),
                'known_baseflow': known_baseflow,
                'qout_median': q_median,
                'qout_max': q_max,
                'qout_max_premium': q_max * 1.5,
                'rp100_premium': rp_flows.get('rp100', 0) * 1.5
            }
            row.update(rp_flows)
            stats_list.append(row)

    elif source == 'GEOGLOWS':
        # Placeholder for GEOGLOWS logic if needed
        print("Reanalysis for GEOGLOWS not yet implemented.")
        pass

    if stats_list:
        out_df = pd.DataFrame(stats_list)
        # Reorder columns
        cols = ['comid', 'known_baseflow', 'qout_median', 'qout_max', 'rp2', 'rp5', 'rp10', 'rp25', 'rp50', 'rp100', 'qout_max_premium', 'rp100_premium']
        # Add any missing columns just in case
        for c in cols:
            if c not in out_df.columns: out_df[c] = 0.0
        
        out_df = out_df[cols]
        out_df.to_csv(reanalysis_path, index=False)
        print(f"Reanalysis file saved to {reanalysis_path}")
    else:
        print("No stats calculated.")



def condense_zarr(comid_list, output_zarr):
    """
    Reads NWM v3.0 Retrospective data from S3 for a specific list of COMIDs,
    condenses it to daily averages, and saves it as a local Zarr store.
    """
    # --- Setup S3 Connection ---
    print("Connecting to NOAA NWM S3 Zarr store...")
    fs = s3fs.S3FileSystem(anon=True)
    s3_path = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr'

    try:
        # Open remote dataset
        ds = xr.open_zarr(s3fs.S3Map(s3_path, s3=fs), consolidated=True)
    except Exception as e:
        print(f"Error opening S3 Zarr: {e}")
        return

    # --- Filter and Process ---
    # Convert list to integers to match feature_id type
    unique_feature_ids = list(set(int(c) for c in comid_list))

    print(f"Subsetting for {len(unique_feature_ids)} COMIDs...")
    # Subset by feature_id (COMID)
    ds_subset = ds.sel(feature_id=unique_feature_ids)

    print("Resampling to daily averages (this may take a moment)...")
    with ProgressBar():
        # Resample from hourly to daily mean as seen in zarr2parquet logic
        daily_streamflow = ds_subset['streamflow'].resample(time='1D').mean()

    # --- Build Final Dataset ---
    final_ds = xr.Dataset({'streamflow': daily_streamflow})

    # --- Add Metadata (Merged from zarr2parquet.py) ---
    final_ds.attrs = {
        'title': 'Selected COMIDs Daily Streamflow (cms)',
        'source': 'NOAA National Water Model v3.0 Retrospective',
        'units': 'cubic meters per second (cms)',
        'temporal_resolution': 'daily average',
        'original_resolution': 'hourly',
        'created_by': 'LHD Processor',
        'selected_points': len(unique_feature_ids)
    }

    final_ds['streamflow'].attrs = {
        'long_name': 'Daily Average Streamflow',
        'units': 'cubic meters per second',
        'standard_name': 'water_volume_transport_in_river_channel'
    }

    final_ds['feature_id'].attrs = {
        'long_name': 'River Reach ID (NWM feature_id)',
        'description': 'Unique identifier for each river reach'
    }

    # --- Prepare for Writing ---
    # Chunking strategy: chunk by COMID, keep time dimension whole for time-series analysis
    final_ds = final_ds.chunk({'time': -1, 'feature_id': 100})

    # Clear encoding chunks to avoid the "mismatched Dask/Zarr chunks" error
    for var in final_ds.variables:
        if 'chunks' in final_ds[var].encoding:
            del final_ds[var].encoding['chunks']

    # --- Save to Zarr ---
    print(f"Saving Zarr store to: {output_zarr}")

    if os.path.exists(output_zarr):
        print("Existing store found. Removing...")
        shutil.rmtree(output_zarr)

    with ProgressBar():
        # Save directly to Zarr without converting to DataFrame/Parquet
        final_ds.to_zarr(output_zarr, consolidated=True)

    print(f"\n✅ Successfully saved daily retrospective Zarr to {output_zarr}")