import os
import s3fs
import pandas as pd
import numpy as np
import xarray as xr
import geoglows as geo
from dask.diagnostics import ProgressBar

def create_reanalysis_file(comid_list, source, package_root, comid_date_map=None, db_manager=None):
    """
    Generates a reanalysis CSV with flow statistics for the given COMIDs.
    Optionally updates the database with the found baseflow values.
    """
    if not comid_list:
        print("No COMIDs provided for reanalysis.")
        return
    
    # Dynamic filename based on source
    filename = "nwm_reanalysis.csv" if source == 'National Water Model' else "geoglows_reanalysis.csv"
    reanalysis_path = os.path.join(package_root, 'data', filename)
    
    # If we are updating with new dates, we might want to overwrite or merge
    # For now, let's assume if comid_date_map is provided, we are doing a second pass
    if os.path.exists(reanalysis_path) and not comid_date_map:
        print(f"Reanalysis file already exists at {reanalysis_path}. Skipping generation.")
        
        # If db_manager is provided, try to update DB from existing file
        if db_manager:
            print("Reading existing reanalysis file to update database...")
            try:
                existing_df = pd.read_csv(reanalysis_path)
                _update_db_from_stats(existing_df.to_dict('records'), db_manager, source)
            except Exception as e:
                print(f"Failed to read existing reanalysis: {e}")
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

            # Log missing IDs
            missing_ids = set(comid_list) - set(valid_ids)
            if missing_ids:
                print(f"Warning: {len(missing_ids)} requested COMIDs were not found in the Zarr index.")
                print(f"Sample missing: {list(missing_ids)[:5]}")
            
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
            # Filter the dataframe for this specific COMID
            site_df = df[df['feature_id'] == comid].copy()
            if site_df.empty: continue

            # Ensure 'streamflow' is numeric and not interpreted as a date
            # This prevents the error where Pandas thinks it's working with datetime objects
            site_df['streamflow'] = pd.to_numeric(site_df['streamflow'], errors='coerce')

            # Drop any NaNs that might have resulted from coercion
            valid_flow = site_df.dropna(subset=['streamflow'])

            if valid_flow.empty:
                print(f"Warning: COMID {comid} exists in Zarr but has no valid streamflow data (all NaNs).")
                continue

            # Now perform numeric calculations
            q_median = valid_flow['streamflow'].median()
            q_max = valid_flow['streamflow'].max()

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
        print(f"Fetching GEOGLOWS data for {len(comid_list)} reaches...")

        for comid in comid_list:
            try:
                # 1. Fetch Retrospective Data (Daily)
                df = geo.data.retrospective(river_id=int(comid))

                if df.empty:
                    print(f"Warning: No GEOGLOWS data for ID {comid}")
                    continue

                df.index = pd.to_datetime(df.index)
                flow_col = df.columns[0]

                # 2. Calculate Basic Stats
                q_median = df[flow_col].median()
                q_max = df[flow_col].max()

                # 3. Known Baseflow Logic
                known_baseflow = q_median
                if comid_date_map:
                    target_date = comid_date_map.get(int(comid))
                    if target_date:
                        try:
                            val = df.loc[target_date]
                            if isinstance(val, pd.Series):
                                known_baseflow = val.mean()
                            else:
                                known_baseflow = val[flow_col]
                        except KeyError:
                            pass

                # 4. Return Periods using GEOGLOWS API
                # Update: Handling index-based return periods (2, 5, 10, 25, 50, 100)
                rp_df = geo.data.return_periods(river_id=int(comid))

                # Ensure the return_period is the index (in case it comes back as a column)
                if 'return_period' in rp_df.columns:
                    rp_df = rp_df.set_index('return_period')

                # The first column contains the flow values (named after the ID)
                flow_values = rp_df.iloc[:, 0]

                rp_flows = {}
                target_rps = [2, 5, 10, 25, 50, 100]

                for rp in target_rps:
                    try:
                        # Look up by the index value (e.g., rp_df.loc[2])
                        rp_flows[f'rp{rp}'] = float(flow_values.loc[rp])
                    except KeyError:
                        rp_flows[f'rp{rp}'] = 0.0

                # 5. Build Row
                row = {
                    'comid': int(comid),
                    'known_baseflow': known_baseflow,
                    'qout_median': q_median,
                    'qout_max': q_max,
                    # Premium logic: 1.5 times the base value
                    'qout_max_premium': q_max * 1.5,
                    'rp100_premium': rp_flows.get('rp100', 0) * 1.5
                }
                row.update(rp_flows)
                stats_list.append(row)

            except Exception as e:
                print(f"Error processing GEOGLOWS ID {comid}: {e}")

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

        if db_manager:
            _update_db_from_stats(stats_list, db_manager, source)
    else:
        print("No stats calculated.")


def _update_db_from_stats(stats_list, db_manager, source):
    """Helper to update database from stats list"""
    print(f"Updating database with baseflow values from {source}...")
    col_name = 'baseflow_nwm' if source == 'National Water Model' else 'baseflow_geo'
    id_col = 'reach_id' if source == 'National Water Model' else 'linkno'
    
    # Ensure we have the sites loaded
    sites_df = db_manager.sites
    
    count = 0
    for row in stats_list:
        try:
            comid = int(row['comid'])
            baseflow = float(row['known_baseflow'])
            
            # Filter where id_col is not null/nan
            valid_sites = sites_df[pd.notna(sites_df[id_col])].copy()
            # Convert to int for comparison (handling floats like 123.0)
            valid_sites['temp_id'] = valid_sites[id_col].astype(float).astype(int)
            matches = valid_sites[valid_sites['temp_id'] == comid]
            
            for _, site in matches.iterrows():
                db_manager.update_site_data(site['site_id'], {col_name: baseflow})
                count += 1
        except Exception as e:
            print(f"Error updating site for COMID {row.get('comid')}: {e}")
    
    db_manager.save()
    print(f"Database updated: {count} sites modified.")


def condense_zarr(comid_list, output_zarr):
    """
    Checks the local Zarr store for existing COMIDs and only downloads/appends
    missing ones from the S3 NWM v3.0 Retrospective data.
    """
    # 1. Identify which COMIDs we actually need to fetch
    existing_ids = []
    if os.path.exists(output_zarr):
        try:
            ds_existing = xr.open_zarr(output_zarr, consolidated=True)
            existing_ids = ds_existing.feature_id.values.tolist()
            print(f"Local Zarr found with {len(existing_ids)} existing COMIDs.")
        except Exception as e:
            print(f"Warning: Could not read existing Zarr, starting fresh: {e}")

    # Filter for IDs not already in the store
    unique_requested = list(set(int(c) for c in comid_list))
    new_ids = [fid for fid in unique_requested if fid not in existing_ids]

    if not new_ids:
        print("✅ All requested COMIDs already exist in the local Zarr store.")
        return

    print(f"Adding {len(new_ids)} new COMIDs to the local store...")

    # 2. Setup S3 Connection for new IDs
    print("Connecting to NOAA NWM S3 Zarr store...")
    fs = s3fs.S3FileSystem(anon=True)
    s3_path = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr'

    try:
        ds_remote = xr.open_zarr(s3fs.S3Map(s3_path, s3=fs), consolidated=True)
    except Exception as e:
        print(f"Error opening S3 Zarr: {e}")
        return

    # 3. Subset and Process only the new IDs
    ds_subset = ds_remote.sel(feature_id=new_ids)
    print("Resampling new data to daily averages...")
    with ProgressBar():
        daily_streamflow = ds_subset['streamflow'].resample(time='1D').mean()

    # 4. Build Dataset with same structure and metadata
    new_ds = xr.Dataset({'streamflow': daily_streamflow})
    new_ds.attrs = {
        'title': 'Selected COMIDs Daily Streamflow (cms)',
        'source': 'NOAA National Water Model v3.0 Retrospective',
        'units': 'cubic meters per second (cms)',
        'temporal_resolution': 'daily average',
        'created_by': 'LHD Processor (Auto-Updater)'
    }

    # Chunking: must match the existing store's chunking for successful appending
    new_ds = new_ds.chunk({'time': -1, 'feature_id': 100})

    # Clear encoding to prevent "mismatched chunks" error
    for var in new_ds.variables:
        if 'chunks' in new_ds[var].encoding:
            del new_ds[var].encoding['chunks']

    # 5. Save/Append to Zarr
    if os.path.exists(output_zarr):
        print(f"Appending new data to {output_zarr}...")
        with ProgressBar():
            # 'a' mode appends to the specified dimension
            new_ds.to_zarr(output_zarr, mode='a', append_dim='feature_id', consolidated=True)
    else:
        print(f"Creating new Zarr store at {output_zarr}...")
        with ProgressBar():
            new_ds.to_zarr(output_zarr, mode='w', consolidated=True)

    print(f"\n✅ Successfully updated Zarr store. Total COMIDs: {len(existing_ids) + len(new_ids)}")
