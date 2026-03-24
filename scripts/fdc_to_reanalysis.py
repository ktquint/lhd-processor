import os
import pandas as pd
import numpy as np
import xarray as xr
import geoglows as geo

def get_exceedance_flows(series):
    """
    Calculate flow values for exceedance probabilities from 10% to 100% in 10% increments.
    An exceedance probability of P% means the value is exceeded P% of the time.
    This corresponds to the (100 - P)th percentile.
    """
    ep_flows = {}
    # 10 to 100 inclusive
    for ep in range(10, 101, 10):
        # e.g., ep=10 -> 90th percentile
        percentile = (100 - ep) / 100.0
        # Calculate quantile
        val = series.quantile(percentile)
        ep_flows[f'q_ep_{ep}'] = val
    return ep_flows

def process_nwm(nwm_csv, zarr_path):
    print(f"Processing NWM data from {nwm_csv}...")
    try:
        df = pd.read_csv(nwm_csv)
    except FileNotFoundError:
        print("NWM reanalysis CSV not found.")
        return

    # Check if we can open the Zarr file
    try:
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except:
            print("Consolidated metadata not found, trying without...")
            ds = xr.open_zarr(zarr_path, consolidated=False)
    except Exception as e:
        print(f"Error opening Zarr file: {e}")
        return

    # Filter for COMIDs present in the CSV
    comids = df['comid'].unique()
    available_ids = ds['feature_id'].values
    valid_ids = [fid for fid in comids if fid in available_ids]

    if not valid_ids:
        print("No valid NWM COMIDs found in Zarr dataset.")
        return

    print(f"Loading NWM streamflow for {len(valid_ids)} COMIDs...")
    try:
        # Load all data at once into memory
        ds_subset = ds.sel(feature_id=valid_ids)
        flow_df = ds_subset.to_dataframe().reset_index()
    except Exception as e:
        print(f"Error reading Zarr dataset: {e}")
        return

    ep_results = []
    # Iterate through each COMID to calculate exceedance flows
    for comid in valid_ids:
        site_flow = flow_df[flow_df['feature_id'] == comid]['streamflow']
        site_flow = pd.to_numeric(site_flow, errors='coerce').dropna()
        
        if site_flow.empty:
            continue
            
        eps = get_exceedance_flows(site_flow)
        eps['comid'] = comid
        ep_results.append(eps)

    if ep_results:
        ep_df = pd.DataFrame(ep_results)
        
        # Merge the new exceedance columns into the original dataframe
        # First, drop any existing q_ep columns to avoid duplicates
        existing_ep_cols = [c for c in ep_df.columns if c != 'comid']
        df = df.drop(columns=[c for c in existing_ep_cols if c in df.columns], errors='ignore')
            
        # Merge
        df = df.merge(ep_df, on='comid', how='left')
        
        # Save back to CSV
        df.to_csv(nwm_csv, index=False)
        print(f"Updated {nwm_csv} with exceedance flows.")
    else:
        print("No flow data found for NWM COMIDs.")

def process_geoglows(geoglows_csv):
    print(f"Processing GEOGLOWS data from {geoglows_csv}...")
    try:
        df = pd.read_csv(geoglows_csv)
    except FileNotFoundError:
        print("GEOGLOWS reanalysis CSV not found.")
        return

    comids = df['comid'].unique()
    
    ep_results = []
    for comid in comids:
        print(f"Fetching GEOGLOWS retrospective data for {comid}...")
        try:
            # Fetch retrospective data from GEOGLOWS API
            site_df = geo.data.retrospective(river_id=int(comid))
            if site_df.empty:
                print(f"No data for {comid}")
                continue
                
            flow_col = site_df.columns[0]
            site_flow = pd.to_numeric(site_df[flow_col], errors='coerce').dropna()
            
            if site_flow.empty:
                continue
                
            eps = get_exceedance_flows(site_flow)
            eps['comid'] = comid
            ep_results.append(eps)
        except Exception as e:
            print(f"Error processing GEOGLOWS ID {comid}: {e}")

    if ep_results:
        ep_df = pd.DataFrame(ep_results)
        
        # Merge new exceedance columns
        existing_ep_cols = [c for c in ep_df.columns if c != 'comid']
        df = df.drop(columns=[c for c in existing_ep_cols if c in df.columns], errors='ignore')
            
        df = df.merge(ep_df, on='comid', how='left')
        
        # Save back to CSV
        df.to_csv(geoglows_csv, index=False)
        print(f"Updated {geoglows_csv} with exceedance flows.")
    else:
        print("No flow data found for GEOGLOWS COMIDs.")

if __name__ == "__main__":
    # Determine paths relative to this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'lhd_processor', 'data')
    
    nwm_csv_path = os.path.join(data_dir, 'nwm_reanalysis.csv')
    geoglows_csv_path = os.path.join(data_dir, 'geoglows_reanalysis.csv')
    zarr_path = os.path.join(data_dir, 'nwm_v3_daily_retrospective.zarr')
    
    process_nwm(nwm_csv_path, zarr_path)
    process_geoglows(geoglows_csv_path)
