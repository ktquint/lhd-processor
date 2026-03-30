import pandas as pd
import numpy as np
import os
import sys
import xarray as xr
import geoglows

# Add the project root to the path so we can import lhd_processor modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lhd_processor.analysis.utils import get_prob_from_Q

def main():
    # Path to the database
    db_path = os.path.join(project_root, 'lhd_processor', 'data', 'lhd_database.xlsx')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    print(f"Reading database from {db_path}...")
    
    # Read the Incidents and Sites sheets
    try:
        incidents_df = pd.read_excel(db_path, sheet_name='Incidents')
        sites_df = pd.read_excel(db_path, sheet_name='Sites')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Initialize new columns for exceedance probabilities if they don't exist
    if 'prob_nwm' not in incidents_df.columns:
        incidents_df['prob_nwm'] = np.nan
    if 'prob_geo' not in incidents_df.columns:
        incidents_df['prob_geo'] = np.nan

    # NWM Zarr path
    nwm_zarr_path = os.path.join(project_root, 'lhd_processor', 'data', 'nwm_v3_daily_retrospective.zarr')
    nwm_ds = None
    if os.path.exists(nwm_zarr_path):
        try:
            nwm_ds = xr.open_zarr(nwm_zarr_path)
            print("Opened NWM Zarr dataset.")
        except Exception as e:
            print(f"Failed to open NWM Zarr: {e}")

    # Group incidents by site_id to process each site once
    grouped = incidents_df.groupby('site_id')
    
    updated_incidents = []
    
    # Debug counter
    processed_count = 0

    for site_id, group in grouped:
        print(f"Processing Site {site_id}...")
        
        # Get site info
        site_info = sites_df[sites_df['site_id'] == site_id]
        if site_info.empty:
            print(f"Site {site_id} not found in Sites sheet. Skipping.")
            updated_incidents.append(group)
            continue
            
        site_row = site_info.iloc[0]
        nwm_id = site_row.get('reach_id')
        geoglows_id = site_row.get('linkno')
        
        # --- NWM Flow Duration Curve ---
        fdc_nwm = None
        if nwm_ds is not None and pd.notna(nwm_id):
            try:
                # Fetch NWM flow series
                flow_series_nwm = nwm_ds.sel(feature_id=int(nwm_id))['streamflow'].to_series()
                
                if not flow_series_nwm.empty:
                    flow_cms = flow_series_nwm.dropna().values
                    # Sort Descending: High Flow -> Low Flow
                    sorted_flow = np.sort(flow_cms)[::-1]
                    n = len(sorted_flow)
                    # Exceedance: 0% -> 100%
                    exceedance = 100.0 * np.arange(1, n + 1) / (n + 1)
                    
                    fdc_nwm = pd.DataFrame({'Exceedance (%)': exceedance, 'Flow (cms)': sorted_flow})
                    
                    # Debug print for first site
                    if processed_count == 0:
                        print(f"--- Debug FDC NWM Site {site_id} ---")
                        print(fdc_nwm.head()) # Should be High Flow, Low Exceedance
                        print(fdc_nwm.tail()) # Should be Low Flow, High Exceedance
            except Exception as e:
                print(f"Error fetching NWM data for site {site_id}: {e}")

        # --- GEOGLOWS Flow Duration Curve ---
        fdc_geo = None
        if pd.notna(geoglows_id):
            try:
                # Fetch GEOGLOWS flow series
                print(f"Fetching GEOGLOWS data for site {site_id}...")
                df_geo = geoglows.data.retrospective(river_id=int(geoglows_id), resolution='daily', bias_corrected=True)
                flow_series_geo = df_geo.iloc[:, 0]
                
                if not flow_series_geo.empty:
                    flow_cms = flow_series_geo.dropna().values
                    # Sort Descending: High Flow -> Low Flow
                    sorted_flow = np.sort(flow_cms)[::-1]
                    n = len(sorted_flow)
                    # Exceedance: 0% -> 100%
                    exceedance = 100.0 * np.arange(1, n + 1) / (n + 1)
                    
                    fdc_geo = pd.DataFrame({'Exceedance (%)': exceedance, 'Flow (cms)': sorted_flow})
            except Exception as e:
                print(f"Error fetching GEOGLOWS data for site {site_id}: {e}")

        # Calculate probabilities for each incident in this group
        for idx, row in group.iterrows():
            # NWM
            flow_nwm = row.get('flow_nwm')
            if pd.notna(flow_nwm) and fdc_nwm is not None:
                # Ensure flow is float
                try:
                    flow_val = float(flow_nwm)
                    prob_nwm = get_prob_from_Q(flow_val, fdc_nwm)
                    group.at[idx, 'prob_nwm'] = prob_nwm
                    if processed_count == 0:
                        print(f"  Incident NWM Flow: {flow_val} -> Prob: {prob_nwm}%")
                except (ValueError, TypeError) as e:
                    print(f"  Could not calculate NWM probability for incident {idx}: {e}")

            # GEOGLOWS
            flow_geo = row.get('flow_geo')
            if pd.notna(flow_geo) and fdc_geo is not None:
                try:
                    flow_val = float(flow_geo)
                    prob_geo = get_prob_from_Q(flow_val, fdc_geo)
                    group.at[idx, 'prob_geo'] = prob_geo
                except (ValueError, TypeError) as e:
                    print(f"  Could not calculate GEOGLOWS probability for incident {idx}: {e}")
        
        updated_incidents.append(group)
        processed_count += 1

    # Concatenate all groups back together
    if updated_incidents:
        final_incidents_df = pd.concat(updated_incidents)
        
        print("Updating Incidents sheet in database...")
        
        try:
            with pd.ExcelWriter(db_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                final_incidents_df.to_excel(writer, sheet_name='Incidents', index=False)
            print("Successfully updated Incidents sheet.")
        except Exception as e:
            print(f"Error writing to Excel file: {e}")
    else:
        print("No incidents processed.")

if __name__ == "__main__":
    main()
