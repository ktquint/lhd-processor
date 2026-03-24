import os
import pandas as pd

def process_crosswalks():
    # Resolve absolute path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'lhd_processor', 'data')

    # Input file paths
    nwm_reanalysis_path = os.path.join(data_dir, 'nwm_reanalysis.csv')
    geoglows_reanalysis_path = os.path.join(data_dir, 'geoglows_reanalysis.csv')
    
    nhd_to_geoglows_cw_path = os.path.join(data_dir, 'nhd_to_geoglows_crosswalk.csv')
    geoglows_to_nhd_cw_path = os.path.join(data_dir, 'geoglows_to_nhd_crosswalk.csv')

    # Output file paths
    nwm_to_geoglows_out_path = os.path.join(data_dir, 'nwm_to_geoglows_reanalysis.csv')
    geoglows_to_nhd_out_path = os.path.join(data_dir, 'geoglows_to_nhd_reanalysis.csv')

    # 1. NWM to GEOGloWS
    print("Processing NWM to GEOGloWS...")
    nwm_df = pd.read_csv(nwm_reanalysis_path)
    nhd_to_geo_cw = pd.read_csv(nhd_to_geoglows_cw_path)

    # Merge NWM data (using NWM/NHD comid) with crosswalk to get GEOGloWS linkno
    nwm_merged = pd.merge(nwm_df, nhd_to_geo_cw, left_on='comid', right_on='nhdplusid', how='inner')
    
    # Drop original NHD comid and crosswalk column, rename linkno to comid
    nwm_merged = nwm_merged.drop(columns=['comid', 'nhdplusid'])
    nwm_merged = nwm_merged.rename(columns={'linkno': 'comid'})
    
    # Reorder so comid is first
    cols = ['comid'] + [c for c in nwm_merged.columns if c != 'comid']
    nwm_merged = nwm_merged[cols]
    
    # Save output
    nwm_merged.to_csv(nwm_to_geoglows_out_path, index=False)
    print(f"Saved {nwm_to_geoglows_out_path}")


    # 2. GEOGloWS to NHD
    print("Processing GEOGloWS to NHD...")
    geoglows_df = pd.read_csv(geoglows_reanalysis_path)
    geo_to_nhd_cw = pd.read_csv(geoglows_to_nhd_cw_path)

    # Merge GEOGloWS data (using GEOGloWS linkno) with crosswalk to get NHD nhdplusid
    geo_merged = pd.merge(geoglows_df, geo_to_nhd_cw, left_on='comid', right_on='linkno', how='inner')
    
    # Drop original GEOGloWS linkno and crosswalk column, rename nhdplusid to comid
    geo_merged = geo_merged.drop(columns=['comid', 'linkno'])
    geo_merged = geo_merged.rename(columns={'nhdplusid': 'comid'})
    
    # Reorder so comid is first
    cols = ['comid'] + [c for c in geo_merged.columns if c != 'comid']
    geo_merged = geo_merged[cols]
    
    # Save output
    geo_merged.to_csv(geoglows_to_nhd_out_path, index=False)
    print(f"Saved {geoglows_to_nhd_out_path}")

if __name__ == '__main__':
    process_crosswalks()
