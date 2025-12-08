import os
import json
import pandas as pd
from ..data_manager import DatabaseManager


def rathcelon_input(db_path, json_output_path, baseflow_method, nwm_parquet):
    """
    Reads the Excel database, creates a temporary CSV bridge for RathCelon,
    and generates the JSON input file with specific conditional logic for
    Streamflow and Flowline sources.
    """
    print("Generating RathCelon input files...")

    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    # 1. Load Data
    db = DatabaseManager(db_path)

    # 2. Filter for ready sites (Must have DEM and Flowline)
    # Using .copy() to avoid SettingWithCopy warnings later
    ready_sites = db.sites.dropna(subset=['dem_dir', 'flowline_path']).copy()

    if ready_sites.empty:
        print("No sites are ready for processing (missing DEM or Flowline).")
        return

    # 3. Create the "Bridge" CSV
    output_dir = os.path.dirname(json_output_path)
    bridge_csv_name = "sites_for_rathcelon.csv"
    bridge_csv_path = os.path.join(output_dir, bridge_csv_name)

    try:
        ready_sites.to_csv(bridge_csv_path, index=False)
        print(f"Created bridge CSV: {bridge_csv_path}")
    except Exception as e:
        print(f"Failed to create bridge CSV: {e}")
        return

    dams_list = []

    # 4. Build the Dictionary for each Dam
    for _, site in ready_sites.iterrows():
        site_id = site['site_id']

        # Determine output directory
        dam_output_dir = site['output_dir']
        if pd.isna(dam_output_dir):
            dam_output_dir = os.path.join(output_dir, "Results")

        # --- BASEFLOW & BANKS LOGIC ---
        # If we calculate via Banks, use_banks=True and we ignore the explicit baseflow value.
        # If we calculate via WSE, use_banks=False and we need the baseflow value.

        baseflow_val = site.get('dem_baseflow')
        if pd.isna(baseflow_val):
            baseflow_val = 0.0
        else:
            baseflow_val = float(baseflow_val)

        if baseflow_method in ["WSE and LiDAR Date", "WSE and Median Daily Flow"]:
            use_banks = False
            # Keep baseflow_val as is
        else:
            # e.g., "2-yr Flow and Bank Estimation"
            use_banks = True
            baseflow_val = None  # JSON will show 'null', RathCelon must handle this

        # --- STREAMFLOW SOURCE LOGIC ---
        sf_source = str(site.get('streamflow_source'))
        fl_source = str(site.get('flowline_source'))

        streamflow_source = None

        if sf_source == "GEOGLOWS" and fl_source == "NHDPlus":
            # Pass the NHD shapefile path so RathCelon can look up the COMID?
            streamflow_source = str(site['flowline_path'])

        elif sf_source == "GEOGLOWS" and fl_source == "GEOGLOWS":
            # Pass None (JSON null), implies RathCelon infers it from geometry/defaults
            streamflow_source = None

        elif sf_source == 'National Water Model' and fl_source == 'NHDPlus':
            # Pass the local parquet file path
            streamflow_source = str(nwm_parquet)

        elif sf_source == 'National Water Model' and fl_source == 'GEOGLOWS':
            # Pass the local parquet file path
            streamflow_source = str(nwm_parquet)

        else:
            # Fallback for unexpected combinations (e.g. USGS)
            streamflow_source = sf_source

        # --- DICTIONARY CONSTRUCTION ---
        dam_dict = {
            "name": str(site['name']),
            "dam_csv": str(bridge_csv_path),
            "dam_id_field": "site_id",
            "dam_id": int(site_id),
            "flowline": str(site['flowline_path']),
            "dem_dir": str(site['dem_dir']),
            "bathy_use_banks": use_banks,
            "output_dir": str(dam_output_dir),
            "process_stream_network": True,
            "find_banks_based_on_landcover": False,
            "create_reach_average_curve_file": False,
            "known_baseflow": baseflow_val,
            "streamflow": streamflow_source
        }

        dams_list.append(dam_dict)

    # 5. Create Final JSON Wrapper
    # Note: Removed top-level 'nwm_parquet' key as requested by your logic
    json_data = {
        "dams": dams_list
    }

    # 6. Save JSON
    try:
        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Successfully created JSON at: {json_output_path}")
        print(f"Total Dams Included: {len(dams_list)}")
    except Exception as e:
        print(f"Failed to write JSON: {e}")
