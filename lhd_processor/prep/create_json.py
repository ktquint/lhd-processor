import os
import json
import pandas as pd
from ..data_manager import DatabaseManager

# Updated signature to accept flowline_source and streamflow_source
def rathcelon_input(db_path, json_output_path, baseflow_method, nwm_parquet, flowline_source, streamflow_source):
    """
    Reads the Excel database, creates a temporary CSV bridge for RathCelon,
    and generates the JSON input file with specific conditional logic.
    """
    print("Generating RathCelon input files...")

    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found.")
        return

    # --- NAMING LOGIC START ---
    # Map full names to short codes
    fl_map = {
        "NHDPlus": "NHD",
        "GEOGLOWS": "TDX"  # GEOGLOWS Flowlines are commonly TDX-Hydro
    }
    sf_map = {
        "National Water Model": "NWM",
        "GEOGLOWS": "GEO"
    }

    # Get codes, defaulting to the full string if not found
    fl_code = fl_map.get(flowline_source, flowline_source)
    sf_code = sf_map.get(streamflow_source, streamflow_source)

    # Construct the dynamic filename
    bridge_csv_name = f"rathcelon_{fl_code}_{sf_code}.csv"
    # --- NAMING LOGIC END ---

    # 1. Load Data
    db = DatabaseManager(db_path)

    # 2. Filter for ready sites
    ready_sites = db.sites.dropna(subset=['dem_dir', 'flowline_path']).copy()

    if ready_sites.empty:
        print("No sites are ready for processing (missing DEM or Flowline).")
        return

    # 3. Create the "Bridge" CSV using the new name
    output_dir = os.path.dirname(json_output_path)
    # bridge_csv_name variable is now defined above
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
        baseflow_val = site.get('dem_baseflow')
        if pd.isna(baseflow_val):
            baseflow_val = 0.0
        else:
            baseflow_val = float(baseflow_val)

        if baseflow_method in ["WSE and LiDAR Date", "WSE and Median Daily Flow"]:
            use_banks = False
        else:
            use_banks = True
            baseflow_val = None

        # --- STREAMFLOW SOURCE LOGIC ---
        sf_source = str(site.get('streamflow_source'))
        fl_source = str(site.get('flowline_source'))

        streamflow_source_path = None

        if sf_source == "GEOGLOWS" and fl_source == "NHDPlus":
            streamflow_source_path = str(site['flowline_path'])

        elif sf_source == "GEOGLOWS" and fl_source == "GEOGLOWS":
            streamflow_source_path = None

        elif sf_source == 'National Water Model' and fl_source == 'NHDPlus':
            streamflow_source_path = str(nwm_parquet)

        elif sf_source == 'National Water Model' and fl_source == 'GEOGLOWS':
            streamflow_source_path = str(nwm_parquet)

        else:
            streamflow_source_path = sf_source

        # --- DICTIONARY CONSTRUCTION ---
        dam_dict = {
            "name": str(site['site_id']), # Ensure this matches your previous fix
            "dam_csv": str(bridge_csv_path),
            "dam_id_field": "site_id",
            "dam_id": int(site_id),
            "flowline": str(site['flowline_path']),
            "dem_dir": str(site['dem_dir']),
            "land_tif": str(site['land_path']),
            "bathy_use_banks": use_banks,
            "output_dir": str(dam_output_dir),
            "process_stream_network": True,
            "find_banks_based_on_landcover": False,
            "create_reach_average_curve_file": False,
            "known_baseflow": baseflow_val,
            "streamflow": streamflow_source_path
        }

        dams_list.append(dam_dict)

    # 5. Create Final JSON Wrapper
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
