import os
import shutil
import json
import pandas as pd
import threading


class DatabaseManager:
    """
    Manages reading and writing to the 4-tab Excel database.
    Now includes support for LiDAR dating.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = threading.Lock()

        # --- Schemas ---
        self.sites_schema = [
            'site_id', 'name', 'latitude', 'longitude', 'weir_length', 'comments',
            'dem_path', 'dem_resolution_m', 'lidar_project',
            'flowline_source', 'streamflow_source',
            'flowline_path_nhd', 'flowline_path_tdx',
            'reach_id', 'linkno',  # Added these to schema
            'baseflow_nwm', 'baseflow_geo',
            'P_known', 'output_dir',
            'baseflow_method', 'lidar_date', 'dem_source_info',
            'land_raster'
        ]

        self.incidents_schema = ['site_id', 'date', 'flow_nwm', 'flow_geo']

        self.xsections_schema = [
            'site_id', 'xs_index', 'slope', 'P_height',
            'Qmin', 'Qmax', 'rating_a', 'rating_b'
        ]

        # Dynamic results
        self.results_schema = [
            'site_id', 'date', 'xs_index',
            'y_t', 'y_flip', 'y_2', 'jump_type'
        ]

        # Map sources to sheet names
        self.source_map = {
            'National Water Model': 'Results_NWM',
            'GEOGLOWS': 'Results_GEOGLOWS'
        }

        self.sites = pd.DataFrame()
        self.incidents = pd.DataFrame()
        self.xsections = pd.DataFrame()
        self.results = {k: pd.DataFrame() for k in self.source_map}

        self.load()

    def _enforce_schema(self, df, schema):
        if df.empty: return pd.DataFrame(columns=schema)
        for col in schema:
            if col not in df.columns: df[col] = None
        return df

    def load(self):
        """Loads data from the Excel file or creates empty structures if new."""
        with self.lock:
            if not os.path.exists(self.filepath):
                self.sites = pd.DataFrame(columns=self.sites_schema)
                self.incidents = pd.DataFrame(columns=self.incidents_schema)
                self.xsections = pd.DataFrame(columns=self.xsections_schema)
                self.results = {k: pd.DataFrame(columns=self.results_schema) for k in self.source_map}
            else:
                try:
                    xls = pd.ExcelFile(self.filepath)

                    def load_sheet(name, schema):
                        if name in xls.sheet_names:
                            df = pd.read_excel(xls, name)
                            if 'date' in df.columns: df['date'] = df['date'].astype(str)
                            return self._enforce_schema(df, schema)
                        return pd.DataFrame(columns=schema)

                    self.sites = load_sheet('Sites', self.sites_schema)
                    self.incidents = load_sheet('Incidents', self.incidents_schema)
                    self.xsections = load_sheet('CrossSections', self.xsections_schema)
                    for source, sheet in self.source_map.items():
                        self.results[source] = load_sheet(sheet, self.results_schema)

                except Exception as e:
                    print(f"Error loading database: {e}")

    def save(self):
        """Saves all DataFrames back to the Excel file."""
        with self.lock:
            if os.path.exists(self.filepath):
                backup = self.filepath.replace('.xlsx', '_backup.xlsx')
                try:
                    shutil.copyfile(self.filepath, backup)
                except:
                    pass

            try:
                with pd.ExcelWriter(self.filepath, engine='openpyxl') as writer:
                    self.sites.to_excel(writer, sheet_name='Sites', index=False)
                    self.incidents.to_excel(writer, sheet_name='Incidents', index=False)
                    self.xsections.to_excel(writer, sheet_name='CrossSections', index=False)
                    for source, sheet in self.source_map.items():
                        self.results[source].to_excel(writer, sheet_name=sheet, index=False)
            except Exception as e:
                print(f"Error saving database: {e}")

    # --- Getters ---
    def get_site(self, site_id):
        with self.lock:
            try:
                site_id = int(site_id)
            except:
                pass
            row = self.sites[self.sites['site_id'] == site_id]
            return row.iloc[0].where(pd.notnull(row.iloc[0]), None).to_dict() if not row.empty else {}

    def get_site_incidents(self, site_id):
        with self.lock:
            try:
                site_id = int(site_id)
            except:
                pass
            return self.incidents[self.incidents['site_id'] == site_id].copy()

    def get_site_results(self, site_id, source):
        with self.lock:
            try:
                site_id = int(site_id)
            except:
                pass
            if source in self.results:
                df = self.results[source]
                return df[df['site_id'] == site_id].copy()
            return pd.DataFrame(columns=self.results_schema)

    # --- Updaters ---
    def update_site_data(self, site_id, data_dict):
        with self.lock:
            site_id = int(site_id)

            # Only keep data that belongs in the Excel schema
            # This keeps your Excel file tidy and prevents "Table with X records" text from being saved
            filtered_dict = {k: v for k, v in data_dict.items() if k in self.sites_schema}

            if site_id not in self.sites['site_id'].values:
                filtered_dict['site_id'] = site_id
                self.sites = pd.concat([self.sites, pd.DataFrame([filtered_dict])], ignore_index=True)
            else:
                idx = self.sites[self.sites['site_id'] == site_id].index[0]
                for key, val in filtered_dict.items():
                    # Explicitly cast to object if necessary to avoid FutureWarning
                    if key in self.sites.columns:
                        if self.sites[key].dtype == 'float64' and isinstance(val, str):
                            self.sites[key] = self.sites[key].astype('object')
                    self.sites.at[idx, key] = val

    def update_site_incidents(self, site_id, updates_df):
        with self.lock:
            site_id = int(site_id)
            # Clear old incidents for this site
            self.incidents = self.incidents[self.incidents['site_id'] != site_id]
            if not updates_df.empty:
                updates_df = updates_df.copy()
                updates_df['site_id'] = site_id
                self.incidents = pd.concat([self.incidents, updates_df], ignore_index=True)

    def update_analysis_results(self, site_id, xs_data, hydraulic_data, source='National Water Model'):
        with self.lock:
            site_id = int(site_id)

            # 1. Update CrossSections
            self.xsections = self.xsections[self.xsections['site_id'] != site_id]
            if xs_data:
                new_xs = pd.DataFrame(xs_data)
                new_xs['site_id'] = site_id
                new_xs = self._enforce_schema(new_xs, self.xsections_schema)
                self.xsections = pd.concat([self.xsections, new_xs], ignore_index=True)

            # 2. Update HydraulicResults
            if source not in self.results:
                source = 'National Water Model'

            current_df = self.results[source]
            current_df = current_df[current_df['site_id'] != site_id]

            if hydraulic_data:
                new_res = pd.DataFrame(hydraulic_data)
                new_res['site_id'] = site_id
                new_res = self._enforce_schema(new_res, self.results_schema)
                current_df = pd.concat([current_df, new_res], ignore_index=True)

            self.results[source] = current_df

    def update_site_results(self, site_id, df, source):
        with self.lock:
            site_id = int(site_id)
            if source in self.results:
                current_df = self.results[source]
                current_df = current_df[current_df['site_id'] != site_id]
                if not df.empty:
                    df = self._enforce_schema(df.copy(), self.results_schema)
                    df['site_id'] = site_id
                    current_df = pd.concat([current_df, df], ignore_index=True)
                self.results[source] = current_df

    def to_json(self, json_output_path, baseflow_method, nwm_parquet, flowline_source, streamflow_source):
        """
        Generates the JSON input file for RathCelon, replacing the functionality of create_json.py.
        """
        print("Generating RathCelon input files...")

        # --- NAMING LOGIC START ---
        # Map full names to short codes
        fl_map = {
            "NHDPlus": "NHD",
            "TDX-Hydro": "TDX"
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

        # 1. Filter for ready sites
        # Ensure we have the necessary columns before filtering
        required_cols = ['dem_path', 'flowline_path_nhd', 'flowline_path_tdx', 'output_dir']
        for col in required_cols:
            if col not in self.sites.columns:
                self.sites[col] = None

        # Determine which flowline path to check based on source
        flowline_col = 'flowline_path_nhd' if flowline_source == 'NHDPlus' else 'flowline_path_tdx'
        
        ready_sites = self.sites.dropna(subset=['dem_path', flowline_col]).copy()

        if ready_sites.empty:
            print("No sites are ready for processing (missing DEM or Flowline).")
            return

        # 2. Create the "Bridge" CSV using the new name
        output_dir = os.path.dirname(json_output_path)
        bridge_csv_path = os.path.join(output_dir, bridge_csv_name)

        try:
            ready_sites.to_csv(bridge_csv_path, index=False)
            print(f"Created bridge CSV: {bridge_csv_path}")
        except Exception as e:
            print(f"Failed to create bridge CSV: {e}")
            return

        dams_list = []

        # 3. Build the Dictionary for each Dam
        for _, site in ready_sites.iterrows():
            site_id = site['site_id']

            # Determine output directory
            dem_dir = os.path.dirname(site['dem_path'])
            dam_output_dir = site['output_dir']
            if pd.isna(dam_output_dir):
                dam_output_dir = os.path.join(output_dir, "Results")

            # --- BASEFLOW & BANKS LOGIC ---
            # Determine baseflow value based on source
            baseflow_val = 0.0
            if streamflow_source == 'National Water Model':
                baseflow_val = site.get('baseflow_nwm')
            elif streamflow_source == 'GEOGLOWS':
                baseflow_val = site.get('baseflow_geo')
            
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
            streamflow_source_path = None

            # Get the correct flowline path
            flowline_path = site[flowline_col]

            if streamflow_source == "GEOGLOWS" and flowline_source == "NHDPlus":
                streamflow_source_path = str(flowline_path)

            elif streamflow_source == "GEOGLOWS" and flowline_source == "TDX-Hydro":
                streamflow_source_path = None

            elif streamflow_source == 'National Water Model' and flowline_source == 'NHDPlus':
                streamflow_source_path = str(nwm_parquet)

            elif streamflow_source == 'National Water Model' and flowline_source == 'TDX-Hydro':
                streamflow_source_path = str(nwm_parquet)

            else:
                streamflow_source_path = streamflow_source

            # --- DICTIONARY CONSTRUCTION ---
            dam_dict = {
                "name": str(site['site_id']),
                "dam_csv": str(bridge_csv_path),
                "dam_id_field": "site_id",
                "dam_id": int(site_id),
                "flowline": str(flowline_path),
                "dem_dir": dem_dir,
                "land_raster": str(site.get('land_raster', '')),
                "bathy_use_banks": use_banks,
                "output_dir": str(dam_output_dir),
                "process_stream_network": True,
                "find_banks_based_on_landcover": False,
                "create_reach_average_curve_file": False,
                "known_baseflow": baseflow_val,
                "streamflow": streamflow_source_path
            }

            dams_list.append(dam_dict)

        # 4. Create Final JSON Wrapper
        json_data = {
            "dams": dams_list
        }

        # 5. Save JSON
        try:
            with open(json_output_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            print(f"Successfully created JSON at: {json_output_path}")
            print(f"Total Dams Included: {len(dams_list)}")
        except Exception as e:
            print(f"Failed to write JSON: {e}")
