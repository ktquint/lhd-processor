import os
import shutil
import pandas as pd
import threading


class DatabaseManager:
    """
    Manages reading and writing to the 4-tab Excel database.
    Now includes support for LiDAR dating and dynamic source sheets.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = threading.Lock()

        # --- Schemas ---
        self.sites_schema = [
            'site_id', 'name', 'latitude', 'longitude', 'weir_length', 'comments',
            'dem_path', 'dem_resolution_m', 'dem_source_info',
            'flowline_path_nhd', 'flowline_path_tdx',
            'flowline_raster_nhd', 'flowline_raster_tdx',
            'reach_id', 'linkno',  # Added these to schema
            'P_known', 'lidar_date', 'lidar_project',
            'land_raster',
            'flowline_source', 'streamflow_source' # Added these to schema
        ]

        self.incidents_schema = ['site_id', 'date', 'flow_nwm', 'flow_geo']

        self.xsections_schema = [
            'site_id', 'xs_index', 'Slope', 'P_height',
            'Qmin', 'Qmax', 'depth_a', 'depth_b'
        ]

        # Dynamic results
        self.results_schema = [
            'site_id', 'date', 'xs_index',
            'y_t', 'y_flip', 'y_2', 'jump_type'
        ]

        # Abbreviation map for sheet names
        self.abbr_map = {
            'NHDPlus': 'NHD',
            'TDX-Hydro': 'TDX',
            'National Water Model': 'NWM',
            'GEOGLOWS': 'GEO'
        }

        self.sites = pd.DataFrame()
        self.incidents = pd.DataFrame()
        
        # Dictionaries to hold DataFrames for each source combination
        # Key = Sheet Name
        self.xsections = {}
        self.results = {}

        self.load()

    def _enforce_schema(self, df, schema):
        if df.empty: return pd.DataFrame(columns=schema)
        # Remove columns not in schema
        df = df[[col for col in df.columns if col in schema]]
        # Add missing columns
        for col in schema:
            if col not in df.columns: df[col] = None
        return df

    def _get_sheet_names(self, flowline, streamflow):
        f = self.abbr_map.get(flowline, flowline)
        s = self.abbr_map.get(streamflow, streamflow)
        return f"CrossSections{f}{s}", f"Results{f}{s}"

    def load(self):
        """Loads data from the Excel file or creates empty structures if new."""
        with self.lock:
            if not os.path.exists(self.filepath):
                self.sites = pd.DataFrame(columns=self.sites_schema)
                self.incidents = pd.DataFrame(columns=self.incidents_schema)
                self.xsections = {}
                self.results = {}
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
                    
                    self.xsections = {}
                    self.results = {}
                    
                    for sheet in xls.sheet_names:
                        if sheet.startswith('CrossSections'):
                            self.xsections[sheet] = load_sheet(sheet, self.xsections_schema)
                        elif sheet.startswith('Results'):
                            self.results[sheet] = load_sheet(sheet, self.results_schema)

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
                    
                    for sheet, df in self.xsections.items():
                        df.to_excel(writer, sheet_name=sheet, index=False)
                        
                    for sheet, df in self.results.items():
                        df.to_excel(writer, sheet_name=sheet, index=False)
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

    def get_site_results(self, site_id, flowline_source, streamflow_source):
        with self.lock:
            try:
                site_id = int(site_id)
            except:
                pass
            
            _, res_sheet = self._get_sheet_names(flowline_source, streamflow_source)
            
            if res_sheet in self.results:
                df = self.results[res_sheet]
                return df[df['site_id'] == site_id].copy()
            return pd.DataFrame(columns=self.results_schema)

    def get_site_xsections(self, site_id, flowline_source, streamflow_source):
        with self.lock:
            try:
                site_id = int(site_id)
            except:
                pass
            
            xs_sheet, _ = self._get_sheet_names(flowline_source, streamflow_source)
            
            if xs_sheet in self.xsections:
                df = self.xsections[xs_sheet]
                return df[df['site_id'] == site_id].copy()
            return pd.DataFrame(columns=self.xsections_schema)

    # --- Updaters ---
    def update_site_data(self, site_id, data_dict):
        with self.lock:
            site_id = int(site_id)

            # Only keep data that belongs in the Excel schema
            filtered_dict = {k: v for k, v in data_dict.items() if k in self.sites_schema}

            if site_id not in self.sites['site_id'].values:
                filtered_dict['site_id'] = site_id
                self.sites = pd.concat([self.sites, pd.DataFrame([filtered_dict])], ignore_index=True)
            else:
                idx = self.sites[self.sites['site_id'] == site_id].index[0]
                for key, val in filtered_dict.items():
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

    def update_site_results(self, site_id, results_df, flowline_source, streamflow_source):
        """
        Updates the results for a specific site and source combination.
        This was missing in the previous version but called in prep/classes.py.
        """
        with self.lock:
            site_id = int(site_id)
            _, res_sheet = self._get_sheet_names(flowline_source, streamflow_source)

            current_res = self.results.get(res_sheet, pd.DataFrame(columns=self.results_schema))
            
            # Remove existing results for this site
            current_res = current_res[current_res['site_id'] != site_id]

            if not results_df.empty:
                new_res = results_df.copy()
                new_res['site_id'] = site_id
                new_res = self._enforce_schema(new_res, self.results_schema)
                current_res = pd.concat([current_res, new_res], ignore_index=True)

            self.results[res_sheet] = current_res

    def update_analysis_results(self, site_id, xs_data, hydraulic_data, flowline_source, streamflow_source):
        with self.lock:
            site_id = int(site_id)
            xs_sheet, res_sheet = self._get_sheet_names(flowline_source, streamflow_source)

            # 1. Update CrossSections
            current_xs = self.xsections.get(xs_sheet, pd.DataFrame(columns=self.xsections_schema))
            current_xs = current_xs[current_xs['site_id'] != site_id]
            
            if xs_data:
                new_xs = pd.DataFrame(xs_data)
                new_xs['site_id'] = site_id
                new_xs = self._enforce_schema(new_xs, self.xsections_schema)
                current_xs = pd.concat([current_xs, new_xs], ignore_index=True)
            
            self.xsections[xs_sheet] = current_xs

            # 2. Update HydraulicResults
            current_res = self.results.get(res_sheet, pd.DataFrame(columns=self.results_schema))
            current_res = current_res[current_res['site_id'] != site_id]

            if hydraulic_data:
                new_res = pd.DataFrame(hydraulic_data)
                new_res['site_id'] = site_id
                new_res = self._enforce_schema(new_res, self.results_schema)
                current_res = pd.concat([current_res, new_res], ignore_index=True)

            self.results[res_sheet] = current_res