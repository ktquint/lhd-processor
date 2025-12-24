import os
import shutil
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
            'baseflow_nwm', 'baseflow_geo',
            'P_known', 'output_dir',
            'baseflow_method', 'lidar_date', 'dem_source_info',
            'land_esa'
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

        self.sites = pd.DataFrame()
        self.incidents = pd.DataFrame()
        self.xsections = pd.DataFrame()
        self.results = pd.DataFrame()

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
                self.results = pd.DataFrame(columns=self.results_schema)
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
                    self.results = load_sheet('HydraulicResults', self.results_schema)

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
                    self.results.to_excel(writer, sheet_name='HydraulicResults', index=False)
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

    def update_analysis_results(self, site_id, xs_data, hydraulic_data):
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
            self.results = self.results[self.results['site_id'] != site_id]
            if hydraulic_data:
                new_res = pd.DataFrame(hydraulic_data)
                new_res['site_id'] = site_id
                new_res = self._enforce_schema(new_res, self.results_schema)
                self.results = pd.concat([self.results, new_res], ignore_index=True)
