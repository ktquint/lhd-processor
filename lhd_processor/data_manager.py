import os
import shutil
import pandas as pd
import threading

import os
import shutil
import threading
import pandas as pd


class DatabaseManager:
    """
    Manages reading and writing to the Excel database.
    Supports dynamic sheets for different flowline/streamflow combinations.
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
            'reach_id', 'linkno',
            'P_known', 'lidar_date', 'lidar_project',
            'land_raster',
            'flowline_source', 'streamflow_source'
        ]

        self.incidents_schema = ['site_id', 'date', 'flow_nwm', 'flow_geo']

        self.xsections_schema = [
            'site_id', 'xs_index', 'Slope', 'P_height',
            'Qmin', 'Qmax', 'depth_a', 'depth_b',
            'vel_a', 'vel_b', 'tw_a', 'tw_b',
            'prob_min', 'prob_max'
        ]

        self.results_schema = [
            'site_id', 'date', 'xs_index',
            'y_t', 'y_flip', 'y_2', 'jump_type',
            'y_1', 'Fr_1'
        ]

        # Sheet name abbreviations
        self.abbr_map = {
            'NHDPlus': 'NHD',
            'TDX-Hydro': 'TDX',
            'National Water Model': 'NWM',
            'GEOGLOWS': 'GEO'
        }

        # DataFrames
        self.sites = pd.DataFrame()
        self.incidents = pd.DataFrame()

        self.xsections = {}
        self.results = {}

        self.empty_results = pd.DataFrame(columns=self.results_schema)
        self.empty_xsections = pd.DataFrame(columns=self.xsections_schema)

        self.load()

    # --------------------------------------------------
    # Utility Functions
    # --------------------------------------------------

    def _safe_concat(self, dfs, schema):
        """Safely concatenate DataFrames ignoring empty/all-NA frames."""
        dfs = [df for df in dfs if not df.empty and not df.dropna(how='all').empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=schema)

    def _enforce_schema(self, df, schema):
        """Ensure dataframe matches schema."""
        if df.empty:
            return pd.DataFrame(columns=schema)

        df = df[[c for c in df.columns if c in schema]].copy()

        for col in schema:
            if col not in df.columns:
                df[col] = None

        return df.astype(object)

    def _get_sheet_names(self, flowline, streamflow):
        f = self.abbr_map.get(flowline, flowline)
        s = self.abbr_map.get(streamflow, streamflow)
        return f"CrossSections{f}{s}", f"Results{f}{s}"

    # --------------------------------------------------
    # Loading / Saving
    # --------------------------------------------------

    def load(self):
        """Loads database from Excel."""
        with self.lock:

            if not os.path.exists(self.filepath):
                self.sites = pd.DataFrame(columns=self.sites_schema)
                self.incidents = pd.DataFrame(columns=self.incidents_schema)
                self.xsections = {}
                self.results = {}
                return

            try:
                xls = pd.ExcelFile(self.filepath)

                def load_sheet(name, schema):
                    if name in xls.sheet_names:
                        df = pd.read_excel(xls, name)
                        if 'date' in df.columns:
                            df['date'] = df['date'].astype(str)
                        return self._enforce_schema(df, schema)
                    return pd.DataFrame(columns=schema)

                self.sites = load_sheet('Sites', self.sites_schema)
                self.incidents = load_sheet('Incidents', self.incidents_schema)

                self.xsections = {}
                self.results = {}

                for sheet in xls.sheet_names:

                    if sheet.startswith("CrossSections"):
                        self.xsections[sheet] = load_sheet(sheet, self.xsections_schema)

                    elif sheet.startswith("Results"):
                        self.results[sheet] = load_sheet(sheet, self.results_schema)

                # Fast site lookup
                if not self.sites.empty:
                    self.sites.set_index('site_id', inplace=True, drop=False)

            except Exception as e:
                print(f"Error loading database: {e}")

    def save(self):
        """Writes database back to Excel."""
        with self.lock:

            if os.path.exists(self.filepath):
                backup = self.filepath.replace(".xlsx", "_backup.xlsx")
                try:
                    shutil.copyfile(self.filepath, backup)
                except:
                    pass

            try:

                with pd.ExcelWriter(self.filepath, engine='openpyxl', mode='w') as writer:

                    self.sites.to_excel(writer, sheet_name='Sites', index=False)
                    self.incidents.to_excel(writer, sheet_name='Incidents', index=False)

                    for sheet, df in self.xsections.items():
                        df.to_excel(writer, sheet_name=sheet, index=False)

                    for sheet, df in self.results.items():
                        df.to_excel(writer, sheet_name=sheet, index=False)

            except Exception as e:
                print(f"Error saving database: {e}")

    # --------------------------------------------------
    # Getters
    # --------------------------------------------------

    def get_site(self, site_id):
        with self.lock:
            try:
                site_id = int(site_id)
            except:
                pass

            if site_id in self.sites.index:
                return self.sites.loc[site_id].where(pd.notnull(self.sites.loc[site_id]), None).to_dict()

            return {}

    def get_site_incidents(self, site_id):
        with self.lock:
            site_id = int(site_id)
            return self.incidents[self.incidents['site_id'] == site_id].copy()

    def get_site_results(self, site_id, flowline_source, streamflow_source):

        with self.lock:

            site_id = int(site_id)

            _, res_sheet = self._get_sheet_names(flowline_source, streamflow_source)

            df = self.results.get(res_sheet, self.empty_results)

            return df[df['site_id'] == site_id].copy()

    def get_site_xsections(self, site_id, flowline_source, streamflow_source):

        with self.lock:

            site_id = int(site_id)

            xs_sheet, _ = self._get_sheet_names(flowline_source, streamflow_source)

            df = self.xsections.get(xs_sheet, self.empty_xsections)

            return df[df['site_id'] == site_id].copy()

    # --------------------------------------------------
    # Updates
    # --------------------------------------------------

    def update_site_data(self, site_id, data_dict):

        with self.lock:

            site_id = int(site_id)

            filtered = {k: v for k, v in data_dict.items() if k in self.sites_schema}

            if site_id not in self.sites.index:

                filtered['site_id'] = site_id

                new_row = pd.DataFrame([filtered])

                self.sites = self._safe_concat([self.sites, new_row], self.sites_schema)

                self.sites.set_index('site_id', inplace=True, drop=False)

            else:

                for key, val in filtered.items():
                    self.sites.at[site_id, key] = val

    def update_site_incidents(self, site_id, updates_df):

        with self.lock:

            site_id = int(site_id)

            self.incidents.drop(
                self.incidents[self.incidents['site_id'] == site_id].index,
                inplace=True
            )

            if not updates_df.empty:

                updates_df = updates_df.copy()

                updates_df['site_id'] = site_id

                self.incidents = self._safe_concat(
                    [self.incidents, updates_df],
                    self.incidents_schema
                )

    def update_site_results(self, site_id, results_df, flowline_source, streamflow_source):

        with self.lock:

            site_id = int(site_id)

            _, res_sheet = self._get_sheet_names(flowline_source, streamflow_source)

            current = self.results.get(res_sheet, self.empty_results.copy())

            current = current[current['site_id'] != site_id]

            if not results_df.empty:

                new_res = results_df.copy()

                new_res['site_id'] = site_id

                new_res = self._enforce_schema(new_res, self.results_schema)

                current = self._safe_concat([current, new_res], self.results_schema)

            self.results[res_sheet] = current

    def update_analysis_results(self, site_id, xs_data, hydraulic_data, flowline_source, streamflow_source):

        with self.lock:

            site_id = int(site_id)

            xs_sheet, res_sheet = self._get_sheet_names(flowline_source, streamflow_source)

            # --- Cross Sections ---

            current_xs = self.xsections.get(xs_sheet, self.empty_xsections.copy())

            current_xs = current_xs[current_xs['site_id'] != site_id]

            if xs_data:

                new_xs = pd.DataFrame(xs_data)

                new_xs['site_id'] = site_id

                new_xs = self._enforce_schema(new_xs, self.xsections_schema)

                current_xs = self._safe_concat([current_xs, new_xs], self.xsections_schema)

            self.xsections[xs_sheet] = current_xs

            # --- Results ---

            current_res = self.results.get(res_sheet, self.empty_results.copy())

            current_res = current_res[current_res['site_id'] != site_id]

            if hydraulic_data:

                new_res = pd.DataFrame(hydraulic_data)

                new_res['site_id'] = site_id

                new_res = self._enforce_schema(new_res, self.results_schema)

                current_res = self._safe_concat([current_res, new_res], self.results_schema)

            self.results[res_sheet] = current_res