import re  # Added for date extraction
import os
import glob
import pandas as pd
from .hydroinformatics import StreamReach
from .download_dem import download_dem
from .download_landuse import download_land_raster
from .dem_baseflow import est_dem_baseflow
from .download_flowline import download_nhdplus, download_tdx_hydro


class Dam:
    """
        Represents a low-head dam during the data preparation phase.
        Reads/Writes to the DatabaseManager.
    """

    def __init__(self, site_id, db_manager):
        self.site_id = int(site_id)
        self.db = db_manager

        # Load site data from the manager
        self.site_data = self.db.get_site(self.site_id)
        if not self.site_data:
            self.site_data = {'site_id': self.site_id}

        # Basic Info
        self.name = self.site_data.get('name')
        self.latitude = float(self.site_data.get('latitude', 0.0))
        self.longitude = float(self.site_data.get('longitude', 0.0))
        self.weir_length = float(self.site_data.get('weir_length', 100.0))

        # Load incidents for processing
        self.incidents_df = self.db.get_site_incidents(self.site_id)

        # Get settings (may be None initially)
        self.output_dir = self.site_data.get('output_dir')
        self.hydrology = self.site_data.get('streamflow_source')
        self.hydrography = self.site_data.get('flowline_source')

        # Placeholders
        self.dam_reach = None
        self.dem_1m = self.site_data.get('dem_1m')
        self.dem_3m = self.site_data.get('dem_3m')
        self.dem_10m = self.site_data.get('dem_10m')
        self.land_cover = self.site_data.get('land_cover')
        self.flowline_NHD = None
        self.flowline_TDX = self.site_data.get('flowline_path')

    def assign_flowlines(self, flowline_dir, VPU_gpkg):
        print(f"Dam {self.site_id}: Assigning flowlines ({self.hydrography})...")
        path = None

        if self.hydrography == 'NHDPlus':
            path = download_nhdplus(self.latitude, self.longitude, flowline_dir)
            self.flowline_NHD = path
        elif 'GEOGLOWS' in [self.hydrography, self.hydrology]:
            path = download_tdx_hydro(self.latitude, self.longitude, flowline_dir, VPU_gpkg)
            self.flowline_TDX = path

        if path:
            self.site_data['flowline_path'] = path

    def assign_dem(self, dem_dir, resolution):
        print(f"Dam {self.site_id}: Checking/Downloading DEM...")
        subdir, titles, res = download_dem(
            self.site_id, self.latitude, self.longitude, self.weir_length, dem_dir, resolution
        )

        if subdir:
            # --- Logic to extract LiDAR Date ---
            # Search for a 4-digit year starting with 19 or 20 (e.g. 1999, 2015)
            # Titles is sometimes a list or string, handle both
            title_str = str(titles)
            date_match = re.search(r'(19|20)\d{2}', title_str)

            if date_match:
                extracted_date = date_match.group(0)
                print(f"  - Found LiDAR Date: {extracted_date}")
                self.site_data['lidar_date'] = extracted_date
            else:
                self.site_data['lidar_date'] = "Unknown"

            # Save full title string for metadata
            self.site_data['dem_source_info'] = title_str
            self.site_data['dem_dir'] = subdir

            # Map resolution to specific columns
            if res is not None:
                if res <= 1.5:
                    self.site_data['dem_1m'] = subdir
                    self.dem_1m = subdir
                elif res <= 5.0:
                    self.site_data['dem_3m'] = subdir
                    self.dem_3m = subdir
                else:
                    self.site_data['dem_10m'] = subdir
                    self.dem_10m = subdir
                self.site_data['final_resolution'] = f"~{res:.2f}m"
            else:
                # Fallback based on request string
                if "1 meter" in resolution:
                    self.dem_1m = subdir
                elif "1/9" in resolution:
                    self.dem_3m = subdir
                else:
                    self.dem_10m = subdir

    def assign_land(self, dem_dir_parent, land_dir):
        print(f"Dam {self.site_id}: Assigning Land Use Data...")

        # Construct the expected path to the final merged DEM file.
        dem_name = f"{self.site_id}_MERGED_DEM.tif"

        # The final merged DEM is expected to be in a subdirectory named after the site_id
        # located inside the main DEM directory (dem_dir_parent).
        dam_dem_subdir = os.path.join(dem_dir_parent, str(self.site_id))
        dem_path = os.path.join(dam_dem_subdir, dem_name)

        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"Merged DEM file not found at expected path: {dem_path}")

        land_raster = download_land_raster(self.site_id, dem_path, land_dir)
        self.site_data['land_path'] = land_raster
        return land_raster

    def create_reach(self, nwm_ds=None, tdx_vpu_map=None):
        print(f'Dam {self.site_id}: Creating Stream Reach object...')

        geoglows_flowline_path = None
        if 'GEOGLOWS' in [self.hydrology, self.hydrography]:
            # Always use the specific flowline path we found/downloaded in assign_flowlines
            geoglows_flowline_path = self.site_data.get('flowline_path')

        self.dam_reach = StreamReach(
            lhd_id=self.site_id,
            latitude=self.latitude,
            longitude=self.longitude,
            data_sources=[self.hydrology],
            geoglows_streams=geoglows_flowline_path,
            nwm_ds=nwm_ds,
            streamflow=True
        )

    def set_dem_baseflow(self, baseflow_method):
        """Calculates baseflow and stores it in the 'Sites' dictionary."""
        current_val = self.site_data.get('dem_baseflow')

        # We generally want to run this if baseflow is missing OR if we want to capture the lidar date
        # If you want to force it to run even if baseflow exists (to get the date), remove the 'if pd.isna' check.
        # For now, I will keep the check, but you can remove it if needed.
        if pd.isna(current_val) or (
                baseflow_method == "WSE and LiDAR Date" and pd.isna(self.site_data.get('lidar_date'))):

            print(f"Dam {self.site_id}: Estimating baseflow via '{baseflow_method}'...")
            try:
                # Unpack the tuple (baseflow, date)
                baseflow, lidar_date = est_dem_baseflow(self.dam_reach, self.hydrology, baseflow_method)

                self.site_data['dem_baseflow'] = baseflow
                self.site_data['baseflow_method'] = baseflow_method

                # If we found a date from the LiDAR data, save it!
                if lidar_date:
                    print(f"  - Saving LiDAR Date to database: {lidar_date}")
                    self.site_data['lidar_date'] = lidar_date
                elif baseflow_method == "WSE and LiDAR Date":
                    self.site_data['lidar_date'] = None

            except Exception as e:
                print(f"Error estimating baseflow: {e}")
        else:
            print(f"Dam {self.site_id}: Baseflow already set ({current_val}).")


    def set_fatal_flows(self):
        """Updates the incidents DataFrame with flow values."""
        if self.incidents_df.empty:
            print(f"Dam {self.site_id}: No incidents found to process.")
            return

        if 'source' in self.incidents_df.columns:
            self.incidents_df['source'] = self.incidents_df['source'].astype(object)

        print(f"Dam {self.site_id}: Retrieving flows for {len(self.incidents_df)} incidents...")

        for index, row in self.incidents_df.iterrows():
            date = row['date']
            # Only fetch if flow is missing
            if pd.isna(row.get('flow')):
                try:
                    flow = self.dam_reach.get_flow_on_date(date, self.hydrology)
                    if isinstance(flow, (int, float)) and not pd.isna(flow):
                        self.incidents_df.at[index, 'flow'] = flow
                        # This line below was causing the warning:
                        self.incidents_df.at[index, 'source'] = self.hydrology
                except Exception as e:
                    print(f"  - Error retrieving flow for {date}: {e}")

    def set_output_dir(self, output_dir):
        self.site_data['output_dir'] = output_dir

    def set_streamflow_source(self, source):
        self.site_data['streamflow_source'] = source
        self.hydrology = source

    def set_flowline_source(self, source):
        self.site_data['flowline_source'] = source
        self.hydrography = source

    def save_changes(self):
        """Commits changes back to the DataManager."""
        self.db.update_site_data(self.site_id, self.site_data)
        self.db.update_site_incidents(self.site_id, self.incidents_df)

    def __repr__(self):
        return f"<Dam ID={self.site_id} Name='{self.name}'>"
