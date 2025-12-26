import os
import re
import json
import datetime
import pandas as pd
import xarray as xr
import geopandas as gpd
import geoglows
from typing import Dict, Any, Optional, List  # Added for proper type hinting
from .download_geospatial_data import (
    download_dem,
    download_nhd_flowline,
    download_tdx_flowline,
    download_land_raster,
    find_water_gpstime
)


class LowHeadDam:
    """
    Represents a low-head dam during the data preparation phase.
    Handles geospatial data assignment and streamflow retrieval.
    """

    def __init__(self, site_id: int, db_manager):
        self.site_id: int = int(site_id)
        self.db = db_manager

        # Explicitly hint this as a Dictionary with Any values to stop the 'int' warning
        self.site_data: Dict[str, Any] = self.db.get_site(self.site_id)

        if not self.site_data:
            self.site_data = {'site_id': self.site_id}

        # Basic Info
        self.name = self.site_data.get('name')
        self.latitude = float(self.site_data.get('latitude', 0.0))
        self.longitude = float(self.site_data.get('longitude', 0.0))
        self.weir_length = float(self.site_data.get('weir_length', 100.0))

        # Identifiers
        self.nwm_id = self.site_data.get('reach_id')
        self.geoglows_id = self.site_data.get('linkno')

        # Paths
        self.output_dir = self.site_data.get('output_dir')
        self.streamflow_source = self.site_data.get('streamflow_source')
        self.flowline_source = self.site_data.get('flowline_source')

        self.dem_path = self.site_data.get('dem_path')
        self.res_meters = self.site_data.get('dem_resolution_m')
        self.lidar_year = self.site_data.get('lidar_year')
        self.land_cover_path = self.site_data.get('land_raster')

        self.flowline_gdf: Optional[gpd.GeoDataFrame] = None
        self.incidents_df = self.db.get_site_incidents(self.site_id)

    def set_streamflow_source(self, source: str):
        self.streamflow_source = source

    def set_flowline_source(self, source: str):
        self.flowline_source = source

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
    # --- STREAMFLOW RETRIEVAL ---

    def get_streamflow(self, date: str, source: Optional[str] = None) -> Optional[float]:
        source = source or self.streamflow_source

        if source == 'GEOGLOWS' and self.geoglows_id:
            try:
                df = geoglows.data.retrospective(
                    river_id=int(self.geoglows_id),
                    start_date=date,
                    end_date=date
                )
                return float(df.iloc[0, 0]) if not df.empty else None
            except Exception as e:
                print(f"GEOGLOWS API Error: {e}")
                return 0.0

        elif source == 'National Water Model' and self.nwm_id:
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                zarr_path = os.path.join(project_root, 'data', 'nwm_v3_daily_retrospective.zarr')
                if os.path.exists(zarr_path):
                    try:
                        ds = xr.open_zarr(zarr_path, consolidated=True)
                    except (KeyError, FileNotFoundError):
                        ds = xr.open_zarr(zarr_path, consolidated=False)
                    
                    try:
                        # Select feature and time
                        val = ds['streamflow'].sel(feature_id=int(self.nwm_id), time=date).values
                        return float(val)
                    except Exception:
                        # Fallback if specific date/ID not found
                        return 0.0
            except Exception as e:
                print(f"NWM Zarr Error: {e}")
                return 0.0
        else:
            return 0.0


    def get_median_streamflow(self, source: Optional[str] = None) -> float:
        source = source or self.streamflow_source

        if source == 'GEOGLOWS' and self.geoglows_id:
            df = geoglows.data.retrospective(river_id=int(self.geoglows_id))
            return float(df.median().iloc[0])

        elif source == 'National Water Model' and self.nwm_id:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            zarr_path = os.path.join(project_root, 'data', 'nwm_v3_daily_retrospective.zarr')
            if os.path.exists(zarr_path):
                try:
                    try:
                        ds = xr.open_zarr(zarr_path, consolidated=True)
                    except (KeyError, FileNotFoundError):
                        ds = xr.open_zarr(zarr_path, consolidated=False)
                    
                    median_val = ds['streamflow'].sel(feature_id=int(self.nwm_id)).median().values
                    return float(median_val)
                except Exception:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0

    def est_dem_baseflow(self, baseflow_method: str):
        print(f"Dam {self.site_id}: Estimating baseflow...")
        found_date = None
        if "lidar date" in baseflow_method.lower():
            lidar_time = find_water_gpstime(self.latitude, self.longitude)
            if lidar_time:
                found_date = (datetime.datetime(1980, 1, 6) + datetime.timedelta(seconds=lidar_time)).strftime(
                    '%Y-%m-%d')
            elif self.lidar_year:
                found_date = f"{self.lidar_year}-06-15"

        if found_date:
            self.site_data['lidar_date'] = found_date
            if self.streamflow_source == 'National Water Model':
                self.site_data['baseflow_nwm'] = self.get_streamflow(found_date, "National Water Model")
            elif self.streamflow_source == 'GEOGLOWS':
                self.site_data['baseflow_geo'] = self.get_streamflow(found_date, "GEOGLOWS")
        else:
            if self.streamflow_source == 'National Water Model':
                self.site_data['baseflow_nwm'] = self.get_median_streamflow("National Water Model")
            elif self.streamflow_source == 'GEOGLOWS':
                self.site_data['baseflow_geo'] = self.get_median_streamflow("GEOGLOWS")

        self.site_data['baseflow_method'] = baseflow_method

    def est_fatal_flows(self, source: Optional[str] = None):
        """
            Retrieves flow for each incident date and updates the site's incidents
            dataframe following the database schema: ['site_id', 'date', 'flow_nwm', 'flow_geo'].
        """
        source = source or self.streamflow_source

        if self.incidents_df.empty:
            print(f"Dam {self.site_id}: No incidents found in the database to process.")
            return

        # Determine which schema column to update based on the source
        flow_col = 'flow_nwm' if source == 'National Water Model' else 'flow_geo'

        print(f"Dam {self.site_id}: Updating {flow_col} for {len(self.incidents_df)} incidents...")

        # Update the dataframe in place
        for idx, row in self.incidents_df.iterrows():
            incident_date = str(row['date'])
            flow_val = self.get_streamflow(incident_date, source=source)

            if flow_val is not None:
                # Update the specific column defined in your incidents_schema
                self.incidents_df.at[idx, flow_col] = float(flow_val)
            else:
                print(f"  - Warning: Could not find {source} flow for {incident_date}")

        # When save_changes() is called, this updated dataframe will be
        # passed to db.update_site_incidents()
        print(f"Dam {self.site_id}: Incident flows updated in memory.")


    # --- GEOSPATIAL ASSIGNMENT ---

    def assign_dem(self, dem_dir: str, resolution: int):
        if self.flowline_gdf is None:
            return

        path, res, project_name = download_dem(self.site_id, self.flowline_gdf, dem_dir, resolution)

        if path:
            self.dem_path = path
            self.res_meters = res

            # Using direct assignment satisfies the IDE linter
            self.site_data['dem_path'] = path
            self.site_data['dem_resolution_m'] = res
            self.site_data['lidar_project'] = project_name
            self.site_data['dem_source_info'] = "USGS 3DEP via Py3DEP"

            year_match = re.search(r'(20\d{2}|FY\d{2})', project_name)
            if year_match:
                year_val = year_match.group(0)
                self.lidar_year = f"20{year_val[-2:]}" if "FY" in year_val else year_val
                self.site_data['lidar_year'] = self.lidar_year

    def assign_flowlines(self, flowline_dir: str, vpu_gpkg: str) -> List[int]:
        """
            Downloads both NHD and TDX flowlines if needed to support
            mixed streamflow sources (NWM and GEOGLOWS).
            Returns a list of all COMIDs/LINKNOs found in the downloaded flowline segments.
        """
        print(f"Dam {self.site_id}: Assigning flowlines for mixed-source support...")
        
        found_ids = []

        if self.flowline_source == 'NHDPlus' or self.streamflow_source == 'National Water Model':
            path_nhd, gdf = download_nhd_flowline(self.latitude, self.longitude, flowline_dir)
            if path_nhd:
                self.site_data['flowline_path_nhd'] = path_nhd
                # Set as primary for DEM clipping if selected
                if self.flowline_source == 'NHDPlus':
                    self.flowline_gdf = gdf

                # EXTRACT STARTING COMID FROM FILENAME
                filename = os.path.basename(path_nhd)
                match = re.search(r'nhd_flowline_(\d+)', filename)
                if match:
                    self.nwm_id = int(match.group(1))
                else:
                    # Fallback to GDF if naming convention is missing
                    self.nwm_id = int(gdf.iloc[0]['nhdplusid'])
                self.site_data['reach_id'] = self.nwm_id
                
                # Collect all IDs in the reach
                if 'nhdplusid' in gdf.columns:
                    found_ids.extend(gdf['nhdplusid'].astype(int).tolist())

        if self.flowline_source == 'TDX-Hydro' or self.streamflow_source == 'GEOGLOWS':
            path_tdx, gdf = download_tdx_flowline(self.latitude, self.longitude, flowline_dir, vpu_gpkg)
            if path_tdx:
                self.site_data['flowline_path_tdx'] = path_tdx
                # Set as primary for DEM clipping if selected
                if self.flowline_source in ['TDX-Hydro', 'GEOGLOWS']:
                    self.flowline_gdf = gdf

                # EXTRACT STARTING LINKNO FROM FILENAME
                filename = os.path.basename(path_tdx)
                match = re.search(r'tdx_flowline_(\d+)', filename)
                if match:
                    self.geoglows_id = int(match.group(1))
                else:
                    # Fallback to GDF if naming convention is missing
                    self.geoglows_id = int(gdf.iloc[0]['LINKNO'])

                self.site_data['linkno'] = self.geoglows_id
                
                # Collect all IDs in the reach
                if 'LINKNO' in gdf.columns:
                    found_ids.extend(gdf['LINKNO'].astype(int).tolist())
        
        return list(set(found_ids))


    def assign_land(self, land_dir: str):
        if self.dem_path:
            land_raster = download_land_raster(self.site_id, str(self.dem_path), land_dir)
            self.land_cover_path = land_raster
            self.site_data['land_raster'] = land_raster

    def save_changes(self):
        """
        Syncs the in-memory state (including mixed-source IDs and updated
        incident flows) back to the central database manager.
        """
        # 1. Sync both IDs back to site_data to ensure they are persisted in the 'Sites' sheet
        self.site_data['reach_id'] = self.nwm_id
        self.site_data['linkno'] = self.geoglows_id

        # 2. Update the main site attributes (Sites sheet)
        # This handles dem_path, baseflow_nwm, baseflow_geo, etc.
        self.db.update_site_data(self.site_id, self.site_data)

        # 3. Update the incident records (Incidents sheet)
        # This persists the flow_nwm and flow_geo values calculated in est_fatal_flows
        if not self.incidents_df.empty:
            self.db.update_site_incidents(self.site_id, self.incidents_df)

        print(f"Dam {self.site_id}: All changes (site metadata and incident flows) saved to database.")


    def to_json(self, output_dir: Optional[str] = None) -> str:
        if output_dir is None:
            output_dir = os.path.join(self.output_dir or "output", str(self.site_id))
        os.makedirs(output_dir, exist_ok=True)

        self.site_data['last_updated'] = datetime.datetime.now().isoformat()
        export_dict = {k: v for k, v in self.site_data.items()
                       if not isinstance(v, (pd.DataFrame, gpd.GeoDataFrame))}

        json_path = os.path.join(output_dir, f"LHD_{self.site_id}_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(export_dict, f, indent=4)
        return json_path
