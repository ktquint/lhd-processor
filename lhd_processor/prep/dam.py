import os
import re
import json
import datetime
import pandas as pd
import geopandas as gpd
from .hydroinformatics import StreamReach
from .download_geospatial_data import (
    download_dem,
    download_nhd_flowline,
    download_tdx_flowline,
    download_land_raster,
    est_dem_baseflow
)


class Dam:
    """
    Represents a low-head dam during the data preparation phase.
    Handles geospatial data assignment and metadata management.
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

        # Settings and Paths
        self.output_dir = self.site_data.get('output_dir')
        self.streamflow_source = self.site_data.get('streamflow_source')
        self.flowline_source = self.site_data.get('flowline_source')

        # Data Placeholders
        self.flowline_gdf = None
        self.dam_reach = None
        self.dem_path = self.site_data.get('dem_path')
        self.res_meters = self.site_data.get('dem_resolution_m')
        self.lidar_year = self.site_data.get('lidar_year')
        self.land_cover_path = self.site_data.get('land_path')
        self.nhd_gdf = None
        self.tdx_gdf = None

    @classmethod
    def from_json(cls, json_path, db_manager):
        """Re-initialize a Dam object from a saved metadata JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(site_id=data['site_id'], db_manager=db_manager)

    def assign_flowlines(self, flowline_dir, vpu_gpkg) -> None:
        print(f"Dam {self.site_id}: Assigning necessary flowlines...")
        if self.flowline_source == 'NHDPlus':
            # 1. Fetch NHD
            path_nhd, gdf_nhd = download_nhd_flowline(self.latitude, self.longitude, flowline_dir)
            if path_nhd:
                self.site_data['flowline_path_nhd'] = path_nhd
                self.nhd_gdf = gdf_nhd

        if self.flowline_source == 'TDX-Hydro' or self.flowline_source == 'GEOGLOWS':
            # 2. Fetch TDX
            path_tdx, gdf_tdx = download_tdx_flowline(self.latitude, self.longitude, flowline_dir, vpu_gpkg)
            if path_tdx:
                self.site_data['flowline_path_tdx'] = path_tdx
                self.tdx_gdf = gdf_tdx


    def assign_dem(self, dem_dir, resolution):
        """
        Downloads 3DEP DEM using flowline extent.
        Consolidates to a single dem_path and extracts lidar project year.
        """
        if self.flowline_gdf is None:
            print(f"  - Error: No flowlines assigned for Dam {self.site_id}. Skipping DEM.")
            return

        print(f"Dam {self.site_id}: Downloading DEM...")
        path, res, project_name = download_dem(self.site_id, self.flowline_gdf, dem_dir, resolution)

        if path:
            self.dem_path = path
            self.res_meters = res
            self.site_data['dem_path'] = path
            self.site_data['dem_resolution_m'] = res
            self.site_data['lidar_project'] = project_name
            self.site_data['dem_source_info'] = "USGS 3DEP via HyRiver"

            # Extract year from project name (e.g., 'FY19' or '2022')
            year_match = re.search(r'(20\d{2}|FY\d{2})', project_name)
            if year_match:
                year_val = year_match.group(0)
                # Convert FY22 to 2022
                self.lidar_year = f"20{year_val[-2:]}" if "FY" in year_val else year_val
                self.site_data['lidar_year'] = self.lidar_year

    def assign_land(self, land_dir):
        """Clips ESA WorldCover to match the current DEM grid."""
        if not self.dem_path or not os.path.exists(self.dem_path):
            print(f"  - Error: No DEM for Dam {self.site_id}. Skipping Land Cover.")
            return

        print(f"Dam {self.site_id}: Aligning Land Use to {self.res_meters}m DEM...")
        land_raster = download_land_raster(self.site_id, str(self.dem_path), land_dir)
        self.land_cover_path = land_raster
        self.site_data['land_path'] = land_raster

    def create_reach(self, nwm_ds=None):
        """Initializes the StreamReach object for hydraulic analysis."""
        print(f"Dam {self.site_id}: Creating Stream Reach object...")

        flowline_path = self.site_data.get('flowline_path')

        self.dam_reach = StreamReach(
            lhd_id=self.site_id,
            latitude=self.latitude,
            longitude=self.longitude,
            data_sources=[self.streamflow_source],
            geoglows_streams=flowline_path if self.flowline_source == 'TDX-Hydro' else None,
            nwm_ds=nwm_ds,
            streamflow=True
        )

    def set_dem_baseflow(self, baseflow_method):
        print(f"Dam {self.site_id}: Calculating baseflow...")

        # Estimate for NWM (National Water Model)
        if self.streamflow_source == 'National Water Model':
            try:
                val_nwm, lidar_date = est_dem_baseflow(self.dam_reach, "National Water Model", baseflow_method)
                self.site_data['baseflow_nwm'] = val_nwm
                if lidar_date: self.site_data['lidar_date'] = lidar_date
            except Exception as e:
                print(f"  - NWM Baseflow Error: {e}")

        if self.streamflow_source == 'GEOGLOWS':
            # Estimate for GEOGLOWS
            try:
                val_geo, _ = est_dem_baseflow(self.dam_reach, "GEOGLOWS", baseflow_method)
                self.site_data['baseflow_geo'] = val_geo
            except Exception as e:
                print(f"  - GEOGLOWS Baseflow Error: {e}")

        self.site_data['baseflow_method'] = baseflow_method


    def to_json(self, output_dir=None):
        """Exports dam metadata and processing state to a JSON file."""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir if self.output_dir else "output", str(self.site_id))

        os.makedirs(output_dir, exist_ok=True)
        self.site_data['last_updated'] = datetime.datetime.now().isoformat()

        # Clean dictionary for JSON serialization
        export_dict = {}
        for k, v in self.site_data.items():
            if isinstance(v, (pd.DataFrame, gpd.GeoDataFrame)):
                export_dict[k] = f"Table ({len(v)} rows)"
            elif hasattr(v, 'wkt'):  # Handle Shapely geometries
                export_dict[k] = v.wkt
            else:
                export_dict[k] = v

        json_path = os.path.join(output_dir, f"LHD_{self.site_id}_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(export_dict, f, indent=4)

        print(f"Dam {self.site_id}: Metadata exported to {json_path}")
        return json_path

    def save_changes(self):
        """Syncs the in-memory state back to the central database/manager."""
        self.db.update_site_data(self.site_id, self.site_data)
        if not self.incidents_df.empty:
            self.db.update_site_incidents(self.site_id, self.incidents_df)

    def __repr__(self):
        return f"<Dam ID={self.site_id} Name='{self.name}' Res={self.res_meters}m>"
