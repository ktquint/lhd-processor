import ast
import pandas as pd
from .hydroinformatics import StreamReach
from .download_dem import download_dem
from .dem_baseflow import est_dem_baseflow
from .download_flowline import download_NHDPlus, download_TDXHYDRO


class Dam:
    """
    Represents a low-head dam and manages its associated geospatial and
    hydrologic data preparation.
    """

    def __init__(self, **kwargs):
        """
        Initialize a Dam object from a dictionary (typically a CSV row).
        """
        # Database info
        try:
            self.ID = int(kwargs['ID'])
        except (KeyError, ValueError):
            raise ValueError("Dam 'ID' is missing or invalid in source data.")

        self.name = kwargs.get('name', None)

        # Geographical info
        self.latitude = float(kwargs['latitude'])
        self.longitude = float(kwargs['longitude'])
        self.city = kwargs.get('city', None)
        self.county = kwargs.get('county', None)
        self.state = kwargs.get('state', None)

        # Fatality info (safely parse list-string)
        try:
            self.fatality_dates = ast.literal_eval(kwargs['fatality_dates'])
        except (ValueError, SyntaxError):
            self.fatality_dates = []

        # Physical information
        self.weir_length = float(kwargs['weir_length'])

        # Data paths (DEMs)
        self.dem_1m = kwargs.get('dem_1m', None)
        self.dem_3m = kwargs.get('dem_3m', None)
        self.dem_10m = kwargs.get('dem_10m', None)
        self.final_titles = kwargs.get('final_titles', None)
        self.final_resolution = kwargs.get('final_resolution', None)

        # Output and Hydro inputs
        self.output_dir = kwargs.get('output_dir', None)
        self.flowline_NHD = kwargs.get('flowline_NHD', None)
        self.flowline_TDX = kwargs.get('flowline_TDX', None)

        # Baseflow and Fatal Flow storage
        self.dem_baseflow_NWM = kwargs.get('dem_baseflow_NWM', None)
        self.dem_baseflow_GEOGLOWS = kwargs.get('dem_baseflow_GEOGLOWS', None)

        self.fatality_flows_NWM = kwargs.get('fatality_flows_NWM', None)
        self.fatality_flows_GEOGLOWS = kwargs.get('fatality_flows_GEOGLOWS', None)

        # Operational attributes (reset for fresh processing)
        self.hydrology = None
        self.hydrography = None
        self.dam_reach = None

    def assign_flowlines(self, flowline_dir: str, VPU_gpkg: str):
        """Downloads/Assigns flowlines based on the selected hydrography source."""
        print(f"Dam {self.ID}: Assigning flowlines ({self.hydrography})...")

        if self.hydrography == 'NHDPlus':
            self.flowline_NHD = download_NHDPlus(self.latitude, self.longitude, flowline_dir)

        elif self.hydrography == 'GEOGLOWS' or self.hydrology == 'GEOGLOWS':
            self.flowline_TDX = download_TDXHYDRO(self.latitude, self.longitude, flowline_dir, VPU_gpkg)

    def assign_dem(self, dem_dir, resolution):
        """Downloads and assigns the DEM based on requested resolution."""
        print(f"Dam {self.ID}: Checking/Downloading DEM...")

        # Call download_dem, which returns resolution_meters
        dem_subdir, self.final_titles, resolution_meters = download_dem(
            self.ID, self.latitude, self.longitude, self.weir_length, dem_dir, resolution
        )

        # Clear existing paths to ensure accuracy
        self.dem_1m = None
        self.dem_3m = None
        self.dem_10m = None
        self.final_resolution = None

        if dem_subdir and resolution_meters is not None:
            # Assign path based on the ACTUAL returned resolution
            if resolution_meters <= 1.5:
                self.dem_1m = dem_subdir
                self.final_resolution = "Digital Elevation Model (DEM) 1 meter"
            elif resolution_meters <= 5.0:
                self.dem_3m = dem_subdir
                self.final_resolution = "National Elevation Dataset (NED) 1/9 arc-second"
            else:
                self.dem_10m = dem_subdir
                self.final_resolution = "National Elevation Dataset (NED) 1/3 arc-second Current"

            print(f"  - Assigned DEM path to category ~{resolution_meters:.2f}m")

        elif dem_subdir:
            # Fallback if resolution calc failed but file exists
            print(f"  - Warning: Could not verify resolution. Defaulting to requested type '{resolution}'.")
            if "1 meter" in resolution:
                self.dem_1m = dem_subdir
            elif "1/9 arc-second" in resolution:
                self.dem_3m = dem_subdir
            else:
                self.dem_10m = dem_subdir
        else:
            print(f"  - DEM assignment failed.")

    def create_reach(self, nwm_ds=None, tdx_vpu_map=None):
        """
        Instantiates a StreamReach object for this Dam.
        """
        print(f'Dam {self.ID}: Creating Stream Reach object...')

        # Build data sources list
        data_sources = [self.hydrology]

        # Determine if we need the GEOGLOWS map
        geoglows_map_path = None
        if 'GEOGLOWS' in data_sources:
            if tdx_vpu_map:
                geoglows_map_path = tdx_vpu_map
            else:
                geoglows_map_path = getattr(self, 'flowline_TDX', None)

        self.dam_reach = StreamReach(
            lhd_id=self.ID,
            latitude=self.latitude,
            longitude=self.longitude,
            data_sources=data_sources,
            geoglows_streams=geoglows_map_path,
            nwm_ds=nwm_ds,
            streamflow=True
        )

    def set_dem_baseflow(self, baseflow_method):
        """Estimates the DEM baseflow if not already calculated."""
        # Map hydrology source to attribute name
        attr_map = {
            'National Water Model': 'dem_baseflow_NWM',
            'GEOGLOWS': 'dem_baseflow_GEOGLOWS'
        }

        baseflow_attr = attr_map.get(self.hydrology)

        if baseflow_attr:
            current_value = getattr(self, baseflow_attr)
            if pd.isna(current_value):
                print(f"Dam {self.ID}: Estimating baseflow via '{baseflow_method}'...")
                baseflow = est_dem_baseflow(self.dam_reach, self.hydrology, baseflow_method)
                setattr(self, baseflow_attr, baseflow)
            else:
                print(f"Dam {self.ID}: Baseflow already set ({current_value}).")
        else:
            print(f"Dam {self.ID}: Unknown hydrology source '{self.hydrology}'. Skipping baseflow.")

    def set_fatal_flows(self):
        """Retrieves flow values for recorded fatality dates."""
        print(f"Dam {self.ID}: Retrieving flows for {len(self.fatality_dates)} fatality dates...")

        valid_dates = []
        valid_flows = []
        skipped_info = {}

        for date in self.fatality_dates:
            flow_result = self.dam_reach.get_flow_on_date(date, self.hydrology)

            if isinstance(flow_result, (int, float)) and not pd.isna(flow_result):
                valid_dates.append(date)
                valid_flows.append(float(flow_result))
            else:
                skipped_info[date] = flow_result

        # Update attributes
        self.fatality_dates = valid_dates

        if self.hydrology == 'National Water Model':
            if pd.isna(self.fatality_flows_NWM) or not self.fatality_flows_NWM:
                self.fatality_flows_NWM = valid_flows
        elif self.hydrology == 'GEOGLOWS':
            if pd.isna(self.fatality_flows_GEOGLOWS) or not self.fatality_flows_GEOGLOWS:
                self.fatality_flows_GEOGLOWS = valid_flows

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir

    def set_streamflow_source(self, hydrology: str):
        self.hydrology = hydrology

    def set_flowline_source(self, hydrography: str):
        self.hydrography = hydrography

    def assign_reach(self, stream_reach):
        self.dam_reach = stream_reach

    def __repr__(self):
        return f"<Dam ID={self.ID} Name='{self.name}'>"
