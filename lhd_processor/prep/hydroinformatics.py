import os
import fiona
import geoglows
import numpy as np
import pandas as pd
import geopandas as gpd
from pygeohydro import NWIS
import hydrosignatures as hs
import matplotlib.pyplot as plt
from shapely.geometry import Point
from typing import List, Optional, Any


def haversine(lat1, lon1, lat2, lon2):
    """
        computes approximate distance (km) between two points.
    """
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0  # Earth radius in km
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _calculate_fdc(flow_data):
    """
        helper function to calculate FDC.
    """
    flow_data = flow_data.dropna()
    sorted_flows = np.sort(flow_data)[::-1]
    ranks = np.arange(1, len(sorted_flows) + 1)
    exceedance = 100 * ranks / (len(sorted_flows) + 1)
    return exceedance, sorted_flows


class StreamReachBase:
    """
    This class represents the base for a stream reach.

    It holds the core initialization and data-agnostic methods
    for streamflow analysis and plotting.
    """

    def __init__(self, lhd_id, latitude, longitude, data_sources, geoglows_streams=None,
                 nwm_ds=None, streamflow=True):
        self.id = lhd_id
        self.latitude = latitude
        self.longitude = longitude
        self.data_sources = data_sources
        self.geoglows_streams_path = geoglows_streams
        self.nwm_ds = nwm_ds
        self.fetch_streamflow = streamflow

        # Private attributes for caching results
        self._geoglows_df: Optional[pd.DataFrame] = None
        self._nwm_df: Optional[pd.DataFrame] = None
        self._reach_id: Optional[Any] = None
        self._linkno: Optional[Any] = None
        self._site_no: Optional[Any] = None  # Note: USGS logic is not fully implemented in source


    # --- STUB PROPERTIES FOR LINTER ---
    # These are overridden by the mixin classes but satisfy the linter
    # so that methods in this base class can use them.
    @property
    def geoglows_df(self) -> Optional[pd.DataFrame]:
        """Property to be overridden by mixin."""
        print("Warning: geoglows_df property not overridden by mixin.")
        return None

    @property
    def nwm_df(self) -> Optional[pd.DataFrame]:
        """Property to be overridden by mixin."""
        print("Warning: nwm_df property not overridden by mixin.")
        return None

    @property
    def reach_id(self) -> Optional[Any]:
        """Property to be overridden by NWM mixin."""
        print("Warning: reach_id property not overridden by mixin.")
        return None

    @property
    def linkno(self) -> Optional[Any]:
        """Property to be overridden by GEOGLOWS mixin."""
        print("Warning: linkno property not overridden by mixin.")
        return None
    # --- END STUB PROPERTIES ---

    def find_nearest_usgs_site(self):
        """Finds the nearest USGS gauge and its metadata."""
        nwis = NWIS()
        # Search for gauges within a 5km radius
        sites = nwis.get_info({"location": f"{self.longitude}, {self.latitude}", "dist": 5})
        if not sites.empty:
            self._site_no = sites.site_no.iloc[0]
            return self._site_no
        return None

    def _get_source_df(self, source) -> Optional[pd.DataFrame]:
        """Helper to get the correct DataFrame based on the source string."""
        if source == "GEOGLOWS":
            # This 'geoglows_df' is a property defined in the GEOGLOWS mixin
            return self.geoglows_df
        elif source == "National Water Model":
            # This 'nwm_df' is a property defined in the NWM mixin
            return self.nwm_df
        else:
            raise ValueError(f"Unknown source: {source}")

    def get_flow_on_date(self, target_date, source) -> float:
        """More efficient method to get flow on a specific date, returning reasons for failure."""
        df = self._get_source_df(source)
        if df is None or df.empty:
            print(f"Dam {self.id}: No data available for source: {source}")
            return np.nan

        # Convert target_date to datetime.date for comparison
        try:
            target_dt = pd.to_datetime(target_date).date()
        except ValueError:
            print(f"Dam {self.id}: Invalid target date format for {target_date}")
            return np.nan

        # Check if the date is within the range of the DataFrame index
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        if not (min_date <= target_dt <= max_date):
            print(f"Dam {self.id}: Date {target_dt} out of range ({min_date} to {max_date})")
            return np.nan

        # Look for the specific date
        match = df[df.index.date == target_dt]

        if not match.empty:
            # Check if the flow value itself is valid (e.g., not NaN)
            flow_value = match.iloc[0]['flow_cms']
            if pd.notna(flow_value):
                return float(flow_value)  # Success! Return the flow
            else:
                print(f"Dam {self.id}: Data exists for {target_dt}, but flow value is missing (NaN)")
                return np.nan
        else:
            # This case might be less common if the date range check passes,
            # but could happen if there are gaps in daily data within the range.
            print(f"Dam {self.id}: No data found for specific date {target_dt} within the range")
            return np.nan

    def get_median_flow(self, source: str) -> float:
        """
        Calculates the median flow for the entire period of record for a given source.
        """
        df = self._get_source_df(source)
        if df is not None and not df.empty:
            return float(df['flow_cms'].median())
        return np.nan

    def get_median_flow_in_range(self, start_date, end_date, source) -> float:
        """More efficient method to get median flow in a date range."""
        df = self._get_source_df(source)
        if df is None: return np.nan

        filtered_df = df.loc[start_date:end_date]
        return float(filtered_df['flow_cms'].median()) if not filtered_df.empty else np.nan

    def get_2yr_return_period(self, source: str) -> float:
        """
        Calculates the 2-year return period discharge (Q2) using the
        Annual Maximum Series method.

        The 2-year flood is defined as the flow magnitude exceeded with
        50% probability in any given year, which corresponds to the
        median of the annual maximums.
        """
        df = self._get_source_df(source)
        if df is None or df.empty:
            print(f"Dam {self.id}: No data available for source: {source} to calc Q2")
            return np.nan

        # 1. Check for sufficient data (at least 2 full years recommended)
        time_span_days = (df.index.max() - df.index.min()).days
        if time_span_days < 365 * 2:
            print(f"Dam {self.id}: Insufficient data length ({time_span_days} days) for return period analysis.")
            return np.nan

        # 2. Resample to Water Years (Oct 1 start) to avoid splitting winter seasons
        # 'AS-OCT' = Annual Start in October
        # noinspection PyBroadException
        try:
            annual_max_series = df['flow_cms'].resample('YS-OCT').max()
        except Exception:
            # Fallback for older pandas versions or non-datetime indices
            annual_max_series = df['flow_cms'].resample('A').max()

        if annual_max_series.empty:
            return np.nan

        # 3. Calculate Q2 (Median of Annual Maxima)
        q2 = annual_max_series.median()

        return float(q2)

    def plot_hydrographs(self):
        """Plots a hydrograph for each available data source."""
        for source in self.data_sources:
            df = self._get_source_df(source)
            if df is not None and not df.empty:
                comid = None
                if source == "GEOGLOWS":
                    comid = self.linkno  # <-- FIX: Use public property
                elif source == "USGS":
                    comid = self._site_no  # (This one has no property)
                elif source == "National Water Model":
                    comid = self.reach_id  # <-- Use public property

                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['flow_cms'], linewidth=1)
                plt.title(f"Streamflow Hydrograph - {source} ID {comid}")
                plt.xlabel("Date")
                plt.ylabel("Streamflow (m³/s)")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    def plot_fdcs(self):
        """Plots a Flow Duration Curve for each available data source on a single plot."""
        plt.figure(figsize=(10, 6))
        for source in self.data_sources:
            df = self._get_source_df(source)
            if df is not None and not df.empty:
                comid = None
                if source == "GEOGLOWS":
                    comid = self.linkno  # <-- FIX: Use public property
                elif source == "USGS":
                    comid = self._site_no  # (This one has no property)
                elif source == "National Water Model":
                    comid = self.reach_id  # <-- Use public property

                flow_data = df['flow_cms'].dropna()
                exceedance, sorted_flows = _calculate_fdc(flow_data)

                plt.plot(exceedance, sorted_flows, label=f"{source} ({comid})")

        plt.yscale('log')
        plt.title(f"Flow Duration Curve Comparison (LHD No. {self.id})")
        plt.xlabel("Exceedance Probability (%)")
        plt.ylabel("Streamflow (m³/s)")
        plt.grid(True, which="both", linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def export_fdcs(self) -> dict:
        """Uses HydroSignatures to calculate FDCs."""
        fdc_results = {}
        for source in self.data_sources:
            df = self._get_source_df(source)
            if df is not None and not df.empty:
                # HydroSignatures returns a clean FDC DataFrame
                fdc_df = hs.compute_fdc(df["flow_cms"])
                fdc_results[source] = fdc_df.to_dict()
        return fdc_results


    def __repr__(self):
        # Accessing _geoglows_df directly to avoid property call in repr
        return (f"Hi, I'm a stream reach\n"
                f"I have the flows: {self._geoglows_df}")


class StreamReachGEOGLOWS:
    """
    Mixin class for handling GEOGLOWS data.
    Provides GEOGLOWS-specific properties and loading methods.
    """
    # --- Hints for Linter ---
    # These attributes are expected to be defined by StreamReachBase
    _linkno: Optional[Any] = None
    _geoglows_df: Optional[pd.DataFrame] = None
    data_sources: List[str] = []
    fetch_streamflow: bool = True
    geoglows_streams_path: Optional[str] = None
    id: int = 0
    longitude: float = 0.0
    latitude: float = 0.0

    # --- End Linter Hints ---

    @property
    def linkno(self) -> Optional[Any]:
        """Lazy-loads the GEOGLOWS link number."""
        if self._linkno is None:
            self._load_geoglows_reach()
        return self._linkno

    @property
    def geoglows_df(self) -> Optional[pd.DataFrame]:
        """Lazy-loads and caches the GEOGLOWS streamflow DataFrame."""
        if self._geoglows_df is None and "GEOGLOWS" in self.data_sources and self.fetch_streamflow:
            self._geoglows_df = self._load_geoglows_flow()
        return self._geoglows_df

    def _load_geoglows_reach(self, force_reload=True):
        """Finds the nearest GEOGLOWS reach and caches its ID and geometry."""
        if self._linkno and not force_reload:
            return

        if not self.geoglows_streams_path:
            print(f"Dam {self.id}: Path to GEOGLOWS streams not provided. Cannot find linkno.")
            return

        print(f"DEBUG: Dam {self.id} is opening GPKG for LINKNO search: {self.geoglows_streams_path}")
        # === 1. Get the target file's CRS ===
        try:
            with fiona.open(self.geoglows_streams_path) as src:
                target_crs = src.crs
        except Exception as e:
            print(f"CRITICAL ERROR: Could not read CRS from {self.geoglows_streams_path}. Error: {e}")
            return

        # === 2. Create your dam point in Lat/Lon (EPSG:4326) ===
        dam_point_latlon = Point(self.longitude, self.latitude)

        # Create a GeoSeries/GeoDataFrame for the point so we can .to_crs() it
        dam_point_gdf = gpd.GeoSeries([dam_point_latlon], crs="EPSG:4326")

        # === 3. Transform the point to the file's CRS ===
        dam_point_projected_gdf = dam_point_gdf.to_crs(target_crs)
        dam_point_projected = dam_point_projected_gdf.iloc[0]

        # === 4. Create your search buffer *in the projected CRS* ===
        # Let's search in a 200-meter radius.
        search_buffer = dam_point_projected.buffer(200)  # 2000 meters

        # === 5. Read the file using the buffer as the filter ===
        # This is the "spatial query"
        gdf = gpd.read_file(self.geoglows_streams_path,
                            bbox=search_buffer.bounds  # Use the buffer's bounds for the fast read
                            )

        if gdf.empty:
            print(f"WARNING: No streams found within 2000m of the dam.")
            return

        # === 6. Proceed with your logic ===

        # This is a *more accurate* filter. The bbox read is fast but
        # might include streams in the corners. This filters to the circle.
        streams_inside_buffer = gdf[gdf.geometry.intersects(search_buffer)]

        if streams_inside_buffer.empty:
            print(f"WARNING: No streams intersected the 2000m buffer.")
            return

        # Calculate distance using the *projected* point
        streams_inside_buffer["distance"] = streams_inside_buffer.geometry.distance(dam_point_projected)

        # Identify the correct column name for Stream Order (case-insensitive)
        order_col = next((c for c in streams_inside_buffer.columns \
                          if c.lower() in ['strmorder', 'streamorder', 'order']),
                         None)

        if order_col:
            max_strm_order = streams_inside_buffer[order_col].max()
            highest_order_streams = streams_inside_buffer[streams_inside_buffer[order_col] == max_strm_order]
        else:
            print(f"Warning: No 'strmOrder' column found. Defaulting to nearest stream.")
            highest_order_streams = streams_inside_buffer

        if highest_order_streams.empty:
            print(f"WARNING: No 'highest_order_streams' found.")
            return

        nearest = highest_order_streams.loc[highest_order_streams['distance'].idxmin()]
        self._linkno = nearest["LINKNO"]

    def _load_geoglows_flow(self) -> Optional[pd.DataFrame]:
        """Fetches and processes GEOGLOWS streamflow data."""
        # Just access the property. This triggers the getter method.
        linkno_val = self.linkno
        if linkno_val is None:
            print(f"Dam {self.id}: Could not find GEOGLOWS linkno. Skipping flow.")
            return None

        # Use the retrieved value
        df = geoglows.data.retrospective(river_id=linkno_val, bias_corrected=True)

        df.index = pd.to_datetime(df.index)

        # You were also using self._linkno in the rename, which might be None
        # if it was just loaded. Use the local variable 'linkno_val' instead.
        df = df.reset_index().rename(columns={"index": "time", int(linkno_val): "flow_cms"})

        df = df.sort_values('time')
        df = df[df['flow_cms'] >= 0]
        return df.set_index('time')


class StreamReachNWM:
    """
    Mixin class for handling National Water Model (NWM) data.
    Provides NWM-specific properties and loading methods.
    """
    # --- Hints for Linter ---
    # These attributes are expected to be defined by StreamReachBase
    _reach_id: Optional[Any] = None
    _nwm_df: Optional[pd.DataFrame] = None
    data_sources: List[str] = []
    fetch_streamflow: bool = True
    nwm_ds: Optional[Any] = None  # Assuming nwm_ds is an xarray.Dataset
    id: int = 0
    longitude: float = 0.0
    latitude: float = 0.0
    # --- End Linter Hints ---

    @property
    def reach_id(self) -> Optional[Any]:
        """Lazy-loads the NWM reach ID."""
        if self._reach_id is None:
            self._load_nwm_reach()
        return self._reach_id

    @property
    def nwm_df(self) -> Optional[pd.DataFrame]:
        """Lazy-loads and caches the NWM streamflow DataFrame."""
        if self._nwm_df is None and "National Water Model" in self.data_sources and self.fetch_streamflow:
            self._nwm_df = self._load_nwm_flow()
        return self._nwm_df

    def _load_nwm_reach(self, force_reload=True):
        """
        Finds the nearest NWM reach ID and point geometry from the
        local nwm_ds (xarray dataset loaded from Parquet).
        """
        if self._reach_id and not force_reload:
            return

        if self.nwm_ds is None:
            print(f"Dam {self.id}: NWM dataset (nwm_ds) must be provided to StreamReach.")
            return

        # Get coordinates and feature_ids from the xarray dataset
        try:
            # feature_id is a 1D coordinate
            feature_ids = self.nwm_ds.feature_id.values
            lats = self.nwm_ds.latitude.values[:, 0]
            lons = self.nwm_ds.longitude.values[:, 0]

        except AttributeError:
            print("ERROR: nwm_ds is missing required coordinates 'latitude', 'longitude', or 'feature_id'.")
            return
        except IndexError:
            print("ERROR: nwm_ds seems to have an empty time dimension. Cannot get coordinates.")
            return

        # Calculate haversine distance from the dam to all points in the dataset
        distances = [haversine(self.latitude, self.longitude, lat, lon) for lat, lon in zip(lats, lons)]

        # Find the index of the closest NWM reach
        min_dist_idx = np.argmin(distances)
        # Get the feature_id (reach_id) for that closest reach
        self._reach_id = feature_ids[min_dist_idx]

    def _load_nwm_flow(self) -> Optional[pd.DataFrame]:
        """Fetches and processes NWM streamflow data."""
        if self.nwm_ds is None:
            print(f"Dam {self.id}: NWM dataset (nwm_ds) must be provided for NWM streamflow.")
            return None

        reach_id_val = self.reach_id
        if reach_id_val is None:
            print(f"Dam {self.id}: Could not find NWM reach_id. Skipping flow.")
            return None

        # This logic remains the same, as nwm_ds is the xarray dataset
        stream = self.nwm_ds['streamflow'].sel(feature_id=int(reach_id_val), time=slice("1979-02-01", None))
        df = stream.compute().to_dataframe().rename(columns={"streamflow": "flow_cms"})
        df.index = pd.to_datetime(df.index)
        df = df[df['flow_cms'] >= 0]
        return df


class StreamReach(StreamReachGEOGLOWS, StreamReachNWM, StreamReachBase):
    """
    This class represents a stream reach.

    It inherits functionality from a Base class and mixins for
    GEOGLOWS and NWM data sources.
    """

    def __init__(self, lhd_id, latitude, longitude, data_sources, geoglows_streams=None,
                 nwm_ds=None, streamflow=True):
        # Call the base class __init__
        super().__init__(lhd_id, latitude, longitude, data_sources, geoglows_streams,
                         nwm_ds, streamflow)


def create_multilayer_gpkg(gdfs: List[gpd.GeoDataFrame],
                           output_path: str,
                           layer_names: Optional[List[str]] = None
                           ) -> None:
    """
        Saves a list of GeoDataFrames to a single GeoPackage file, with each
        GeoDataFrame as its own layer.
    """
    # --- 1. Input Validation ---
    if not output_path.lower().endswith('.gpkg'):
        raise ValueError("Output file path must end with '.gpkg'")

    if not gdfs:
        raise ValueError("The list of GeoDataFrames cannot be empty.")

    if layer_names:
        if len(gdfs) != len(layer_names):
            raise ValueError("The number of layer names must match the number of GeoDataFrames.")
    else:
        # Create default layer names if none are provided
        layer_names = [f"layer_{i + 1}" for i in range(len(gdfs))]

    # --- 2. Remove Existing File
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file at: {output_path}")

    # --- 3. Write Each GeoDataFrame to a Layer ---
    print(f"Creating GeoPackage at: {output_path}")
    for gdf, layer_name in zip(gdfs, layer_names):
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
            print(f"Warning: Skipping empty or invalid GeoDataFrame for layer '{layer_name}'.")
            continue

        try:
            gdf.to_file(filename=output_path,
                        driver="GPKG",
                        layer=layer_name)
            print(f"  - Successfully wrote layer: '{layer_name}'")
        except Exception as e:
            print(f"  - Failed to write layer '{layer_name}'. Error: {e}")

    print("GeoPackage creation complete.")


def create_gpkg_from_lists(
        lists_of_gdfs: List[List[gpd.GeoDataFrame]],
        output_path: str,
        layer_names: Optional[List[str]] = None
) -> None:
    """
        Combines lists of GeoDataFrames and saves each combined list as a
        separate layer in a single GeoPackage file.
    """
    merged_gdfs = []
    print("--- Merging lists of GeoDataFrames ---")
    for i, gdf_list in enumerate(lists_of_gdfs):
        if not gdf_list:
            print(f"Warning: Inner list at index {i} is empty, skipping.")
            continue

        # Ensure all elements are GeoDataFrames before concatenating
        valid_gdfs = [g for g in gdf_list if isinstance(g, gpd.GeoDataFrame) and not g.empty]
        if not valid_gdfs:
            print(f"Warning: Inner list at index {i} contains no valid GeoDataFrames, skipping.")
            continue

        # We assume all gdfs in a list share the same CRS and take it from the first one.
        crs = valid_gdfs[0].crs
        # Use pandas.concat to merge the list of GeoDataFrames
        merged_gdf = gpd.GeoDataFrame(
            pd.concat(valid_gdfs, ignore_index=True), crs=crs
        )
        merged_gdfs.append(merged_gdf)
        print(f"Merged list {i + 1} into a single GeoDataFrame with {len(merged_gdf)} features.")

    # Now call the original function with the list of merged GeoDataFrames
    create_multilayer_gpkg(merged_gdfs, output_path, layer_names)
