import gc
import os
import re
import datetime
import geoglows
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from typing import Dict, Any, Optional, List

from .download_geospatial_data import (download_dem,
                                       download_nhd_flowline,
                                       download_tdx_flowline,
                                       download_land_raster,
                                       find_water_gpstime)

try:
    import gdal
    import gdal_array
except ImportError:
    from osgeo import gdal, ogr, osr, gdal_array


# helper functions to rasterize the flowlines
def read_raster_w_gdal(input_raster: str):
    """Reads raster data into an array and returns spatial metadata."""
    dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray()
    ncols, nrows = dataset.RasterXSize, dataset.RasterYSize
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * abs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0]
    xur = xll + ncols * geotransform[1]
    lat = abs((yll + yur) / 2.0)
    raster_proj = dataset.GetProjectionRef()
    del dataset, band
    return raster_array, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, raster_proj


def write_output_raster(s_output_filename, raster_data, dem_geotransform, dem_projection):
    """Writes a numpy array to a GeoTIFF file."""
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(raster_data.dtype) or gdal.GDT_Float32
    n_rows, n_cols = raster_data.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(s_output_filename, xsize=n_cols, ysize=n_rows, bands=1, eType=gdal_dtype)
    ds.SetGeoTransform(dem_geotransform)
    ds.SetProjection(dem_projection)
    ds.GetRasterBand(1).WriteArray(raster_data)
    ds.FlushCache()
    del ds


def clean_strm_raster(strm_tif: str, clean_strm_tif: str) -> None:
    """Robust filter to remove single cells and redundant thickness for ARC compatibility."""
    (SN, ncols, nrows, cellsize, yll, yur, xll, xur, lat, gt, proj) = read_raster_w_gdal(strm_tif)
    B = np.zeros((nrows + 2, ncols + 2))
    B[1:nrows + 1, 1:ncols + 1] = np.where(SN > 0, SN, 0)

    # Free memory of original raster array
    del SN
    gc.collect()

    (RR, CC) = np.where(B > 0)
    num_nonzero = len(RR)

    for filterpass in range(2):
        for x in range(num_nonzero):
            r, c = RR[x], CC[x]
            if B[r, c] > 0:
                if B[r, c + 1] == 0 and B[r, c - 1] == 0:
                    if (B[r + 1, c - 1:c + 2].sum() == 0 and B[r - 1, c] > 0) or \
                            (B[r - 1, c - 1:c + 2].sum() == 0 and B[r + 1, c] > 0):
                        B[r, c] = 0
                elif B[r + 1, c] == B[r, c] and (B[r + 1, c + 1] == B[r, c] or B[r + 1, c - 1] == B[r, c]):
                    if sum(B[r + 1, c - 1:c + 2]) == B[r, c] * 2:
                        B[r + 1, c] = 0
    write_output_raster(clean_strm_tif, B[1:nrows + 1, 1:ncols + 1], gt, proj)


def create_arc_strm_raster(StrmSHP, output_raster_path, DEM_File, value_field):
    """
        Rasterizes a shapefile to match the extent and resolution of a DEM.
    """
    gdf = gpd.read_file(StrmSHP)
    with rasterio.open(DEM_File) as ref:
        meta, ref_transform, out_shape, crs = ref.meta.copy(), ref.transform, (ref.height, ref.width), ref.crs
    if gdf.crs != crs: gdf = gdf.to_crs(crs)
    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[value_field])]
    raster = rasterize(shapes=shapes, out_shape=out_shape, fill=0, transform=ref_transform, dtype='int32')
    meta.update({"driver": "GTiff", "dtype": "int32", "count": 1, "nodata": 0})
    with rasterio.open(output_raster_path, 'w', **meta) as dst: dst.write(raster, 1)


class LowHeadDam:
    """
    Represents a low-head dam during the data preparation phase.
    Handles geospatial data assignment and streamflow retrieval.
    """

    def __init__(self, site_id: int, db_manager):
        self.site_id: int = int(site_id)
        self.db = db_manager
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

        # Results container for multi-source support
        self.results: Dict[str, pd.DataFrame] = {}
        for source in ['National Water Model', 'GEOGLOWS']:
            if hasattr(self.db, 'get_site_results'):
                if source == 'National Water Model':
                    fl_source_for_call = 'NHDPlus'
                else:
                    fl_source_for_call = 'TDX-Hydro'

                self.results[source] = self.db.get_site_results(self.site_id, fl_source_for_call, source)
            else:
                self.results[source] = pd.DataFrame()

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
                # geoglows returns a dataframe with datetime index
                # If date is 'YYYY-MM-DD', it might return one or more rows (if hourly)
                # But retrospective is usually daily or hourly.
                # Let's assume daily for now or take mean if multiple
                if not df.empty:
                    # The column name varies, so take the first column
                    val = float(df.iloc[:, 0].mean())
                    return val
                return None
            except Exception as e:
                print(f"GEOGLOWS API Error (Dam {self.site_id}): {e}")
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
                        val = ds['streamflow'].sel(feature_id=int(self.nwm_id), time=date).values
                        return float(val)
                    except Exception:
                        # Fallback if specific date/ID not found
                        return 0.0
                else:
                    print(f"Warning: Zarr file NOT FOUND at {zarr_path}")
                    return 0.0
            except Exception as e:
                print(f"NWM Zarr Error (Dam {self.site_id}): {e}")
                return 0.0
        else:
            # Only print if we expected to find something but didn't have IDs
            if source:
                print(f"Warning: Missing ID for {source} (Dam {self.site_id})")
            return 0.0


    def get_median_streamflow(self, source: Optional[str] = None) -> float:
        source = source or self.streamflow_source

        if source == 'GEOGLOWS' and self.geoglows_id:
            try:
                df = geoglows.data.retrospective(river_id=int(self.geoglows_id))
                if not df.empty:
                    return float(df.iloc[:, 0].median())
                return 0.0
            except Exception as e:
                print(f"GEOGLOWS Median Error (Dam {self.site_id}): {e}")
                return 0.0

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
        # Check if baseflow is already recorded
        target_col = 'baseflow_nwm' if self.streamflow_source == 'National Water Model' else 'baseflow_geo'
        existing_val = self.site_data.get(target_col)

        if pd.notna(existing_val) and float(existing_val) != 0.0:
            return

        found_date = None
        if "lidar date" in baseflow_method.lower():
            existing_lidar_date = self.site_data.get('lidar_date')
            if existing_lidar_date:
                found_date = existing_lidar_date
            else:
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
        source = source or self.streamflow_source

        if self.incidents_df.empty:
            return

        flow_col = 'flow_nwm' if source == 'National Water Model' else 'flow_geo'

        for idx, row in self.incidents_df.iterrows():
            existing_val = row.get(flow_col)
            if pd.notna(existing_val) and float(existing_val) != 0.0:
                continue

            incident_date = str(row['date'])
            if ' ' in incident_date:
                incident_date = incident_date.split(' ')[0]

            flow_val = self.get_streamflow(incident_date, source=source)

            if flow_val is not None:
                self.incidents_df.at[idx, flow_col] = float(flow_val)
            else:
                print(f"Warning: Could not find {source} flow for Dam {self.site_id} on {incident_date}")

    # --- GEOSPATIAL ASSIGNMENT ---

    def _try_load_existing_flowline(self):
        path = None
        src = str(self.flowline_source) if self.flowline_source else ""

        # 1. Check if source is actually a path (User error or legacy data)
        if src and (os.sep in src or '/' in src) and os.path.exists(src):
            path = src
        
        # 2. Standard Check
        elif src == 'NHDPlus':
            path = self.site_data.get('flowline_path_nhd')
        elif src in ['TDX-Hydro', 'GEOGLOWS']:
            path = self.site_data.get('flowline_path_tdx')
            
        # 3. Fuzzy Check (if source is "nhd_flowline_..." or similar but not a full path, or just "NHD")
        if not path and src:
            if 'nhd' in src.lower():
                path = self.site_data.get('flowline_path_nhd')
            elif 'tdx' in src.lower():
                path = self.site_data.get('flowline_path_tdx')

        if path and os.path.exists(path):
            try:
                self.flowline_gdf = gpd.read_file(path)
            except Exception as e:
                print(f"Dam {self.site_id}: Failed to load existing flowline: {e}")

    def assign_dem(self, dem_dir: str, resolution: int):
        if self.flowline_gdf is None:
            self._try_load_existing_flowline()

        if self.flowline_gdf is None:
            print(f"Dam {self.site_id}: Skipping DEM assignment (Missing Flowlines).")
            return

        path, res, project_name = download_dem(self.site_id, self.flowline_gdf, dem_dir, resolution)

        if path:
            self.dem_path = path
            self.res_meters = res
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
        found_ids = []

        if self.flowline_source == 'NHDPlus' or self.streamflow_source == 'National Water Model':
            path_nhd = self.site_data.get('flowline_path_nhd')
            gdf = None

            if path_nhd and os.path.exists(path_nhd):
                try:
                    gdf = gpd.read_file(path_nhd)
                except Exception as e:
                    print(f"Dam {self.site_id}: Failed to read existing NHD flowline ({e}). Redownloading...")
                    path_nhd = None

            if not path_nhd:
                path_nhd, gdf = download_nhd_flowline(self.latitude, self.longitude, flowline_dir)

            if path_nhd:
                self.site_data['flowline_path_nhd'] = path_nhd
                if self.flowline_source == 'NHDPlus':
                    self.flowline_gdf = gdf

                filename = os.path.basename(path_nhd)
                match = re.search(r'nhd_flowline_(\d+)', filename)
                if match:
                    self.nwm_id = int(match.group(1))
                elif gdf is not None and not gdf.empty:
                    self.nwm_id = int(gdf.iloc[0]['nhdplusid'])
                self.site_data['reach_id'] = self.nwm_id

                if gdf is not None and 'nhdplusid' in gdf.columns:
                    found_ids.extend(gdf['nhdplusid'].astype(int).tolist())

        if self.flowline_source == 'TDX-Hydro' or self.streamflow_source == 'GEOGLOWS':
            path_tdx = self.site_data.get('flowline_path_tdx')
            gdf = None

            if path_tdx and os.path.exists(path_tdx):
                try:
                    gdf = gpd.read_file(path_tdx)
                except Exception as e:
                    print(f"Dam {self.site_id}: Failed to read existing TDX flowline ({e}). Redownloading...")
                    path_tdx = None

            if not path_tdx:
                path_tdx, gdf = download_tdx_flowline(self.latitude, self.longitude, flowline_dir, vpu_gpkg, site_id=self.site_id)

            if path_tdx:
                self.site_data['flowline_path_tdx'] = path_tdx
                if self.flowline_source in ['TDX-Hydro', 'GEOGLOWS']:
                    self.flowline_gdf = gdf

                filename = os.path.basename(path_tdx)
                match = re.search(r'tdx_flowline_(\d+)', filename)
                if match:
                    self.geoglows_id = int(match.group(1))
                elif gdf is not None and not gdf.empty:
                    self.geoglows_id = int(gdf.iloc[0]['LINKNO'])

                self.site_data['linkno'] = self.geoglows_id

                if gdf is not None and 'LINKNO' in gdf.columns:
                    found_ids.extend(gdf['LINKNO'].astype(int).tolist())

        return list(set(found_ids))


    def assign_land(self, land_dir: str):
        if self.dem_path:
            land_raster = download_land_raster(self.site_id, str(self.dem_path), land_dir)
            self.land_cover_path = land_raster
            self.site_data['land_raster'] = land_raster
            
    def generate_stream_raster(self):
        """Generates the clean stream raster required for ARC."""
        if not self.dem_path or not os.path.exists(self.dem_path):
            print(f"Dam {self.site_id}: Skipping stream raster generation (Missing DEM).")
            return

        # Determine flowline path, ID field, and ID value
        flowline_path = None
        rivid_field = None
        identifier = None
        prefix = None

        if self.flowline_source == 'TDX-Hydro':
            flowline_path = self.site_data.get('flowline_path_tdx')
            rivid_field = 'LINKNO'
            identifier = self.geoglows_id
            prefix = 'geo'
        else:
            flowline_path = self.site_data.get('flowline_path_nhd')
            rivid_field = 'nhdplusid'
            identifier = self.nwm_id
            prefix = 'nhd'

        if not flowline_path or not os.path.exists(flowline_path):
             print(f"Dam {self.site_id}: Skipping stream raster generation (Missing Flowline).")
             return
        
        if not identifier:
             print(f"Dam {self.site_id}: Skipping stream raster generation (Missing ID).")
             return

        # Output directory is the same as flowline directory
        strm_dir = os.path.dirname(flowline_path)
        
        # Naming convention: prefix_{id}_clean.tif
        clean_strm_tif = os.path.join(strm_dir, f'{prefix}_{identifier}_clean.tif')
        raw_strm_tif = clean_strm_tif.replace('_clean.tif', '.tif')

        print(f"Dam {self.site_id}: Generating stream raster at {clean_strm_tif}...")
        try:
            create_arc_strm_raster(flowline_path, raw_strm_tif, self.dem_path, rivid_field)
            clean_strm_raster(raw_strm_tif, clean_strm_tif)
            self.site_data['strm_tif_clean'] = clean_strm_tif
        except Exception as e:
            print(f"Dam {self.site_id}: Error generating stream raster: {e}")


    def save_changes(self):
        self.site_data['reach_id'] = self.nwm_id
        self.site_data['linkno'] = self.geoglows_id
        
        # Ensure sources are updated in the dictionary so they get saved to DB
        self.site_data['flowline_source'] = self.flowline_source
        self.site_data['streamflow_source'] = self.streamflow_source
        
        # Ensure land raster path is saved
        self.site_data['land_raster'] = self.land_cover_path

        self.db.update_site_data(self.site_id, self.site_data)

        if not self.incidents_df.empty:
            self.db.update_site_incidents(self.site_id, self.incidents_df)

        if hasattr(self.db, 'update_site_results'):
            for source, df in self.results.items():
                if not df.empty:
                    # Same issue here: update_site_results needs flowline_source too.
                    # We need to be consistent with what we did in __init__
                    
                    if source == 'National Water Model':
                        fl_source_for_call = 'NHDPlus'
                    else:
                        fl_source_for_call = 'TDX-Hydro'
                        
                    self.db.update_site_results(self.site_id, df, fl_source_for_call, source)

        self.db.save()
