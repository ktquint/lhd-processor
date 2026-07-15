import gc
import os
import re
import datetime
import geoglows
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
from typing import Dict, Any, Optional, List

from .download_geospatial_data import (download_dem,
                                       download_nhd_flowline,
                                       download_tdx_flowline,
                                       make_constant_land_raster,
                                       find_water_gpstime)
from .nwm_api import load_api_key, fetch_retrospective_daily_mean, fetch_percentile_flows

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

        # Per-flowline-source DEM/land raster (each is built to fit that
        # source's flowline extent, so NHDPlus and TDX-Hydro get their own).
        self.dem_path_nhd = self.site_data.get('dem_path_nhd')
        self.dem_path_tdx = self.site_data.get('dem_path_tdx')
        self.land_raster_nhd = self.site_data.get('land_raster_nhd')
        self.land_raster_tdx = self.site_data.get('land_raster_tdx')

        # Ensure flowline rasters are loaded from DB if available
        self.flowline_raster_nhd = self.site_data.get('flowline_raster_nhd')
        self.flowline_raster_tdx = self.site_data.get('flowline_raster_tdx')

        self.flowline_path_nhd = self.site_data.get('flowline_path_nhd')
        self.flowline_path_tdx = self.site_data.get('flowline_path_tdx')

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
            api_key = load_api_key()
            if not api_key:
                print(f"Warning: NWM_API_KEY not set (Dam {self.site_id})")
                return 0.0
            val = fetch_retrospective_daily_mean(int(self.nwm_id), date, api_key)
            return float(val) if val is not None else 0.0
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
            api_key = load_api_key()
            if not api_key:
                return 0.0
            flows = fetch_percentile_flows([int(self.nwm_id)], api_key)
            median_val = flows.get(int(self.nwm_id), {}).get(50)
            return float(median_val) if median_val is not None else 0.0
        else:
            return 0.0

    def est_dem_baseflow(self, baseflow_method: str):
        # Check if baseflow is already recorded
        is_nwm = self.streamflow_source == 'National Water Model'
        target_col = 'baseflow_nwm' if is_nwm else 'baseflow_geo'
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

        flow_val = 0.0
        if found_date:
            self.site_data['lidar_date'] = found_date
            flow_val = self.get_streamflow(found_date, self.streamflow_source)
        else:
            flow_val = self.get_median_streamflow(self.streamflow_source)

        self.site_data[target_col] = flow_val
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

    def _dem_land_key(self) -> str:
        """Which per-source DEM/land-raster slot applies to the active
        flowline_source. Mirrors ArcDam's own flowline_source == 'TDX-Hydro'
        check for picking flowline_raster_nhd vs flowline_raster_tdx -- 'Both'
        and anything else default to 'nhd', same as there."""
        return 'tdx' if self.flowline_source == 'TDX-Hydro' else 'nhd'

    def _try_load_existing_flowline(self):
        path = None
        src = str(self.flowline_source) if self.flowline_source else ""

        # 1. Check if source is actually a path (User error or legacy data)
        if src and (os.sep in src or '/' in src) and os.path.exists(src):
            path = src
        
        # 2. Standard Check
        elif src == 'NHDPlus' or src == 'Both':
            path = self.site_data.get('flowline_path_nhd')
            if not path and src == 'Both':
                path = self.site_data.get('flowline_path_tdx')
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
        # A dam prepped under both NHDPlus and TDX-Hydro has two DEMs, since
        # each is built to fit that source's own flowline extent (see
        # _dem_land_key) -- so check the per-source slot, not just the
        # generic column, which only ever remembers the most recent source.
        key = self._dem_land_key()
        keyed_col = f'dem_path_{key}'
        keyed_path = self.site_data.get(keyed_col)

        # Reuse an already-built merged/warped DEM instead of re-querying the
        # TNM API and re-warping every run. Check both the recorded path and
        # the deterministic on-disk path, in case site_data is stale or
        # missing (e.g. migrating a dam that predates the per-source split).
        expected_path = os.path.join(dem_dir, str(self.site_id), f"LHD_{self.site_id}_3DEP_DEM.tif")
        existing_path = None
        if keyed_path and os.path.exists(keyed_path):
            existing_path = keyed_path
        elif os.path.exists(expected_path):
            existing_path = expected_path

        if existing_path:
            print(f"Dam {self.site_id}: DEM already exists at {existing_path}, skipping download.")
            self.dem_path = existing_path
            self.site_data['dem_path'] = existing_path
            self.site_data[keyed_col] = existing_path
            return

        if self.flowline_gdf is None:
            self._try_load_existing_flowline()

        if self.flowline_gdf is None:
            print(f"Dam {self.site_id}: Skipping DEM assignment (Missing Flowlines).")
            return

        path, res, dem_meta = download_dem(self.site_id, self.flowline_gdf, dem_dir, resolution)

        if path:
            self.dem_path = path
            self.res_meters = res
            self.site_data['dem_path'] = path
            self.site_data[keyed_col] = path
            self.site_data['dem_resolution_m'] = res
            self.site_data['dem_source_info'] = dem_meta.get('dataset') or "USGS TNM API"
            self.site_data['dem_tile_count'] = dem_meta.get('tile_count')
            self.site_data['dem_tile_files'] = "; ".join(dem_meta.get('tile_files', []))

            projects = dem_meta.get('projects', [])
            project_name = "; ".join(projects)
            self.site_data['lidar_project'] = project_name

            # NOTE: this is the DEM product's USGS *publication* date, not the LiDAR
            # flight/acquisition date. Keep it separate from `lidar_date`, which
            # est_dem_baseflow() populates from the point-cloud GPS time (find_water_gpstime)
            # and treats as authoritative -- if present it skips that more precise lookup.
            pub_dates = dem_meta.get('publication_dates', [])
            self.site_data['dem_publication_date'] = "; ".join(pub_dates)

            year_match = re.search(r'(20\d{2}|FY\d{2})', project_name)
            if year_match:
                year_val = year_match.group(0)
                self.lidar_year = f"20{year_val[-2:]}" if "FY" in year_val else year_val
                self.site_data['lidar_year'] = self.lidar_year

    def assign_flowlines(self, flowline_dir: str, vpu_gpkg: str) -> List[int]:
        found_ids = []

        if self.flowline_source in ['NHDPlus', 'Both'] or self.streamflow_source in ['National Water Model', 'Both']:
            path_nhd = self.site_data.get('flowline_path_nhd')
            gdf = None

            if path_nhd and os.path.exists(path_nhd):
                try:
                    gdf = gpd.read_file(path_nhd)
                except Exception as e:
                    print(f"Dam {self.site_id}: Failed to read existing NHD flowline ({e}). Redownloading...")
                    path_nhd = None

            if not path_nhd:
                # Pass tuple (1, 2) for 1km upstream and 2km downstream
                # Pass site_id to create subdirectory
                path_nhd, gdf = download_nhd_flowline(self.latitude, self.longitude, flowline_dir, distance_km=(1, 2), site_id=self.site_id)

            if path_nhd:
                self.flowline_path_nhd = path_nhd
                self.site_data['flowline_path_nhd'] = path_nhd
                if self.flowline_source in ['NHDPlus', 'Both']:
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

        if self.flowline_source in ['TDX-Hydro', 'Both'] or self.streamflow_source in ['GEOGLOWS', 'Both']:
            path_tdx = self.site_data.get('flowline_path_tdx')
            gdf = None

            if path_tdx and os.path.exists(path_tdx):
                try:
                    gdf = gpd.read_file(path_tdx)
                except Exception as e:
                    print(f"Dam {self.site_id}: Failed to read existing TDX flowline ({e}). Redownloading...")
                    path_tdx = None

            if not path_tdx:
                # Pass tuple (1, 2) for 1km upstream and 2km downstream
                # Pass site_id to create subdirectory
                path_tdx, gdf = download_tdx_flowline(self.latitude, self.longitude, flowline_dir, vpu_gpkg, distance_km=(1, 2), site_id=self.site_id)

            if path_tdx:
                self.flowline_path_tdx = path_tdx
                self.site_data['flowline_path_tdx'] = path_tdx
                if self.flowline_source in ['TDX-Hydro', 'GEOGLOWS', 'Both']:
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
            land_raster = make_constant_land_raster(self.site_id, str(self.dem_path), land_dir)
            self.land_cover_path = land_raster
            self.site_data['land_raster'] = land_raster
            self.site_data[f'land_raster_{self._dem_land_key()}'] = land_raster
            
    def generate_stream_raster(self):
        """Generates the clean stream raster required for ARC."""
        if not self.dem_path or not os.path.exists(self.dem_path):
            print(f"Dam {self.site_id}: Skipping stream raster generation (Missing DEM).")
            return

        # Define configurations to process
        configs = []
        
        # Check NHD
        path_nhd = self.site_data.get('flowline_path_nhd')
        if path_nhd and os.path.exists(path_nhd) and self.nwm_id:
            configs.append({
                'source': 'NHDPlus',
                'path': path_nhd,
                'field': 'nhdplusid',
                'id': self.nwm_id,
                'prefix': 'nhd',
                'target_attr': 'flowline_raster_nhd'
            })
            
        # Check TDX
        path_tdx = self.site_data.get('flowline_path_tdx')
        if path_tdx and os.path.exists(path_tdx) and self.geoglows_id:
            configs.append({
                'source': 'TDX-Hydro',
                'path': path_tdx,
                'field': 'LINKNO',
                'id': self.geoglows_id,
                'prefix': 'tdx',
                'target_attr': 'flowline_raster_tdx'
            })

        if not configs:
             print(f"Dam {self.site_id}: Skipping stream raster generation (No valid flowlines/IDs found).")
             return

        for cfg in configs:
            strm_dir = os.path.dirname(cfg['path'])
            clean_strm_tif = os.path.join(strm_dir, f"{cfg['prefix']}_{int(cfg['id'])}_clean.tif")
            raw_strm_tif = clean_strm_tif.replace('_clean.tif', '.tif')

            if os.path.exists(clean_strm_tif):
                print(f"Dam {self.site_id}: {cfg['source']} stream raster already exists at {clean_strm_tif}, skipping regeneration.")
                setattr(self, cfg['target_attr'], clean_strm_tif)
                self.site_data[cfg['target_attr']] = clean_strm_tif
                continue

            print(f"Dam {self.site_id}: Generating {cfg['source']} stream raster at {clean_strm_tif}...")
            try:
                create_arc_strm_raster(cfg['path'], raw_strm_tif, self.dem_path, cfg['field'])
                clean_strm_raster(raw_strm_tif, clean_strm_tif)
                
                # Update instance and site_data
                setattr(self, cfg['target_attr'], clean_strm_tif)
                self.site_data[cfg['target_attr']] = clean_strm_tif
                
                # Delete the raw raster after cleaning
                if os.path.exists(raw_strm_tif):
                    try:
                        os.remove(raw_strm_tif)
                        print(f"Dam {self.site_id}: Deleted raw stream raster {raw_strm_tif}")
                    except Exception as e:
                        print(f"Dam {self.site_id}: Warning: Could not delete raw stream raster: {e}")

            except Exception as e:
                print(f"Dam {self.site_id}: Error generating {cfg['source']} stream raster: {e}")


    def save_changes(self):
        self.site_data['reach_id'] = self.nwm_id
        self.site_data['linkno'] = self.geoglows_id
        
        # Ensure sources are updated in the dictionary so they get saved to DB
        self.site_data['flowline_source'] = self.flowline_source
        self.site_data['streamflow_source'] = self.streamflow_source
        
        # Ensure land raster path is saved
        self.site_data['land_raster'] = self.land_cover_path
        
        # Ensure stream raster paths are saved
        if self.flowline_raster_nhd:
            self.site_data['flowline_raster_nhd'] = self.flowline_raster_nhd
        if self.flowline_raster_tdx:
            self.site_data['flowline_raster_tdx'] = self.flowline_raster_tdx

        if self.flowline_path_nhd:
            self.site_data['flowline_path_nhd'] = self.flowline_path_nhd
        if self.flowline_path_tdx:
            self.site_data['flowline_path_tdx'] = self.flowline_path_tdx

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
