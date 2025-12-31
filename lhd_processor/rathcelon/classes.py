# build-in imports
import os
import sys
import stat
import platform
from pathlib import Path

# third-party imports
from arc import Arc  # automated rating curve generator

try:
    import gdal
    import gdal_array
except ImportError:
    from osgeo import gdal, ogr, osr, gdal_array

import fiona
import rasterio
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import box, shape, Point, LineString, MultiLineString
from pyproj import CRS, Geod, Transformer
from shapely.ops import nearest_points, linemerge, transform
from rasterio.features import rasterize



def get_raster_info(dem_tif: str):
    """Retrieves the geographic details and projection of a raster."""
    with rasterio.open(dem_tif) as dataset:
        geoTransform = dataset.transform
        ncols = dataset.width
        nrows = dataset.height
        minx = geoTransform.c
        dx = geoTransform.a
        maxy = geoTransform.f
        dy = geoTransform.e
        maxx = minx + dx * ncols
        miny = maxy + dy * nrows
        Rast_Projection = dataset.crs.to_wkt()
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection


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
    (RR, CC) = np.where(B > 0)
    num_nonzero = len(RR)

    for filterpass in range(2):
        for x in range(num_nonzero):
            r, c = RR[x], CC[x]
            if B[r, c] > 0:
                # Remove single hanging cells
                if B[r, c + 1] == 0 and B[r, c - 1] == 0:
                    if (B[r + 1, c - 1:c + 2].sum() == 0 and B[r - 1, c] > 0) or \
                            (B[r - 1, c - 1:c + 2].sum() == 0 and B[r + 1, c] > 0):
                        B[r, c] = 0
                # Remove redundant thick cells
                elif B[r + 1, c] == B[r, c] and (B[r + 1, c + 1] == B[r, c] or B[r + 1, c - 1] == B[r, c]):
                    if sum(B[r + 1, c - 1:c + 2]) == B[r, c] * 2:
                        B[r + 1, c] = 0
    write_output_raster(clean_strm_tif, B[1:nrows + 1, 1:ncols + 1], gt, proj)


def create_arc_strm_raster(StrmSHP, output_raster_path, DEM_File, value_field):
    """Rasterizes a shapefile to match the extent and resolution of a DEM."""
    gdf = gpd.read_file(StrmSHP)
    with rasterio.open(DEM_File) as ref:
        meta, ref_transform, out_shape, crs = ref.meta.copy(), ref.transform, (ref.height, ref.width), ref.crs
    if gdf.crs != crs: gdf = gdf.to_crs(crs)
    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[value_field])]
    raster = rasterize(shapes=shapes, out_shape=out_shape, fill=0, transform=ref_transform, dtype='int32')
    meta.update({"driver": "GTiff", "dtype": "int32", "count": 1, "nodata": 0})
    with rasterio.open(output_raster_path, 'w', **meta) as dst: dst.write(raster, 1)


class RathCelonDam:
    def __init__(self, dam_row: dict):
        """Initializes dam processing using a row from the Excel database."""

        def safe_p(p): return Path(p) if pd.notna(p) and p != "" else None

        self.dam_row = dam_row
        self.name = dam_row.get('name')

        # Assumption: First column of the Excel row is the Dam ID
        self.id_field = list(dam_row.keys())[0]
        self.dam_id = int(dam_row.get(self.id_field))

        # Physical and Path specs
        self.weir_length = dam_row.get('weir_length')
        self.dem_dir = safe_p(dam_row.get('dem_dir'))
        self.output_dir = safe_p(dam_row.get('output_dir'))
        self.land_raster = safe_p(dam_row.get('land_raster'))

        # --- DYNAMIC SOURCE SELECTION ---
        self.flowline_source = dam_row.get('flowline_source', 'NHDPlus')
        self.streamflow_source = dam_row.get('streamflow_source', 'National Water Model')

        # 1. Select Flowline Path & ID Field
        if self.flowline_source == 'TDX-Hydro':
            self.flowline = safe_p(dam_row.get('flowline_path_tdx'))
            self.rivid_field = 'LINKNO'
        else:
            # Default to NHDPlus
            self.flowline = safe_p(dam_row.get('flowline_path_nhd'))
            self.rivid_field = 'nhdplusid'

        # Fallback if specific columns are empty but generic 'flowline' exists
        if not self.flowline:
            self.flowline = safe_p(dam_row.get('flowline'))
            if self.flowline:
                # Try to guess ID field from path name or default to NHD
                if 'TDX' in str(self.flowline) or 'tdx' in str(self.flowline):
                    self.rivid_field = 'LINKNO'
                else:
                    self.rivid_field = 'nhdplusid'

        # 2. Select Streamflow Reanalysis File
        # We assume the reanalysis files are in the package 'data' directory
        package_root = Path(__file__).resolve().parent.parent
        data_dir = package_root / 'data'

        if self.streamflow_source == 'GEOGLOWS':
            self.streamflow = data_dir / 'geoglows_reanalysis.csv'
        else:
            self.streamflow = data_dir / 'nwm_reanalysis.csv'

        # Allow override if 'streamflow' column points to a valid file
        custom_sf = safe_p(dam_row.get('streamflow'))
        if custom_sf and custom_sf.exists() and custom_sf.is_file():
            self.streamflow = custom_sf

    def _create_arc_input_txt(self, q_bf_col, q_max_col):
        """Generates the ARC input text file pointing to the Master database."""
        x_dist = int(10 * self.weir_length)
        with open(self.arc_input, 'w') as f:
            f.write('#ARC_Inputs\n')
            f.write(f'DEM_File\t{self.dem_tif}\n')
            f.write(f'Stream_File\t{self.strm_tif_clean}\n')
            f.write(f'LU_Raster_SameRes\t{self.land_tif}\n')
            f.write(f'Flow_File\t{self.streamflow}\n')
            f.write(f'Flow_File_ID\t{self.rivid_field}\n')
            f.write(f'Flow_File_BF\t{q_bf_col}\n')
            f.write(f'Flow_File_QMax\t{q_max_col}\n')
            f.write(f'X_Section_Dist\t{x_dist}\n')
            f.write(f'BATHY_Out_File\t{self.bathy_tif}\n')
            f.write(f'XS_Out_File\t{self.xs_txt}\n')

    def _process_geospatial_data(self):
        """Prepares rasters without duplicating the CSV database."""
        self.strm_tif = os.path.join(self.strm_dir, f'{self.dam_id}_STRM.tif')
        self.strm_tif_clean = self.strm_tif.replace('.tif', '_Clean.tif')
        self.land_tif = self.land_raster

        if not os.path.exists(self.strm_tif):
            create_arc_strm_raster(str(self.flowline), self.strm_tif, self.dem_tif, self.rivid_field)

        if not os.path.exists(self.strm_tif_clean):
            clean_strm_raster(self.strm_tif, self.strm_tif_clean)

        self._create_arc_input_txt("qout_median", "rp100")

    def process_dam(self):
        """Setup directories and assess the dam."""
        for sd in ['STRM', 'ARC_InputFiles', 'Bathymetry', 'VDT', 'XS']:
            folder = self.output_dir / self.name / sd
            setattr(self, f"{sd.lower().split('_')[0]}_dir", str(folder))
            folder.mkdir(parents=True, exist_ok=True)

        for dem in [f for f in os.listdir(self.dem_dir) if f.endswith('.tif')]:
            self.dem_tif = str(self.dem_dir / dem)
            self.arc_input = os.path.join(self.arc_dir, f'ARC_Input_{self.dam_id}.txt')
            self.bathy_tif = os.path.join(self.bathy_dir, f'{self.dam_id}_Bathy.tif')
            self.xs_txt = os.path.join(self.xs_dir, f'{self.dam_id}_XS.txt')

            self._process_geospatial_data()
            if not os.path.exists(self.bathy_tif):
                Arc(self.arc_input).run()
