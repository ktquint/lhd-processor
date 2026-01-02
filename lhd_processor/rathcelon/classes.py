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


# --- GEOSPATIAL UTILITIES ---

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
                if B[r, c + 1] == 0 and B[r, c - 1] == 0:
                    if (B[r + 1, c - 1:c + 2].sum() == 0 and B[r - 1, c] > 0) or \
                            (B[r - 1, c - 1:c + 2].sum() == 0 and B[r + 1, c] > 0):
                        B[r, c] = 0
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


def create_mannings_esa(manning_txt):
    """Writes standard Manning's n look-up table for ESA WorldCover."""
    with open(manning_txt, 'w') as f:
        f.write('LC_ID\tDescription\tManning_n\n')
        f.write(
            '10\tTree Cover\t0.120\n20\tShrubland\t0.070\n30\tGrassland\t0.035\n40\tCropland\t0.035\n50\tBuilt-up\t0.150\n')
        f.write(
            '60\tBare / Sparse Vegetation\t0.040\n70\tSnow and Ice\t0.025\n80\tPermanent Water Bodies\t0.045\n90\tHerbaceous Wetland\t0.100\n')
        f.write('95\tMangroves\t0.150\n100\tMoss and Lichen\t0.040\n')


def move_upstream(point, current_link, distance, G):
    """Walks a fixed distance upstream through a directed graph network."""
    remaining = distance
    curr_pt = point
    link = current_link
    while remaining > 0:
        if link not in G.nodes: return curr_pt, link
        seg_geom = None
        in_edges = list(G.in_edges(link, data=True))
        if in_edges: seg_geom = in_edges[0][2].get('geometry')
        if not seg_geom: return curr_pt, link
        if hasattr(seg_geom, 'geom_type') and seg_geom.geom_type.startswith("Multi"):
            merged = linemerge(seg_geom)
            seg_line = merged if merged.geom_type == 'LineString' else max(merged.geoms, key=lambda g: g.length)
        else:
            seg_line = seg_geom
        proj = seg_line.project(curr_pt)
        available = proj
        if available >= remaining:
            new_pt = seg_line.interpolate(proj - remaining)
            remaining = 0
        else:
            remaining -= available
            new_pt = Point(seg_line.coords[0])
            ups = list(G.in_edges(link))
            if not ups: return new_pt, link
            link = ups[0][0]
        curr_pt = new_pt
    return curr_pt, link


class RathCelonDam:
    def __init__(self, dam_row: dict):
        """Initializes dam processing using a row from the Excel database."""

        def safe_p(p):
            return Path(p) if pd.notna(p) and p != "" else None

        self.dam_row = dam_row
        self.name = dam_row.get('name')
        self.id_field = list(dam_row.keys())[0]
        self.dam_id = int(dam_row.get(self.id_field))

        self.weir_length = dam_row.get('weir_length', 30)
        self.latitude = dam_row.get('latitude')
        self.longitude = dam_row.get('longitude')
        self.dem_dir = safe_p(dam_row.get('dem_dir'))
        self.output_dir = safe_p(dam_row.get('output_dir'))
        self.land_raster = safe_p(dam_row.get('land_raster'))

        self.flowline_source = dam_row.get('flowline_source', 'NHDPlus')
        self.streamflow_source = dam_row.get('streamflow_source', 'National Water Model')

        # ARC specific defaults
        self.create_reach_average_curve_file = 'False'
        self.bathy_use_banks = 'True'
        self.find_banks_based_on_landcover = 'True'

        if self.flowline_source == 'TDX-Hydro':
            self.flowline = safe_p(dam_row.get('flowline_path_tdx'))
            self.rivid_field = 'LINKNO'
        else:
            self.flowline = safe_p(dam_row.get('flowline_path_nhd'))
            self.rivid_field = 'nhdplusid'

        package_root = Path(__file__).resolve().parent.parent
        data_dir = package_root / 'data'

        if self.streamflow_source == 'GEOGLOWS' and self.flowline_source == 'TDX-Hydro':
            self.streamflow = data_dir / 'geoglows_reanalysis.csv'
        elif self.streamflow_source == 'GEOGLOWS' and self.flowline_source == 'NHDPlus':
            self.streamflow = data_dir / 'geoglows_to_nhd_reanalysis.csv'
        elif self.streamflow_source == 'National Water Model' and self.flowline_source == 'TDX-Hydro':
            self.streamflow = data_dir / 'nwm_to_geoglows_reanalysis.csv'
        else:
            self.streamflow = data_dir / 'nwm_reanalysis.csv'

    def _create_arc_input_txt(self, comid, Q_baseflow, Q_max):
        """Full implementation of ARC input generation with all original parameters."""
        x_section_dist = int(10 * self.weir_length)

        with open(self.arc_input, 'w') as out_file:
            out_file.write('#ARC_Inputs\n')
            out_file.write(f'DEM_File\t{self.dem_tif}\n')
            out_file.write(f'Stream_File\t{self.strm_tif_clean}\n')
            out_file.write(f'LU_Raster_SameRes\t{self.land_tif}\n')
            out_file.write(f'LU_Manning_n\t{self.manning_n_txt}\n')
            out_file.write(f'Flow_File\t{self.streamflow}\n')
            out_file.write(f'Flow_File_ID\t{comid}\n')
            out_file.write(f'Flow_File_BF\t{Q_baseflow}\n')
            out_file.write(f'Flow_File_QMax\t{Q_max}\n')
            out_file.write(f'Spatial_Units\tdeg\n')
            out_file.write(f'X_Section_Dist\t{x_section_dist}\n')
            out_file.write(f'Degree_Manip\t6.1\n')
            out_file.write(f'Degree_Interval\t1.5\n')
            out_file.write(f'Low_Spot_Range\t2\n')
            out_file.write(f'Str_Limit_Val\t1\n')
            out_file.write(f'Gen_Dir_Dist\t10\n')
            out_file.write(f'Gen_Slope_Dist\t10\n\n')

            out_file.write('#VDT_Output_File_and_CurveFile\n')
            out_file.write('VDT_Database_NumIterations\t30\n')
            out_file.write(f'VDT_Database_File\t{self.vdt_txt}\n')
            out_file.write(f'Print_VDT_Database\t{self.vdt_txt}\n')
            out_file.write(f'Print_Curve_File\t{self.curvefile_csv}\n')
            out_file.write(f'Reach_Average_Curve_File\t{self.create_reach_average_curve_file}\n\n')

            out_file.write('#Bathymetry_Information\n')
            out_file.write('Bathy_Trap_H\t0.20\n')
            out_file.write(f'Bathy_Use_Banks\t{self.bathy_use_banks}\n')
            if self.find_banks_based_on_landcover:
                out_file.write(f'FindBanksBasedOnLandCover\t{self.find_banks_based_on_landcover}\n')
            out_file.write(f'AROutBATHY\t{self.bathy_tif}\n')
            out_file.write(f'BATHY_Out_File\t{self.bathy_tif}\n')
            out_file.write(f'XS_Out_File\t{self.xs_txt}\n')

    def _find_strm_up_downstream(self):
        """Extracts the specific hydraulic cross-sections after ARC run."""
        vdt_df = pd.read_csv(self.vdt_txt, sep='\t')
        curve_data_df = pd.read_csv(self.curvefile_csv)

        dam_flowline_gdf = gpd.read_file(self.flowline)
        projected_crs = dam_flowline_gdf.estimate_utm_crs()
        dam_flowline_gdf = dam_flowline_gdf.to_crs(projected_crs)
        dam_point = gpd.GeoSeries([Point(self.longitude, self.latitude)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]

        # Merge flowlines into a single geometry
        merged_geom = linemerge(dam_flowline_gdf.geometry.tolist())

        # Handle MultiLineString if merge wasn't perfect
        if merged_geom.geom_type == 'MultiLineString':
            merged_geom = min(merged_geom.geoms, key=lambda g: g.distance(dam_point))

        # Project dam point onto the line
        start_dist = merged_geom.project(dam_point)

        target_points = []

        # Downstream points (4 points)
        for i in range(1, 5):
            target_dist = start_dist + (i * self.weir_length)
            if target_dist > merged_geom.length:
                target_dist = merged_geom.length
            target_points.append(merged_geom.interpolate(target_dist))

        # Upstream point (1 point)
        us_dist = start_dist - self.weir_length
        if us_dist < 0:
            us_dist = 0
        target_points.append(merged_geom.interpolate(us_dist))

        minx, miny, maxx, maxy, dx, dy, _, _, _, proj_wkt = get_raster_info(self.dem_tif)

        def pt_to_rc(pt):
            return int((maxy - pt.y) / abs(dy)), int((pt.x - minx) / dx)

        final_indices = []
        for pt in target_points:
            r, c = pt_to_rc(pt)
            curve_data_df['d_pix'] = np.sqrt((curve_data_df['Row'] - r) ** 2 + (curve_data_df['Col'] - c) ** 2)
            final_indices.append(curve_data_df['d_pix'].idxmin())

        extracted_curve = curve_data_df.loc[final_indices].copy()
        target_ids = extracted_curve['XS_ID'].unique()
        self.vdt_gdf = vdt_df[vdt_df['XS_ID'].isin(target_ids)].copy()

        x_base, y_base = float(minx) + 0.5 * dx, float(maxy) - 0.5 * abs(dy)
        extracted_curve['X'] = x_base + extracted_curve['Col'] * dx
        extracted_curve['Y'] = y_base - extracted_curve['Row'] * abs(dy)
        transformer = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
        extracted_curve[['Lon', 'Lat']] = extracted_curve.apply(
            lambda r: pd.Series(transformer.transform(r['X'], r['Y'])), axis=1)
        self.curve_data_gdf = gpd.GeoDataFrame(extracted_curve,
                                               geometry=gpd.points_from_xy(extracted_curve.Lon, extracted_curve.Lat),
                                               crs="EPSG:4326")

        xs_lines = []
        with open(self.xs_txt, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                try:
                    meta = lines[i].strip().split('\t')
                    if int(meta[0]) in target_ids:
                        coords = [tuple(map(float, c.split(','))) for c in lines[i + 1].strip().split(' ')]
                        xs_lines.append({'XS_ID': int(meta[0]), 'geometry': LineString(coords)})
                except:
                    continue
        self.xs_gdf = gpd.GeoDataFrame(xs_lines, crs=projected_crs).to_crs("EPSG:4326")

    def process_dam(self):
        """Execution loop for processing a single dam."""
        print(f"Processing Dam: {self.dam_id}")
        dirs = {sd: self.output_dir / str(self.dam_id) / sd for sd in
                ['STRM', 'ARC_InputFiles', 'Bathymetry', 'VDT', 'XS', 'LAND']}
        for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

        self.manning_n_txt = str(dirs['LAND'] / 'Manning_n.txt')
        create_mannings_esa(self.manning_n_txt)

        dems = [f for f in os.listdir(self.dem_dir) if f.endswith('.tif')]
        for dem in dems:
            print(f"  Processing DEM: {dem}")
            self.dem_tif = str(self.dem_dir / dem)
            self.arc_input = str(dirs['ARC_InputFiles'] / f'ARC_Input_{self.dam_id}.txt')
            self.bathy_tif = str(dirs['Bathymetry'] / f'{self.dam_id}_Bathy.tif')
            self.strm_tif_clean = str(dirs['STRM'] / f'{self.dam_id}_STRM_Clean.tif')
            self.vdt_txt = str(dirs['VDT'] / f'{self.dam_id}_VDT.txt')
            self.curvefile_csv = str(dirs['VDT'] / f'{self.dam_id}_Curve.csv')
            self.xs_txt = str(dirs['XS'] / f'{self.dam_id}_XS.txt')
            self.land_tif = str(self.land_raster)

            if not os.path.exists(self.strm_tif_clean):
                print("    Creating clean stream raster...")
                raw_strm = self.strm_tif_clean.replace('_Clean.tif', '.tif')
                create_arc_strm_raster(str(self.flowline), raw_strm, self.dem_tif, self.rivid_field)
                clean_strm_raster(raw_strm, self.strm_tif_clean)

            if not os.path.exists(self.bathy_tif):
                print("    Running ARC simulation...")
                self._create_arc_input_txt(self.rivid_field, "known_baseflow", "rp100")
                arc_runner = Arc(self.arc_input, quiet=False)
                try:
                    arc_runner.set_log_level('info')
                except AttributeError:
                    pass
                arc_runner.run()

            print("    Extracting hydraulic cross-sections...")
            self._find_strm_up_downstream()
            self.vdt_gdf.to_csv(str(dirs['VDT'] / f'{self.dam_id}_Local_VDT.csv'), index=False)
            self.curve_data_gdf.to_file(str(dirs['VDT'] / f'{self.dam_id}_Local_Curve.gpkg'))
            self.xs_gdf.to_file(str(dirs['XS'] / f'{self.dam_id}_Local_XS.gpkg'))
        print(f"Finished Dam: {self.dam_id}")
