# build-in imports
import os  # working with the operating system
import sys  # working with the interpreter

# third-party imports
from arc import Arc  # automated rating curve generator (arc)

try:
    import gdal  # geospatial data abstraction library (gdal)
    import gdal_array
except ImportError:
    from osgeo import gdal, ogr, osr, gdal_array  # import from osgeo if direct import doesn't work

import stat  # interpreting file status
import fiona  # reading/writing vector spatial data
import platform  # access information about system
import rasterio  # raster data access/manipulation
import numpy as np  # numerical + py = numpy
import pandas as pd  # tabular data manipulation
import networkx as nx  # network/graph analysis
import geopandas as gpd  # geospatial + pandas = geopandas
from pathlib import Path  # OOP - programming + filesystem paths
from shapely.geometry import box, shape  # geometric shapes + manipulation
from pyproj import CRS, Geod, Transformer  # CRS and transformations
from shapely.geometry import Point, LineString, MultiLineString  # , shape, mapping
from shapely.ops import nearest_points, linemerge, transform  # , split
from rasterio.features import rasterize

# local imports
from . import esa_download_processing as esa


def get_raster_info(dem_tif: str):
    print(dem_tif)
    with rasterio.open(dem_tif) as dataset:
        geoTransform = dataset.transform  # affine transform
        ncols = dataset.width
        nrows = dataset.height
        minx = geoTransform.c
        dx = geoTransform.a
        maxy = geoTransform.f
        dy = geoTransform.e
        maxx = minx + dx * ncols
        miny = maxy + dy * nrows
        Rast_Projection = dataset.crs.to_wkt()  # wkt format

    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection


def move_upstream(point, current_link, distance, G):
    # same logic as your original while-block, but for a fixed 'distance'
    remaining = distance
    curr_pt = point
    link = current_link

    while remaining > 0:
        # retrieve segment geometry for this link
        if 'geometry' in G.nodes[link]:
            seg_geom = G.nodes[link]['geometry']
        else:
            # fallback to edge data
            for u, v, data in list(G.out_edges(link, data=True)) + list(G.in_edges(link, data=True)):
                if 'geometry' in data:
                    seg_geom = data['geometry']
                    break
            else:
                raise RuntimeError(f"No geometry found for link {link}")

        # flatten MultiLine if needed and handle both outcomes
        if hasattr(seg_geom, 'geom_type') and seg_geom.geom_type.startswith("Multi"):
            merged = linemerge(seg_geom)
            if merged.geom_type == 'LineString':
                seg_line = merged
            elif merged.geom_type == 'MultiLineString':
                # pick the longest line segment
                seg_line = max(merged.geoms, key=lambda g: g.length)
            else:
                raise TypeError(f"Unexpected geometry after linemerge: {merged.geom_type}")
        else:
            seg_line = seg_geom

        # ensure seg_line endpoints with current point at end
        coords = list(seg_line.coords)
        if not Point(coords[-1]).equals_exact(curr_pt, tolerance=1e-6):
            coords.reverse()
            seg_line = LineString(coords)

        # how far along this segment we are
        proj = seg_line.project(curr_pt)
        available = proj

        if int(available) >= remaining:
            # can step within this segment
            new_pt = seg_line.interpolate(proj - remaining)
            remaining = 0
        else:
            # go to upstream end of this segment
            remaining -= available
            new_pt = Point(seg_line.coords[0])
            # step to upstream link
            ups = list(G.in_edges(link))
            if not ups:
                curr_pt = new_pt
                return curr_pt, link
                # raise RuntimeError("Ran out of upstream network")
            # pick first, or implement smarter selection
            link = ups[0][0]
        curr_pt = new_pt

    return curr_pt, link


def read_raster_w_gdal(input_raster: str):
    """
    Retrieves the geographic details of a raster using GDAL in a slightly different way than Get_Raster_Details()

    Parameters
    ----------
    input_raster: str
        The file name and full path to the raster you are analyzing

    Returns
    -------
    raster_array: arr
        A numpy array of the values in the first band of the raster you are analyzing
    ncols: int
        The raster width in pixels
    nrows: int
        The raster height in pixels
    cellsize: float
        The pixel size of the raster longitudinally
    yll: float
        The lowest latitude of the raster
    yur: float
        The latitude of the top left corner of the top pixel of the raster
    xll: float
        The longitude of the top left corner of the top pixel of the raster
    xur: float
        The highest longitude of the raster
    lat: float
        The average of the yur and yll latitude values
    geoTransform: list
        A list of geotransform characteristics for the raster
    raster_proj:str
        The projection system reference for the raster
    """
    try:
        dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError("Field Raster File cannot be read?")
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")

    # Retrieve dimensions of cell size and cell count then close DEM dataset
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray()

    # global ncols, nrows, cellsize, yll, yur, xll, xur
    ncols = dataset.RasterXSize
    nrows = dataset.RasterYSize

    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * abs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0]
    xur = xll + ncols * geotransform[1]
    lat = abs((yll + yur) / 2.0)

    raster_proj = dataset.GetProjectionRef()

    # close datasets
    del dataset
    del band

    print(f'Spatial Data for Raster File:\n'
          f'\tncols = {ncols}\n'
          f'\tnrows = {nrows}\n'
          f'\tcellsize = {cellsize}\n'
          f'\tyll = {yll}\n'
          f'\tyur = {yur}\n'
          f'\txll = {xll}\n'
          f'\txur = {xur}')

    return raster_array, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, raster_proj


def clean_strm_raster(strm_tif: str, clean_strm_tif: str) -> None:
    print('\nCleaning up the Stream File.')
    (SN, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) \
        = read_raster_w_gdal(strm_tif)

    # Create an array that is slightly larger than the STRM Raster Array
    B = np.zeros((nrows + 2, ncols + 2))

    # Imbed the STRM Raster within the Larger Zero Array
    B[1:(nrows + 1), 1:(ncols + 1)] = SN

    # Added this because sometimes the non-stream values end up as -9999
    B = np.where(B > 0, B, 0)
    # (RR,CC) = B.nonzero()
    (RR, CC) = np.where(B > 0)
    num_nonzero = len(RR)

    for filterpass in range(2):
        # First pass is just to get rid of single cells hanging out not doing anything
        p_count = 0
        p_percent = (num_nonzero + 1) / 100.0
        n = 0
        for x in range(num_nonzero):
            if x >= p_count * p_percent:
                p_count = p_count + 1
                print(f' {p_count}', end=" ")
            r = RR[x]
            c = CC[x]
            V = B[r, c]
            if V > 0:
                # Left and Right cells are zeros
                if B[r, c + 1] == 0 and B[r, c - 1] == 0:
                    # The bottom cells are all zeros as well, but there is a cell directly above that is legit
                    if (B[r + 1, c - 1] + B[r + 1, c] + B[r + 1, c + 1]) == 0 and B[r - 1, c] > 0:
                        B[r, c] = 0
                        n += 1
                    # The top cells are all zeros as well, but there is a cell directly below that is legit
                    elif (B[r - 1, c - 1] + B[r - 1, c] + B[r - 1, c + 1]) == 0 and B[r + 1, c] > 0:
                        B[r, c] = 0
                        n += 1
                # top and bottom cells are zeros
                if B[r, c] > 0 and B[r + 1, c] == 0 and B[r - 1, c] == 0:
                    # All cells on the right are zero, but there is a cell to the left that is legit
                    if (B[r + 1, c + 1] + B[r, c + 1] + B[r - 1, c + 1]) == 0 and B[r, c - 1] > 0:
                        B[r, c] = 0
                        n += 1
                    elif (B[r + 1, c - 1] + B[r, c - 1] + B[r - 1, c - 1]) == 0 and B[r, c + 1] > 0:
                        B[r, c] = 0
                        n += 1
        print(f'\nFirst pass removed {n} cells')

        # This pass is to remove all the redundant cells
        n = 0
        p_count = 0
        p_percent = (num_nonzero + 1) / 100.0
        for x in range(num_nonzero):
            if x >= p_count * p_percent:
                p_count = p_count + 1
                print(f' {p_count}', end=" ")
            r = RR[x]
            c = CC[x]
            V = B[r, c]
            if V > 0:
                if B[r + 1, c] == V and (B[r + 1, c + 1] == V or B[r + 1, c - 1] == V):
                    if sum(B[r + 1, c - 1:c + 2]) == 0:
                        B[r + 1, c] = 0
                        n += 1
                elif B[r - 1, c] == V and (B[r - 1, c + 1] == V or B[r - 1, c - 1] == V):
                    if sum(B[r - 1, c - 1:c + 2]) == 0:
                        B[r - 1, c] = 0
                        n += 1
                elif B[r, c + 1] == V and (B[r + 1, c + 1] == V or B[r - 1, c + 1] == V):
                    if sum(B[r - 1:r + 1, c + 2]) == 0:
                        B[r, c + 1] = 0
                        n += 1
                elif B[r, c - 1] == V and (B[r + 1, c - 1] == V or B[r - 1, c - 1] == V):
                    if sum(B[r - 1:r + 1, c - 2]) == 0:
                        B[r, c - 1] = 0
                        n += 1

        print(f'\nSecond pass removed {n} redundant cells')

    print(f'Writing Output File {clean_strm_tif}')
    write_output_raster(clean_strm_tif, B[1:nrows + 1, 1:ncols + 1], dem_geotransform, dem_projection)


# def write_output_raster(s_output_filename: str, raster_data, dem_geotransform, dem_projection, s_file_format='GTiff'):

#     # Construct the file with the appropriate data shape
#     s_output_type = gdal_array.NumericTypeCodeToGDALTypeCode(raster_data.dtype)
#     n_rows, n_cols = raster_data.shape
#     with gdal.GetDriverByName(s_file_format).Create(s_output_filename, xsize=n_cols, ysize=n_rows,
#                                                     bands=1, eType=s_output_type) as dst:
#         # Set the geotransform
#         dst.SetGeoTransform(dem_geotransform)

#         # Set the spatial reference
#         dst.SetProjection(dem_projection)

#         # Write the data to the file
#         dst.GetRasterBand(1).WriteArray(raster_data)

def write_output_raster(
        s_output_filename: str,
        raster_data,
        dem_geotransform,
        dem_projection,
        *,
        s_file_format: str = "GTiff",
        nodata=None,
        creation_options=("TILED=YES", "COMPRESS=LZW")
):
    # Map numpy dtype -> GDAL type
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(raster_data.dtype)
    if gdal_dtype is None:
        # Default to Float32 if GDAL can't infer it (e.g., object dtype)
        raster_data = np.asarray(raster_data, dtype=np.float32)
        gdal_dtype = gdal.GDT_Float32

    n_rows, n_cols = raster_data.shape
    driver = gdal.GetDriverByName(s_file_format)
    if driver is None:
        raise RuntimeError(f"GDAL driver not found: {s_file_format}")

    ds = driver.Create(
        s_output_filename,
        xsize=n_cols,
        ysize=n_rows,
        bands=1,
        eType=gdal_dtype,
        options=list(creation_options) if creation_options else None
    )
    if ds is None:
        raise RuntimeError(f"Failed to create dataset: {s_output_filename}")

    try:
        ds.SetGeoTransform(dem_geotransform)
        ds.SetProjection(dem_projection)

        band = ds.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(nodata)

        band.WriteArray(raster_data)
        band.FlushCache()
    finally:
        del band
        del ds
        # band = None
        # ds = None


def update_crs(dem_tif: str, lc_tif: str) -> str:
    # Load the projection of the DEM file
    with rasterio.open(dem_tif) as src:
        dem_projection = src.crs
    src.close()

    # re-project the LAND file raster, if necessary
    with rasterio.open(lc_tif) as src:
        current_crs = src.crs
    src.close()

    if current_crs != dem_projection:
        input_raster = gdal.Open(lc_tif)
        updated_lc_tif = f"{lc_tif[:-4]}_new.tif"
        output_raster = updated_lc_tif
        gdal.Warp(output_raster, input_raster, dstSRS=dem_projection)
        # Closes the files
        del input_raster

        # delete the old LAND raster, if it was replaced and change the name
        os.remove(lc_tif)
        lc_tif = updated_lc_tif

    return lc_tif


def create_arc_strm_raster(StrmSHP: str, output_raster_path: str, DEM_File: str, value_field: str):
    # Load the shapefile
    gdf = gpd.read_file(StrmSHP)

    # Check if value_field is valid
    if value_field is None or value_field not in gdf.columns:
        raise ValueError(
            f"Invalid value_field: '{value_field}'. Not found in shapefile columns: {gdf.columns.to_list()}")

    # if values are bigger than 32 bit intergers, subtract 5000000000000 from the value_field
    if gdf[value_field].max() > 2147483647:
        gdf[value_field] = gdf[value_field] - 5000000000000

    # Load the reference raster to get transform, shape, and CRS
    with rasterio.open(DEM_File) as ref:
        meta = ref.meta.copy()
        ref_transform = ref.transform
        out_shape = (ref.height, ref.width)
        crs = ref.crs

    print(f"gdf.crs: {gdf.crs}")
    print(f"raster crs: {crs}")
    print(f"gdf.crs == crs? {gdf.crs == crs}")

    # Reproject the shapefile to match reference CRS
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Prepare shapes (geometry, value)
    shapes = [(geom, value if value is not None else 0)
              for geom, value in zip(gdf.geometry, gdf[value_field])]

    # Rasterize
    raster = rasterize(shapes=shapes, out_shape=out_shape, fill=0, transform=ref_transform, dtype='int32')

    # Update metadata
    meta.update({"driver": "GTiff", "dtype": "int32", "count": 1, "compress": "lzw", "nodata": 0})

    # Write to file
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(raster, 1)

    print(f"Raster written to {output_raster_path}")


def create_arc_lc_raster(lc_tif: str, land_tif: str, projWin_extents, ncols: int, nrows: int):
    """
    Creates a land cover raster that is clipped to a specified extent and cell size
    ... (omitted docstring)
    """

    # 1. Define the options object for cleaner code (requires GDAL 2.x+)
    # The key to speed is 'NUM_THREADS=ALL_CPUS'
    options = gdal.TranslateOptions(
        projWin=projWin_extents,
        width=ncols,
        height=nrows,
        # Using creation options to enable parallel processing and compression
        creationOptions=['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS']
    )

    # 2. Call gdal.Translate directly on the input filename (lc_tif)
    # This replaces the two lines: ds = gdal.Open(lc_tif) and del ds
    gdal.Translate(land_tif, lc_tif, options=options)

    return


def create_mannings(manning_txt: str):
    with open(manning_txt, 'w') as out_file:
        out_file.write('LC_ID\tDescription\tManning_n\n')
        out_file.write('11\tWater\t0.030\n')
        out_file.write('21\tDev_Open_Space\t0.013\n')
        out_file.write('22\tDev_Low_Intensity\t0.050\n')
        out_file.write('23\tDev_Med_Intensity\t0.075\n')
        out_file.write('24\tDev_High_Intensity\t0.100\n')
        out_file.write('31\tBarren_Land\t0.030\n')
        out_file.write('41\tDecid_Forest\t0.120\n')
        out_file.write('42\tEvergreen_Forest\t0.120\n')
        out_file.write('43\tMixed_Forest\t0.120\n')
        out_file.write('52\tShrub\t0.050\n')
        out_file.write('71\tGrass_Herb\t0.030\n')
        out_file.write('81\tPasture_Hay\t0.040\n')
        out_file.write('82\tCultivated_Crops\t0.035\n')
        out_file.write('90\tWoody_Wetlands\t0.100\n')
        out_file.write('95\tEmergent_Herb_Wet\t0.100')


def create_mannings_esa(manning_txt: str):
    with open(manning_txt, 'w') as out_file:
        out_file.write('LC_ID\tDescription\tManning_n\n')
        out_file.write('10\tTree Cover\t0.120\n')
        out_file.write('20\tShrubland\t0.050\n')
        out_file.write('30\tGrassland\t0.030\n')
        out_file.write('40\tCropland\t0.035\n')
        out_file.write('50\tBuiltup\t0.075\n')
        out_file.write('60\tBare\t0.030\n')
        out_file.write('70\tSnowIce\t0.030\n')
        out_file.write('80\tWater\t0.030\n')
        out_file.write('90\tEmergent_Herb_Wet\t0.100\n')
        out_file.write('95\tMangroves\t0.100\n')
        out_file.write('100\tMossLichen\t0.100')


class RathCelonDam:
    def __init__(self, **kwargs):
        def safe_path(p):
            if pd.isna(p) or p == "" or p is None:
                return None
            return Path(p)
        # required parameters
        self.name = kwargs['name']
        self.csv_path = safe_path(kwargs['dam_csv'])  # Use safe_path
        self.id_field = kwargs['dam_id_field']
        self.dam_id = kwargs['dam_id']
        self.flowline = safe_path(kwargs['flowline'])  # Use safe_path
        self.dem_dir = safe_path(kwargs['dem_dir'])  # Use safe_path
        self.land_raster = safe_path(kwargs['land_raster'])
        self.output_dir = safe_path(kwargs['output_dir'])  # Use safe_path

        # OPTIONAL parameters
        self.streamflow = safe_path(kwargs.get('streamflow'))
        self.bathy_use_banks = kwargs.get('bathy_use_banks', False)
        self.flood_waterlc_and_strm_cells = kwargs.get('flood_waterlc_and_strm_cells', False)
        self.process_stream_network = kwargs.get('process_stream_network', False)
        self.find_banks_based_on_landcover = kwargs.get('find_banks_based_on_landcover', True)
        self.create_reach_average_curve_file = kwargs.get('create_reach_average_curve_file', False)
        self.known_baseflow = kwargs.get('known_baseflow', None)
        self.known_channel_forming_discharge = kwargs.get('known_channel_forming_discharge', None)
        self.upstream_elevation_change_threshold = 1.0  # kwargs.get('upstream_elevation_change_threshold', 0.1)


        # internal attributes
        self.rivid_field = None  # Will be set in process_dam

        # folder locations:
        self.arc_dir = None
        self.bathy_dir = None
        self.strm_dir = None
        self.land_dir = None
        self.flow_dir = None
        self.vdt_dir = None
        self.esa_lc_dir = None
        self.xs_dir = None
        self.manning = None

        self.flowline_gdf = None

    def _create_arc_input_txt(self, comid, Q_baseflow, Q_max):

        x_section_dist = int(10 * self.weir_length)

        with open(self.arc_input, 'w') as out_file:
            out_file.write('#ARC_Inputs\n')
            out_file.write(f'DEM_File\t{self.dem_tif}\n')
            out_file.write(f'Stream_File\t{self.strm_tif_clean}\n')
            out_file.write(f'LU_Raster_SameRes\t{self.land_tif}\n')
            out_file.write(f'LU_Manning_n\t{self.manning}\n')
            out_file.write(f'Flow_File\t{self.reanalysis_csv}\n')
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

    def _find_strm_up_downstream(self, n_cross_sections=4):

        # Read the VDT and Curve data into DataFrames
        vdt_df = pd.read_csv(self.vdt_txt)
        curve_data_df = pd.read_csv(self.curvefile_csv)

        # Read the dam reanalysis flow file
        dam_reanalysis_df = pd.read_csv(self.reanalysis_csv)

        # Merge flow parameters
        merged_df = pd.merge(curve_data_df, dam_reanalysis_df, on='COMID', how='left')
        if self.known_baseflow is None:
            # Check if 'rp2' exists, if not, fallback or error
            if 'rp2' not in merged_df.columns:
                print("Warning: 'rp2' column not found in reanalysis data. Cannot calculate 'tw_rp2'.")
                # As a fallback, maybe set to a default or skip, here I'll skip calculation
                merged_df['tw_rp2'] = np.nan
            else:
                merged_df['tw_rp2'] = merged_df['tw_a'] * (merged_df['rp2'] ** merged_df['tw_b'])
        else:
            merged_df['tw_known_baseflow'] = merged_df['tw_a'] * (self.known_baseflow ** merged_df['tw_b'])

        # Read stream shapefile and dam locations
        dam_flowline_gdf = gpd.read_file(self.dam_shp, engine='fiona')
        dam_gdf = pd.read_csv(self.csv_path)
        dam_gdf = gpd.GeoDataFrame(dam_gdf, geometry=gpd.points_from_xy(dam_gdf['longitude'], dam_gdf['latitude']),
                                   crs="EPSG:4269")

        # Convert to a projected CRS for accurate distance calculations
        projected_crs = dam_flowline_gdf.estimate_utm_crs()
        dam_flowline_gdf = dam_flowline_gdf.to_crs(projected_crs)
        dam_gdf = dam_gdf.to_crs(projected_crs)

        # Filter to the specific dam
        dam_gdf = dam_gdf[dam_gdf[self.id_field] == self.dam_id]
        if dam_gdf.empty:
            raise ValueError("No matching dam found for the given dam_id.")

        dam_gdf = dam_gdf.reset_index(drop=True)
        dam_point = dam_gdf.geometry.iloc[0]

        # **1. Build a Directed Graph Using LINKNO and DSLINKNO**
        G = nx.DiGraph()

        # Use the rivid_field already set in the class
        rivid_field = self.rivid_field
        if rivid_field == 'LINKNO':
            ds_rivid_field = 'DSLINKNO'
        else:
            ds_rivid_field = 'dnhydroseq'

        for _, row in dam_flowline_gdf.iterrows():
            link_id = row[rivid_field]
            ds_link_id = row[ds_rivid_field]
            geometry = row.geometry

            if rivid_field == 'hydroseq':
                if isinstance(geometry, LineString):
                    geometry = LineString(list(geometry.coords)[::-1])
                elif isinstance(geometry, MultiLineString):
                    # Merge parts to a single LineString if possible
                    merged = linemerge(geometry)
                    if isinstance(merged, LineString):
                        geometry = LineString(list(merged.coords)[::-1])
                    else:
                        # If still multipart, you might need custom handling,
                        # but as a fallback just keep as is or raise an error
                        geometry = merged  # or handle differently

            if ds_link_id > 0:  # Ignore terminal reaches
                G.add_edge(link_id, ds_link_id, geometry=geometry, weight=geometry.length)

        # Find the Closest Stream to the Dam**
        dam_flowline_gdf['distance'] = dam_flowline_gdf.distance(dam_point)
        closest_stream = dam_flowline_gdf.loc[dam_flowline_gdf['distance'].idxmin()]
        start_link = closest_stream[rivid_field]

        # Get the dam’s intersection point on the stream
        current_point = nearest_points(closest_stream.geometry, dam_point)[0]
        # Assume current_link is the stream segment containing current_point (the dam intersection)
        current_link = start_link

        dam_ids = []
        link_nos = []
        points_of_interest = []

        weir_length = dam_gdf['weir_length'].values[0]

        # Loop for each downstream cross-section
        for i in range(1, n_cross_sections + 1):
            # We want to move downstream the length of the weir from the current point
            remaining_distance_to_travel = weir_length

            print(f"Calculating point {i * remaining_distance_to_travel} meters downstream of the dam.")

            # Traverse the network until we've moved the full tw meters
            while remaining_distance_to_travel > 0:
                # Get downstream edges from the current link
                downstream_edges = list(G.out_edges(current_link, data=True))
                # print(G.out_edges(current_link))
                if not downstream_edges:
                    print(f"At failure, current_link = {current_link}")
                    print(G.out_edges(current_link))

                    raise ValueError(
                        f"Not enough downstream stream length for cross-section {i} (link {current_link})."
                    )
                # Select the first valid downstream edge (adjust this logic as needed)
                next_link = None
                for edge in downstream_edges:
                    _, candidate_next_link, _ = edge
                    if candidate_next_link in G:
                        next_link = candidate_next_link
                        break
                if next_link is None:
                    raise ValueError(f"No valid downstream link from link {current_link}.")

                # Retrieve geometry for the current segment (from current_link to next_link)
                seg_geom = shape(G[current_link][next_link]['geometry'])

                # Merge or unwrap multipart geometries safely
                if seg_geom.geom_type.startswith('Multi'):
                    merged_geom = linemerge(seg_geom)
                    if isinstance(merged_geom, MultiLineString):
                        # Flatten all coords into a single list (or handle each part individually if needed)
                        seg_coords = [pt for line in merged_geom.geoms for pt in line.coords]
                    else:  # It successfully became a LineString
                        seg_coords = list(merged_geom.coords)
                else:
                    seg_coords = list(seg_geom.coords)

                # Ensure the segment’s coordinates are ordered so that its start is near current_point
                if not Point(seg_coords[0]).equals_exact(current_point, tolerance=0.2):
                    seg_coords.reverse()

                seg_line = LineString(seg_coords)

                # Find how far along seg_line our current_point lies
                proj_distance = seg_line.project(current_point)
                # print(f'proj_distance: {proj_distance}')

                distance_remaining_in_seg = seg_line.length - proj_distance

                if distance_remaining_in_seg >= remaining_distance_to_travel:
                    # The target point lies within the current segment.
                    new_point = seg_line.interpolate(proj_distance + remaining_distance_to_travel)
                    # Update our current point and finish the "walk" for this cross-section.
                    current_point = new_point
                    remaining_distance_to_travel = 0
                else:
                    # Use up the remainder of this segment and continue into the next one.
                    remaining_distance_to_travel -= distance_remaining_in_seg
                    # Set current_point to the end of the segment.
                    current_point = Point(seg_coords[-1])
                    # Update current_link to the downstream link we just traversed.
                    current_link = next_link

            # At this point, current_point has moved exactly tw meters from its previous location.
            # Record the result (convert to EPSG:4326 if needed).
            downstream_point = (gpd.GeoSeries([current_point], crs=projected_crs).geometry.iloc[0])
            dam_ids.append(self.dam_id)
            link_nos.append(current_link)
            points_of_interest.append(downstream_point)

        # downstream_points now contains cross-sections spaced exactly tw meters apart along the stream.
        (minx, miny, maxx, maxy, dx, dy, _, _, _, crs_str) = get_raster_info(self.dem_tif)
        cellsize_x, cellsize_y = abs(float(dx)), abs(float(dy))
        crs = CRS.from_user_input(crs_str)

        # Compute upper-left cell center in raster coordinates
        x_base = float(minx) + 0.5 * cellsize_x
        y_base = float(maxy) - 0.5 * cellsize_y

        # Compute all X/Ys in raster space (regardless of CRS)
        curve_data_df['X'] = x_base + curve_data_df['Col'] * cellsize_x
        curve_data_df['Y'] = y_base - curve_data_df['Row'] * cellsize_y

        vdt_df['X'] = x_base + vdt_df['Col'] * cellsize_x
        vdt_df['Y'] = y_base - vdt_df['Row'] * cellsize_y

        # Open the cross-section file and read the end point lat and lons
        XS_Out_File_df = pd.read_csv(self.xs_txt, sep='\t')

        XS_Out_File_df['X1'] = x_base + XS_Out_File_df['c1'] * cellsize_x
        XS_Out_File_df['Y1'] = y_base - XS_Out_File_df['r1'] * cellsize_y
        XS_Out_File_df['X2'] = x_base + XS_Out_File_df['c2'] * cellsize_x
        XS_Out_File_df['Y2'] = y_base - XS_Out_File_df['r2'] * cellsize_y

        # Convert to Lat/Lon
        if crs.is_geographic:
            # CRS is already in lat/lon
            curve_data_df['Lon'] = curve_data_df['X']
            curve_data_df['Lat'] = curve_data_df['Y']
            vdt_df['Lon'] = vdt_df['X']
            vdt_df['Lat'] = vdt_df['Y']
            XS_Out_File_df['Lon1'] = XS_Out_File_df['X1']
            XS_Out_File_df['Lat1'] = XS_Out_File_df['Y1']
            XS_Out_File_df['Lon2'] = XS_Out_File_df['X2']
            XS_Out_File_df['Lat2'] = XS_Out_File_df['Y2']
        else:
            # Projected -> convert to WGS84
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            # print("vdt_df shape:", vdt_df.shape)
            # print(vdt_df.head())

            # print(transformer.transform(vdt_df.iloc[0]['X'], vdt_df.iloc[0]['Y']))
            curve_data_df[['Lon', 'Lat']] = curve_data_df.apply(
                lambda row_i: pd.Series(transformer.transform(row_i['X'], row_i['Y'])), axis=1)
            vdt_df[['Lon', 'Lat']] = vdt_df.apply(
                lambda row_i: pd.Series(transformer.transform(row_i['X'], row_i['Y'])),
                axis=1
            )
            XS_Out_File_df[['Lon1', 'Lat1']] = XS_Out_File_df.apply(
                lambda row_i: pd.Series(transformer.transform(row_i['X1'], row_i['Y1'])), axis=1)
            XS_Out_File_df[['Lon2', 'Lat2']] = XS_Out_File_df.apply(
                lambda row_i: pd.Series(transformer.transform(row_i['X2'], row_i['Y2'])), axis=1)

            # x, y = XS_Out_File_df.loc[0, 'X1'], XS_Out_File_df.loc[0, 'Y1']
            # lon, lat = transformer.transform(x, y)
            # print(f"X1 = {x}, Y1 = {y} → Lon = {lon}, Lat = {lat}")

        self.curve_data_gdf = gpd.GeoDataFrame(curve_data_df,
                                               geometry=gpd.points_from_xy(curve_data_df['Lon'], curve_data_df['Lat']),
                                               crs="EPSG:4269")
        self.vdt_gdf = gpd.GeoDataFrame(vdt_df, geometry=gpd.points_from_xy(vdt_df['Lon'], vdt_df['Lat']),
                                        crs="EPSG:4269")

        # Convert all to projected CRS
        self.curve_data_gdf, self.vdt_gdf = (gdf.to_crs(projected_crs) for gdf in [self.curve_data_gdf, self.vdt_gdf])

        # ---------------------------------------------------------
        # SIMPLIFIED UPSTREAM SELECTION: One Weir Length Upstream
        # ---------------------------------------------------------

        # Reset to the dam location
        current_point_geom = nearest_points(closest_stream.geometry, dam_point)[0]
        current_link = start_link

        # store dam origin so we can always measure from it
        origin_point_geom = current_point_geom
        origin_link = current_link

        # Set target distance to exactly one weir length
        target_distance_upstream = self.weir_length

        print(f"Calculating upstream point exactly {target_distance_upstream} meters upstream of the dam.")

        # Move upstream by the target distance
        try:
            upstream_point_geom, upstream_link = move_upstream(
                current_point_geom,
                current_link,
                target_distance_upstream,
                G
            )

            # Reproject the result to the target CRS
            upstream_point = (
                gpd.GeoSeries([upstream_point_geom], crs=projected_crs)
                .geometry.iloc[0]
            )

            # Record the point
            dam_ids.append(self.dam_id)
            link_nos.append(upstream_link)
            points_of_interest.append(upstream_point)

        except Exception as e:
            print(f"Error moving upstream: {e}")
            # Fallback: keep dam location if upstream movement fails
            dam_ids.append(self.dam_id)
            link_nos.append(origin_link)
            points_of_interest.append(gpd.GeoSeries([origin_point_geom], crs=projected_crs).geometry.iloc[0])

        # ---------------------------------------------------------
        # End Simplified Logic
        # ---------------------------------------------------------

        # **Save the Downstream Points as a Shapefile**
        self.downstream_gdf = gpd.GeoDataFrame(
            data={'dam_id': dam_ids, rivid_field: link_nos, 'geometry': points_of_interest},
            crs=projected_crs)

        # **4. Find Nearest VDT and Curve Data Points to the points of interest**
        vdt_gdfs = []
        curve_data_gdfs = []
        # Extract target point from the GeoDataFrame
        for pt in self.downstream_gdf.geometry:
            # compute distances
            dists_curve = self.curve_data_gdf.geometry.distance(pt)
            dists_vdt = self.vdt_gdf.geometry.distance(pt)

            # grab the *index* of the nearest cell
            idx_curve = dists_curve.idxmin()
            idx_vdt = dists_vdt.idxmin()

            # select exactly that one row
            nearest_curves = self.curve_data_gdf.loc[[idx_curve]]
            nearest_vdt = self.vdt_gdf.loc[[idx_vdt]]

            curve_data_gdfs.append(nearest_curves)
            vdt_gdfs.append(nearest_vdt)

        # combine the VDT gdfs and curve data gdfs into one a piece
        self.vdt_gdf = pd.concat(vdt_gdfs)
        self.curve_data_gdf = pd.concat(curve_data_gdfs)

        # filter the XS_Out_File_df to only include cross-sections with 'row' and 'col' combinations that match the curve_data_gdf
        # keep only rows whose (Row,Col) is in curve_data_gdf
        XS_Out_File_df = (XS_Out_File_df.merge(self.curve_data_gdf[['COMID', 'Row', 'Col']].drop_duplicates(),
                                               on=['COMID', 'Row', 'Col'], how='inner'))

        # Step 1: Build geometry from lon/lat
        xs_lines = [LineString([Point(row['Lon1'], row['Lat1']), Point(row['Lon2'], row['Lat2'])])
                    for _, row in XS_Out_File_df.iterrows()]

        XS_Out_File_df['geometry'] = xs_lines

        # set crs to EPSG:4326 (WGS84) since we have lat lon not proj. coords
        self.xs_gdf = gpd.GeoDataFrame(XS_Out_File_df, geometry='geometry', crs="EPSG:4269")

        # then set the crs to match the dem proj.
        self.xs_gdf = self.xs_gdf.to_crs(self.rast_proj)  # or to projected_crs if defined separately


    def _process_geospatial_data(self):

        # Get the Spatial Information from the DEM Raster
        (minx, miny, maxx, maxy, dx, dy, ncols, nrows, dem_geoTransform, dem_projection) \
            = get_raster_info(self.dem_tif)

        projWin_extents = [minx, maxy, maxx, miny]
        # outputBounds = [minx, miny, maxx, maxy]  #https://gdal.org/api/python/osgeo.gdal.html

        # self.rivid_field = None # This is now set in process_dam

        # Create Land Dataset
        if os.path.isfile(self.land_tif):
            print(f'{self.land_tif} Already Exists')
        else:
            print(f'Creating {self.land_tif}')
            # Let's make sure all the GIS data is using the same coordinate system as the DEM
            self.lc_tif = update_crs(self.dem_tif, self.lc_tif)
            create_arc_lc_raster(self.lc_tif, self.land_tif, projWin_extents, ncols, nrows)

        # now we need to figure out if our dam_flowline and dem_reanalysis_flowfile exists and if not, create it
        if os.path.isfile(self.dam_shp) and os.path.isfile(self.reanalysis_csv):
            print(f'{self.dam_shp} Already Exists\n'
                  f'{self.reanalysis_csv} Already Exists')
            self.dam_gdf = gpd.read_file(self.dam_shp)

            # Ensure rivid_field is set correctly, even if files already exist
            if 'LINKNO' in self.dam_gdf.columns:
                self.rivid_field = 'LINKNO'
            else:
                self.rivid_field = 'hydroseq'

            if self.rivid_field == 'hydroseq':
                # subtract 5000000000000 from the rivid_field in the dam_gdf
                self.dam_gdf[self.rivid_field] = self.dam_gdf[self.rivid_field]
                # subtract 5000000000000 from the dnhydroseq field in the dam_gdf if it exists
                if 'dnhydroseq' in self.dam_gdf.columns:
                    self.dam_gdf['dnhydroseq'] = self.dam_gdf['dnhydroseq']

            self.rivids = self.dam_gdf[self.rivid_field].values

            print(f"Using existing files. Set rivid_field to: {self.rivid_field}")
            print(self.rivids)

        elif self.flowline_gdf is not None and os.path.isfile(self.dam_shp) is False and os.path.isfile(
                self.reanalysis_csv) is False:

            # This block now correctly uses self.rivid_field which was set in process_dam
            print('Running Function: Process_and_Write_Retrospective_Data_for_Dam')
            from . import streamflow_processing
            streamflow_processing.Process_and_Write_Retrospective_Data_for_Dam(self)

        elif self.flowline_gdf is None:
            raise ValueError("process_stream_network is False, but dam shapefile and flow file do not exist. "
                             "Please set process_stream_network to True or provide existing files.")

        # Create Stream Raster
        if os.path.isfile(self.strm_tif):
            print(f'{self.strm_tif} Already Exists')
        else:
            print(f'Creating {self.strm_tif}\n'
                  f'\tby rasterizing {self.dam_shp}')
            print(f'rivid_field: {self.rivid_field}')
            create_arc_strm_raster(self.dam_shp, self.strm_tif, self.dem_tif, self.rivid_field)

        # Clean Stream Raster
        if os.path.isfile(self.strm_tif_clean):
            print(f'{self.strm_tif_clean} Already Exists')
        else:
            print(f'Creating {self.strm_tif_clean}')
            clean_strm_raster(self.strm_tif, self.strm_tif_clean)

        # Get the unique values for all the stream ids
        (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = read_raster_w_gdal(
            self.strm_tif_clean)
        (RR, CC) = S.nonzero()
        num_strm_cells = len(RR)
        print(f'num_strm_cells: {num_strm_cells}')
        COMID_Unique = np.unique(S)
        # COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero
        COMID_Unique = COMID_Unique[np.where(COMID_Unique > 0)]
        COMID_Unique = np.sort(COMID_Unique).astype(int)
        num_comids = len(COMID_Unique)
        print(f'num_comids: {num_comids}')

        # Create the Bathy Input File

        # Let's extract the weir length real quick
        dam_df = pd.read_csv(self.csv_path)
        # Filter to the specific dam
        dam_df = dam_df[dam_df[self.id_field] == self.dam_id]
        self.weir_length = dam_df['weir_length'].values[0]

        print(f'Creating ARC Input File: {self.arc_input}')

        if not self.bathy_use_banks and self.known_baseflow is None:
            Q_bf_param = 'qout_median'

        elif self.bathy_use_banks and self.known_baseflow is None:
            Q_bf_param = 'rp2'

        elif not self.bathy_use_banks and self.known_baseflow is not None:
            Q_bf_param = 'known_baseflow'

        elif self.bathy_use_banks and self.known_channel_forming_discharge is not None:
            Q_bf_param = 'known_channel_forming_discharge'

        else:
            print('Error: bathy_use_banks and known_baseflow are both set to True.  Please check your inputs.\n')
            print('You want to pair known_baseflow with bathy_use_banks set to False.')
            sys.exit("Terminating: Invalid input combination.")

        self._create_arc_input_txt("COMID", Q_bf_param, 'rp2')

    def process_dam(self):
        # Folder Management
        self.arc_dir = os.path.join(self.output_dir, f'{self.name}', 'ARC_InputFiles')
        self.bathy_dir = os.path.join(self.output_dir, f'{self.name}', 'Bathymetry')
        self.strm_dir = os.path.join(self.output_dir, f'{self.name}', 'STRM')
        self.land_dir = os.path.join(self.output_dir, f'{self.name}', 'LAND')
        self.flow_dir = os.path.join(self.output_dir, f'{self.name}', 'FLOW')
        self.vdt_dir = os.path.join(self.output_dir, f'{self.name}', 'VDT')
        self.esa_lc_dir = os.path.join(self.output_dir, f'{self.name}', 'ESA_LC')
        self.xs_dir = os.path.join(self.output_dir, f'{self.name}', 'XS')

        # make directories
        dirs_to_make = [self.arc_dir, self.bathy_dir, self.strm_dir, self.land_dir,
                        self.flow_dir, self.vdt_dir, self.esa_lc_dir, self.xs_dir]
        for d in dirs_to_make:
            os.makedirs(d, exist_ok=True)

        # Datasets that can be good for a large domain
        self.manning = os.path.join(self.land_dir, 'AR_Manning_n_MED.txt')

        # Create a Baseline Manning N File
        print(f'Creating Manning n file: {self.manning}')
        create_mannings_esa(self.manning)

        # --- FIXED ---
        # Determine rivid_field based on flowline file name
        # This needs to be set regardless of whether process_stream_network is True or False
        if 'NHD' in self.flowline.name:
            self.rivid_field = 'hydroseq'
        else:
            self.rivid_field = 'LINKNO'
        print(f"Using rivid_field: {self.rivid_field}")
        # --- END FIX ---

        # This is the list of all the DEM files we will go through
        DEM_List = os.listdir(self.dem_dir)

        # Before we get too far ahead, let's make sure that our DEMs and Flowlines have the same coordinate system
        # we will assume that all DEMs in the DEM list have the same coordinate system
        self.flowline_gdf = None  # Initialize as None

        if self.process_stream_network:
            print(f'Reading in stream file: {self.flowline.name}')

            # Get the first DEM file to determine extent and CRS
            test_dem = next((file for file in DEM_List if file.endswith('.tif')), None)
            if test_dem is None:
                raise ValueError("No .tif files found in DEM directory")

            test_dem_path = os.path.join(self.dem_dir, test_dem)

            with rasterio.open(test_dem_path) as src:
                dem_bounds = src.bounds  # (left, bottom, right, top)
                dem_crs = src.crs  # Can be passed directly to GeoDataFrame

            # Create bounding box and GeoDataFrame
            dem_bbox_geom = box(*dem_bounds)
            dem_bbox_gdf = gpd.GeoDataFrame(geometry=[dem_bbox_geom], crs=dem_crs)

            # # Get DEM bounds and CRS using GDAL (more efficient than opening full raster)
            # raster_dataset = gdal.Open(test_dem_path)
            # gt = raster_dataset.GetGeoTransform()
            #
            # # Get the bounds of the raster (xmin, ymin, xmax, ymax)
            # xmin = gt[0]
            # xmax = xmin + gt[1] * raster_dataset.RasterXSize
            # ymin = gt[3] + gt[5] * raster_dataset.RasterYSize
            # ymax = gt[3]
            #
            # # Get DEM CRS
            # dem_proj = raster_dataset.GetProjection()
            # dem_spatial_ref = osr.SpatialReference()
            # dem_spatial_ref.ImportFromWkt(dem_proj)
            # dem_spatial_ref.AutoIdentifyEPSG()
            # dem_epsg_code = dem_spatial_ref.GetAuthorityCode(None)
            #
            # # Close the raster dataset
            # del raster_dataset
            #
            # # Create bounding box in DEM CRS
            # dem_bbox = (xmin, ymin, xmax, ymax)
            #
            # dem_bounds_geom = box(*dem_bbox)
            # dem_bbox_gdf = gpd.GeoDataFrame(geometry=[dem_bounds_geom], crs=f"EPSG:{dem_epsg_code}")

            print(f'DEM bounds: {dem_bounds}')
            print(f'DEM CRS: {dem_crs}')

            layer_name = None
            # Read stream file with bbox filter (much more efficient!)
            if self.flowline.name.endswith(".gdb"):
                with fiona.Env():
                    layers = fiona.listlayers(self.flowline)
                    if "geoglowsv2" in layers:
                        layer_name = "geoglowsv2"
                    elif "NHDFlowline" in layers:
                        layer_name = "NHDFlowline"
                    elif self.rivid_field == 'hydroseq' and "NHDFlowline" in layers:
                        layer_name = "NHDFlowline"
                    elif self.rivid_field == 'LINKNO' and "geoglowsv2" in layers:
                        layer_name = "geoglowsv2"
                    elif len(layers) == 1:
                        layer_name = layers[0]
                    else:
                        print(
                            f"Warning: Could not auto-determine layer in {self.flowline}. Found: {layers}. Trying first layer.")
                        layer_name = layers[0]

                    with fiona.open(self.flowline, layer=layer_name) as src:
                        stream_crs = src.crs
            elif self.flowline.name.endswith(".shp") or self.flowline.name.endswith(".gpkg"):
                if self.flowline.name.endswith(".gpkg"):
                    layers = fiona.listlayers(self.flowline)
                    if "geoglowsv2" in layers:
                        layer_name = "geoglowsv2"
                    elif "NHDFlowline" in layers:
                        layer_name = "NHDFlowline"
                    elif self.rivid_field == 'hydroseq' and "NHDFlowline" in layers:
                        layer_name = "NHDFlowline"
                    elif self.rivid_field == 'LINKNO' and "geoglowsv2" in layers:
                        layer_name = "geoglowsv2"
                    elif len(layers) == 1:
                        layer_name = layers[0]
                    else:
                        print(
                            f"Warning: Could not auto-determine layer in {self.flowline}. Found: {layers}. Trying first layer.")
                        layer_name = layers[0]

                stream_meta = gpd.read_file(self.flowline, rows=1, layer=layer_name)
                stream_crs = stream_meta.crs
            else:
                raise ValueError(f"Unsupported flowline file type: {self.flowline.name}")

            dem_bbox_gdf = dem_bbox_gdf.to_crs(stream_crs)
            bbox_geom = dem_bbox_gdf.geometry.iloc[0].bounds

            self.flowline_gdf = gpd.read_file(self.flowline, layer=layer_name, engine='fiona', bbox=bbox_geom)

            # if the river id field has values greater than 32-bit int, subtract 5000000000000 from it
            if self.rivid_field == 'hydroseq':
                if self.flowline_gdf['hydroseq'].max() > 2147483647:
                    self.flowline_gdf['hydroseq'] = self.flowline_gdf['hydroseq'] - 5000000000000
                    # do the same for dnhydroseq if it exists
                    if 'dnhydroseq' in self.flowline_gdf.columns:
                        self.flowline_gdf['dnhydroseq'] = self.flowline_gdf['dnhydroseq'] - 5000000000000
                    print('Adjusted hydroseq values by subtracting 5000000000000 to fit within 32-bit integer range.')

            print(f'Filtered stream features: {len(self.flowline_gdf)} (vs reading entire file)')

            # removing any lingering NoneType geometries
            self.flowline_gdf = self.flowline_gdf[~self.flowline_gdf.geometry.isna()]

            print('Converting the coordinate system of the stream file to match the DEM files, if necessary')

            # Check if stream CRS matches DEM CRS
            if self.flowline_gdf.crs != dem_crs:
                print("DEM and Stream Network have different coordinate systems...")
                print(f"Stream CRS: {self.flowline_gdf.crs}, and DEM CRS: {dem_crs}")
                # Reproject the stream data to match DEM CRS
                self.flowline_gdf = self.flowline_gdf.to_crs(dem_crs)

        # Now go through each DEM dataset
        for DEM in DEM_List:
            if DEM.endswith('.tif'):
                self._assess_dam(DEM)

        # delete the ESA_LC_Folder and the data in it
        # Loop through all files in the directory and remove them
        for file in Path(self.esa_lc_dir).glob("*"):
            try:
                if file.is_file():
                    # Adjust file permissions before deletion
                    if platform.system() == "Windows":
                        os.chmod(file, stat.S_IWRITE)  # Remove read-only attribute on Windows
                    else:
                        os.chmod(file, stat.S_IWUSR)  # Give user write permission on Unix systems
                    os.remove(file)
                    print(f"process_dam: Deleted file: {file}")
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
        if os.path.exists(self.esa_lc_dir):
            # Adjust file permissions before deletion
            if platform.system() == "Windows":
                os.chmod(self.esa_lc_dir, stat.S_IWRITE)  # Remove read-only attribute on Windows
            else:
                os.chmod(self.esa_lc_dir, stat.S_IWUSR)  # Give user write permission on Unix systems
            os.rmdir(self.esa_lc_dir)
            print(f"process_dam: Deleted empty folder: {self.esa_lc_dir}")
        else:
            print(f"process_dam: Folder {self.esa_lc_dir} does not exist.")

    def _assess_dam(self, dem):
        dem_name = dem
        # FileName = os.path.splitext(DEM_Name)[0] # grabs just name without extension
        self.dem_tif = os.path.join(self.dem_dir, dem_name)

        # Input Dataset
        self.arc_input = os.path.join(self.arc_dir, f'ARC_Input_{self.dam_id}.txt')

        # Datasets to be Created
        self.dam_shp = os.path.join(self.strm_dir, f"{self.dam_id}_StrmShp.gpkg")
        self.reanalysis_csv = os.path.join(self.flow_dir, f"{self.dam_id}_Reanalysis.csv")

        self.strm_tif = os.path.join(self.strm_dir, f'{self.dam_id}_STRM_Raster.tif')
        self.strm_tif_clean = self.strm_tif.replace('.tif', '_Clean.tif')
        self.land_tif = self.land_raster

        self.vdt_txt = os.path.join(self.vdt_dir, f'{self.dam_id}_VDT_Database.txt')
        self.curvefile_csv = os.path.join(self.vdt_dir, f'{self.dam_id}_CurveFile.csv')

        # these are the files that will be created by the code
        local_vdt_shp = os.path.join(self.vdt_dir, f'{self.dam_id}_Local_VDT_Database.gpkg')
        local_cf_shp = os.path.join(self.vdt_dir, f'{self.dam_id}_Local_CurveFile.gpkg')

        self.bathy_tif = os.path.join(self.bathy_dir, f'{self.dam_id}_ARC_Bathy.tif')

        self.xs_txt = os.path.join(self.xs_dir, f'{self.dam_id}_XS_Out.txt')
        local_xs_shp = os.path.join(self.xs_dir, f'{self.dam_id}_Local_XS_Lines.gpkg')

        # Get the details of the DEM file
        lon_1, lat_1, lon_2, lat_2, dx, dy, ncols, nrows, geoTransform, self.rast_proj = get_raster_info(self.dem_tif)

        # set up a WGS84 ellipsoid
        geod = Geod(ellps='WGS84')

        raster_crs = CRS.from_wkt(self.rast_proj)  # or use src.crs directly if you have it

        if raster_crs.is_geographic:
            # Use geod.inv as before (your original logic), assuming lon/lat in degrees
            _, _, res_x_m = geod.inv(lon_1, lat_1, lon_1 + dx, lat_1)
            _, _, res_y_m = geod.inv(lon_1, lat_1, lon_1, lat_1 + dy)
        else:
            # It's already in meters — just use dx, dy directly
            res_x_m = dx
            res_y_m = abs(dy)

        print(f"Dam Assessment: Exact pixel size of DEM: {res_x_m:.3f} m × {res_y_m:.3f} m")

        # Calculate the average distance increment for upstream processing
        self.avg_dist_upstream = (res_x_m + res_y_m) / 2

        # Download and Process Land Cover Data
        self.lc_tif = None
        if not os.path.exists(self.land_tif):

            # Get geometry in original projection
            geom = esa.Get_Polygon_Geometry(lon_1, lat_1, lon_2, lat_2)

            # Check if raster projection is WGS 84
            wgs84_crs = CRS.from_epsg(4326)

            if raster_crs != wgs84_crs:
                # Convert geometry to WGS 84
                transformer = Transformer.from_crs(raster_crs, wgs84_crs, always_xy=True)
                geom = transform(transformer.transform, geom)

            self.lc_tif = esa.Download_ESA_WorldLandCover(self.esa_lc_dir, geom)

        # This function sets-up the Input files for ARC and FloodSpreader
        # It also does some of the geospatial processing
        self._process_geospatial_data()

        # read in the reanalysis streamflow and break the code if the dataframe is empty or if the streamflow is all 0
        dem_reanalysis_flowfile_df = pd.read_csv(self.reanalysis_csv)
        if (dem_reanalysis_flowfile_df.empty
                or 'rp2' not in dem_reanalysis_flowfile_df.columns  # Check for a key flow col
                or len(dem_reanalysis_flowfile_df.index) == 0):
            print(
                f"Dam_Assessment: Results for {dem} are not possible because we don't have valid streamflow estimates...")
            return

        # Create our Curve and VDT Database data
        if not os.path.exists(self.bathy_tif) or not os.path.exists(self.vdt_txt) or not os.path.exists(
                self.curvefile_csv):
            print(f'Dam_Assessment: Cannot find bathy file, so creating {self.bathy_tif}')
            arc = Arc(self.arc_input)
            arc.run()  # Runs ARC

        # Check if ARC ran successfully by seeing if the outputs exist
        if not os.path.exists(self.vdt_txt) or not os.path.exists(self.curvefile_csv):
            print(f"ARC run appears to have failed. Skipping post-processing for {self.dam_id}.")
            return

        # Now we need to use the dam_shp and vdt_txt to find the stream cells at distance increments below the dam
        self._find_strm_up_downstream()

        # output the results to shapefiles
        self.vdt_gdf.to_file(local_vdt_shp)
        self.curve_data_gdf.to_file(local_cf_shp)
        self.xs_gdf.to_file(local_xs_shp)
