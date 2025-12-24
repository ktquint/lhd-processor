# Program downloads esa world land cover datasets and also creates a water mask
# https://esa-worldcover.org/en/data-access

# use requests library to download them
import requests  # conda install anaconda::requests
from tqdm.auto import tqdm  # provides a progressbar     #conda install conda-forge::tqdm
from shapely.geometry import Polygon  # , LineString    #conda install conda-forge::shapely
# import os  # Added for Path and os.path.isfile
# from pathlib import Path  # Added for Path

try:
    import gdal  # conda install conda-forge::gdal
    import gdal_array
except ImportError:
    from osgeo import gdal, gdal_array

# local imports
from .classes import *


def Geom_Based_On_Country(country, Shapefile_Use):
    ne = gpd.read_file(Shapefile_Use)

    # get AOI geometry (Italy in this case)
    geom = ne[ne.NAME == country].iloc[0].geometry
    return geom


def Download_ESA_WorldLandCover(output_dir, geom, year=2021):
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    # load natural earth low res shapefile

    # load worldcover grid
    url = f'{s3_url_prefix}/esa_worldcover_grid.geojson'
    grid = gpd.read_file(url, crs="epsg:4326")

    # get grid tiles intersecting AOI
    tiles = grid[grid.intersects(geom)]
    print(tiles)

    # select version tag, based on the year
    version = {2020: 'v100',
               2021: 'v200'}[year]

    lc_list = []
    # just open with gdal
    # gdal warp to change output bounds.
    for tile in tqdm(tiles.ll_tile, desc="Downloading ESA Tiles"):
        url = f"{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        out_fn = Path(output_dir) / Path(url).name
        lc_list.append(str(out_fn))

        if os.path.isfile(out_fn):
            print('Already Exists: ' + str(out_fn))
        else:
            # Use stream=True to download chunk-by-chunk and write directly to disk
            r = requests.get(url, allow_redirects=True, stream=True)

            # Raise an HTTPError for bad responses (4xx or 5xx)
            r.raise_for_status()

            with open(out_fn, 'wb') as f:
                # Write file content in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

    # let's merge the list of tiles together
    LandCoverFile = os.path.join(output_dir, "merged_ESA_LC.tif")

    # Merge rasters
    merged_raster = gdal.Warp(LandCoverFile, lc_list, options=gdal.WarpOptions(format='GTiff'))

    # Ensure data is written and file is closed properly
    if merged_raster:
        merged_raster.FlushCache()  # Save changes
        del merged_raster  # Close dataset

    return LandCoverFile


def Create_Water_Mask(lc_file, waterboundary_file, water_value):
    (RastArray, n_cols, n_rows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection) = read_raster_w_gdal(
        lc_file)
    RastArray = np.where(RastArray == water_value, 1, 0)  # Streams are identified with zeros
    write_output_raster(waterboundary_file, RastArray, geotransform, Rast_Projection)
    return


def Get_Polygon_Geometry(lon_1, lat_1, lon_2, lat_2):
    return Polygon([[min(lon_1, lon_2), min(lat_1, lat_2)], [min(lon_1, lon_2), max(lat_1, lat_2)],
                    [max(lon_1, lon_2), max(lat_1, lat_2)], [max(lon_1, lon_2), min(lat_1, lat_2)]])


if __name__ == "__main__":

    # Just leave blank if using Option 1 or 2 below
    if len(sys.argv) > 1:
        DEM_File = sys.argv[1]
        print(f'Input DEM File: {DEM_File}')
    else:
        DEM_File = 'NED_n39w090_Clipped.tif'
        print(f'Did not input DEM, going with default: {DEM_File}')

    year = 2021  # setting this to 2020 will download the v100 product instead
    output_folder = 'ESA_LC'  # use current directory or set a different one to store downloaded files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    '''
    ###Option 1 - Get Geometry from a Shapefile
    #Get Geometry based on Country and Shapefile
    geom = Geom_Based_On_Country('Cyprus', 'ne_110m_admin_0_countries.shp')


    ###Option 2 - Get Geometry from Lat/Long Bounding Coordinates
    #Get Geometry based on Latitude and Longitude
    lat_1 = 42.5
    lat_2 = 43.0 
    lon_1 = -106.0 
    lon_2 = -106.5
    d = {'col1': ['name1'], 'geometry': LineString([[lon_1, lat_1], [lon_2, lat_2]])}
    geom = Get_Polygon_Geometry(lon_1, lat_1, lon_2, lat_2)
    '''

    ###Option 3 - Get Geometry from Raster File
    (lon_1, lat_1, lon_2, lat_2, dx, dy, ncols, nrows, geoTransform, Rast_Projection) = get_raster_info(DEM_File)
    geom = Get_Polygon_Geometry(lon_1, lat_1, lon_2, lat_2)

    lc_list = Download_ESA_WorldLandCover(output_folder, geom, year)

    for lc_file in lc_list:
        lc_file_str = str(lc_file)

        if DEM_File != '':
            LAND_File_Clipped = lc_file_str.replace('.tif', '_Clipped.tif')
            if os.path.isfile(LAND_File_Clipped):
                print(f'Already Exists: {LAND_File_Clipped}')
            else:
                print(f'Creating: {LAND_File_Clipped}')
                create_arc_lc_raster(lc_file_str, LAND_File_Clipped, [lon_1, lat_2, lon_2, lat_1], ncols, nrows)
            # lc_file_str = LAND_File_Clipped

        '''
        waterboundary_file = lc_file_str.replace('.tif','_wb.tif')
        if os.path.isfile(waterboundary_file):
            print('Already Exists: ' + str(waterboundary_file))
        else:
            print('Creating ' + str(waterboundary_file))
            Create_Water_Mask(lc_file_str, waterboundary_file, 80)
        '''
