import os
import requests
from pathlib import Path
from tqdm.auto import tqdm
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer, CRS
from shapely.ops import transform  # Explicitly import transform
import rasterio  # Needed for _get_raster_info

try:
    import gdal
except ImportError:
    from osgeo import gdal  # Fallback for GDAL


# --- UTILITY FUNCTIONS ---

def _get_raster_info(dem_tif: str):
    """
    Retrieves essential geospatial metadata from a DEM file using rasterio.
    (Logic adapted from rathcelon/classes.py)
    """
    with rasterio.open(dem_tif) as dataset:
        geoTransform = dataset.transform
        ncols = dataset.width
        nrows = dataset.height
        minx = geoTransform.c  # xmin
        dx = geoTransform.a
        maxy = geoTransform.f  # ymax
        dy = geoTransform.e
        maxx = minx + dx * ncols
        miny = maxy + dy * nrows
        Rast_Projection = dataset.crs.to_wkt()

    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection


def _get_polygon_geometry(lon_1: float, lat_1: float, lon_2: float, lat_2: float) -> Polygon:
    """Creates a Shapely Polygon from two corner coordinates (in EPSG:4326)."""
    return Polygon([[min(lon_1, lon_2), min(lat_1, lat_2)],
                    [min(lon_1, lon_2), max(lat_1, lat_2)],
                    [max(lon_1, lon_2), max(lat_1, lat_2)],
                    [max(lon_1, lon_2), min(lat_1, lat_2)]])


def _download_and_save_tile(url: str, out_fn: Path) -> str:
    """Downloads a single tile file in chunks."""
    r = requests.get(url, allow_redirects=True, stream=True)
    r.raise_for_status()

    with open(out_fn, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return str(out_fn)


def _download_esa_land_cover(geom: Polygon, output_dir: str, dem_crs: CRS, dem_proj_wkt: str,
                             year: int = 2021) -> str | None:
    """Downloads and merges ESA WorldCover tiles that intersect the given geometry."""

    # ... (file path and initial check setup)
    land_cover_file_name = "merged_ESA_LC.tif"
    merged_output_path = os.path.join(output_dir, land_cover_file_name)

    if os.path.exists(merged_output_path):
        print(f"Using existing raw merged land cover file: {merged_output_path}. Skipping download.")
        return merged_output_path

    print("Raw merged land cover file not found. Starting download/merge process...")
    os.makedirs(output_dir, exist_ok=True)

    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    grid_url = f'{s3_url_prefix}/esa_worldcover_grid.geojson'

    # Reproject the geometry to WGS 84 if DEM is not already in it, to align with ESA grid
    wgs84_crs = CRS.from_epsg(4326)

    if dem_crs != wgs84_crs:
        transformer = Transformer.from_crs(dem_crs, wgs84_crs, always_xy=True)

        # --- ROBUST FIX: Define a nested function for transformation ---
        # This resolves the 'unfilled parameters' error by correctly matching the
        # function signature expected by shapely.ops.transform (which passes single points).
        def reproject_func(x, y, z=None):
            return transformer.transform(x, y, z)

        geom = transform(reproject_func, geom)
        # --- END FIX ---

    try:
        grid = gpd.read_file(grid_url, crs="epsg:4326")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ESA grid file: {e}")
        return None

    tiles = grid[grid.intersects(geom)]

    if tiles.empty:
        print("WARNING: No ESA WorldCover tiles intersect the bounding box.")
        return None

    version = {2020: 'v100', 2021: 'v200'}[year]

    temp_downloaded_tiles = []

    # 3. Download individual tiles
    for tile in tqdm(tiles.ll_tile, desc="Downloading ESA Tiles"):
        url = f"{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        out_fn = Path(output_dir) / Path(url).name

        if os.path.isfile(out_fn):
            temp_downloaded_tiles.append(str(out_fn))
            continue

        try:
            downloaded_path = _download_and_save_tile(url, out_fn)
            temp_downloaded_tiles.append(downloaded_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {tile} (Request failed): {e}")

    # 4. Merge all downloaded tiles (using optimized GDAL Warp)
    if temp_downloaded_tiles:
        print(f"Merging {len(temp_downloaded_tiles)} land cover tiles...")

        merged_raster = gdal.Warp(
            merged_output_path,
            temp_downloaded_tiles,
            options=gdal.WarpOptions(
                format='GTiff',
                creationOptions=['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS']
            )
        )

        if merged_raster:
            merged_raster.FlushCache()
            del merged_raster

        print("Merge complete.")
        return merged_output_path

    return None


def _create_aligned_land_raster(
        raw_lc_tif: str,
        aligned_land_tif: str,
        proj_win_extents: list,
        ncols: int,
        nrows: int
):
    """
    Clips and resamples the raw merged Land Cover TIFF (raw_lc_tif) to match
    the extent and grid of the target DEM.
    """

    print(f"Creating aligned land raster: {aligned_land_tif}")

    # Use gdal.Translate with optimization options for clipping and resampling
    options = gdal.TranslateOptions(
        projWin=proj_win_extents,
        width=ncols,
        height=nrows,
        creationOptions=['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS']
    )

    gdal.Translate(aligned_land_tif, raw_lc_tif, options=options)

    print(f"Aligned Land Cover saved to: {aligned_land_tif}")

    return aligned_land_tif


# --- MAIN EXPORTED FUNCTION ---

def download_land_raster(lhd_id: int, dem_path: str, land_dir: str):
    """
    Download and save the final Land Cover TIF.

    Parameters:
    lhd_id (int): The ID of the low-head dam.
    dem_path (str): Path to the DEM file.
    land_dir (str): Directory where the final LAND_Raster.tif will be placed.

    Returns:
    str: The full path to the final aligned Land Cover TIFF file.
    """

    # 1. DEFINE FILE PATHS AND CHECK FINAL OUTPUT (LAND_Raster.tif)
    final_land_tif_name = f"{lhd_id}_LAND_Raster.tif"
    final_land_tif_path = os.path.join(land_dir, final_land_tif_name)

    if os.path.exists(final_land_tif_path):
        print(f"Final Land Raster already exists: {final_land_tif_path}")
        return final_land_tif_path

    # --- INTERMEDIATE RAW FILE LOCATION ---
    # The raw merged ESA file is saved in a 'raw_esa' subdirectory inside land_dir
    # to separate it from the final processed raster.
    raw_esa_dir = os.path.join(land_dir, 'raw_esa')
    os.makedirs(raw_esa_dir, exist_ok=True)

    # 2. GET ESA BOUNDING BOX FROM DEM INFO
    print(f"Extracting bounds from DEM: {dem_path}")
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection_WKT) \
        = _get_raster_info(dem_path)

    proj_win_extents = [minx, maxy, maxx, miny]
    dem_crs = CRS.from_wkt(Rast_Projection_WKT)

    # 3. DOWNLOAD OR REUSE RAW ESA DATA
    # _download_esa_land_cover handles the download/reuse/merge logic
    raw_lc_tif_path = _download_esa_land_cover(
        geom=_get_polygon_geometry(minx, miny, maxx, maxy),
        output_dir=raw_esa_dir,
        dem_crs=dem_crs,
        dem_proj_wkt=Rast_Projection_WKT
    )

    if not raw_lc_tif_path:
        raise RuntimeError("Failed to download or find raw merged ESA land cover data.")

    # 4. CREATE THE ALIGNED LAND RASTER
    # This clips and aligns the raw data to the specific DEM grid
    _create_aligned_land_raster(
        raw_lc_tif=raw_lc_tif_path,
        aligned_land_tif=final_land_tif_path,
        proj_win_extents=proj_win_extents,
        ncols=ncols,
        nrows=nrows
    )

    # 5. CLEAN UP INTERMEDIATE RAW TILE FILES
    print("Cleaning up individual downloaded ESA tiles...")
    for item in os.listdir(raw_esa_dir):
        item_path = os.path.join(raw_esa_dir, item)
        # Delete only the individual tile files (those ending in .tif but not merged_ESA_LC.tif)
        if item.endswith('.tif') and item != 'merged_ESA_LC.tif':
            os.remove(item_path)

    return final_land_tif_path