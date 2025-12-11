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
    with rasterio.open(dem_tif) as dataset:
        geoTransform = dataset.transform
        ncols = dataset.width
        nrows = dataset.height
        minx = geoTransform.c  # x_min
        dx = geoTransform.a
        maxy = geoTransform.f  # y_max
        dy = geoTransform.e
        maxx = minx + dx * ncols
        miny = maxy + dy * nrows
        Rast_Projection = dataset.crs.to_wkt()

    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection


def _get_polygon_geometry_from_bounds(minx: float, miny: float, maxx: float, maxy: float, crs: CRS) -> gpd.GeoSeries:
    """Creates a GeoSeries containing a Shapely Polygon from bounds in the native CRS."""
    geom = Polygon([
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny]
    ])
    # Returns a GeoSeries with the correct CRS.
    return gpd.GeoSeries([geom], crs=crs)


def _download_and_save_tile(url: str, out_fn: Path) -> str:
    """Downloads a single tile file in chunks."""
    r = requests.get(url, allow_redirects=True, stream=True)
    r.raise_for_status()

    with open(out_fn, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return str(out_fn)


def _download_esa_land_cover(geom: Polygon, output_dir: str, dem_crs: CRS, year: int = 2021) -> str | None:
    """
    Downloads and merges ESA WorldCover tiles that intersect the given geometry.
    The input `geom` is a shapely Polygon object in the CRS specified by `dem_crs`.
    """

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

    # --- FIX: Reproject the input geometry (which is in dem_crs) to WGS84 ---
    wgs84_crs = CRS.from_epsg(4326)

    # 1. Check if reprojection is needed
    if dem_crs != wgs84_crs:
        print(f"Reprojecting DEM bounds from {dem_crs.to_string()} to EPSG:4326 for spatial query.")
        transformer = Transformer.from_crs(dem_crs, wgs84_crs, always_xy=True)

        # Function to transform points (required by shapely.ops.transform)
        def reproject_func(x, y, z=None):
            return transformer.transform(x, y, z)

        # Transform the Polygon geometry
        geom_wgs84 = transform(reproject_func, geom)
    else:
        geom_wgs84 = geom
    # --- END FIX ---

    try:
        grid = gpd.read_file(grid_url, crs="epsg:4326")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ESA grid file: {e}")
        return None

    # Use the correctly reprojected WGS84 geometry for intersection
    tiles = grid[grid.intersects(geom_wgs84)]

    if tiles.empty:
        print("WARNING: No ESA WorldCover tiles intersect the bounding box.")
        return None

    # ... (rest of the function is unchanged, handles download and merge)
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
# ... (function remains unchanged)
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
    raw_esa_dir = os.path.join(land_dir, 'raw_esa')
    os.makedirs(raw_esa_dir, exist_ok=True)

    # 2. GET ESA BOUNDING BOX FROM DEM INFO
    print(f"Extracting bounds from DEM: {dem_path}")
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection_WKT) \
        = _get_raster_info(dem_path)

    proj_win_extents = [minx, maxy, maxx, miny]
    dem_crs = CRS.from_wkt(Rast_Projection_WKT)

    # Use the native bounds and CRS to create the polygon
    dem_gs = _get_polygon_geometry_from_bounds(minx, miny, maxx, maxy, dem_crs)
    dem_polygon = dem_gs.iloc[0]

    # 3. DOWNLOAD OR REUSE RAW ESA DATA
    # _download_esa_land_cover now handles reprojection internally
    raw_lc_tif_path = _download_esa_land_cover(
        geom=dem_polygon,
        output_dir=raw_esa_dir,
        dem_crs=dem_crs,
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

    # 5. CLEAN UP INTERMEDIATE RAW TILE FILES (Including the raw merged file)
    print("Cleaning up individual downloaded ESA tiles...")
    merged_esa_filename = "merged_ESA_LC.tif"
    raw_lc_tif_path_full = os.path.join(raw_esa_dir, merged_esa_filename)

    for item in os.listdir(raw_esa_dir):
        item_path = os.path.join(raw_esa_dir, item)
        # Delete only the individual tile files (those ending in .tif but not merged_ESA_LC.tif)
        if item.endswith('.tif') and item != merged_esa_filename:
            os.remove(item_path)

    # Delete the large intermediate raw merged file
    if os.path.exists(raw_lc_tif_path_full):
        try:
            os.remove(raw_lc_tif_path_full)
            print(f"Removed intermediate raw merged file: {raw_lc_tif_path_full}")
        except Exception as e:
            print(f"Warning: Could not remove {raw_lc_tif_path_full}: {e}")

    return final_land_tif_path