import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_dem_to_nad83(src_path, dst_crs="EPSG:4269"):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Use a temporary file for safety
        temp_path = src_path.replace(".tif", "_tmp.tif")

        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)

    # Overwrite the original file
    os.replace(temp_path, src_path)


def batch_reproject_dems(parent_folder):
    for subdir, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(".tif"):
                input_path = os.path.join(subdir, file)
                print(f"Reprojecting: {input_path}")
                reproject_dem_to_nad83(input_path)


# Example usage
batch_reproject_dems(r"E:\LHD_Project\LHD_DEMs")
