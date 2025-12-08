import geopandas as gpd
import os
import pandas as pd

# Directory with your GeoPackage files
directory = "E:/TDX_HYDRO"

# Collect all .gpkg files in the directory
gpkg_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.gpkg')]

# This will hold all reprojected GeoDataFrames
all_gdfs = []

for file in gpkg_files:
    # Read the first (or only) layer from the GeoPackage
    gdf = gpd.read_file(file)

    # Reproject to EPSG:4326 if needed
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    elif gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)  # assume it's 4326 if no CRS is defined

    all_gdfs.append(gdf)

# Concatenate all GeoDataFrames into one
merged_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")

# Save the merged GeoDataFrame to a new GeoPackage
merged_gdf.to_file("E:/TDX_HYDRO/streams.gpkg", driver="GPKG")
