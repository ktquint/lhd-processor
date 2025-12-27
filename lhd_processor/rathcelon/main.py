"""
Here is a list of where Low Head Dams are located.  What we're wanting to do:

1.) Identify the stream reach(es) that are associated with the Low-Head Dam.
2.) Determine the approximate Top-Width (TW) of the Dam (or the stream itself)
3.) Determine the average (base) flow for the stream as well as the seasonal high flow (not a flood, just a normal high-flow).
4.) Go downstream approximately TW distance and pull a perpendicular cross-section.
5.) Go downstream another TW distance (2*TW from the dam) and pull another perpendicular cross-section.
6.) Go downstream another TW distance (3*TW from the dam) and pull another perpendicular cross-section.
7.) For each cross-section estimate the bathymetry.
8.) For each cross-section calculate the rating curve of the cross-section.  Slope can likely be calculated from steps 4-6.

"""

# build-in imports
import os       # working with the operating system
import argparse # parses command-line args

# third-party imports
from geopandas import GeoDataFrame  # object for geospatial data + tabular data

try:
    import gdal                         # geospatial data abstraction library (gdal)
except ImportError:
    from osgeo import gdal, ogr, osr    # import from osgeo if direct import doesn't work

import json         # json encoding/decoding
from shapely.geometry import Point, LineString #, MultiLineString, shape, mapping

# local imports
from .classes import RathCelonDam


def create_flowfile(main_flowfile: str, flowfile_name: str, output_id: int|str, q_param: str) -> None:
    with open(main_flowfile, 'r') as infile:
        lines = infile.readlines()

    headers = lines[0].strip().split(',')

    try:
        q_index = headers.index(q_param)
        id_index = headers.index(output_id)
    except ValueError as e:
        raise ValueError(f"Missing excepted column in input file: {e}")

    with open(flowfile_name, 'w') as outfile:
        outfile.write(f"{output_id},{q_param}")
        for line in lines[1:]:
            values = line.strip().split(',')
            outfile.write(f"{values[id_index]},{values[q_index]}\n")



def find_nearest_idx(point, tree, gdf: GeoDataFrame):
    """Find the nearest index and corresponding data for a given point using spatial indexing."""
    nearest_idx = tree.nearest(point)  # Directly query with the geometry
    return nearest_idx, gdf.iloc[nearest_idx]


# Function to get a point strictly on the network
def get_point_on_stream(line, target_distance) -> Point:
    """
    Walks along a LineString and finds a point at a given distance,
    ensuring it follows the stream path without deviating.
    """
    current_distance = 0  # Start distance tracking

    for i in range(len(line.coords) - 1):
        start_pt = Point(line.coords[i])
        end_pt = Point(line.coords[i + 1])

        segment = LineString([start_pt, end_pt])
        segment_length = segment.length

        # If target_distance falls within this segment
        if current_distance + segment_length >= target_distance:
            # Compute exact location on this segment
            remaining_distance = target_distance - current_distance
            return segment.interpolate(remaining_distance)

        current_distance += segment_length  # Update walked distance

    return Point(line.coords[-1])  # If we exceed the length, return last point


def walk_stream_for_point(line, target_distance: float) -> Point:
    """
        Walks along a LineString segment-by-segment, stopping at the exact
        cumulative distance to ensure the point stays on the stream network.
    """
    current_distance = 0

    for i in range(len(line.coords) - 1):
        start_pt = Point(line.coords[i])
        end_pt = Point(line.coords[i + 1])
        segment = LineString([start_pt, end_pt])
        segment_length = segment.length

        # If our target point is within this segment
        if current_distance + segment_length >= target_distance:
            remaining_distance = target_distance - current_distance
            return segment.interpolate(remaining_distance)  # Stay exactly on the stream

        current_distance += segment_length  # Move forward

    return Point(line.coords[-1])  # Return last point if over length


def process_json_input(json_file):
    """Process input from a JSON file."""
    with open(json_file, 'r') as file:
        print(f'Opening {file}')
        data = json.load(file)
        print(data)

    # Helper to handle NaNs/None/Empty strings safely
    def safe_normpath(path_val):
        if isinstance(path_val, str) and path_val.strip():
            return os.path.normpath(path_val)
        return None

    dams = data.get("dams", [])
    for dam in dams:
        dam_name = dam.get("name")

        # Update these lines to use the safe helper
        dam_csv = safe_normpath(dam.get("dam_csv"))
        flowline = safe_normpath(dam.get("flowline"))
        dem_dir = safe_normpath(dam.get("dem_dir"))
        output_dir = safe_normpath(dam.get("output_dir"))
        land_raster = safe_normpath(dam.get("land_raster"))

        dam_dict = {
            "name": dam_name,
            "dam_csv": dam_csv,
            "dam_id_field": dam.get("dam_id_field"),
            "dam_id": int(dam.get("dam_id")),
            "flowline": flowline,
            "dem_dir": dem_dir,
            "land_raster": land_raster,
            "output_dir": output_dir,
            "bathy_use_banks": dam.get("bathy_use_banks", False),
            "flood_waterlc_and_strm_cells": dam.get("flood_waterlc_and_strm_cells", False),
            "process_stream_network": dam.get("process_stream_network", False),
            "find_banks_based_on_landcover": dam.get("find_banks_based_on_landcover", True),
            "create_reach_average_curve_file": dam.get("create_reach_average_curve_file", False),
            "known_baseflow": dam.get("known_baseflow", None),
            "known_channel_forming_discharge": dam.get("known_channel_forming_discharge", None),
            "upstream_elevation_change_threshold": dam.get("upstream_elevation_change_threshold", 1.0),
            "streamflow": safe_normpath(dam.get("streamflow")),
        }

        # Ensure the output directory exists (check if it's not None first)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Processing dam: {dam_name} with parameters: {dam_dict}")

        # Call your existing processing logic here
        dam_i = RathCelonDam(**dam_dict)
        dam_i.process_dam()


def normalize_path(path):
    if path:
        return os.path.normpath(path)
    return None


def process_cli_arguments(args):
    """Process input from CLI arguments."""
    output_dir = args.output_dir
    dam_name = args.dam
    dam_dict = {
        "name": dam_name,
        "dam_csv": normalize_path(args.dam_csv),
        "dam_id_field": args.dam_id_field,
        "dam_id": args.dam_id,
        "flowline": normalize_path(args.flowline),
        "dem_dir": normalize_path(args.dem_dir),
        "land_raster": normalize_path(args.land_raster),
        "bathy_use_banks": args.bathy_use_banks,
        "output_dir": normalize_path(output_dir),
        "process_stream_network": args.process_stream_network,
        "find_banks_based_on_landcover": args.find_banks_based_on_landcover,
        "create_reach_average_curve_file": args.create_reach_average_curve_file,
        "known_baseflow": args.known_baseflow,
        "known_channel_forming_discharge": args.known_channel_forming_discharge,
        "upstream_elevation_change_threshold": args.upstream_elevation_change_threshold,
        "streamflow": normalize_path(args.streamflow) if args.streamflow else None,
    }

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing dam: {args.dam} with parameters: {dam_dict}")
    print(f"Results will be saved in: {output_dir}")

    # Call the existing processing logic here
    # process_dam(dam_dict)
    dam_i = RathCelonDam(**dam_dict)
    dam_i.process_dam()


def main():
    parser = argparse.ArgumentParser(description="Process rating curves on streams below a dam.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand for JSON input
    json_parser = subparsers.add_parser("json", help="Process watersheds from a JSON file")
    json_parser.add_argument("json_file", type=str, help="Path to the JSON file")

    # Subcommand for CLI input
    cli_parser = subparsers.add_parser("cli", help="Process watershed parameters via CLI")
    cli_parser.add_argument("dam", type=str, help="Dam name")
    cli_parser.add_argument("dam_csv", type=str, help="Path to the dam csv file")
    cli_parser.add_argument("dam_id_field", type=str, help="Name of the csv field with the dam id")
    cli_parser.add_argument("dam_id", type=int, help="ID of the dam in the damn_id_field")
    cli_parser.add_argument("flowline", type=str, help="Path to the flowline shapefile")
    cli_parser.add_argument("dem_dir", type=str, help="Directory containing DEM files")
    cli_parser.add_argument("land_raster", type=str, help="Path to an existing land-use raster file (e.g., {ID}_Land_Raster.tif) to use instead of downloading ESA data.")
    cli_parser.add_argument("output_dir", type=str, help="Directory where results will be saved")
    cli_parser.add_argument("--bathy_use_banks", action="store_true", help="Use bathy banks for processing")
    cli_parser.add_argument("--process_stream_network", action="store_true", help="Clean DEM data before processing")
    cli_parser.add_argument("--find_banks_based_on_landcover", action="store_true", help="Use landcover data for finding banks when estimating bathymetry")
    cli_parser.add_argument("--create_reach_average_curve_file", action="store_true", help="Create a reach average curve file instead of one that varies for each stream cell")
    cli_parser.add_argument("--known_baseflow", type=float, default=None, help="Known baseflow value")
    cli_parser.add_argument("--known_channel_forming_discharge", type=float, default=None, help="Known channel forming discharge value")
    cli_parser.add_argument("--upstream_elevation_change_threshold", type=float, help="The upstream elevation change used to identify the appropriate upstream cross-section, default is 1.0 meters", default=1.0)
    cli_parser.add_argument("--streamflow", type=str, default=None, help="Path to a local .gpkg or .parquet file containing streamflow data")

    args = parser.parse_args()

    if args.command == "json":
        print(f'Processing {args.json_file}')
        process_json_input(args.json_file)
    elif args.command == "cli":
        process_cli_arguments(args)

if __name__ == "__main__":
    main()
