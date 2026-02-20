# build-in imports
import ast
import os
import gc
from pathlib import Path

# third-party imports
from arc import Arc  # automated-rating-curve-generator

try:
    import gdal
    import gdal_array
except ImportError:
    from osgeo import gdal, ogr, osr, gdal_array

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString


# --- GEOSPATIAL UTILITIES ---

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


class ArcDam:
    def __init__(self, dam_row: dict, results_dir, flowline_source, streamflow_source, baseflow_method):
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
        self.dem_path = safe_p(dam_row.get('dem_path'))
        
        # Ensure output_dir is a Path object
        self.output_dir = Path(results_dir) if results_dir else None
        
        self.land_raster = safe_p(dam_row.get('land_raster'))
        if flowline_source == 'TDX-Hydro':
            self.strm_tif_clean = safe_p(dam_row.get('flowline_raster_tdx'))
        else:
            self.strm_tif_clean = safe_p(dam_row.get('flowline_raster_nhd'))

        # Use the shared Manning's n file in the LAND folder
        if self.land_raster:
            self.manning_n_txt = self.land_raster.parent.parent / 'Manning_n.txt'
        else:
            self.manning_n_txt = None
            
        self.flowline_source = flowline_source
        self.streamflow_source = streamflow_source
        self.baseflow_method = baseflow_method

        # ARC specific defaults
        self.create_reach_average_curve_file = 'False'
        self.bathy_use_banks = 'False'
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

        # files that will be made with arc or used in arc input
        self.arc_input = None
        self.bathy_tif = None
        self.vdt_txt = None
        self.curvefile_csv = None
        self.xs_txt = None
        self.vdt_gdf = None
        self.curve_data_gdf = None
        self.xs_gdf = None

    def _create_arc_input_txt(self, Q_baseflow, Q_max):
        """Full implementation of ARC input generation with all original parameters."""
        x_section_dist = int(10 * self.weir_length)
        
        with open(self.arc_input, 'w') as out_file:
            out_file.write('#ARC_Inputs\n')
            out_file.write(f'DEM_File\t{self.dem_path}\n')
            out_file.write(f'Stream_File\t{self.strm_tif_clean}\n')
            out_file.write(f'LU_Raster_SameRes\t{self.land_raster}\n')
            out_file.write(f'LU_Manning_n\t{self.manning_n_txt}\n')
            out_file.write(f'Flow_File\t{self.streamflow}\n')
            out_file.write(f'Flow_File_ID\tcomid\n')
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
        print(f"    Extracting hydraulic cross-sections for Dam {self.dam_id}...")

        # Ensure files exist (using Path.exists())
        if not self.vdt_txt or not self.vdt_txt.exists():
            print(f"    ❌ VDT file missing: {self.vdt_txt}")
            return
        if not self.curvefile_csv or not self.curvefile_csv.exists():
            print(f"    ❌ Curve file missing: {self.curvefile_csv}")
            return
        if not self.flowline or not self.flowline.exists():
            print(f"    ❌ Flowline file missing: {self.flowline}")
            return
        if not self.dem_path or not self.dem_path.exists():
            print(f"    ❌ DEM file missing: {self.dem_path}")
            return

        # Load Curve (Load VDT later to save memory)
        try:
            curve_df = pd.read_csv(self.curvefile_csv)
            curve_df.columns = [c.strip() for c in curve_df.columns]
            if curve_df.empty:
                print("    ❌ Curve file is empty.")
                return
        except Exception as e:
            print(f"    ❌ Error loading Curve: {e}")
            return

        # Load Flowline
        flowline_gdf = gpd.read_file(self.flowline)
        if flowline_gdf.empty:
            print("    ❌ Flowline GeoDataFrame is empty.")
            return

        # Project flowline
        projected_crs = None
        try:
            projected_crs = flowline_gdf.estimate_utm_crs()
        except:
            with rasterio.open(self.dem_path) as ds:
                projected_crs = CRS.from_wkt(ds.crs.to_wkt())
        flowline_gdf = flowline_gdf.to_crs(projected_crs)

        # Project dam point
        dam_point = gpd.GeoSeries([Point(self.longitude, self.latitude)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]

        # Merge flowline
        merged_geom = linemerge(flowline_gdf.geometry.tolist())
        if merged_geom.geom_type == 'MultiLineString':
            merged_geom = min(merged_geom.geoms, key=lambda g: g.distance(dam_point))

        if self.flowline_source == 'TDX-Hydro':
            merged_geom = LineString(list(merged_geom.coords)[::-1])

        start_dist = merged_geom.project(dam_point)

        # Generate target points
        target_points = [merged_geom.interpolate(min(start_dist + i * self.weir_length, merged_geom.length)) for i in
                         range(1, 5)]
        target_points.append(merged_geom.interpolate(max(start_dist - self.weir_length, 0)))
        
        print(f"    Generated {len(target_points)} target points along flowline.")

        # Define labels
        point_labels = [f"Downstream{i}" for i in range(1, 5)]
        point_labels.append("Upstream")

        # Transform target points to DEM
        with rasterio.open(self.dem_path) as ds:
            geoTransform = ds.transform
            minx, dx = geoTransform.c, geoTransform.a
            maxy, dy = geoTransform.f, geoTransform.e
            dem_crs = CRS.from_wkt(ds.crs.to_wkt())

        transformer = Transformer.from_crs(projected_crs, dem_crs, always_xy=True)
        target_points_dem = [Point(transformer.transform(pt.x, pt.y)) for pt in target_points]

        def pt_to_rc(pt):
            return int((maxy - pt.y) / abs(dy)), int((pt.x - minx) / dx)

        # Calculate Row/Col for target points
        tp_gdf = gpd.GeoDataFrame(geometry=target_points, crs=projected_crs)
        tp_gdf['Relative_Loc'] = point_labels
        rc_values = [pt_to_rc(pt) for pt in target_points_dem]
        tp_gdf['Row'] = [x[0] for x in rc_values]
        tp_gdf['Col'] = [x[1] for x in rc_values]

        # Match points to Curve (Spatial match to find correct Row/Col in Curve)
        final_indices = []

        # Optimization: Use numpy arrays and squared distance
        c_rows = curve_df['Row'].values
        c_cols = curve_df['Col'].values

        for idx, row in tp_gdf.iterrows():
            r = row['Row']
            c = row['Col']
            # d_pix calculation
            d2 = (c_rows - r) ** 2 + (c_cols - c) ** 2
            best_pos = np.argmin(d2)
            best_idx = curve_df.index[best_pos]
            final_indices.append(best_idx)
            
        print(f"    Matched {len(final_indices)} points to Curve file.")

        extracted_curve = curve_df.loc[final_indices].copy()
        extracted_curve['Relative_Loc'] = tp_gdf['Relative_Loc'].values

        # Free memory of full curve_df
        del curve_df
        del c_rows
        del c_cols
        gc.collect()

        # Determine ID field (COMID or XS_ID)
        id_col = 'COMID' if 'COMID' in extracted_curve.columns else 'XS_ID'

        # Load VDT NOW
        try:
            vdt_df = pd.read_csv(self.vdt_txt, sep=',')
            vdt_df.columns = [c.strip() for c in vdt_df.columns]
        except Exception as e:
            print(f"    ❌ Error loading VDT: {e}")
            return

        # Filter VDT
        target_ids = extracted_curve[id_col].unique()
        self.vdt_gdf = vdt_df[vdt_df[id_col].isin(target_ids)].copy()
        self.vdt_gdf = self.vdt_gdf.merge(extracted_curve[[id_col, 'Relative_Loc']].drop_duplicates(), on=id_col,
                                          how='left')

        # Free memory of full vdt_df
        del vdt_df
        gc.collect()

        # Create Curve GeoDataFrame
        x_base = float(minx) + 0.5 * dx
        y_base = float(maxy) - 0.5 * abs(dy)
        extracted_curve['X'] = x_base + extracted_curve['Col'] * dx
        extracted_curve['Y'] = y_base - extracted_curve['Row'] * abs(dy)

        transformer_to_wgs = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        extracted_curve[['Lon', 'Lat']] = extracted_curve.apply(
            lambda r: pd.Series(transformer_to_wgs.transform(r['X'], r['Y'])), axis=1)
        self.curve_data_gdf = gpd.GeoDataFrame(extracted_curve,
                                               geometry=gpd.points_from_xy(extracted_curve.Lon, extracted_curve.Lat),
                                               crs="EPSG:4326")

        # Convert VDT to GeoDataFrame
        geom_map = dict(zip(self.curve_data_gdf[id_col], self.curve_data_gdf.geometry))
        self.vdt_gdf = gpd.GeoDataFrame(self.vdt_gdf, geometry=self.vdt_gdf[id_col].map(geom_map), crs="EPSG:4326")

        # XS Extraction
        xs_lines = []
        if self.xs_txt.exists():
            try:
                xs_df = pd.read_csv(self.xs_txt, sep='\t')
                xs_df.columns = [c.strip() for c in xs_df.columns]
                
                print(f"    Found {len(xs_df)} rows in XS file.")

                xs_id_col = 'COMID' if 'COMID' in xs_df.columns else 'XS_ID'

                # Merge XS with extracted_curve on ID, Row, Col
                merged_xs = xs_df.merge(extracted_curve[[id_col, 'Row', 'Col', 'Relative_Loc']],
                                        left_on=[xs_id_col, 'Row', 'Col'],
                                        right_on=[id_col, 'Row', 'Col'],
                                        how='inner')
                                        
                print(f"    Matched {len(merged_xs)} rows in XS file to target points.")

                # Free memory of full xs_df
                del xs_df
                gc.collect()

                transformer_ll = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)

                for _, row in merged_xs.iterrows():
                    try:
                        comid = int(row[xs_id_col])

                        def ensure_list(val):
                            if isinstance(val, str) and val.startswith('['):
                                return ast.literal_eval(val)
                            elif isinstance(val, (float, int)):
                                return [val]
                            else:
                                return val

                        xs1 = ensure_list(row['XS1_Profile'])
                        n1 = ensure_list(row['Manning_N_Raster1'])
                        xs2 = ensure_list(row['XS2_Profile']) if 'XS2_Profile' in row else None
                        n2 = ensure_list(row['Manning_N_Raster2']) if 'Manning_N_Raster2' in row else None
                        ord_dist = ensure_list(row['Ordinate_Dist'])

                        if xs1 is None or ord_dist is None or len(xs1) < 2:
                            print(f"    ⚠️ Skipping row {comid}: Invalid XS data.")
                            continue

                        x1 = x_base + row['c1'] * dx
                        y1 = y_base - row['r1'] * abs(dy)
                        x2 = x_base + row['c2'] * dx
                        y2 = y_base - row['r2'] * abs(dy)

                        lon1, lat1 = transformer_ll.transform(x1, y1)
                        lon2, lat2 = transformer_ll.transform(x2, y2)

                        line_geom = LineString([(x1, y1), (x2, y2)])

                        xs_lines.append({
                            'COMID': comid,
                            'Relative_Loc': row['Relative_Loc'],
                            'Row': int(row['Row']),
                            'Col': int(row['Col']),
                            'geometry': line_geom,
                            'XS1_Profile': str(xs1),
                            'Ordinate_Dist': str(ord_dist),
                            'Manning_N_Raster1': str(n1),
                            'XS2_Profile': str(xs2) if xs2 is not None else None,
                            'Manning_N_Raster2': str(n2) if n2 is not None else None,
                            'r1': int(row['r1']),
                            'c1': int(row['c1']),
                            'r2': int(row['r2']),
                            'c2': int(row['c2']),
                            'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2,
                            'Lon1': lon1, 'Lat1': lat1, 'Lon2': lon2, 'Lat2': lat2
                        })
                    except Exception as e:
                        print(f"    ⚠️ Error parsing XS row: {e}")

            except Exception as e:
                print(f"    ⚠️ Error processing XS file: {e}")
        else:
             print(f"    ⚠️ XS file not found: {self.xs_txt}")

        if xs_lines:
            self.xs_gdf = gpd.GeoDataFrame(xs_lines, crs=projected_crs).to_crs("EPSG:4326")
            print(f"    Successfully extracted {len(self.xs_gdf)} cross-sections.")
        else:
            print("    ⚠️ No XS lines extracted.")
            self.xs_gdf = gpd.GeoDataFrame(columns=['geometry', 'COMID'], crs="EPSG:4326")


    def process_dam(self):
        """Execution loop for processing a single dam."""
        print(f"Processing Dam: {self.dam_id}")
        
        if not self.output_dir:
             print("  ❌ Output directory not specified.")
             return

        # Ensure output_dir is a Path
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        dirs = {sd: self.output_dir / str(self.dam_id) / sd for sd in
                ['ARC_InputFiles', 'Bathymetry', 'VDT', 'XS']}
        for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

        if not self.dem_path or not self.dem_path.exists():
            print(f"  ❌ DEM not found: {self.dem_path}")
            return

        print(f"  Processing DEM: {self.dem_path.name}")
        
        # Keep these as Path objects
        self.arc_input = dirs['ARC_InputFiles'] / f'ARC_Input_{self.dam_id}.txt'
        self.bathy_tif = dirs['Bathymetry'] / f'{self.dam_id}_Bathy.tif'
        self.vdt_txt = dirs['VDT'] / f'{self.dam_id}_VDT.txt'
        self.curvefile_csv = dirs['VDT'] / f'{self.dam_id}_Curve.csv'
        self.xs_txt = dirs['XS'] / f'{self.dam_id}_XS.txt'

        if not self.bathy_tif.exists():
            print("    Running ARC simulation...")
            if self.baseflow_method == 'WSE and LiDAR Date':
                Q_baseflow = "known_baseflow"
            elif self.baseflow_method == "WSE and Median Daily Flow":
                Q_baseflow = "qout_median"
            else:
                Q_baseflow = "rp2"
            self._create_arc_input_txt(Q_baseflow, "rp100")
            
            # Convert to string for Arc if it requires string input
            arc_runner = Arc(str(self.arc_input), quiet=True)
            try:
                arc_runner.set_log_level('info')
            except AttributeError:
                pass
            arc_runner.run()

            # Explicitly clean up ARC runner if possible
            del arc_runner
            gc.collect()

        print("    Extracting hydraulic cross-sections...")
        self._find_strm_up_downstream()

        if self.vdt_gdf is not None:
            self.vdt_gdf.to_file(dirs['VDT'] / f'{self.dam_id}_Local_VDT_Database.gpkg')
        if self.curve_data_gdf is not None:
            self.curve_data_gdf.to_file(dirs['VDT'] / f'{self.dam_id}_Local_Curve.gpkg')
        if self.xs_gdf is not None:
            self.xs_gdf.to_file(dirs['XS'] / f'{self.dam_id}_Local_XS.gpkg')
            
        # Final cleanup for this dam (Moved here)
        if self.vdt_gdf is not None: del self.vdt_gdf
        if self.curve_data_gdf is not None: del self.curve_data_gdf
        if self.xs_gdf is not None: del self.xs_gdf
        gc.collect()

        print(f"Finished Dam: {self.dam_id}")
