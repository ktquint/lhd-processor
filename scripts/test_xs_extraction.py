# import os
# import sys
# import ast
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point, LineString
# from shapely.ops import linemerge
# from pyproj import CRS, Transformer
# import rasterio
# import numpy as np
#
# # --- FIX PROJ database for pyproj ---
# try:
#     conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
#     proj_lib = os.path.join(conda_prefix, 'share', 'proj')
#     if not os.path.exists(proj_lib):
#         proj_lib = "/opt/homebrew/Cellar/micromamba/2.4.0/envs/lhd-environment/share/proj"
#     if os.path.exists(proj_lib):
#         os.environ['PROJ_LIB'] = proj_lib
#         import pyproj
#         pyproj.datadir.set_data_dir(proj_lib)
#         print(f"✅ Set PROJ_LIB to: {proj_lib}")
#         print(f"✅ pyproj data dir set to: {pyproj.datadir.get_data_dir()}")
#     else:
#         print(f"⚠️ Warning: Could not find proj data at {proj_lib}")
# except Exception as e:
#     print(f"⚠️ Warning: Failed to set PROJ_LIB: {e}")
#
# ID_FIELD = "COMID"
#
# class MockRathCelonDam:
#     def __init__(self, dam_id, output_dir, dem_dir, flowline_path, latitude, longitude):
#         self.dam_id = dam_id
#         self.output_dir = output_dir
#         self.dem_dir = dem_dir
#         self.flowline = flowline_path
#         self.latitude = latitude
#         self.longitude = longitude
#         self.weir_length = 30
#
#         self.vdt_txt = os.path.join(output_dir, str(dam_id), 'VDT', f'{dam_id}_VDT.txt')
#         self.curvefile_csv = os.path.join(output_dir, str(dam_id), 'VDT', f'{dam_id}_Curve.csv')
#         self.xs_txt = os.path.join(output_dir, str(dam_id), 'XS', f'{dam_id}_XS.txt')
#
#         if os.path.exists(self.dem_dir):
#             dems = [f for f in os.listdir(self.dem_dir) if f.endswith('.tif')]
#             self.dem_tif = os.path.join(self.dem_dir, dems[0]) if dems else None
#             if self.dem_tif:
#                 print(f"Using DEM: {self.dem_tif}")
#             else:
#                 print(f"⚠️ Warning: No DEM found in {self.dem_dir}")
#         else:
#             print(f"⚠️ Warning: DEM directory not found: {self.dem_dir}")
#             self.dem_tif = None
#
#         self.vdt_gdf = None
#         self.curve_data_gdf = None
#         self.xs_gdf = None
#
#     def _find_strm_up_downstream(self):
#         print("\n--- Starting Extraction Test ---")
#
#         if not os.path.exists(self.vdt_txt) or not os.path.exists(self.curvefile_csv) or not os.path.exists(self.flowline) or not self.dem_tif:
#             print("❌ Missing input files.")
#             return
#
#         # --------------------------
#         # Load Curve and VDT
#         # --------------------------
#         print("Loading VDT and Curve data...")
#         vdt_df = pd.read_csv(self.vdt_txt, sep=',')
#         curve_df = pd.read_csv(self.curvefile_csv)
#         vdt_df.columns = [c.strip() for c in vdt_df.columns]
#         curve_df.columns = [c.strip() for c in curve_df.columns]
#
#         # --------------------------
#         # Load Flowline
#         # --------------------------
#         print("Loading Flowline...")
#         flowline_gdf = gpd.read_file(self.flowline)
#         if flowline_gdf.empty:
#             print("❌ Flowline GeoDataFrame is empty.")
#             return
#
#         # Project flowline
#         try:
#             projected_crs = flowline_gdf.estimate_utm_crs()
#         except:
#             with rasterio.open(self.dem_tif) as ds:
#                 projected_crs = CRS.from_wkt(ds.crs.to_wkt())
#         flowline_gdf = flowline_gdf.to_crs(projected_crs)
#
#         # Project dam point
#         dam_point = gpd.GeoSeries([Point(self.longitude, self.latitude)], crs="EPSG:4326").to_crs(projected_crs).iloc[0]
#
#         # Merge flowline
#         merged_geom = linemerge(flowline_gdf.geometry.tolist())
#         if merged_geom.geom_type == 'MultiLineString':
#             merged_geom = min(merged_geom.geoms, key=lambda g: g.distance(dam_point))
#         start_dist = merged_geom.project(dam_point)
#
#         # Generate target points along flowline
#         target_points = [merged_geom.interpolate(min(start_dist + i * self.weir_length, merged_geom.length)) for i in range(1,5)]
#         target_points.append(merged_geom.interpolate(max(start_dist - self.weir_length,0)))
#
#         # --------------------------
#         # Transform target points to DEM
#         # --------------------------
#         with rasterio.open(self.dem_tif) as ds:
#             geoTransform = ds.transform
#             minx, dx = geoTransform.c, geoTransform.a
#             maxy, dy = geoTransform.f, geoTransform.e
#             dem_crs = CRS.from_wkt(ds.crs.to_wkt())
#         transformer = Transformer.from_crs(projected_crs, dem_crs, always_xy=True)
#         target_points_dem = [Point(transformer.transform(pt.x, pt.y)) for pt in target_points]
#
#         def pt_to_rc(pt):
#             return int((maxy - pt.y)/abs(dy)), int((pt.x - minx)/dx)
#
#         # Save debug layers
#         debug_dir = os.path.join(self.output_dir, "debug_extraction")
#         os.makedirs(debug_dir, exist_ok=True)
#         gpd.GeoDataFrame(geometry=[merged_geom], crs=projected_crs).to_file(os.path.join(debug_dir, "flowline.gpkg"))
#         gpd.GeoDataFrame(geometry=[dam_point], crs=projected_crs).to_file(os.path.join(debug_dir, "dam_point.gpkg"))
#
#         # Calculate Row/Col and save target points
#         tp_gdf = gpd.GeoDataFrame(geometry=target_points, crs=projected_crs)
#         rc_values = [pt_to_rc(pt) for pt in target_points_dem]
#         tp_gdf['Row'] = [x[0] for x in rc_values]
#         tp_gdf['Col'] = [x[1] for x in rc_values]
#
#         # Match points to Curve
#         final_indices = []
#         for idx, row in tp_gdf.iterrows():
#             r = row['Row']
#             c = row['Col']
#             curve_df['d_pix'] = np.sqrt((curve_df['Row'] - r)**2 + (curve_df['Col'] - c)**2)
#             best_idx = curve_df['d_pix'].idxmin()
#             final_indices.append(best_idx)
#
#         extracted_curve = curve_df.loc[final_indices].copy()
#         tp_gdf['COMID'] = extracted_curve[ID_FIELD].values
#         tp_gdf.to_file(os.path.join(debug_dir, "target_points.gpkg"))
#         target_ids = extracted_curve[ID_FIELD].unique()
#
#         # Filter VDT
#         self.vdt_gdf = vdt_df[vdt_df[ID_FIELD].isin(target_ids)].copy()
#
#         # --------------------------
#         # XS lines extraction (match exactly target_points by COMID + Row/Col)
#         # --------------------------
#         xs_lines = []
#         if os.path.exists(self.xs_txt):
#             xs_df = pd.read_csv(self.xs_txt, sep='\t')
#             xs_df.columns = [c.strip() for c in xs_df.columns]
#
#             # Compute raster coordinates for XS endpoints
#             with rasterio.open(self.dem_tif) as ds:
#                 geoTransform = ds.transform
#                 minx, dx = geoTransform.c, geoTransform.a
#                 maxy, dy = geoTransform.f, geoTransform.e
#
#             x_base = float(minx) + 0.5 * dx
#             y_base = float(maxy) - 0.5 * abs(dy)
#
#             # Filter XS to relevant COMIDs first
#             target_comids = extracted_curve[ID_FIELD].unique()
#             xs_df = xs_df[xs_df[ID_FIELD].isin(target_comids)].copy()
#
#             # Transformer for Lat/Lon
#             transformer_ll = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
#
#             matched_xs_rows = []
#             # Iterate over each target point to find the closest XS in that COMID
#             for _, target_row in extracted_curve.iterrows():
#                 t_comid = target_row[ID_FIELD]
#                 t_r, t_c = target_row['Row'], target_row['Col']
#
#                 # Subset XS for this COMID
#                 subset = xs_df[xs_df[ID_FIELD] == t_comid].copy()
#                 if subset.empty:
#                     continue
#
#                 # Calculate distance from target point to XS midpoint (approx)
#                 subset['dist_sq'] = ((subset['r1'] + subset['r2']) / 2 - t_r)**2 + \
#                                     ((subset['c1'] + subset['c2']) / 2 - t_c)**2
#
#                 best_match = subset.loc[subset['dist_sq'].idxmin()].copy()
#                 best_match['Target_Row'] = t_r
#                 best_match['Target_Col'] = t_c
#                 matched_xs_rows.append(best_match)
#
#             xs_df = pd.DataFrame(matched_xs_rows) if matched_xs_rows else pd.DataFrame()
#
#             for _, row in xs_df.iterrows():
#                 try:
#                     comid = int(row[ID_FIELD])
#
#                     def ensure_list(val):
#                         if isinstance(val, str) and val.startswith('['):
#                             return ast.literal_eval(val)
#                         elif isinstance(val, (float, int)):
#                             return [val]
#                         else:
#                             return val
#
#                     xs1 = ensure_list(row['XS1_Profile'])
#                     n1 = ensure_list(row['Manning_N_Raster1'])
#                     xs2 = ensure_list(row['XS2_Profile']) if 'XS2_Profile' in row else None
#                     n2 = ensure_list(row['Manning_N_Raster2']) if 'Manning_N_Raster2' in row else None
#                     ord_dist = ensure_list(row['Ordinate_Dist'])
#
#                     # Skip if empty
#                     if xs1 is None or ord_dist is None or len(xs1) < 2:
#                         print(f"⚠️ XS row COMID {comid} has insufficient points, skipping.")
#                         continue
#
#                     # Compute X/Y coordinates for endpoints
#                     x1 = x_base + row['c1'] * dx
#                     y1 = y_base - row['r1'] * abs(dy)
#                     x2 = x_base + row['c2'] * dx
#                     y2 = y_base - row['r2'] * abs(dy)
#
#                     # Compute Lat/Lon
#                     lon1, lat1 = transformer_ll.transform(x1, y1)
#                     lon2, lat2 = transformer_ll.transform(x2, y2)
#
#                     line_geom = LineString([(x1, y1), (x2, y2)])
#
#                     xs_lines.append({
#                         'COMID': comid,
#                         'Row': int(row['Target_Row']),
#                         'Col': int(row['Target_Col']),
#                         'geometry': line_geom,
#                         'XS1_Profile': str(xs1),
#                         'Ordinate_Dist': str(ord_dist),
#                         'Manning_N_Raster1': str(n1),
#                         'XS2_Profile': str(xs2) if xs2 is not None else None,
#                         'Manning_N_Raster2': str(n2) if n2 is not None else None,
#                         'r1': int(row['r1']),
#                         'c1': int(row['c1']),
#                         'r2': int(row['r2']),
#                         'c2': int(row['c2']),
#                         'X1': x1,
#                         'Y1': y1,
#                         'X2': x2,
#                         'Y2': y2,
#                         'Lon1': lon1,
#                         'Lat1': lat1,
#                         'Lon2': lon2,
#                         'Lat2': lat2
#                     })
#
#                 except Exception as e:
#                     print(f"⚠️ Failed to parse XS row COMID {row[ID_FIELD]}: {e}")
#
#         if xs_lines:
#             self.xs_gdf = gpd.GeoDataFrame(xs_lines, crs=projected_crs)
#             self.xs_gdf.to_file(os.path.join(debug_dir, "extracted_xs.gpkg"))
#             print(f"  Saved extracted XS lines with attributes to {os.path.join(debug_dir, 'extracted_xs.gpkg')}")
#         else:
#             print("  ⚠️ No XS lines found matching target points.")
#
#
# if __name__ == "__main__":
#     DAM_ID = 1
#     OUTPUT_DIR = "/Volumes/KenDrive/lhd_testing/Results_nwm_nhd"
#     DEM_DIR = f"/Volumes/KenDrive/LHD_Project/DEM/{DAM_ID}"
#     FLOWLINE_PATH = "/Volumes/KenDrive/lhd_testing/STRM/nhd_flowline_2789571.gpkg"
#     LATITUDE = 37.9356167
#     LONGITUDE = -122.0475322
#
#     mock = MockRathCelonDam(DAM_ID, OUTPUT_DIR, DEM_DIR, FLOWLINE_PATH, LATITUDE, LONGITUDE)
#     mock._find_strm_up_downstream()
