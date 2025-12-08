import math
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union

def round_sigfig(num, sig_figs):
    if num == 0:
        return 0
    else:
        return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)

def hydraulic_jump_type(y_2, y_t, y_flip):
    """Classifies the hydraulic jump type based on tailwater depth."""
    if y_t < y_2:
        return 'A'
    elif y_t == y_2:
        return 'B'
    elif y_2 < y_t < y_flip:
        return 'C'
    elif y_t >= y_flip:
        return 'D'
    else:
        return 'N/A'

def rating_curve_intercept(Q: float, L: float, P: float, a: float, b: float, which: str) -> float:
    # Local import to avoid circular dependency
    from .hydraulics import compute_flip_and_conjugate
    y_flip, y_2 = compute_flip_and_conjugate(Q, L, P)
    y_t = a * Q ** b
    if which == 'flip':
        return y_flip - y_t
    elif which == 'conjugate':
        return y_2 - y_t
    else:
        raise ValueError("which must be 'flip' or 'conjugate'")

def get_prob_from_Q(Q, df):
    return df.loc[(df['Flow (cfs)'] - Q).abs().idxmin(), 'Exceedance (%)']

def fuzzy_merge(left, right, tol=2):
    """Perform fuzzy merge based on Row and Col coordinates within tolerance."""
    result_rows = []
    right_cols_to_add = [col for col in right.columns if col not in ['COMID', 'Row', 'Col']]

    for comid, group_left in left.groupby('COMID'):
        group_right = right[right['COMID'] == comid].copy()

        if group_right.empty:
            for col in right_cols_to_add:
                group_left = group_left.copy()
                group_left[col] = np.nan
            result_rows.append(group_left)
            continue

        for idx, row_left in group_left.iterrows():
            row_diff = abs(group_right['Row'] - row_left['Row'])
            col_diff = abs(group_right['Col'] - row_left['Col'])
            matches = group_right[(row_diff <= tol) & (col_diff <= tol)]

            if not matches.empty:
                if len(matches) > 1:
                    distances = row_diff + col_diff
                    closest_idx = distances.idxmin()
                    match = matches.loc[closest_idx]
                else:
                    match = matches.iloc[0]

                combined_row = row_left.copy()
                for col in right_cols_to_add:
                    combined_row[col] = match[col]
                result_rows.append(combined_row.to_frame().T)
            else:
                row_with_nans = row_left.copy()
                for col in right_cols_to_add:
                    row_with_nans[col] = np.nan
                result_rows.append(row_with_nans.to_frame().T)

    if result_rows:
        return pd.concat(result_rows, ignore_index=True)
    else:
        return pd.DataFrame()

def merge_databases(cf_database, xs_database):
    cf_df = pd.read_csv(cf_database)
    xs_df = pd.read_csv(xs_database, sep='\t')
    return pd.merge(xs_df, cf_df, on=['COMID', 'Row', 'Col'])

def merge_arc_results(curve_file: str, local_vdt: str, cross_section: str) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Reads and merges VDT, CurveFile, and XS data."""
    vdt_gdf = gpd.read_file(local_vdt)
    rc_gdf = gpd.read_file(curve_file)
    xs_gdf = gpd.read_file(cross_section)

    # Clean list-like strings
    list_columns = ['XS1_Profile', 'Manning_N_Raster1', 'XS2_Profile', 'Manning_N_Raster2']
    for col in list_columns:
        if col in xs_gdf.columns:
            xs_gdf[col] = xs_gdf[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else x)

    # Drop duplicates
    if 'Ordinate_Dist.1' in xs_gdf.columns and 'Ordinate_Dist' in xs_gdf.columns:
        if xs_gdf['Ordinate_Dist'].equals(xs_gdf['Ordinate_Dist.1']):
            xs_gdf = xs_gdf.drop(columns=['Ordinate_Dist.1'])

    # Merges
    first_merge = fuzzy_merge(rc_gdf, vdt_gdf, tol=2)
    results_gdf = fuzzy_merge(first_merge, xs_gdf, tol=2)

    if 'geometry' in results_gdf.columns:
        results_gdf = gpd.GeoDataFrame(results_gdf, geometry='geometry')

    results_gdf = results_gdf.sort_values(by=["Row", "Col"]).reset_index(drop=True)

    # Ensure upstream/downstream order
    if not results_gdf.empty and results_gdf['DEM_Elev'][0] < results_gdf['DEM_Elev'][len(results_gdf) - 1]:
        results_gdf = results_gdf[::-1].reset_index(drop=True)

    return results_gdf
