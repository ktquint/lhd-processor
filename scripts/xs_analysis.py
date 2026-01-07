import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from figures import merge_arc_results


def plot_water_surface_elevations(df_row):
    lhd_id = df_row['ID']
    y_1 = df_row['XS1_Profile']
    # y_1 = y_1[::-1]
    y_2 = df_row['XS2_Profile']
    xs_elevation = y_1 + y_2

    x_1 = [0 - j * df_row['Ordinate_Dist'] for j in range(len(y_1))]
    x_2 = [0 + j * df_row['Ordinate_Dist'] for j in range(len(y_2))]
    xs_lateral = x_1 + x_2
    for i in range(1, 31):
        wse_i = df_row[f'wse_{i}']
        wse_list = [wse_i, wse_i]
        lateral_list = [min(xs_lateral), max(xs_lateral)]
        plt.plot(lateral_list, wse_list, color='dodgerblue', linestyle='--')
    plt.plot(xs_lateral, xs_elevation, color='black')
    plt.xlabel('Lateral Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f'Water Surface Elevation at LHD No. {lhd_id}')
    plt.show()


def plot_downstream(cf_local, vdt_database, xs_database, id):
    """
        all inputs are dataframes
    """
    row_col = cf_local['Row'].tolist()
    col_col = cf_local['Col'].tolist()

    indices = []
    for i in range(len(row_col)):
        row, col = row_col[i], col_col[i]
        index = vdt_database[(vdt_database['Row'] == row) & (vdt_database['Col'] == col)].index[0]
        indices.append(index)

    start_index = min(indices)
    end_index = max(indices)

    vdt_trimmed = vdt_database.iloc[start_index:end_index + 1]  # +1 to include the end row
    xs_trimmed = xs_database.iloc[start_index:end_index + 1]

    downstream_dist = [0]
    total_dist = 0
    wse = vdt_trimmed['Elev'].tolist()
    row_list = vdt_database['Row'].tolist()
    col_list = vdt_database['Col'].tolist()
    xs_1 = xs_trimmed['elev_a'].tolist()
    xs_2 = xs_trimmed['elev_b'].tolist()
    bed_elev = [min(xs_1[0] + xs_2[0])]

    for i in range(len(vdt_trimmed['Row'].tolist()) - 1):
        min_elev = min(xs_1[i + 1] + xs_2[i + 1])
        bed_elev.append(min_elev)
        dist = (row_list[i + 1] - row_list[i]) ** 2 + (col_list[i + 1] - col_list[i]) ** 2
        total_dist += np.sqrt(dist)
        downstream_dist.append(total_dist)
    wse = wse[::-1]
    bed_elev = bed_elev[::-1]

    plt.plot(downstream_dist, wse, color='dodgerblue', label='Water Surface')
    plt.plot(downstream_dist, bed_elev, color='black', label='Estimated Bed Elevation')
    plt.xlabel('Downstream Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.title(f"Water Surface Elevation Along Streamline at LHD No. {id}")
    plt.show()


def run_successful(run_results_dir):
    lhd_id = os.path.basename(run_results_dir)
    xs_gpkg = os.path.join(run_results_dir, "XS", f"{lhd_id}_Local_XS.gpkg")

    if not os.path.exists(xs_gpkg):
        return False  # or raise an error if preferred

    try:
        xs_gdf = gpd.read_file(xs_gpkg)
        return not xs_gdf.empty
    except Exception as e:
        print(f"Error reading DBF for {lhd_id}: {e}")
        return False


def create_xs_figs(xs_1m, xs_10m, lhd_id):
    for i in range(0, len(xs_1m)):
        # first let's extract the data from the 1-m data
        y_1 = xs_1m.at[i, 'XS1_Profile']
        y_1 = y_1[::-1]
        y_2 = xs_1m.at[i, 'XS2_Profile']
        xs_elevation_1 = y_1 + y_2

        x_1 = [0 - j * xs_1m.at[i, 'Ordinate_Dist'] for j in range(len(y_1))]
        x_1 = x_1[::-1]
        x_2 = [0 + j * xs_1m.at[i, 'Ordinate_Dist'] for j in range(len(y_2))]
        xs_lateral_1 = x_1 + x_2
        plt.plot(xs_lateral_1, xs_elevation_1, label=f'1-m Resolution Cross-section')
        # next we'll get the info from the 10-m data
        y_1 = xs_10m.at[i, 'XS1_Profile']
        y_1 = y_1[::-1]
        y_2 = xs_10m.at[i, 'XS2_Profile']
        xs_elevation_10 = y_1 + y_2

        x_1 = [0 - j * xs_10m.at[i, 'Ordinate_Dist'] for j in range(len(y_1))]
        x_1 = x_1[::-1]
        x_2 = [0 + j * xs_10m.at[i, 'Ordinate_Dist'] for j in range(len(y_2))]
        xs_lateral_10 = x_1 + x_2
        plt.plot(xs_lateral_10, xs_elevation_10, label=f'10-m Resolution Cross-section')

        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'Cross-Sections for LHD No. {lhd_id}')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()


def create_rating_curves(rc_1m, rc_10m, lhd_id):

    for i in range(0, len(rc_1m)):
        q_1m = []
        d_1m = []
        q_10m = []
        d_10m = []
        for j in range(1, 31):
            if rc_1m.at[i, f"wse_{j}"] > 0:
                q_1m.append(rc_1m.at[i, f"q_{j}"])
                d_1m.append(rc_1m.at[i, f"wse_{j}"] - rc_1m.at[i, 'elev_a'][0])

                q_10m.append(rc_10m.at[i, f"q_{j}"])
                d_10m.append(rc_10m.at[i, f"wse_{j}"] - rc_10m.at[i, 'elev_a'][0])

        Q = np.linspace(1, q_1m[-1], 100)
        y_1m = rc_1m.at[i, 'depth_a'] * Q ** rc_1m.at[i, 'depth_b']
        plt.plot(Q, y_1m, label="CF 1-m", color='orange')
        plt.plot(q_1m, d_1m, 'o', label="VDT 1-m", color='red')

        Q = np.linspace(1, q_10m[-1], 100)
        y_10m = rc_10m.at[i, 'depth_a'] * Q ** rc_10m.at[i, 'depth_b']
        plt.plot(Q, y_10m, label="CF 1-m", color='green')
        plt.plot(q_10m, d_10m, 'o', label="VDT 1-m", color='dodgerblue')

        plt.xlabel('Flow (m$^{3}$/s)')
        plt.ylabel('Depth (m)')
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        plt.title(label=f'Rating Curves for LHD No. {lhd_id}')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()


def compare_1m_to_10m(one_meter_results, ten_meter_results):
    """
        inputs are file paths (e.g., LHD_Project/LHD_Results and LHD_1m/LHD_Results)
    """
    # first lets list out the dams we have results for in each set
    one_meter_runs = [d for d in os.listdir(one_meter_results)]
    ten_meter_runs = [d for d in os.listdir(ten_meter_results)]
    # then we'll trim down to just the runs we have in common
    common_runs = list(set(one_meter_runs).intersection(set(ten_meter_runs)))
    print(common_runs)

    # okay, let's loop through each run and do some magic...
    for lhd_id in common_runs:
        one_meter_dir = os.path.join(one_meter_results, lhd_id)
        ten_meter_dir = os.path.join(ten_meter_results, lhd_id)
        if run_successful(one_meter_dir) and run_successful(ten_meter_dir):

            vdt_1m = os.path.join(one_meter_results, lhd_id, 'VDT', f'{lhd_id}_Local_VDT_Database.gpkg')
            cf_1m = os.path.join(one_meter_results, lhd_id, 'VDT', f'{lhd_id}_Local_Curve.gpkg')
            xs_1m = os.path.join(one_meter_results, lhd_id, 'XS', f'{lhd_id}_Local_XS.gpkg')

            vdt_10m = os.path.join(ten_meter_results, lhd_id, 'VDT', f'{lhd_id}_Local_VDT_Database.gpkg')
            cf_10m = os.path.join(ten_meter_results, lhd_id, 'VDT', f'{lhd_id}_Local_Curve.gpkg')
            xs_10m = os.path.join(ten_meter_results, lhd_id, 'XS', f'{lhd_id}_Local_XS.gpkg')

            print(f"Working on {lhd_id}")
            gdf_1m = merge_arc_results(cf_1m, vdt_1m, xs_1m)
            gdf_10m = merge_arc_results(cf_10m, vdt_10m, xs_10m)
            create_xs_figs(gdf_1m, gdf_10m, lhd_id)
            # create_rating_curves(gdf_1m, gdf_10m, lhd_id)


def count_good_files(results_dir):
    rath_runs = [d for d in os.listdir(results_dir)]

    # okay, let's loop through each run and do some magic...
    for lhd_id in rath_runs:
        lhd_result = os.path.join(results_dir, lhd_id)

        if run_successful(lhd_result):
            print(f"{lhd_id} was ran successfully")


compare_1m_to_10m("E:/LHD_1-m_NWM/LHD_Results", "E:/LHD_10-m_NWM/LHD_Results")
# count_good_files("E:\LHD_1-3_arc-second - Copy\LHD_Results")


