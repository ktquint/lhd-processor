import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import pyproj
import geoglows
from matplotlib.ticker import FixedLocator
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.optimize import fsolve

# Internal imports
from .hydraulics import compute_flip_and_conjugate, dam_height
from .utils import (
    merge_arc_results,
    merge_databases,
    hydraulic_jump_type,
    round_sigfig,
    rating_curve_intercept,
    get_prob_from_Q
)


def add_north_arrow(ax, size=0.1, loc_x=0.1, loc_y=0.9):
    """Adds a north arrow to the map."""
    import matplotlib.patheffects as pe
    ax.annotate(
        'N',
        xy=(loc_x, loc_y), xytext=(loc_x, loc_y - size),
        xycoords='axes fraction', textcoords='axes fraction',
        ha='center', va='center',
        fontsize=16, fontweight='bold',
        arrowprops=dict(facecolor='black', width=5, headwidth=15)
    )


class CrossSection:
    def __init__(self, index, xs_row, parent_dam):
        self.parent_dam = parent_dam
        self.id = self.parent_dam.id
        self.fig_dir = self.parent_dam.fig_dir
        self.L = self.parent_dam.weir_length
        self.hydrology = self.parent_dam.hydrology

        # geospatial info
        self.lat = xs_row['Lat']
        self.lon = xs_row['Lon']
        self.index = index
        if self.index == 0:
            self.location = 'Upstream'
            self.distance = "some"
            self.fatal_qs = None
        else:
            self.location = 'Downstream'
            self.distance = int(self.index * self.L)
            self.fatal_qs = self.parent_dam.fatal_flows

        # rating curve info
        val_a = xs_row['depth_a']
        val_b = xs_row['depth_b']
        self.a = float(val_a[0]) if isinstance(val_a, list) else float(val_a)
        self.b = float(val_b[0]) if isinstance(val_b, list) else float(val_b)
        self.max_Q = xs_row['QMax']
        self.slope = round_sigfig(xs_row['Slope'], 3)

        # cross-section plot info
        self.wse = xs_row['Elev']
        INVALID_THRESHOLD = -1e5
        y_1 = xs_row['XS1_Profile']
        y_2 = xs_row['XS2_Profile']
        x_1 = [-1 * xs_row['Ordinate_Dist'] - j * xs_row['Ordinate_Dist'] for j in range(len(y_1))]
        x_2 = [0 + j * xs_row['Ordinate_Dist'] for j in range(len(y_2))]

        # delete any points that contain missing data
        x = x_1[::-1] + x_2
        y = y_1[::-1] + y_2

        x_clean = []
        y_clean = []
        for xi, yi in zip(x, y):
            if yi > INVALID_THRESHOLD:
                x_clean.append(xi)
                y_clean.append(yi)

        self.elevation = y_clean
        self.bed_elevation = min(y_clean) if y_clean else 0
        self.lateral = x_clean

        # Determine wse intersection
        wse_left, wse_lat_left = [], []
        i = 0
        while i < len(y_1) and y_1[i] <= self.wse:
            wse_left.append(self.wse)
            wse_lat_left.append(x_1[i])
            i += 1

        wse_right, wse_lat_right = [], []
        i = 0
        while i < len(y_2) and y_2[i] <= self.wse:
            wse_right.append(self.wse)
            wse_lat_right.append(x_2[i])
            i += 1

        self.water_elevation = wse_left[::-1] + wse_right
        self.water_lateral = wse_lat_left[::-1] + wse_lat_right
        self.P = None

    def set_dam_height(self, P):
        self.P = P

    def plot_cross_section(self):
        fig, ax = plt.subplots()
        ax.plot(self.lateral, self.elevation, color='black', label=f'Downstream Slope: {self.slope}')
        ax.plot(self.water_lateral, self.water_elevation, color='cyan', linestyle='--',
                label=f'Water Surface Elevation: {round(self.wse, 1)} m')
        ax.set_xlim(-1.5 * self.L, 1.5 * self.L)
        ax.set_xlabel('Lateral Distance (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        ax.legend(loc='upper right')

        fname = f"{'US' if self.index == 0 else 'DS'}_XS_{self.index if self.index > 0 else 'LHD'}_{self.id}.png"
        if self.index == 0: fname = f"US_XS_LHD_{self.id}.png"

        fig.savefig(os.path.join(self.fig_dir, fname))
        return fig

    def create_rating_curve(self):
        x = np.linspace(0.01, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y,
                 label=f'Rating Curve {self.distance} meters {self.location}: $y = {self.a:.3f} x^{{{self.b:.3f}}}$')

    def plot_flip_sequent(self, ax):
        Qs = np.linspace(0.01, self.max_Q, 100)
        Y_Ts = self.a * Qs ** self.b
        Y_Flips = []
        Y_Conjugates = []

        for Q in Qs:
            Y_Flip, Y_Conj = compute_flip_and_conjugate(Q, self.L, self.P)
            Y_Flips.append(Y_Flip)
            Y_Conjugates.append(Y_Conj)

        ax.plot(Qs * 35.315, np.array(Y_Flips) * 3.281, label="Flip Depth", color='gray', linestyle='--')
        ax.plot(Qs * 35.315, Y_Ts * 3.281, label="Tailwater Depth", color='dodgerblue', linestyle='-')
        ax.plot(Qs * 35.315, np.array(Y_Conjugates) * 3.281, label="Sequent Depth", color='gray', linestyle='-')
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Discharge (ft$^{3}$/s)')
        ax.set_ylabel('Depth (ft)')
        ax.set_title(f'Submerged Hydraulic Jumps at Low-Head Dam No. {self.id}')

    def plot_fatal_flows(self, ax):
        if self.fatal_qs is None or len(self.fatal_qs) == 0:
            return
        np_fatal_qs = np.array(self.fatal_qs)
        fatal_d = self.a * np_fatal_qs ** self.b
        ax.scatter(np_fatal_qs * 35.315, fatal_d * 3.281, label="Recorded Fatality", marker='o', facecolors='none',
                   edgecolors='black')

    def create_combined_fig(self):
        fig, ax = plt.subplots()
        self.plot_flip_sequent(ax)
        self.plot_fatal_flows(ax)
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(self.fig_dir, f"RC_{self.index}_LHD_{self.id}.png"))
        return fig

    def get_dangerous_flow_range(self):
        """Calculates Qmin and Qmax for Type C jump (in cfs)."""
        Q_i = np.array([0.001])
        # Solve for intersection of Tailwater with Conjugate (Qmin)
        Q_min = fsolve(rating_curve_intercept, Q_i, args=(self.L, self.P, self.a, self.b, 'conjugate'))[0]
        # Solve for intersection of Tailwater with Flip Bucket (Qmax)
        Q_max = fsolve(rating_curve_intercept, Q_i, args=(self.L, self.P, self.a, self.b, 'flip'))[0]
        return Q_min * 35.315, Q_max * 35.315

    def plot_fdc(self, ax: Axes):
        flow_data = self.parent_dam.get_flow_data()
        if flow_data is None or flow_data.empty:
            print(f"No flow data available to plot FDC for Dam {self.id}")
            return

        flow_cms = flow_data.dropna().values
        flow_cfs = flow_cms * 35.315
        sorted_flow = np.sort(flow_cfs)[::-1]
        n = len(sorted_flow)
        exceedance = 100 * np.arange(1, n + 1) / (n + 1)

        fdc_df = pd.DataFrame({'Exceedance (%)': exceedance, 'Flow (cfs)': sorted_flow})
        ax.plot(fdc_df['Exceedance (%)'], fdc_df['Flow (cfs)'], label='FDC', color='dodgerblue')

        try:
            Q_conj, Q_flip = self.get_dangerous_flow_range()
            P_flip = get_prob_from_Q(Q_flip, fdc_df)
            P_conj = get_prob_from_Q(Q_conj, fdc_df)
            ax.axvline(x=P_flip, color='black', linestyle='--', label=f'Flip and Conjugate Depth Intersections')
            ax.axvline(x=P_conj, color='black', linestyle='--')
        except Exception as e:
            print(f"Could not calc dangerous intersections: {e}")

        ax.set_ylabel('Discharge (cfs)')
        ax.set_yscale("log")
        ax.set_xlabel('Exceedance Probability (%)')
        ax.set_title(f'Flow-Duration Curve for Low-Head Dam No. {self.id}')
        ax.grid(True, which="both", linestyle='--')
        ax.legend()

    def create_combined_fdc(self):
        fig, ax = plt.subplots()
        self.plot_fdc(ax)
        fig.savefig(os.path.join(self.fig_dir, f"FDC_{self.index}_LHD_{self.id}.png"))
        return fig


class Dam:
    def __init__(self, lhd_id, lhd_csv, hydrology, est_dam, base_results_dir):
        self.id = int(lhd_id)
        self.hydrology = hydrology
        lhd_df = pd.read_csv(lhd_csv)
        id_row = lhd_df[lhd_df['ID'] == self.id].reset_index(drop=True)

        self.results_dir = base_results_dir
        self.fig_dir = os.path.join(self.results_dir, str(self.id), "FIGS")
        os.makedirs(self.fig_dir, exist_ok=True)

        # Load COMID table
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        comid_path = os.path.join(project_root, 'data', 'comid_table.csv')
        self.nwm_id = None
        self.geoglows_id = None

        if os.path.exists(comid_path):
            comid_df = pd.read_csv(comid_path)
            comid_df['ID'] = pd.to_numeric(comid_df['ID'], errors='coerce')
            match = comid_df[comid_df['ID'] == self.id]
            if not match.empty:
                self.nwm_id = match.iloc[0]['reach_id']
                self.geoglows_id = match.iloc[0]['linkno']

        # Dam Height setup
        if est_dam:
            self.P = None
        else:
            self.P = id_row['P_known'].values[0] / 3.218

        self.latitude = id_row['latitude'].values[0]
        self.longitude = id_row['longitude'].values[0]
        self.cross_sections = []
        self.weir_length = id_row['weir_length'].values[0]
        self.fatality_dates = ast.literal_eval(id_row['fatality_dates'].values[0])

        # Hydrology
        if hydrology == "GEOGLOWS":
            self.fatal_flows = ast.literal_eval(id_row.at[0, 'fatality_flows_GEOGLOWS'])
            baseflow_val = id_row.at[0, 'dem_baseflow_GEOGLOWS']
        elif hydrology == "USGS":
            self.fatal_flows = ast.literal_eval(id_row.at[0, 'fatality_flows_USGS'])
            baseflow_val = id_row.at[0, 'dem_baseflow_USGS']
        else:  # NWM
            self.fatal_flows = ast.literal_eval(id_row.at[0, 'fatality_flows_NWM'])
            baseflow_val = id_row.at[0, 'dem_baseflow_NWM']

        try:
            self.known_baseflow = float(baseflow_val)
            if pd.isna(self.known_baseflow): raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"Invalid baseflow value '{baseflow_val}' for Dam {self.id}")

        # VDT + XS Info
        vdt_gpkg = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Local_VDT_Database.gpkg")
        rc_gpkg = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Local_CurveFile.gpkg")
        xs_gpkg = os.path.join(self.results_dir, str(self.id), "XS", f"{self.id}_Local_XS_Lines.gpkg")

        self.dam_gdf = merge_arc_results(rc_gpkg, vdt_gpkg, xs_gpkg)
        self.xs_gpkg = xs_gpkg
        self.bathy_tif = os.path.join(self.results_dir, str(self.id), "Bathymetry", f"{self.id}_ARC_Bathy.tif")

        for index, row in self.dam_gdf.iterrows():
            self.cross_sections.append(CrossSection(index, row, self))

        # Dam Height Calculation & Qmin/Qmax Recording
        for i in range(1, len(self.cross_sections)):
            s_i = self.cross_sections[i].slope
            lhd_df.loc[lhd_df['ID'] == self.id, f's_{i}'] = s_i
            y_ts, y_flips, y_2s, jump_types = [], [], [], []

            if est_dam:
                delta_wse_i = self.cross_sections[i].wse - self.cross_sections[0].wse
                y_i = self.cross_sections[i].wse - self.cross_sections[i].bed_elevation
                P_i = dam_height(self.known_baseflow, self.weir_length, delta_wse_i, y_i)

                if P_i < 0.1 or P_i > 100:
                    raise ValueError(f"Calculated dam height ({P_i:.2f}m) is unrealistic.")

                self.cross_sections[i].set_dam_height(P_i)
                lhd_df.loc[lhd_df['ID'] == self.id, f'P_{i}'] = P_i * 3.281

                # --- NEW: Calculate and record Qmin/Qmax ---
                try:
                    q_min, q_max = self.cross_sections[i].get_dangerous_flow_range()
                    lhd_df.loc[lhd_df['ID'] == self.id, f'Qmin_{i}'] = q_min
                    lhd_df.loc[lhd_df['ID'] == self.id, f'Qmax_{i}'] = q_max
                except Exception as e:
                    print(f"Could not calculate Qmin/Qmax for XS {i}: {e}")
                    lhd_df.loc[lhd_df['ID'] == self.id, f'Qmin_{i}'] = None
                    lhd_df.loc[lhd_df['ID'] == self.id, f'Qmax_{i}'] = None
                # -------------------------------------------

                # Jumps
                for flow in self.fatal_flows:
                    y_t = self.cross_sections[i].a * flow ** self.cross_sections[i].b
                    y_flip, y_2 = compute_flip_and_conjugate(flow, self.weir_length, P_i)
                    y_ts.append(float(y_t))
                    y_flips.append(float(y_flip))
                    y_2s.append(float(y_2))
                    jump_types.append(hydraulic_jump_type(y_2, y_t, y_flip))
            else:
                self.cross_sections[i].set_dam_height(self.P)
                # Same logic would apply here if you were running with known P

            lhd_df.loc[lhd_df['ID'] == self.id, f'y_t_{i}'] = str(y_ts)
            lhd_df.loc[lhd_df['ID'] == self.id, f'y_flip_{i}'] = str(y_flips)
            lhd_df.loc[lhd_df['ID'] == self.id, f'y_2_{i}'] = str(y_2s)
            lhd_df.loc[lhd_df['ID'] == self.id, f'type_{i}'] = str(jump_types)

        lhd_df.to_csv(lhd_csv, index=False)

    def get_flow_data(self):
        flow_series = None
        if self.hydrology == 'National Water Model':
            if self.nwm_id and not pd.isna(self.nwm_id):
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(current_dir)
                    parquet_path = os.path.join(project_root, 'data', 'nwm_daily_retrospective.parquet')
                    if os.path.exists(parquet_path):
                        df = pd.read_parquet(parquet_path)
                        if 'feature_id' in df.index.names:
                            flow_series = df.xs(int(self.nwm_id), level='feature_id')['streamflow']
                except Exception as e:
                    print(f"Error reading NWM data: {e}")

        elif self.hydrology == 'GEOGLOWS':
            if self.geoglows_id and not pd.isna(self.geoglows_id):
                try:
                    df = geoglows.data.retrospective(river_id=int(self.geoglows_id), bias_corrected=True)
                    flow_series = df.iloc[:, 0]
                except Exception as e:
                    print(f"Error fetching GEOGLOWS data: {e}")
        return flow_series

    def plot_rating_curves(self):
        for cross_section in self.cross_sections[1:]:
            cross_section.create_rating_curve()
        plt.xlabel('Flow (m$^{3}$/s)')
        plt.ylabel('Depth (m)')
        plt.title(f'Rating Curves for LHD No. {self.id}')
        plt.legend(title="Rating Curve Equations", loc='best', fontsize='small')
        plt.savefig(os.path.join(self.fig_dir, f"Rating Curves for LHD No. {self.id}.png"))

    def plot_cross_sections(self):
        for cross_section in self.cross_sections:
            cross_section.plot_cross_section()

    def plot_all_curves(self):
        for cross_section in self.cross_sections[1:]:
            cross_section.create_combined_fig()

    def plot_map(self):
        strm_gpkg = os.path.join(self.results_dir, str(self.id), "STRM", f"{self.id}_StrmShp.gpkg")
        strm_gdf = gpd.read_file(strm_gpkg).to_crs('EPSG:3857')
        xs_gdf = gpd.read_file(self.xs_gpkg).to_crs('EPSG:3857')

        # Bounds
        buffer = 100
        minx, miny, maxx, maxy = xs_gdf.total_bounds

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(minx - buffer * 2, maxx + buffer * 2)
        ax.set_ylim(miny - buffer, maxy + buffer)

        ctx.add_basemap(ax, crs='EPSG:3857', source="Esri.WorldImagery", zorder=0)

        gdf_upstream = xs_gdf.iloc[[0]]
        gdf_downstream = xs_gdf.iloc[1:]
        strm_gdf.plot(ax=ax, color='green', markersize=100, edgecolor='black', zorder=2, label="Flowline")
        gdf_upstream.plot(ax=ax, color='red', markersize=100, edgecolor='black', zorder=2, label="Upstream")
        gdf_downstream.plot(ax=ax, color='dodgerblue', markersize=100, edgecolor='black', zorder=2, label="Downstream")

        # Ticks (simplified)
        proj = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xticks_lon = [proj.transform(x, yticks[0])[0] for x in xticks]
        yticks_lat = [proj.transform(xticks[0], y)[1] for y in yticks]

        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xticklabels([f"{lon:.4f}" for lon in xticks_lon])
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.set_yticklabels([f"{lat:.4f}" for lat in yticks_lat])

        ax.set_title(f"Cross-Section Locations for LHD No. {self.id}")
        ax.legend(title="Cross-Section Location", loc='upper right')

        # Scalebar
        ax.add_artist(ScaleBar(1, units="m", dimension="si-length", location="lower left"))

        # North Arrow
        import matplotlib.patheffects as pe
        ax.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.05), xycoords='axes fraction',
                    textcoords='axes fraction', ha='center', va='center', fontsize=22, fontweight='bold',
                    color="black", arrowprops=dict(facecolor='black', edgecolor='white', width=8, headwidth=25),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
                    ).set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, f"LHD No. {self.id} Location.png"))
        return fig

    def plot_water_surface(self):
        cf_csv = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_CurveFile.csv")
        xs_txt = os.path.join(self.results_dir, str(self.id), "XS", f"{self.id}_XS_Out.txt")
        database_df = merge_databases(cf_csv, xs_txt)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(database_df.index, database_df['DEM_Elev'], color='dodgerblue', label='DEM Elevation')
        ax.plot(database_df.index, database_df['BaseElev'], color='black', label='Bed Elevation')

        upstream_xs = self.dam_gdf.iloc[0]
        upstream_idx = \
        database_df[(database_df['Row'] == upstream_xs['Row']) & (database_df['Col'] == upstream_xs['Col'])].index[0]
        ax.axvline(x=upstream_idx, color='red', linestyle='--', label=f'Upstream Cross-Section')

        for i in range(1, len(self.dam_gdf)):
            ds_xs = self.dam_gdf.iloc[i]
            ds_idx = database_df[(database_df["Row"] == ds_xs['Row']) & (database_df["Col"] == ds_xs['Col'])].index[0]
            label = 'Downstream Cross-Sections' if i == 1 else ""
            ax.axvline(x=ds_idx, color='cyan', linestyle='--', label=label)

        ax.legend()
        ax.set_xlabel("Distance Downstream (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"Water Surface Profile for LHD No. {self.id}")
        fig.tight_layout()
        return fig
