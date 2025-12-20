import os
import pyproj
import geoglows
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import fsolve
from matplotlib.ticker import FixedLocator
from matplotlib_scalebar.scalebar import ScaleBar

# Internal imports
from .hydraulics import (solve_weir_geometry,
                         solve_y2_jump,
                         weir_H,
                         compute_y_flip,
                         solve_y1_downstream)

from .utils import (merge_arc_results,
                    merge_databases,
                    hydraulic_jump_type,
                    round_sigfig,
                    get_prob_from_Q)


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
            self.distance = int(self.L)
            self.fatal_qs = None
        else:
            self.location = 'Downstream'
            self.distance = int(self.index * self.L)
            self.fatal_qs = self.parent_dam.fatal_flows

        # Legacy rating curve info (kept for schema compatibility)
        val_a = xs_row.get('depth_a', 0)
        val_b = xs_row.get('depth_b', 0)
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

        self.elevation = np.array(y_clean)
        self.bed_elevation = min(y_clean) if y_clean else 0
        self.elevation_shifted = self.elevation - self.bed_elevation
        self.lateral = np.array(x_clean)

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
        self.H = None

        # --- LOAD VDT DATA (Dynamic Columns) ---
        q_list = []
        wse_list = []

        # Find all columns starting with 'q_' and extract their suffix
        # xs_row.index gives us the column names since xs_row is a Series
        q_cols = [col for col in xs_row.index if str(col).startswith('q_')]

        for q_col in q_cols:
            # Safely extract the suffix (e.g. '1' from 'q_1')
            parts = q_col.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                suffix = parts[1]
                wse_col = f"wse_{suffix}"

                if wse_col in xs_row:
                    q_val = xs_row[q_col]
                    wse_val = xs_row[wse_col]

                    # Filter valid data
                    if pd.notnull(q_val) and pd.notnull(wse_val):
                        q_list.append(float(q_val))
                        wse_list.append(float(wse_val))

        if q_list:
            # Convert to numpy arrays and sort by Flow (Q)
            inds = np.argsort(q_list)
            self.vdt_Q = np.array(q_list)[inds]
            sorted_wse = np.array(wse_list)[inds]

            # Calculate Depth = WSE - Bed Elevation
            self.vdt_depth = sorted_wse - self.bed_elevation
            # Ensure no negative depths due to minor discrepancies
            self.vdt_depth[self.vdt_depth < 0] = 0
        else:
            self.vdt_Q = np.array([])
            self.vdt_depth = np.array([])
            print(f"Warning: No valid q_X / wse_X columns found for XS {self.index}.")

    def set_dam_height(self, P):
        self.P = P

    def set_head(self, H):
        self.H = H

    def get_tailwater_depth(self, Q):
        """Interpolates tailwater depth from the VDT data."""
        if len(self.vdt_Q) == 0:
            # Fallback to power law if VDT is missing
            return self.a * Q ** self.b

        # Linear interpolation
        return np.interp(Q, self.vdt_Q, self.vdt_depth)

    def plot_cross_section(self, save=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.lateral, self.elevation, color='black', label='Cross-Section with Bathymetry Estimation')
        ax.plot(self.water_lateral, self.water_elevation, color='cyan', linestyle='--',
                label=f'Water Surface Elevation: {round(self.wse, 1)} m')
        ax.set_xlim(-1.5 * self.L, 1.5 * self.L)
        ax.set_xlabel('Lateral Distance (m)')
        ax.set_ylabel('Elevation (m)')
        # ax.set_title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        ax.set_title(f'Downstream Slope: {self.slope}')
        ax.legend(loc='upper right')

        if save:
            fname = f"{'US' if self.index == 0 else 'DS'}_XS_{self.index if self.index > 0 else 'LHD'}_{self.id}.png"
            if self.index == 0:
                fname = f"US_XS_LHD_{self.id}.png"

            fig.savefig(os.path.join(self.fig_dir, fname), dpi=300, bbox_inches='tight')
            return fig
        else:
            return None

    def create_rating_curve(self):
        # Use VDT data if available, otherwise fallback to linspace + power law
        if len(self.vdt_Q) > 0:
            x = self.vdt_Q
            y = self.vdt_depth
            label = f'Rating Curve (VDT Data) {self.distance}m {self.location}'
        else:
            x = np.linspace(0.01, self.max_Q, 100)
            y = self.a * x ** self.b
            label = f'Rating Curve (Power Law) {self.distance}m {self.location}'

        plt.plot(x, y, label=label)

    def plot_flip_sequent(self, ax):
        if len(self.vdt_Q) > 0:
            Qs = self.vdt_Q
            # Use interpolated depth (though at exact VDT points, interp is exact)
            Y_Ts = self.vdt_depth
        else:
            Qs = np.linspace(0.01, self.max_Q, 100)
            Y_Ts = self.a * Qs ** self.b

        Y_Flips = []
        Y_Conjugates = []

        for i, Q in enumerate(Qs):
            # Q is flow, Y_Ts[i] is tailwater

            # Recalculate Hydraulic Jump characteristics for this Q
            Y_Flip = compute_y_flip(Q, self.L, self.P)

            Y_Conj1 = solve_y1_downstream(Q,  self.L, self.P, self.lateral, self.elevation_shifted)
            Y_Conj2 = solve_y2_jump(Q, Y_Conj1, self.lateral, self.elevation_shifted)

            Y_Flips.append(Y_Flip)
            Y_Conjugates.append(Y_Conj2)

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
        # Use new get_tailwater_depth method
        fatal_d = np.array([self.get_tailwater_depth(q) for q in np_fatal_qs])

        ax.scatter(np_fatal_qs * 35.315, fatal_d * 3.281, label="Recorded Fatality", marker='o', facecolors='none',
                   edgecolors='black')

    def create_combined_fig(self, save=True):
        fig, ax = plt.subplots(figsize=(12, 8))
        self.plot_flip_sequent(ax)
        self.plot_fatal_flows(ax)
        ax.legend(loc='upper left')
        if save:
            fname = os.path.join(self.fig_dir, f"RC_{self.index}_LHD_{self.id}.png")
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            return fig
        else:
            return None

    def get_dangerous_flow_range(self):
        """Calculates Qmin and Qmax for Type C jump (in cfs) using VDT interpolation."""

        def obj_func(Q_guess, which):
            Q = Q_guess[0] if isinstance(Q_guess, (list, np.ndarray)) else Q_guess
            if Q <= 0.001: return 1e6

            # 1. Get Tailwater from VDT Interpolation
            y_t = self.get_tailwater_depth(Q)

            # 2. Get Hydraulics for this Q
            if which == 'flip':
                y_target = compute_y_flip(Q, self.L, self.P)
            elif which == 'conjugate':
                y_1 = solve_y1_downstream(Q, self.L, self.P, self.lateral, self.elevation_shifted)
                y_target = solve_y2_jump(Q, y_1, self.lateral, self.elevation_shifted)
            else:
                return 1e6

            return y_target - y_t

        Q_i = np.array([10.0]) # Initial guess

        # Q_max: Tailwater drops below Flip Bucket (y_t = y_flip)
        Q_max = fsolve(obj_func, Q_i, args=('flip',))[0]

        # Q_min: Tailwater drops below Sequent Depth (y_t = y_2)
        Q_min = fsolve(obj_func, Q_i, args=('conjugate',))[0]

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
            # Safety check if solver failed (returned negative or tiny flow)
            if Q_conj > 1 and Q_flip > 1:
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
    def __init__(self, lhd_id, db_manager, base_results_dir):
        self.id = int(lhd_id)
        self.db = db_manager
        self.results_dir = base_results_dir

        # Load Data
        self.site_data = self.db.get_site(self.id)
        self.incidents_df = self.db.get_site_incidents(self.id)

        if not self.site_data:
            raise ValueError(f"Dam {self.id} not found in database.")

        self.latitude = self.site_data['latitude']
        self.longitude = self.site_data['longitude']
        self.weir_length = self.site_data['weir_length']
        self.hydrology = self.site_data.get('streamflow_source', 'National Water Model')
        self.baseflow = float(self.site_data.get('dem_baseflow', 0))

        self.fatal_flows = self.incidents_df['flow'].dropna().tolist()
        self.fig_dir = os.path.join(self.results_dir, str(self.id), "FIGS")
        os.makedirs(self.fig_dir, exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        comid_path = os.path.join(project_root, 'data', 'comid_table.csv')
        self.nwm_id = None
        self.geoglows_id = None
        if os.path.exists(comid_path):
            try:
                comid_df = pd.read_csv(comid_path)
                comid_df['ID'] = pd.to_numeric(comid_df['ID'], errors='coerce')
                match = comid_df[comid_df['ID'] == self.id]
                if not match.empty:
                    self.nwm_id = match.iloc[0]['reach_id']
                    self.geoglows_id = match.iloc[0]['linkno']
            except:
                pass

        # Load Geometry
        vdt_gpkg = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Local_VDT_Database.gpkg")
        rc_gpkg = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Local_CurveFile.gpkg")
        xs_gpkg = os.path.join(self.results_dir, str(self.id), "XS", f"{self.id}_Local_XS_Lines.gpkg")
        self.xs_gpkg = xs_gpkg

        self.dam_gdf = merge_arc_results(rc_gpkg, vdt_gpkg, xs_gpkg)
        self.cross_sections = []
        for index, row in self.dam_gdf.iterrows():
            self.cross_sections.append(CrossSection(index, row, self))


    def load_results(self):
        """
            Loads existing analysis results (P, H) from the database so we can plot without re-calc.
        """
        if self.db.xsections.empty:
            return

        # Filter for this dam
        xs_data = self.db.xsections[self.db.xsections['site_id'] == self.id]
        if xs_data.empty:
            return

        for xs in self.cross_sections:
            row = xs_data[xs_data['xs_index'] == xs.index]
            if not row.empty:
                try:
                    p_val = row.iloc[0]['P_height']
                    if pd.notnull(p_val):
                        P = float(p_val)
                        xs.set_dam_height(P)
                        # Estimate H based on baseflow and P (needed for some plots)
                        if self.baseflow > 0:
                            H = weir_H(self.baseflow, self.weir_length, P)
                            xs.set_head(H)
                except Exception as e:
                    print(f"Error loading parameters for XS {xs.index}: {e}")

    def run_analysis(self, est_dam=True):
        """
            Performs the computationally expensive hydraulic analysis.
        """
        xs_data_list = []
        hydro_results_list = []

        for i, xs in enumerate(self.cross_sections):
            xs_info = {
                'xs_index': i,
                'slope': xs.slope,
                'rating_a': xs.a,
                'rating_b': xs.b
            }

            if i > 0:
                # 1. Calculate Dam Height (P) and Head (H)
                if est_dam:
                    delta_wse = self.cross_sections[0].wse - xs.wse
                    y_i = xs.wse - xs.bed_elevation
                    try:
                        H_i, P_i = solve_weir_geometry(self.baseflow, self.weir_length, y_i, delta_wse)

                        # Drowning check
                        y_1 = solve_y1_downstream(self.baseflow, self.weir_length, P_i,
                                                  xs.lateral, xs.elevation_shifted)
                        if y_1 > P_i:
                            print(f"Warning: Dam {self.id} XS {i} appears drowned.")

                        xs.set_dam_height(P_i)
                        xs.set_head(H_i)
                        xs_info['P_height'] = P_i

                    except Exception as e:
                        print(f"Solver failed for Dam {self.id} XS {i}: {e}")
                        xs_info['P_height'] = None
                else:
                    known_p = self.site_data.get('P_known', 2.0)
                    xs.set_dam_height(known_p)
                    xs_info['P_height'] = known_p

                # 2. Calculate Dangerous Flow Range
                if xs.P:
                    try:
                        q_min, q_max = xs.get_dangerous_flow_range()
                        xs_info['Qmin'] = q_min
                        xs_info['Qmax'] = q_max
                    except:
                        pass

                # 3. Calculate Incident Hydraulics
                for _, inc_row in self.incidents_df.iterrows():
                    Q = inc_row['flow']
                    date = inc_row['date']

                    if pd.notna(Q) and xs.P:
                        try:
                            # Use VDT interpolation for y_t
                            y_t = xs.get_tailwater_depth(Q)

                            y_1 = solve_y1_downstream(Q, xs.L, xs.P, xs.lateral, xs.elevation_shifted)
                            y_2 = solve_y2_jump(Q, y_1, xs.lateral, xs.elevation_shifted)
                            y_flip = compute_y_flip(Q, xs.L, xs.P)
                            jump = hydraulic_jump_type(y_2, y_t, y_flip)

                            hydro_results_list.append({
                                'date': date,
                                'xs_index': i,
                                'y_t': y_t,
                                'y_flip': y_flip,
                                'y_2': y_2,
                                'jump_type': jump
                            })
                        except:
                            pass

            xs_data_list.append(xs_info)

        # Save to the DB object
        self.db.update_analysis_results(self.id, xs_data_list, hydro_results_list)

    def get_flow_data(self):
        flow_series = None
        if self.hydrology == 'National Water Model':
            if self.nwm_id and not pd.isna(self.nwm_id):
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(current_dir)
                    parquet_path = os.path.join(project_root, 'data', 'nwm_v3_daily_retrospective.parquet')
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
        ax.add_artist(ScaleBar(1, units="m", dimension="si-length", location="lower left"))

        import matplotlib.patheffects as pe
        ax.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.05), xycoords='axes fraction',
                    textcoords='axes fraction', ha='center', va='center', fontsize=22, fontweight='bold',
                    color="black", arrowprops=dict(facecolor='black', edgecolor='white', width=8, headwidth=25),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
                    ).set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        fig.tight_layout()
        fig.savefig(os.path.join(self.fig_dir, f"LHD No. {self.id} Location.png"))
        return fig

    def plot_water_surface(self, save=True):
        cf_csv = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_CurveFile.csv")
        xs_txt = os.path.join(self.results_dir, str(self.id), "XS", f"{self.id}_XS_Out.txt")
        database_df = merge_databases(cf_csv, xs_txt)

        fig, ax = plt.subplots(figsize=(12, 10))
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

        if save:
            fname = os.path.join(self.fig_dir, f"{self.id}_WSP.png")
            fig.savefig(os.path.join(self.fig_dir, fname), dpi=300, bbox_inches='tight')
            return fig
        else:
            return None