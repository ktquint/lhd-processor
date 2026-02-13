import os
import ast
import pyproj
import geoglows
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
from matplotlib_scalebar.scalebar import ScaleBar

# Internal imports
from .hydraulics import (solve_weir_geom,
                         solve_y2_adv,
                         calc_y2_simp,
                         weir_H_simp,
                         compute_y_flip,
                         solve_y1_adv,
                         solve_y1_simp,
                         rating_curve_intercepts_simp,
                         rating_curve_intercept_adv,
                         calc_froude_custom,
                         solve_Fr_simp)

from .utils import (merge_arc_results,
                    merge_databases,
                    hydraulic_jump_type,
                    round_sigfig,
                    get_prob_from_Q)


class CrossSection:
    INVALID_THRESHOLD = -1e5

    def __init__(self, xs_row, parent_dam, index=None):
        self.parent_dam = parent_dam
        
        # If index is not provided, try to get it from the row (name/index)
        if index is None:
            if hasattr(xs_row, 'name'):
                self.index = xs_row.name
            else:
                # Fallback if no index is provided and row has no name
                self.index = 0 
        else:
            self.index = index
        
        self._init_metadata(xs_row)
        self._init_rating_curve(xs_row)
        self._init_geometry(xs_row)
        self._init_hydraulic_profiles()
        self._init_vdt_data(xs_row)

        self.P = None
        self.H = None # head at baseflow condition


    def _init_metadata(self, xs_row):
        self.id = self.parent_dam.id
        self.fig_dir = self.parent_dam.fig_dir
        self.L = self.parent_dam.weir_length
        self.hydrology = self.parent_dam.hydrology
        self.calc_mode = self.parent_dam.calc_mode
        self.nwm_id = self.parent_dam.nwm_id
        self.geoglows_id = self.parent_dam.geoglows_id

        # geospatial info
        self.lat = xs_row['Lat']
        self.lon = xs_row['Lon']

        # FIX: Handle NaN in Index or Weir Length to prevent "float NaN to int" crash
        safe_idx = self.index if pd.notnull(self.index) else 0
        safe_L = self.L if pd.notnull(self.L) else 10.0

        # Use Relative_Loc if available, otherwise fallback to index logic
        if 'Relative_Loc' in xs_row and pd.notnull(xs_row['Relative_Loc']):
            self.location_label = xs_row['Relative_Loc']
            if self.location_label == 'Upstream':
                self.location = 'Upstream'
                self.distance = int(safe_L)
                self.fatal_qs = None
            else:
                self.location = 'Downstream'
                # Extract number from "DownstreamX"
                try:
                    num = int(self.location_label.replace('Downstream', ''))
                    self.distance = int(num * safe_L)
                except ValueError:
                    self.distance = int(safe_idx * safe_L) # Fallback
                self.fatal_qs = self.parent_dam.fatal_flows
        else:
            # Fallback to old logic if Relative_Loc is missing
            if safe_idx == 0:
                self.location = 'Upstream'
                self.location_label = 'Upstream'
                self.distance = int(safe_L)
                self.fatal_qs = None
            else:
                self.location = 'Downstream'
                self.location_label = f'Downstream{safe_idx}'
                self.distance = int(safe_idx * safe_L)
                self.fatal_qs = self.parent_dam.fatal_flows

    def _init_rating_curve(self, xs_row):
        # rating curve equation D = a * Q**b
        val_a = xs_row.get('depth_a', 0)
        val_b = xs_row.get('depth_b', 0)
        
        # New variables
        val_vel_a = xs_row.get('vel_a', 0)
        val_vel_b = xs_row.get('vel_b', 0)
        val_tw_a = xs_row.get('tw_a', 0)
        val_tw_b = xs_row.get('tw_b', 0)

        def _parse_val(v):
            if isinstance(v, str):
                v = v.strip()
                # Handle string representation of list like '[1.23]'
                if v.startswith('[') and v.endswith(']'):
                    try:
                        v = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        pass
            return float(v[0]) if isinstance(v, list) else float(v)

        self.a = _parse_val(val_a)
        self.b = _parse_val(val_b)
        
        self.vel_a = _parse_val(val_vel_a)
        self.vel_b = _parse_val(val_vel_b)
        self.tw_a = _parse_val(val_tw_a)
        self.tw_b = _parse_val(val_tw_b)
        
        # load the dam's flow series
        self.flow_series = self.parent_dam.flow_series
        
        if self.flow_series is not None and not self.flow_series.empty:
            vals = self.flow_series.dropna().values
            if len(vals) > 0:
                self.Qmin = np.min(vals)
                self.Qmax = np.max(vals)
            else:
                self.Qmin = 0.0
                self.Qmax = 100.0
        else:
            self.Qmin = 0.0
            self.Qmax = 100.0
            
        self.slope = round_sigfig(xs_row['Slope'], 3)

    def _init_geometry(self, xs_row):
        # cross-section plot info
        val_elev = xs_row['Elev']
        if isinstance(val_elev, str):
            val_elev = val_elev.strip()
            if val_elev.startswith('[') and val_elev.endswith(']'):
                try:
                    val_elev = ast.literal_eval(val_elev)[0]
                except:
                    pass
        self.wse = float(val_elev)
        
        # Raw Profiles (Center -> Outwards)
        def parse_profile(val):
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    try:
                        return ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        pass
            if isinstance(val, (list, np.ndarray)):
                return val
            return [] if pd.isna(val) else [val]

        self.y_1_raw = parse_profile(xs_row['XS1_Profile'])  # Center -> Left
        self.y_2_raw = parse_profile(xs_row['XS2_Profile'])  # Center -> Right
        
        val_dist = xs_row['Ordinate_Dist']
        if isinstance(val_dist, str):
            try:
                val_dist = ast.literal_eval(val_dist)
            except (ValueError, SyntaxError):
                pass
        self.dist = float(val_dist[0]) if isinstance(val_dist, list) else float(val_dist)

        # --- COORDINATE SYSTEM (0 to Total Width) ---
        # Calculate Offset to shift min_x to 0
        self.offset = len(self.y_1_raw) * self.dist

        # Map Raw Indices to Global X Coordinates
        self.x_1_coords = [self.offset - (i + 1) * self.dist for i in range(len(self.y_1_raw))]
        self.x_2_coords = [self.offset + i * self.dist for i in range(len(self.y_2_raw))]

        # --- WATER SURFACE (Center-Out Logic) ---
        wse_x_left = []
        for i, elev in enumerate(self.y_1_raw):
            if self.wse >= elev > self.INVALID_THRESHOLD:
                wse_x_left.append(self.x_1_coords[i])
            else:
                break

        wse_x_right = []
        for i, elev in enumerate(self.y_2_raw):
            if self.wse >= elev > self.INVALID_THRESHOLD:
                wse_x_right.append(self.x_2_coords[i])
            else:
                break

        self.water_lateral = wse_x_left[::-1] + wse_x_right
        self.water_elevation = [self.wse] * len(self.water_lateral)

        # --- FULL BED PROFILE (For Plotting) ---
        x_full = self.x_1_coords[::-1] + self.x_2_coords
        y_full = self.y_1_raw[::-1] + self.y_2_raw

        x_clean = []
        y_clean = []
        for xi, yi in zip(x_full, y_full):
            if yi > self.INVALID_THRESHOLD:
                x_clean.append(xi)
                y_clean.append(yi)

        self.elevation = np.array(y_clean)
        self.bed_elevation = min(y_clean) if y_clean else 0
        self.elevation_shifted = self.elevation - self.bed_elevation
        self.lateral = np.array(x_clean)

    def _init_hydraulic_profiles(self):
        # --- PREPARE HYDRAULIC PROFILES (Shifted & Cleaned) ---
        # Creates relative depth profiles (Depth = Elev - Bed_Elev) for the solver.
        self.y_1_shifted = self._clean_shift(self.y_1_raw, self.bed_elevation)
        self.y_2_shifted = self._clean_shift(self.y_2_raw, self.bed_elevation)

    def _clean_shift(self, profile, bed):
        valid = []
        if not profile: return np.array([0.0, 0.0])
        for v in profile:
            if v <= self.INVALID_THRESHOLD:
                break
            valid.append(v)
        if not valid:
            return np.array([0.0, 0.0])
        return np.array(valid) - bed

    def _init_vdt_data(self, xs_row):
        # --- LOAD VDT DATA ---
        q_list = []
        wse_list = []
        q_cols = [col for col in xs_row.index if str(col).startswith('q_')]

        for q_col in q_cols:
            parts = q_col.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                suffix = parts[1]
                wse_col = f"wse_{suffix}"
                if wse_col in xs_row:
                    q_val = xs_row[q_col]
                    wse_val = xs_row[wse_col]
                    if pd.notnull(q_val) and pd.notnull(wse_val):
                        q_list.append(float(q_val))
                        wse_list.append(float(wse_val))

        if q_list:
            inds = np.argsort(q_list)
            self.vdt_Q = np.array(q_list)[inds]
            sorted_wse = np.array(wse_list)[inds]
            self.vdt_depth = sorted_wse - self.bed_elevation
            self.vdt_depth[self.vdt_depth < 0] = 0
        else:
            self.vdt_Q = np.array([])
            self.vdt_depth = np.array([])
            # print(f"Warning: No valid q_X / wse_X columns found for XS {self.index}.")

    def set_dam_height(self, P):
        self.P = P

    def set_weir_head(self, H):
        self.H = H


    def plot_cross_section(self, save=True):
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        ax.plot(self.lateral, self.elevation, color='black', label='Cross-Section with Bathymetry Estimation')

        if self.location == 'Downstream':
            try:
                # Use explicit upstream lookup
                upstream_xs = self.parent_dam.get_upstream_xs()
                wse_upstream = upstream_xs.wse

                us_x_left = []
                for i, elev in enumerate(self.y_1_raw):
                    if wse_upstream >= elev > -1e5:
                        us_x_left.append(self.x_1_coords[i])
                    else:
                        break

                us_x_right = []
                for i, elev in enumerate(self.y_2_raw):
                    if wse_upstream >= elev > -1e5:
                        us_x_right.append(self.x_2_coords[i])
                    else:
                        break

                upstream_lateral = us_x_left[::-1] + us_x_right

                if upstream_lateral:
                    Q_b = self.parent_dam.baseflow
                    y_t = self.wse - self.bed_elevation
                    delta_wse = wse_upstream - self.wse
                    H, P = solve_weir_geom(Q_b, self.L, y_t, delta_wse)

                    crest_elev_val = wse_upstream - H
                    upstream_elevs = [wse_upstream] * len(upstream_lateral)
                    crest_elevs = [crest_elev_val] * len(upstream_lateral)

                    ax.plot(upstream_lateral, crest_elevs, color='black', linestyle='--',
                            label=f'Crest Elevation: {round(crest_elev_val, 1)} m')
                    ax.plot(upstream_lateral, upstream_elevs, color='red', linestyle='--',
                            label=f'Upstream Water Surface Elevation: {round(wse_upstream, 1)} m')

            except Exception as e:
                print(f"Could not plot derived dam lines for XS {self.index}: {e}")

        if len(self.water_lateral) > 0:
            ax.plot(self.water_lateral, self.water_elevation, color='dodgerblue', linestyle='--',
                    label=f'Water Surface Elevation: {round(self.wse, 1)} m')

        thalweg_x = self.offset
        ax.set_xlim(thalweg_x - self.L * 2.5, thalweg_x + self.L * 2.5)

        ax.set_xlabel('Lateral Distance (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'Downstream Slope: {self.slope}')
        ax.legend(loc='upper right')

        if save:
            fname = f"{self.location_label}_XS_{self.id}.png"
            fig.savefig(os.path.join(self.fig_dir, fname), dpi=300, bbox_inches='tight')
            return fig
        else:
            return None


    def get_tailwater_depth(self, Q):
        """
        Calculates tailwater depth using the Power Law function (y = a*x^b).
        Ignores VDT database values.
        """
        return self.a * Q ** self.b


    def create_rating_curve(self):
        """
        Plots the rating curve using the Power Law function (y = a*x^b).
        Ignores VDT database values.
        """
        Qmin_plot = self.Qmin if self.Qmin != 0 else 0.01
        Qs = np.linspace(Qmin_plot, self.Qmax, 100)
        Ys = self.get_tailwater_depth(Qs)
        label = f'Rating Curve (Power Law) {self.distance}m {self.location}'
        plt.plot(Qs, Ys, label=label)


    def plot_flip_sequent(self, ax):
        """
        Plots the Flip vs Sequent depth analysis using the Power Law for tailwater.
        Ignores VDT database values.
        """
        # Always use linspace and power function
        Qmin_plot = self.Qmin if self.Qmin != 0 else 0.01
        Qs = np.linspace(Qmin_plot, self.Qmax, 100)
        Y_Ts = self.get_tailwater_depth(Qs)

        Y_Flips = []
        Y_Conjugates = []

        for i, Q in enumerate(Qs):
            Y_Flip = compute_y_flip(Q, self.L, self.P)
            # Use shifted profiles
            # Recalculate H for this Q
            # NOTE: This assumes H varies with Q, which it does (H = (Q/CL)^(2/3))
            # The stored self.H is only for baseflow conditions!
            # We need to calculate H for the current Q to get the correct sequent depth.
            
            # Using simplified weir equation to estimate H for a given Q
            # Q = C * L * H^(3/2)  =>  H = (Q / (C * L))^(2/3)
            # Assuming C ~ 1.8 (SI) or 3.3 (Imperial)? 
            # The hydraulics module likely has a function for this.
            # weir_H_simp(Q, L) is available.
            
            H_current = weir_H_simp(Q, self.L)
            
            if self.calc_mode == "Advanced":
                Y_Conj2 = solve_y2_adv(Q, self.L, H_current, self.P, self.y_1_shifted, self.y_2_shifted, self.dist)
            else:
                Y_Conj2 = calc_y2_simp(H_current, self.P)

            Y_Flips.append(Y_Flip)
            Y_Conjugates.append(Y_Conj2)

        ax.plot(Qs, np.array(Y_Flips), label="Flip Depth", color='gray', linestyle='--')
        ax.plot(Qs, Y_Ts, label="Tailwater Depth", color='dodgerblue', linestyle='-')
        ax.plot(Qs, np.array(Y_Conjugates), label="Sequent Depth", color='gray', linestyle='-')
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Discharge (m$^{3}$/s)', fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_title(f'Submerged Hydraulic Jumps at Low-Head Dam No. {self.id}', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)



    def plot_fatal_flows(self, ax):
        if self.fatal_qs is None or len(self.fatal_qs) == 0:
            return
        np_fatal_qs = np.array(self.fatal_qs)
        fatal_d = np.array([self.get_tailwater_depth(q) for q in np_fatal_qs])
        ax.scatter(np_fatal_qs, fatal_d, label="Recorded Fatality", marker='o', facecolors='none',
                   edgecolors='black')

    def create_combined_fig(self, save=True):
        fig = Figure(figsize=(13, 6.5))
        ax = fig.add_subplot(111)
        self.plot_flip_sequent(ax)
        self.plot_fatal_flows(ax)
        ax.legend(loc='upper left', prop={'size': 12})
        if save:
            fname = os.path.join(self.fig_dir, f"RC_{self.location_label}_LHD_{self.id}.png")
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            return fig
        else:
            return None

    def get_dangerous_flow_range(self):
        if self.calc_mode == "Advanced":
            return rating_curve_intercept_adv(self.L, self.P, self.a, self.b,
                                              self.y_1_shifted, self.y_2_shifted, self.dist,
                                              self.Qmin, self.Qmax)
        elif self.calc_mode == "Simplified":
            return rating_curve_intercepts_simp(self.L, self.P, self.a, self.b,
                                                self.Qmin, self.Qmax)
        else:
            return None


    def plot_fdc(self, ax: Axes):
        flow_data = self.parent_dam.flow_series
        if flow_data is None or flow_data.empty:
            print(f"No flow data available to plot FDC for Dam {self.id}")
            return

        flow_cms = flow_data.dropna().values
        sorted_flow = np.sort(flow_cms)[::-1]
        n = len(sorted_flow)
        exceedance = 100 * np.arange(1, n + 1) / (n + 1)

        fdc_df = pd.DataFrame({'Exceedance (%)': exceedance, 'Flow (cms)': sorted_flow})
        ax.plot(fdc_df['Exceedance (%)'], fdc_df['Flow (cms)'], label='FDC', color='dodgerblue')

        try:
            Q_conj, Q_flip = self.get_dangerous_flow_range()
            
            # Use parent dam's absolute min/max to clamp or validate
            # (Optional: You could clamp Q_conj/Q_flip here if they are outside observed range)

            if Q_conj > 1 and Q_flip > 1:
                P_flip = get_prob_from_Q(Q_flip, fdc_df)
                P_conj = get_prob_from_Q(Q_conj, fdc_df)
                ax.axvline(x=P_flip, color='black', linestyle='--', label=f'Flip and Conjugate Depth Intersections')
                ax.axvline(x=P_conj, color='black', linestyle='--')
        except Exception as e:
            print(f"Could not calc dangerous intersections: {e}")

        ax.set_ylabel('Discharge (m$^{3}$/s)', fontsize=12)
        ax.set_yscale("log")
        ax.set_xlabel('Exceedance Probability (%)', fontsize=12)
        ax.set_title(f'Flow-Duration Curve for Low-Head Dam No. {self.id}', fontsize=14)
        ax.grid(True, which="both", linestyle='--')
        ax.legend(prop={'size': 12})
        ax.tick_params(axis='both', which='major', labelsize=12)

    def create_combined_fdc(self):
        fig = Figure(figsize=(13, 6.5))
        ax = fig.add_subplot(111)
        self.plot_fdc(ax)
        fig.savefig(os.path.join(self.fig_dir, f"FDC_{self.location_label}_LHD_{self.id}.png"), dpi=300, bbox_inches='tight')
        return fig


class Dam:
    def __init__(self, lhd_id, db_manager, base_results_dir,
                 calc_mode, flowline_source, streamflow_source):

        self.id = int(lhd_id)
        self.db = db_manager
        self.results_dir = os.path.normpath(base_results_dir)
        self.calc_mode = calc_mode
        self.flowline_source = flowline_source
        self.streamflow_source = streamflow_source

        self.site_data = self.db.get_site(self.id)
        self.incidents_df = self.db.get_site_incidents(self.id)

        if not self.site_data:
            raise ValueError(f"Dam {self.id} not found in database.")

        self.latitude = self.site_data['latitude']
        self.longitude = self.site_data['longitude']
        self.weir_length = self.site_data['weir_length']
        self.hydrology = streamflow_source
        self.nwm_id = self.site_data['reach_id']
        self.geoglows_id = self.site_data['linkno']

        if self.hydrology == 'National Water Model':
            self.baseflow = float(self.site_data.get('baseflow_nwm', 0))
            self.fatal_flows = self.incidents_df['flow_nwm'].dropna().tolist()
        else:
            self.baseflow = float(self.site_data.get('baseflow_geo', 0))
            self.fatal_flows = self.incidents_df['flow_geo'].dropna().tolist()

        self.fig_dir = os.path.join(self.results_dir, str(self.id), "FIGS")
        os.makedirs(self.fig_dir, exist_ok=True)


        # Helper to check for .gpkg or .shp
        def check_path(path):
            # Check if path exists and is not empty (if it's a file)
            if os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
            
            # If it's a gpkg, try shp
            if path.endswith('.gpkg'):
                shp_path = path.replace('.gpkg', '.shp')
                if os.path.exists(shp_path) and os.path.isfile(shp_path) and os.path.getsize(shp_path) > 0:
                    return shp_path
            
            # Return original path (will fail later if it doesn't exist)
            return path

        vdt_gpkg = check_path(os.path.normpath(os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Local_VDT_Database.gpkg")))
        rc_gpkg = check_path(os.path.normpath(os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Local_Curve.gpkg")))
        xs_gpkg = check_path(os.path.normpath(os.path.join(self.results_dir, str(self.id), "XS", f"{self.id}_Local_XS.gpkg")))
        
        for p in [vdt_gpkg, rc_gpkg, xs_gpkg]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file not found: {p}")

        self.xs_gpkg = xs_gpkg

        self.dam_gdf = merge_arc_results(rc_gpkg, vdt_gpkg, xs_gpkg)
        
        # REVERSE ORDER: Upstream (0) -> Downstream (N)
        self.dam_gdf = self.dam_gdf.iloc[::-1].reset_index(drop=True)

        # --- Initialize Flow Data ---
        self.flow_series = self.get_flow_data()
        if self.flow_series is not None and not self.flow_series.empty:
            self.Qmin_abs = self.flow_series.min()
            self.Qmax_abs = self.flow_series.max()
        else:
            self.Qmin_abs = 0.0
            self.Qmax_abs = 10000.0 # Default fallback

        self.cross_sections = []
        for index, row in self.dam_gdf.iterrows():
            self.cross_sections.append(CrossSection(row, self, index=index))

    def get_upstream_xs(self):
        # Try to find by location label
        for xs in self.cross_sections:
            if xs.location == 'Upstream':
                return xs
        # Fallback: return the first one (assuming it was sorted/reversed correctly)
        if self.cross_sections:
            return self.cross_sections[0]
        return None

    def load_results(self):
        # Use the new getter with source parameters
        xs_data = self.db.get_site_xsections(self.id, self.flowline_source, self.streamflow_source)
        
        if xs_data.empty:
            return
        
        upstream_xs = self.get_upstream_xs()
        
        for xs in self.cross_sections:
            row = xs_data[xs_data['xs_index'] == xs.index]
            if not row.empty:
                try:
                    # this should never happen lol
                    p_val = row.iloc[0]['P_height']
                    if pd.notnull(p_val):
                        P = float(p_val)
                        xs.set_dam_height(P)
                        
                    # Let's re-calculate H based on P and baseflow
                    if xs.P is not None:
                        # We need delta_wse to calculate H properly using solve_weir_geom
                        
                        # Let's try to re-run the basic geometry solve for this XS
                        if xs != upstream_xs:
                            delta_wse = upstream_xs.wse - xs.wse
                            y_i = xs.wse - xs.bed_elevation
                            
                            H_i, P_i = solve_weir_geom(self.baseflow, self.weir_length, y_i, delta_wse)
                            xs.set_weir_head(H_i)
                            # We can also update P just in case
                            # xs.set_dam_height(P_i)

                except Exception as e:
                    print(f"Error loading parameters for XS {xs.index}: {e}")

    def run_analysis(self):
        xs_data_list = []
        hydro_results_list = []
        
        upstream_xs = self.get_upstream_xs()
        
        for i, xs in enumerate(self.cross_sections):
            
            # Determine export index based on location label
            if xs.location == 'Upstream':
                export_idx = 0
            else:
                try:
                    # Extract number from "DownstreamX"
                    export_idx = int(xs.location_label.replace('Downstream', ''))
                except ValueError:
                    export_idx = i # Fallback
            
            xs_info = {
                'site_id': self.id,
                'xs_index': export_idx,
                'Slope': xs.slope,
                'depth_a': xs.a,
                'depth_b': xs.b,
                'vel_a': xs.vel_a,
                'vel_b': xs.vel_b,
                'tw_a': xs.tw_a,
                'tw_b': xs.tw_b,
                'P_height': None
            }
            if xs != upstream_xs:
                # the difference in water surface elevation
                # going from the upstream cross-section to
                # the current cross-section
                delta_wse = upstream_xs.wse - xs.wse
                y_i = xs.wse - xs.bed_elevation
                try:
                    # calculate the dam height and weir head at baseflow conditions
                    H_i, P_i = solve_weir_geom(self.baseflow, self.weir_length,
                                               y_i, delta_wse)

                    # solve for y_1 to see if the weir is drowned
                    if self.calc_mode == "Advanced":
                        y_1 = solve_y1_adv(self.baseflow, self.weir_length, H_i, P_i,
                                           xs.y_1_shifted, xs.y_2_shifted, xs.dist)
                    else: # self.calc_mode == "Simplified":
                        y_1 = solve_y1_simp(H_i, P_i)
                    if y_1 > P_i:
                        print(f"Warning: Dam {self.id} XS {i} appears drowned.")

                    # save the dam height to the cross-section
                    xs.set_dam_height(P_i)
                    xs.set_weir_head(H_i)
                    xs_info['P_height'] = P_i

                except Exception as e:
                    print(f"Solver failed for Dam {self.id} XS {i}: {e}")
                    xs_info['P_height'] = None

                if xs.P:
                    try:
                        q_min, q_max = xs.get_dangerous_flow_range()
                        xs_info['Qmin'] = q_min
                        xs_info['Qmax'] = q_max
                    except:
                        pass

                for _, inc_row in self.incidents_df.iterrows():
                    if self.hydrology == 'National Water Model':
                        Q = inc_row['flow_nwm']
                    else:
                        Q = inc_row['flow_geo']
                    date = inc_row['date']
                    if pd.notna(Q) and xs.P:
                        try:
                            y_t = xs.get_tailwater_depth(Q)
                            # Updated to pass shifted profiles
                            
                            # Recalculate H for this Q
                            H_current = weir_H_simp(Q, xs.L)
                            
                            if self.calc_mode == "Advanced":
                                y_1_curr = solve_y1_adv(Q, xs.L, H_current, xs.P, xs.y_1_shifted, xs.y_2_shifted, xs.dist)
                                y_2 = solve_y2_adv(Q, xs.L, H_current, xs.P, xs.y_1_shifted, xs.y_2_shifted, xs.dist)
                                Fr_1 = calc_froude_custom(Q, y_1_curr, xs.y_1_shifted, xs.y_2_shifted, xs.dist)
                            else:
                                y_1_curr = solve_y1_simp(H_current, xs.P)
                                y_2 = calc_y2_simp(H_current, xs.P)
                                Fr_1 = solve_Fr_simp(H_current, xs.P)


                            y_flip = compute_y_flip(Q, xs.L, xs.P)
                            jump = hydraulic_jump_type(y_2, y_t, y_flip)
                            hydro_results_list.append({
                                'site_id': self.id,
                                'date': date,
                                'xs_index': export_idx,
                                'y_t': y_t,
                                'y_flip': y_flip,
                                'y_1': y_1_curr,
                                'y_2': y_2,
                                'Fr_1': Fr_1,
                                'jump_type': jump,
                            })
                        except:
                            pass
            xs_data_list.append(xs_info)
        return xs_data_list, hydro_results_list

    def get_flow_data(self):
        flow_series = None
        if self.hydrology == 'National Water Model':
            if self.nwm_id and not pd.isna(self.nwm_id):
                try:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(current_dir)
                    zarr_path = os.path.join(project_root, 'data', 'nwm_v3_daily_retrospective.zarr')
                    if os.path.exists(zarr_path):
                        ds = xr.open_zarr(zarr_path)
                        if 'streamflow' in ds:
                            # Assuming the zarr structure has feature_id as a dimension or coordinate,
                            # and we want to select by it.
                            # Adjust selection logic based on actual Zarr structure if needed.
                            # Typically: ds.sel(feature_id=int(self.nwm_id)).streamflow.to_series()
                            # But let's be safe and check if feature_id is a dim.
                            
                            # Note: xarray selection is lazy, so this is efficient.
                            try:
                                flow_series = ds.sel(feature_id=int(self.nwm_id))['streamflow'].to_series()
                            except Exception as e:
                                print(f"Error selecting ID {self.nwm_id} from Zarr: {e}")

                except Exception as e:
                    print(f"Error reading NWM Zarr data: {e}")
        elif self.hydrology == 'GEOGLOWS':
            if self.geoglows_id and not pd.isna(self.geoglows_id):
                try:
                    df = geoglows.data.retrospective(river_id=int(self.geoglows_id),
                                                     resolution='daily',
                                                     bias_corrected=True)
                    flow_series = df.iloc[:, 0]
                except Exception as e:
                    print(f"Error fetching GEOGLOWS data: {e}")
        return flow_series

    def plot_rating_curves(self):
        for cross_section in self.cross_sections[1:]:
            cross_section.create_rating_curve()
        plt.xlabel('Flow (m$^{3}$/s)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title(f'Rating Curves for LHD No. {self.id}', fontsize=14)
        plt.legend(title="Rating Curve Equations", loc='best', fontsize=12, title_fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(os.path.join(self.fig_dir, f"Rating Curves for LHD No. {self.id}.png"))

    def plot_cross_sections(self):
        for cross_section in self.cross_sections:
            cross_section.plot_cross_section()

    def plot_all_curves(self):
        for cross_section in self.cross_sections[1:]:
            cross_section.create_combined_fig()

    def plot_map(self):
        # strm_gpkg = os.path.join(self.results_dir, str(self.id), "STRM", f"{self.id}_StrmShp.gpkg")
        # strm_gdf = gpd.read_file(strm_gpkg).to_crs('EPSG:3857')
        xs_gdf = gpd.read_file(self.xs_gpkg).to_crs('EPSG:3857')
        
        # Reverse to match Dam order
        xs_gdf = xs_gdf.iloc[::-1].reset_index(drop=True)

        buffer = 100
        minx, miny, maxx, maxy = xs_gdf.total_bounds
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xlim(minx - buffer * 2, maxx + buffer * 2)
        ax.set_ylim(miny - buffer, maxy + buffer)
        ctx.add_basemap(ax, crs='EPSG:3857', source="Esri.WorldImagery", zorder=0)

        if 'Relative_Loc' in xs_gdf.columns:
            gdf_upstream = xs_gdf[xs_gdf['Relative_Loc'] == 'Upstream']
            gdf_downstream = xs_gdf[xs_gdf['Relative_Loc'] != 'Upstream']
        else:
            gdf_upstream = xs_gdf.iloc[[0]]
            gdf_downstream = xs_gdf.iloc[1:]

        # strm_gdf.plot(ax=ax, color='green', markersize=100, edgecolor='black', zorder=2, label="Flowline")
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
        fig.savefig(os.path.join(self.fig_dir, f"LHD No. {self.id} Location.png"), dpi=300, bbox_inches='tight')
        return fig

    def plot_water_surface(self, save=True):
        cf_csv = os.path.join(self.results_dir, str(self.id), "VDT", f"{self.id}_Curve.csv")
        xs_txt = os.path.join(self.results_dir, str(self.id), "XS", f"{self.id}_XS.txt")
        database_df = merge_databases(cf_csv, xs_txt)
        fig = Figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.plot(database_df.index, database_df['DEM_Elev'], color='dodgerblue', label='DEM Elevation')
        ax.plot(database_df.index, database_df['BaseElev'], color='black', label='Bed Elevation')

        ds_label_added = False
        for i, xs in enumerate(self.cross_sections):
            if i >= len(self.dam_gdf):
                continue

            row_data = self.dam_gdf.iloc[i]
            matches = database_df[(database_df['Row'] == row_data['Row']) & (database_df['Col'] == row_data['Col'])]

            if not matches.empty:
                idx = matches.index[0]
                if xs.location == 'Upstream':
                    ax.axvline(x=idx, color='red', linestyle='--', label='Upstream Cross-Section')
                else:
                    label = 'Downstream Cross-Sections' if not ds_label_added else ""
                    ax.axvline(x=idx, color='cyan', linestyle='--', label=label)
                    ds_label_added = True

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
