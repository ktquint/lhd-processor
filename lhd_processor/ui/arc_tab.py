import gc
import os
import threading
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed

from . import utils
from ..data_manager import DatabaseManager

try:
    from ..prep.lhd_arc import ArcDam
except Exception as e:
    print(f"Warning: Could not import ARC. Error: {e}")
    ArcDam = None

try:
    from ..prep.nwm_roughness import get_regional_manning_n
except Exception as e:
    print(f"Warning: Could not import NWM roughness lookup. Error: {e}")
    get_regional_manning_n = None

from ..prep.download_geospatial_data import CONSTANT_LC_CODE, make_constant_land_raster

# Module-level widgets
arc_xlsx_entry = None
arc_results_entry = None
arc_flowline_var = None
arc_streamflow_var = None
arc_baseflow_var = None
arc_ep_entry = None
arc_manning_mode_var = None
arc_manning_entry = None
arc_manning_label = None
arc_run_button = None
arc_stop_button = None

stop_event = threading.Event()

def setup_arc_tab(parent_tab):
    global arc_xlsx_entry, arc_results_entry, arc_flowline_var, arc_streamflow_var, arc_baseflow_var, arc_ep_entry
    global arc_manning_mode_var, arc_manning_entry, arc_manning_label
    global arc_run_button, arc_stop_button

    arc_frame = ttk.LabelFrame(parent_tab, text="Step 2: Automated Rating Curves")
    arc_frame.pack(pady=10, padx=10, fill="x")
    arc_frame.columnconfigure(1, weight=1)

    # Row 0: Excel Database
    ttk.Label(arc_frame, text="Excel Database:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    arc_xlsx_entry = ttk.Entry(arc_frame)
    arc_xlsx_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    
    def check_arc_xlsx_exists(_):
        path = arc_xlsx_entry.get()
        if path and not os.path.isfile(path):
            messagebox.showwarning("Path Not Found", f"The specified Excel file does not exist:\n{path}")
    arc_xlsx_entry.bind("<FocusOut>", check_arc_xlsx_exists)

    ttk.Button(arc_frame, text="Select...", command=select_arc_xlsx).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Results Folder
    ttk.Label(arc_frame, text="Results Folder:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    arc_results_entry = ttk.Entry(arc_frame)
    arc_results_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(arc_frame, text="Select...", command=select_arc_results_dir).grid(row=1, column=2, padx=5, pady=5)

    # Row 2: Flowline Source & Streamflow Source
    opts_frame = ttk.Frame(arc_frame)
    opts_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=5)
    opts_frame.columnconfigure(1, weight=1)
    opts_frame.columnconfigure(3, weight=1)

    ttk.Label(opts_frame, text="Flowline Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    arc_flowline_var = tk.StringVar(value="NHDPlus")
    ttk.Combobox(opts_frame, textvariable=arc_flowline_var, state="readonly", values=("NHDPlus", "TDX-Hydro")).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Label(opts_frame, text="Streamflow Source (locked to Flowline Source):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    arc_streamflow_var = tk.StringVar(value="National Water Model")
    arc_streamflow_combo = ttk.Combobox(opts_frame, textvariable=arc_streamflow_var, state="disabled",
                                         values=("National Water Model", "GEOGLOWS"))
    arc_streamflow_combo.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

    # NHDPlus reaches only have NWM reach IDs; TDX-Hydro reaches only have
    # GEOGLOWS linknos -- so the two aren't independent choices, and picking
    # them separately is how a dam ends up processed with mismatched
    # flowline/streamflow data.
    def _sync_arc_streamflow_source(*args):
        arc_streamflow_var.set("GEOGLOWS" if arc_flowline_var.get() == "TDX-Hydro" else "National Water Model")

    arc_flowline_var.trace_add("write", _sync_arc_streamflow_source)
    _sync_arc_streamflow_source()

    # Row 3: Baseflow Estimation
    bf_frame = ttk.Frame(arc_frame)
    bf_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
    bf_frame.columnconfigure(1, weight=1)
    bf_frame.columnconfigure(3, weight=1)

    ttk.Label(bf_frame, text="Baseflow Estimation:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    arc_baseflow_var = tk.StringVar(value="WSE and LiDAR Date")
    bf_values = ("WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation", "WSE and Exceedance Probability")
    ttk.Combobox(bf_frame, textvariable=arc_baseflow_var, state="readonly", values=bf_values).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    ep_label = ttk.Label(bf_frame, text="Exceedance Probability (%):")
    arc_ep_entry = ttk.Entry(bf_frame, width=10)
    arc_ep_entry.insert(0, "50")

    def on_bf_method_change(*args):
        if arc_baseflow_var.get() == "WSE and Exceedance Probability":
            ep_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            arc_ep_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)
        else:
            ep_label.grid_remove()
            arc_ep_entry.grid_remove()

    arc_baseflow_var.trace_add("write", on_bf_method_change)
    # Call once to set initial state
    on_bf_method_change()

    # Row 4: Manning's n
    mn_frame = ttk.Frame(arc_frame)
    mn_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=5)
    mn_frame.columnconfigure(1, weight=1)
    mn_frame.columnconfigure(3, weight=1)

    ttk.Label(mn_frame, text="Manning's n Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    arc_manning_mode_var = tk.StringVar(value="Uniform Value")
    mn_mode_combo = ttk.Combobox(mn_frame, textvariable=arc_manning_mode_var, state="readonly",
                                  values=("Uniform Value", "Regionalized (NWM)"))
    mn_mode_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    arc_manning_label = ttk.Label(mn_frame, text="Manning's n:")
    arc_manning_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    arc_manning_entry = ttk.Entry(mn_frame, width=10)
    arc_manning_entry.insert(0, "0.035")
    arc_manning_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

    def on_manning_mode_change(*args):
        if arc_manning_mode_var.get() == "Regionalized (NWM)":
            arc_manning_label.config(text="Fallback n (reach not in NWM data):")
        else:
            arc_manning_label.config(text="Manning's n:")

    arc_manning_mode_var.trace_add("write", on_manning_mode_change)
    on_manning_mode_change()

    # Row 5: Buttons
    btn_frame = ttk.Frame(arc_frame)
    btn_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=10)
    btn_frame.columnconfigure(0, weight=1)
    btn_frame.columnconfigure(1, weight=1)

    arc_run_button = ttk.Button(btn_frame, text="2. Run ARC", command=start_arc_thread)
    arc_run_button.grid(row=0, column=0, padx=5, sticky=tk.EW)

    arc_stop_button = ttk.Button(btn_frame, text="STOP Processing", command=stop_arc_thread, state=tk.DISABLED)
    arc_stop_button.grid(row=0, column=1, padx=5, sticky=tk.EW)

def select_arc_xlsx():
    f = filedialog.askopenfilename(filetypes=[("Excel Database", "*.xlsx")])
    if f:
        arc_xlsx_entry.delete(0, tk.END)
        arc_xlsx_entry.insert(0, f)
        # Try to auto-fill results
        try:
            project_path = os.path.dirname(f)
            arc_results_entry.delete(0, tk.END)
            arc_results_entry.insert(0, os.path.join(project_path, "Results"))
        except (OSError, AttributeError):
            pass

def select_arc_results_dir():
    d = filedialog.askdirectory()
    if d:
        arc_results_entry.delete(0, tk.END)
        arc_results_entry.insert(0, d)

def create_mannings_table(manning_txt, manning_n_value):
    """Writes the Manning's n lookup table ARC keys off the constant land raster.

    Only two codes exist on that raster's effective land-use array: the
    constant placeholder fill (CONSTANT_LC_CODE) and 80/"water", which ARC
    always force-overlays onto every stream cell regardless of the source
    landcover raster (see Automated_Rating_Curve_Generator.py's stream-cell
    overlay). Both map to the same resolved n for this dam.
    """
    with open(manning_txt, 'w') as f:
        f.write('LC_ID\tDescription\tManning_n\n')
        f.write(f'{CONSTANT_LC_CODE}\tConstant\t{manning_n_value}\n')
        f.write(f'80\tWater\t{manning_n_value}\n')

def process_single_dam_arc(dam_dict, results_dir, flowline_source, streamflow_source, baseflow_method, ep_value=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['GDAL_CACHEMAX'] = '256'

    dam_id = int(dam_dict.get(list(dam_dict.keys())[0]))
    dam_name = dam_dict.get('name', "Unknown Dam")
    try:
        if ep_value is not None:
            dam_dict['ep_value'] = ep_value
        dam_i = ArcDam(dam_dict, results_dir, flowline_source, streamflow_source, baseflow_method)
        dam_i.process_dam()
        del dam_i
        gc.collect()
        return True, dam_id, dam_name, None
    except Exception as e:
        gc.collect()
        return False, dam_id, dam_name, str(e)

def threaded_run_arc():
    try:
        stop_event.clear()
        arc_run_button.config(state=tk.DISABLED)
        arc_stop_button.config(state=tk.NORMAL)

        excel_loc = arc_xlsx_entry.get()
        results_loc = arc_results_entry.get()
        fl_source = arc_flowline_var.get()
        sf_source = arc_streamflow_var.get()
        bf_method = arc_baseflow_var.get()
        manning_mode = arc_manning_mode_var.get()
        manning_n = arc_manning_entry.get()

        ep_value = None
        if bf_method == "WSE and Exceedance Probability":
            try:
                ep_value = float(arc_ep_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid Exceedance Probability value.")
                return

        if not os.path.exists(excel_loc):
            messagebox.showerror("Error", "Excel database not found.")
            return

        try:
            manning_n_value = float(manning_n)
        except ValueError:
            messagebox.showerror("Error", "Invalid Manning's n value.")
            return

        # Fallback only -- used for a dam that has no land raster at all yet
        # (Step 1 was never run for it). Normally each dam's Manning's n file
        # is written next to its own land raster (see dam_land_dirs below),
        # wherever Step 1 was actually told to put it (LAND, LAND_NWM,
        # LAND_GEO, ...), not this hardcoded default.
        project_path = os.path.dirname(excel_loc)
        default_land_folder = os.path.join(project_path, "LAND")
        os.makedirs(default_land_folder, exist_ok=True)

        df = pd.read_excel(excel_loc)
        dams = df.to_dict(orient='records')
        count_to_run = len(dams)

        # DEM and land raster are each built to fit one flowline source's
        # extent (see PrepDam._dem_land_key), so pick the column matching
        # this run -- falling back to the old shared column for dams that
        # predate the per-source split.
        dem_col = 'dem_path_tdx' if fl_source == 'TDX-Hydro' else 'dem_path_nhd'
        land_col = 'land_raster_tdx' if fl_source == 'TDX-Hydro' else 'land_raster_nhd'

        # Land raster is a constant placeholder aligned to the DEM grid (the
        # actual n value is resolved via the LU_Manning_n lookup table below,
        # not the raster's pixel values). If a dam's DEM was ever regenerated
        # with a different extent since its land raster was made, the two go
        # out of alignment and ARC fails with "Rows do not Match!" -- so
        # re-verify/rebuild it here every time ARC is (re)run, whichever
        # Manning's n source is selected.
        utils.set_status("Verifying land cover rasters match current DEMs...")
        dam_land_dirs = {}
        for dam in dams:
            dam_id = int(dam.get(list(dam.keys())[0]))
            dem_path = dam.get(dem_col) if pd.notna(dam.get(dem_col)) else dam.get('dem_path')
            if not dem_path or pd.isna(dem_path) or not os.path.exists(dem_path):
                continue
            existing_land_raster = dam.get(land_col) if pd.notna(dam.get(land_col)) else dam.get('land_raster')
            if existing_land_raster and pd.notna(existing_land_raster):
                site_land_dir = os.path.dirname(os.path.dirname(existing_land_raster))
            else:
                site_land_dir = default_land_folder
            dam_land_dirs[dam_id] = site_land_dir
            try:
                new_land_raster = make_constant_land_raster(dam_id, dem_path, site_land_dir)
                dam['land_raster'] = new_land_raster
                dam[land_col] = new_land_raster
            except Exception as e:
                utils.set_status(f"Warning: Could not verify/build land raster for dam {dam_id}: {e}")

        # Per-dam Manning's n actually resolved for this run -- persisted to the
        # Excel DB below so it doesn't only live in the LAND/*.txt files.
        resolved_manning_n = {}

        if manning_mode == "Regionalized (NWM)":
            if get_regional_manning_n is None:
                messagebox.showerror("Error", "NWM roughness lookup is unavailable (missing dependency).")
                return
            if 'reach_id' not in df.columns:
                messagebox.showerror("Error", "Excel database has no 'reach_id' column. Run Step 1 with NHDPlus/National Water Model selected first.")
                return

            reach_ids = [r for r in df['reach_id'].tolist() if pd.notna(r)]
            utils.set_status(f"Looking up NWM channel roughness for {len(set(reach_ids))} reaches...")
            regional_n = get_regional_manning_n(reach_ids, status_callback=utils.set_status)
            utils.set_status(f"Found NWM roughness for {len(regional_n)} of {len(set(reach_ids))} reaches; "
                              f"using fallback n={manning_n_value} for the rest.")

            for dam in dams:
                dam_id = int(dam.get(list(dam.keys())[0]))
                reach_id = dam.get('reach_id')
                n_value = regional_n.get(int(reach_id)) if pd.notna(reach_id) else None
                if n_value is None:
                    n_value = manning_n_value
                dam_land_dir = dam_land_dirs.get(dam_id, default_land_folder)
                manning_txt = os.path.join(dam_land_dir, f'Manning_n_{dam_id}.txt')
                create_mannings_table(manning_txt, n_value)
                dam['manning_n_txt'] = manning_txt
                dam['manning_n_value'] = n_value
                resolved_manning_n[dam_id] = (n_value, manning_txt)
            utils.set_status("Created per-dam regionalized Manning's n files.")
        else:
            # Uniform value applied to all dams, but still written next to
            # each dam's own land raster rather than one shared location --
            # keeps every dam's ARC_Input self-contained within its own
            # LAND/LAND_NWM/LAND_GEO folder.
            for dam in dams:
                dam_id = int(dam.get(list(dam.keys())[0]))
                dam_land_dir = dam_land_dirs.get(dam_id, default_land_folder)
                manning_txt = os.path.join(dam_land_dir, f'Manning_n_{dam_id}.txt')
                create_mannings_table(manning_txt, manning_n_value)
                # Clear out any per-dam override left over from a prior
                # Regionalized (NWM) run -- otherwise ArcDam.__init__ prefers
                # that stale path over this run's uniform-value file.
                dam['manning_n_txt'] = manning_txt
                dam['manning_n_value'] = manning_n_value
                resolved_manning_n[dam_id] = (manning_n_value, manning_txt)
            utils.set_status("Created per-dam uniform Manning's n files.")

        total_cores = os.cpu_count() or 1
        worker_count = max(1, int(total_cores / 3))

        utils.set_status(f"Initializing Dask Workers ({worker_count}) for {count_to_run} dams...")

        run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_results = {}  # dam_id -> (success, err)

        with LocalCluster(processes=True, threads_per_worker=1, n_workers=worker_count) as cluster:
            with Client(cluster) as client:
                utils.set_status(f"Processing... (Dashboard: {client.dashboard_link})")
                futures = client.map(process_single_dam_arc, dams, results_dir=results_loc, flowline_source=fl_source, streamflow_source=sf_source, baseflow_method=bf_method, ep_value=ep_value)

                processed_so_far = 0
                for future in dask_as_completed(futures):
                    if stop_event.is_set():
                        client.cancel(futures)
                        break

                    try:
                        success, dam_id, name, err = future.result()
                        run_results[dam_id] = (success, err)
                        processed_so_far += 1
                        if success:
                            utils.set_status(f"Finished {name} ({processed_so_far}/{count_to_run})")
                        else:
                            utils.set_status(f"Failed {name} ({processed_so_far}/{count_to_run}): {err}")
                    except Exception as e:
                        processed_so_far += 1
                        utils.set_status(f"Worker crashed on a task ({processed_so_far}/{count_to_run}): {e}")

        # Persist run provenance (what was actually fed into the hydraulic model)
        # back to the Excel DB, so it's fully documented alongside the results.
        utils.set_status("Saving run provenance to Excel database...")
        try:
            db = DatabaseManager(excel_loc)
            for dam_id, (n_value, n_txt) in resolved_manning_n.items():
                success, err = run_results.get(dam_id, (None, None))
                status = "Success" if success else (f"Failed: {err}" if success is False else "Not Run")
                db.update_site_data(dam_id, {
                    'arc_flowline_source': fl_source,
                    'arc_streamflow_source': sf_source,
                    'baseflow_method': bf_method,
                    'ep_value': ep_value,
                    'manning_n_mode': manning_mode,
                    'manning_n_value': n_value,
                    'manning_n_txt': n_txt,
                    'arc_run_date': run_timestamp,
                    'arc_run_status': status,
                })
            db.save()
        except Exception as e:
            utils.set_status(f"Warning: ARC run finished but failed to save provenance to Excel DB: {e}")

        if stop_event.is_set():
            messagebox.showwarning("Stopped", "Processing stopped by user.")
        else:
            utils.set_status("Batch complete.")
            messagebox.showinfo("Success", f"Finished processing {count_to_run} dams.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
    finally:
        arc_run_button.config(state=tk.NORMAL)
        arc_stop_button.config(state=tk.DISABLED)

def stop_arc_thread():
    if messagebox.askyesno("Stop", "Stop processing after current tasks finish?"):
        stop_event.set()

def start_arc_thread():
    threading.Thread(target=threaded_run_arc, daemon=True).start()
