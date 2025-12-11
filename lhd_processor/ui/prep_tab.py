import os
import json
import threading
import gc
import pandas as pd
import tkinter as tk
from dask.distributed import Client, LocalCluster, as_completed
from hsclient import HydroShare
from tkinter import ttk, filedialog, messagebox

# CLEAN IMPORT: Importing directly from the package
from ..prep import Dam as PrepDam, rathcelon_input
from ..data_manager import DatabaseManager
from . import utils

# Import Rathcelon carefully
try:
    from rathcelon.classes import Dam as RathcelonDam
except Exception as e:
    print(f"Warning: Could not import RathCelon. Error: {e}")
    RathcelonDam = None

# Module-level widgets
project_entry = None
database_entry = None
dem_entry = None
strm_entry = None
land_use_entry = None
results_entry = None
json_entry = None
flowline_var = None
dd_var = None
streamflow_var = None
baseflow_var = None
prep_run_button = None
rath_json_entry = None
rath_run_button = None
rath_stop_button = None

# Global Threading Event for Stopping
stop_event = threading.Event()


def setup_prep_tab(parent_tab):
    """Constructs the UI for the Prep tab."""
    global project_entry, database_entry, dem_entry, strm_entry, land_use_entry, results_entry
    global json_entry, flowline_var, dd_var, streamflow_var, baseflow_var
    global prep_run_button, rath_json_entry, rath_run_button, rath_stop_button

    # --- Step 1 Frame ---
    prep_frame = ttk.LabelFrame(parent_tab, text="Step 1: Prepare Data")
    prep_frame.pack(pady=10, padx=10, fill="x")

    paths_frame = ttk.Frame(prep_frame)
    paths_frame.pack(pady=5, padx=10, fill="x")

    # Configure columns: Label (0), Entry (1-Expands), Button (2)
    paths_frame.columnconfigure(1, weight=1)

    # Helper to make rows with Label -> Entry -> Button
    # CHANGED: Added must_exist parameter
    def add_path_row(row, label_text, cmd, is_file=False, must_exist=True):
        # 1. Label
        ttk.Label(paths_frame, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)

        # 2. Entry (Editable)
        entry = ttk.Entry(paths_frame)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)

        # CHANGED: Passed must_exist to the util
        utils.bind_path_validation(entry, is_file=is_file, must_exist=must_exist)

        # 3. Browse Button
        ttk.Button(paths_frame, text="Select...", command=cmd).grid(row=row, column=2, padx=5, pady=5)

        return entry

    # --- INPUTS (Strict Check: Red if missing) ---
    project_entry = add_path_row(0, "Project Folder:", select_project_dir, is_file=False, must_exist=True)
    database_entry = add_path_row(1, "Database (.xlsx):", select_database, is_file=True, must_exist=True)

    # --- OUTPUTS (Loose Check: Blue if missing, as they will be created) ---
    dem_entry = add_path_row(2, "DEM Folder:", select_dem_dir, is_file=False, must_exist=False)
    strm_entry = add_path_row(3, "Hydrography Folder:", select_strm_dir, is_file=False, must_exist=False)
    land_use_entry = add_path_row(4, "Land Use Folder:", select_land_use_dir, is_file=False, must_exist=False)
    results_entry = add_path_row(5, "Results Folder:", select_results_dir, is_file=False, must_exist=False)
    json_entry = add_path_row(6, "RathCelon Input (.json):", select_json_file, is_file=True, must_exist=False)

    # Dropdowns
    hydro_frame = ttk.Frame(prep_frame)
    hydro_frame.pack(pady=5, padx=10, fill="x")
    hydro_frame.columnconfigure(1, weight=1)

    def add_combo(row, label, values, default):
        ttk.Label(hydro_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(hydro_frame, textvariable=var, state="readonly", values=values)
        cb.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
        return var

    flowline_var = add_combo(0, "Flowline Source:", ("NHDPlus", "GEOGLOWS"), "NHDPlus")
    dd_var = add_combo(1, "DEM Resolution:", ("1 m", "1/9 arc-second (~3 m)", "1/3 arc-second (~10 m)"), "1 m")
    streamflow_var = add_combo(2, "Streamflow Source:", ("National Water Model", "GEOGLOWS"), "National Water Model")
    baseflow_var = add_combo(3, "Baseflow Estimation:",
                             ("WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation"),
                             "WSE and LiDAR Date")
    baseflow_combo = hydro_frame.winfo_children()[-1]

    utils.ToolTip(baseflow_combo, "Choose how to estimate baseflow:\n"
                                  "- LiDAR Date: Uses flow on the specific day the LiDAR was flown.\n"
                                  "- Median: Uses long-term median daily flow.")

    prep_run_button = ttk.Button(prep_frame, text="1. Prepare Data & Create Input File", command=start_prep_thread,
                                 style="Accent.TButton")
    prep_run_button.pack(pady=10, padx=10, fill="x")

    # --- Step 2 Frame ---
    rath_frame = ttk.LabelFrame(parent_tab, text="Step 2: Run RathCelon")
    rath_frame.pack(pady=10, padx=10, fill="x")

    # Configure columns for Label/Entry/Button
    rath_frame.columnconfigure(1, weight=1)

    # Row 0: Input File Selection (Strict Check because this is an INPUT for step 2)
    ttk.Label(rath_frame, text="Input File (.json):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    rath_json_entry = ttk.Entry(rath_frame)
    rath_json_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    # Bind validation manually here since it's outside the helper loop
    utils.bind_path_validation(rath_json_entry, is_file=True, must_exist=True)

    ttk.Button(rath_frame, text="Select...", command=select_rath_json).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Action Buttons (Using a sub-frame to keep them centered/expanded independently)
    btn_frame = ttk.Frame(rath_frame)
    btn_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=10)
    btn_frame.columnconfigure(0, weight=1)
    btn_frame.columnconfigure(1, weight=1)

    rath_run_button = ttk.Button(btn_frame, text="2. Run RathCelon", command=start_rath_thread, style="Accent.TButton")
    rath_run_button.grid(row=0, column=0, padx=5, sticky=tk.EW)

    rath_stop_button = ttk.Button(btn_frame, text="STOP Processing", command=stop_rath_thread, state=tk.DISABLED)
    rath_stop_button.grid(row=0, column=1, padx=5, sticky=tk.EW)


# --- Event Handlers (File Dialogs) ---

def select_project_dir():
    path = filedialog.askdirectory()
    if not path: return
    project_entry.delete(0, tk.END)
    project_entry.insert(0, path)

    try:
        files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
        if files:
            db_path = os.path.join(path, files[0])
            database_entry.delete(0, tk.END)
            database_entry.insert(0, db_path)
            json_path = os.path.splitext(db_path)[0] + '.json'
            json_entry.delete(0, tk.END)
            json_entry.insert(0, json_path)
            rath_json_entry.delete(0, tk.END)
            rath_json_entry.insert(0, json_path)

        dem_entry.delete(0, tk.END)
        dem_entry.insert(0, os.path.join(path, "DEM"))
        strm_entry.delete(0, tk.END)
        strm_entry.insert(0, os.path.join(path, "STRM"))
        if land_use_entry:
            land_use_entry.delete(0, tk.END)
            land_use_entry.insert(0, os.path.join(path, "LAND"))
        results_entry.delete(0, tk.END)
        results_entry.insert(0, os.path.join(path, "Results"))
        utils.set_status("Project paths loaded.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load paths: {e}")


def select_database():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f:
        database_entry.delete(0, tk.END)
        database_entry.insert(0, f)
        json_path = os.path.splitext(f)[0] + '.json'
        json_entry.delete(0, tk.END)
        json_entry.insert(0, json_path)
        rath_json_entry.delete(0, tk.END)
        rath_json_entry.insert(0, json_path)


def select_dem_dir():
    d = filedialog.askdirectory()
    if d:
        dem_entry.delete(0, tk.END)
        dem_entry.insert(0, d)


def select_strm_dir():
    d = filedialog.askdirectory()
    if d:
        strm_entry.delete(0, tk.END)
        strm_entry.insert(0, d)

def select_land_use_dir():
    d = filedialog.askdirectory()
    if d:
        land_use_entry.delete(0, tk.END)
        land_use_entry.insert(0, d)

def select_results_dir():
    d = filedialog.askdirectory()
    if d: results_entry.delete(0, tk.END)
    results_entry.insert(0, d)


def select_json_file():
    f = filedialog.asksaveasfilename(filetypes=[("JSON", "*.json")], defaultextension=".json")
    if f:
        json_entry.delete(0, tk.END)
        json_entry.insert(0, f)


def select_rath_json():
    f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
    if f:
        rath_json_entry.delete(0, tk.END)
        rath_json_entry.insert(0, f)


# --- Stop Handler ---

def stop_rath_thread():
    """Sets the stop event to halt processing."""
    if messagebox.askyesno("Stop Processing",
                           "Are you sure you want to stop? Current tasks will complete, but the queue will be cleared."):
        stop_event.set()
        utils.set_status("Stopping... Waiting for running workers to finish.")


# --- Logic Functions ---

def process_single_dam_rathcelon(dam_dict):
    """Worker for Dask."""

    # --- Performance Fix: Force single-threaded libraries ---
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # --- Memory Fix: Limit GDAL Cache per worker ---
    os.environ['GDAL_CACHEMAX'] = '256'

    dam_name = dam_dict.get('name', "Unknown Dam")
    try:
        from rathcelon.classes import Dam as RD
        dam_i = RD(**dam_dict)
        dam_i.process_dam()

        del dam_i
        gc.collect()

        return True, dam_name, None
    except Exception as e:
        gc.collect()
        return False, dam_name, str(e)


def threaded_prepare_data():
    try:
        xlsx_path = database_entry.get()
        flowline_source = flowline_var.get()
        dem_resolution = dd_var.get()
        streamflow_source = streamflow_var.get()
        dem_folder = dem_entry.get()
        strm_folder = strm_entry.get()
        land_folder = land_use_entry.get() if land_use_entry else None
        results_folder = results_entry.get()
        baseflow_method = baseflow_var.get()

        if not os.path.exists(xlsx_path):
            messagebox.showerror("Error", f"Database file not found:\n{xlsx_path}")
            return

        utils.set_status("Inputs validated. Loading database...")
        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        if land_folder:
            os.makedirs(land_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        db = DatabaseManager(xlsx_path)

        hydroshare_id = "88759266f9c74df8b5bb5f52d142ba8e"
        ui_dir = os.path.dirname(os.path.realpath(__file__))
        package_root = os.path.dirname(ui_dir)
        data_dir = os.path.join(package_root, 'data')
        os.makedirs(data_dir, exist_ok=True)

        nwm_parquet = os.path.join(data_dir, 'nwm_v3_daily_retrospective.parquet')
        vpu_filename = "vpu-boundaries.gpkg"
        tdx_vpu_map = os.path.join(data_dir, vpu_filename)
        nwm_ds = None

        if streamflow_source == 'National Water Model':
            if not os.path.exists(nwm_parquet):
                utils.set_status("Downloading NWM Parquet...")
                try:
                    hs = HydroShare()
                    resource = hs.resource(hydroshare_id)
                    resource.file_download(path='nwm_v3_daily_retrospective.parquet', save_path=data_dir)
                except Exception as e:
                    messagebox.showerror("Download Failed", f"Failed to download NWM file: {e}")
                    return

            utils.set_status("Loading NWM dataset...")
            try:
                nwm_df = pd.read_parquet(nwm_parquet)
                nwm_ds = nwm_df.to_xarray()
            except Exception as e:
                utils.set_status("Error loading NWM dataset.")
                print(e)

        if 'GEOGLOWS' in [streamflow_source, flowline_source]:
            if not os.path.exists(tdx_vpu_map):
                utils.set_status("Downloading VPU Map...")
                try:
                    hs = HydroShare()
                    resource = hs.resource(hydroshare_id)
                    resource.file_download(path=vpu_filename, save_path=data_dir)
                except Exception as e:
                    messagebox.showerror("Download Failed", f"Failed to download VPU map: {e}")
                    return

        total_dams = len(db.sites)
        processed_count = 0

        for i, site_id in enumerate(db.sites['site_id']):
            try:
                utils.set_status(f"Prep: Dam {site_id} ({i + 1}/{total_dams})...")

                dam = PrepDam(site_id, db)
                dam.set_streamflow_source(streamflow_source)
                dam.set_flowline_source(flowline_source)
                dam.set_output_dir(results_folder)
                dam.assign_flowlines(strm_folder, tdx_vpu_map)
                dam.assign_dem(dem_folder, dem_resolution)
                dam.assign_land(dem_folder, land_folder)

                if not any([dam.dem_1m, dam.dem_3m, dam.dem_10m]):
                    print(f"Skipping Dam {site_id}: No DEM.")
                    continue

                if streamflow_source == 'National Water Model' and nwm_ds is None:
                    pass
                else:
                    dam.create_reach(nwm_ds, tdx_vpu_map)
                    dam.set_dem_baseflow(baseflow_method)
                    dam.set_fatal_flows()

                dam.save_changes()
                processed_count += 1

            except Exception as e:
                print(f"Error prepping Dam {site_id}: {e}")

        if processed_count > 0:
            utils.set_status("Saving database...")
            db.save()

            json_loc = json_entry.get()
            rathcelon_input(
                xlsx_path,
                json_loc,
                baseflow_method,
                nwm_parquet,
                flowline_source,
                streamflow_source
            )

            utils.set_status(f"Prep complete. {processed_count} dams processed.")
            messagebox.showinfo("Success", f"Prep complete. Database updated.")
        else:
            utils.set_status("No dams processed successfully.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
        print(e)
    finally:
        prep_run_button.config(state=tk.NORMAL)


def threaded_run_rathcelon():
    try:
        # 1. Reset Stop Event at start
        stop_event.clear()

        # 2. Toggle Buttons
        rath_run_button.config(state=tk.DISABLED)
        rath_stop_button.config(state=tk.NORMAL)

        json_loc = rath_json_entry.get()
        if not os.path.exists(json_loc):
            messagebox.showerror("Error", "JSON file not found.")
            return

        with open(json_loc, 'r') as f:
            data = json.load(f)

        dams = data.get("dams", [])
        total_dams = len(dams)

        dams_to_process = []
        skipped_count = 0

        utils.set_status("Checking for existing results...")

        for dam in dams:
            site_id = str(dam.get("dam_id"))
            output_dir = dam.get("output_dir")
            expected_file = os.path.join(output_dir, site_id, "XS", f"{site_id}_Local_XS_Lines.gpkg")

            if os.path.exists(expected_file):
                skipped_count += 1
            else:
                dams_to_process.append(dam)

        count_to_run = len(dams_to_process)

        if count_to_run == 0:
            utils.set_status(f"All {total_dams} dams are already processed!")
            messagebox.showinfo("Complete", "All dams in this file have existing results.")
            return

        total_cores = os.cpu_count() or 1
        worker_count = 1 # max(1, int(total_cores / 3))

        utils.set_status(
            f"Skipping {skipped_count}. Initializing Dask (Workers: {worker_count}) for {count_to_run} dams...")

        with LocalCluster(processes=True, threads_per_worker=1, n_workers=worker_count) as cluster:
            with Client(cluster) as client:

                print(f"Dask Dashboard: {client.dashboard_link}")
                utils.set_status(f"Processing {count_to_run} dams... (Dashboard: {client.dashboard_link})")

                futures = client.map(process_single_dam_rathcelon, dams_to_process)

                success_count = 0
                processed_so_far = 0

                for future in as_completed(futures):
                    # --- STOP CHECK ---
                    if stop_event.is_set():
                        client.cancel(futures)  # Attempt to cancel pending tasks
                        utils.set_status("Processing halted by user.")
                        break  # Break the monitoring loop

                    success, name, err = future.result()
                    processed_so_far += 1

                    if success:
                        success_count += 1
                        utils.set_status(f"Finished {name} ({processed_so_far}/{count_to_run})")
                    else:
                        utils.set_status(f"Error on {name}: {err}")

        # Check if we exited early
        if stop_event.is_set():
            messagebox.showwarning("Stopped", f"Processing stopped by user.\nCompleted this session: {success_count}")
        else:
            total_success = success_count + skipped_count
            utils.set_status(f"Done. Ran {success_count}, Skipped {skipped_count}. Total: {total_success}/{total_dams}")
            messagebox.showinfo("Success", f"Batch complete.\nRan: {success_count}\nSkipped: {skipped_count}")

    except Exception as e:
        utils.set_status(f"Error: {e}")
        messagebox.showerror("Error", str(e))
    finally:
        # Reset Buttons
        rath_run_button.config(state=tk.NORMAL)
        rath_stop_button.config(state=tk.DISABLED)


# Thread Starters
def start_prep_thread():
    prep_run_button.config(state=tk.DISABLED)
    threading.Thread(target=threaded_prepare_data, daemon=True).start()


def start_rath_thread():
    # Button state is toggled inside the threaded function now
    threading.Thread(target=threaded_run_rathcelon, daemon=True).start()
