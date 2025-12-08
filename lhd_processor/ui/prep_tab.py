import os
import json
import threading
import tkinter as tk
import concurrent.futures
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from hsclient import HydroShare

# CLEAN IMPORT: Importing directly from the package, not internal files
from ..prep import Dam as PrepDam, rathcelon_input
from . import utils

# Import Rathcelon carefully
try:
    from rathcelon.classes import Dam as RathcelonDam
except ImportError:
    RathcelonDam = None

# Module-level widgets
project_entry = None
database_entry = None
dem_entry = None
strm_entry = None
results_entry = None
json_entry = None
flowline_var = None
dd_var = None
streamflow_var = None
baseflow_var = None
prep_run_button = None
rath_json_entry = None
rath_run_button = None


def setup_prep_tab(parent_tab):
    """Constructs the UI for the Prep tab."""
    global project_entry, database_entry, dem_entry, strm_entry, results_entry
    global json_entry, flowline_var, dd_var, streamflow_var, baseflow_var
    global prep_run_button, rath_json_entry, rath_run_button

    # --- Step 1 Frame ---
    prep_frame = ttk.LabelFrame(parent_tab, text="Step 1: Prepare Data")
    prep_frame.pack(pady=10, padx=10, fill="x")

    paths_frame = ttk.Frame(prep_frame)
    paths_frame.pack(pady=5, padx=10, fill="x")
    paths_frame.columnconfigure(1, weight=1)

    # Helper to make rows
    def add_path_row(row, label, cmd):
        btn = ttk.Button(paths_frame, text=label, command=cmd)
        btn.grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        entry = ttk.Entry(paths_frame)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
        return entry

    project_entry = add_path_row(0, "Select Project Folder", select_project_dir)
    database_entry = add_path_row(1, "Select Database (.csv)", select_database)
    dem_entry = add_path_row(2, "Select DEM Folder", select_dem_dir)
    strm_entry = add_path_row(3, "Select Hydrography Folder", select_strm_dir)
    results_entry = add_path_row(4, "Select Results Folder", select_results_dir)
    json_entry = add_path_row(5, "RathCelon Input (.json)", select_json_file)

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

    prep_run_button = ttk.Button(prep_frame, text="1. Prepare Data & Create Input File", command=start_prep_thread,
                                 style="Accent.TButton")
    prep_run_button.pack(pady=10, padx=10, fill="x")

    # --- Step 2 Frame ---
    rath_frame = ttk.LabelFrame(parent_tab, text="Step 2: Run RathCelon")
    rath_frame.pack(pady=10, padx=10, fill="x")
    rath_frame.columnconfigure(1, weight=1)

    ttk.Button(rath_frame, text="Select Input File (.json)", command=select_rath_json).grid(row=0, column=0, padx=5,
                                                                                            pady=5)
    rath_json_entry = ttk.Entry(rath_frame)
    rath_json_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    rath_run_button = ttk.Button(rath_frame, text="2. Run RathCelon", command=start_rath_thread, style="Accent.TButton")
    rath_run_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)


# --- Event Handlers (File Dialogs) ---

def select_project_dir():
    path = filedialog.askdirectory()
    if not path: return
    project_entry.delete(0, tk.END);
    project_entry.insert(0, path)

    try:
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            db_path = os.path.join(path, csv_files[0])
            database_entry.delete(0, tk.END);
            database_entry.insert(0, db_path)
            json_path = os.path.splitext(db_path)[0] + '.json'
            json_entry.delete(0, tk.END);
            json_entry.insert(0, json_path)
            rath_json_entry.delete(0, tk.END);
            rath_json_entry.insert(0, json_path)

        dem_entry.delete(0, tk.END);
        dem_entry.insert(0, os.path.join(path, "DEM"))
        strm_entry.delete(0, tk.END);
        strm_entry.insert(0, os.path.join(path, "STRM"))
        results_entry.delete(0, tk.END);
        results_entry.insert(0, os.path.join(path, "Results"))
        utils.set_status("Project paths loaded.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load paths: {e}")


def select_database():
    f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if f: database_entry.delete(0, tk.END); database_entry.insert(0, f)


def select_dem_dir():
    d = filedialog.askdirectory()
    if d: dem_entry.delete(0, tk.END); dem_entry.insert(0, d)


def select_strm_dir():
    d = filedialog.askdirectory()
    if d: strm_entry.delete(0, tk.END); strm_entry.insert(0, d)


def select_results_dir():
    d = filedialog.askdirectory()
    if d: results_entry.delete(0, tk.END); results_entry.insert(0, d)


def select_json_file():
    f = filedialog.asksaveasfilename(filetypes=[("JSON", "*.json")], defaultextension=".json")
    if f: json_entry.delete(0, tk.END); json_entry.insert(0, f)


def select_rath_json():
    f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
    if f: rath_json_entry.delete(0, tk.END); rath_json_entry.insert(0, f)


# --- Logic Functions ---

def process_single_dam_rathcelon(dam_dict):
    """Worker for multiprocessing."""
    dam_name = dam_dict.get('name', "Unknown Dam")
    try:
        # Re-import to ensure visibility in worker process
        from rathcelon.classes import Dam as RD
        dam_i = RD(**dam_dict)
        dam_i.process_dam()
        return True, dam_name, None
    except Exception as e:
        return False, dam_name, str(e)


def threaded_prepare_data():
    try:
        # 1. Get Values
        lhd_csv = database_entry.get()
        flowline_source = flowline_var.get()
        dem_resolution = dd_var.get()
        streamflow_source = streamflow_var.get()
        dem_folder = dem_entry.get()
        strm_folder = strm_entry.get()
        results_folder = results_entry.get()
        baseflow_method = baseflow_var.get()

        if not os.path.exists(lhd_csv):
            messagebox.showerror("Error", f"Database file not found:\n{lhd_csv}")
            return

        utils.set_status("Inputs validated. Starting data prep...")

        # 2. Create directories
        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        lhd_df = pd.read_csv(lhd_csv)
        final_df = lhd_df.copy()

        # Fix object types
        try:
            sample_dam_dict = PrepDam(**lhd_df.iloc[0].to_dict()).__dict__
            cols_to_update = [key for key in sample_dam_dict.keys() if key in final_df.columns]
            for col in cols_to_update:
                if final_df[col].dtype != 'object':
                    final_df[col] = final_df[col].astype(object)
        except Exception:
            pass

        # 3. Handle Data Downloads (NWM/GEOGLOWS)
        hydroshare_id = "88759266f9c74df8b5bb5f52d142ba8e"

        ui_dir = os.path.dirname(os.path.realpath(__file__))
        package_root = os.path.dirname(ui_dir)
        data_dir = os.path.join(package_root, 'data')
        os.makedirs(data_dir, exist_ok=True)

        nwm_parquet = os.path.join(data_dir, 'nwm_daily_retrospective.parquet')
        vpu_filename = "vpu-boundaries.gpkg"
        tdx_vpu_map = os.path.join(data_dir, vpu_filename)
        nwm_ds = None

        if streamflow_source == 'National Water Model':
            if not os.path.exists(nwm_parquet):
                utils.set_status("Downloading NWM Parquet...")
                try:
                    hs = HydroShare()
                    resource = hs.resource(hydroshare_id)
                    resource.file_download(path='nwm_daily_retrospective.parquet', save_path=data_dir)
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

        # 4. Processing Loop
        total_dams = len(lhd_df)
        processed_count = 0

        for i, row in lhd_df.iterrows():
            dam_id = row.get("ID", f"Row_{i}")
            try:
                utils.set_status(f"Prep: Dam {dam_id} ({i + 1}/{total_dams})...")

                dam = PrepDam(**row.to_dict())
                dam.set_streamflow_source(streamflow_source)
                dam.set_flowline_source(flowline_source)

                dam.assign_flowlines(strm_folder, tdx_vpu_map)
                dam.assign_dem(dem_folder, dem_resolution)

                if not any([dam.dem_1m, dam.dem_3m, dam.dem_10m]):
                    print(f"Skipping Dam {dam_id}: No DEM.")
                    continue

                dam.set_output_dir(results_folder)

                # Hydrology Checks
                needs_reach = False
                if streamflow_source == 'National Water Model':
                    if pd.isna(row.get('dem_baseflow_NWM')) or pd.isna(row.get('fatality_flows_NWM')):
                        needs_reach = True
                elif streamflow_source == 'GEOGLOWS':
                    if pd.isna(row.get('dem_baseflow_GEOGLOWS')) or pd.isna(row.get('fatality_flows_GEOGLOWS')):
                        needs_reach = True

                if needs_reach:
                    if streamflow_source == 'National Water Model' and nwm_ds is None:
                        pass
                    else:
                        dam.create_reach(nwm_ds, dam.flowline_TDX)
                        dam.set_dem_baseflow(baseflow_method)
                        dam.set_fatal_flows()

                # Update DataFrame
                for key, value in dam.__dict__.items():
                    if isinstance(value, (list, np.ndarray)):
                        final_df.loc[i, key] = str(value)
                    else:
                        final_df.loc[i, key] = value

                processed_count += 1

            except Exception as e:
                print(f"Error prepping Dam {dam_id}: {e}")

        # 5. Save Output
        if processed_count > 0:
            final_df.to_csv(lhd_csv, index=False)
            json_loc = json_entry.get()
            # UPDATED: Use the imported function directly
            rathcelon_input(lhd_csv, json_loc, baseflow_method, nwm_parquet)
            utils.set_status(f"Prep complete. {processed_count} dams processed.")
            messagebox.showinfo("Success", f"Prep complete. Saved to {json_loc}")
        else:
            utils.set_status("No dams processed successfully.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
        print(e)
    finally:
        prep_run_button.config(state=tk.NORMAL)


def threaded_run_rathcelon():
    try:
        json_loc = rath_json_entry.get()
        if not os.path.exists(json_loc):
            messagebox.showerror("Error", "JSON file not found.")
            return

        with open(json_loc, 'r') as f:
            data = json.load(f)

        dams = data.get("dams", [])
        total = len(dams)
        utils.set_status(f"Starting parallel RathCelon for {total} dams...")

        success_count = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_dam = {executor.submit(process_single_dam_rathcelon, d): d for d in dams}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_dam)):
                success, name, err = future.result()
                if success:
                    success_count += 1
                    utils.set_status(f"Finished {name} ({i + 1}/{total})")
                else:
                    utils.set_status(f"Error on {name}: {err}")

        utils.set_status(f"Completed. {success_count} dams processed.")
        messagebox.showinfo("Success", f"Processed {success_count} dams.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
        messagebox.showerror("Error", str(e))
    finally:
        rath_run_button.config(state=tk.NORMAL)


# Thread Starters
def start_prep_thread():
    prep_run_button.config(state=tk.DISABLED)
    threading.Thread(target=threaded_prepare_data, daemon=True).start()


def start_rath_thread():
    rath_run_button.config(state=tk.DISABLED)
    threading.Thread(target=threaded_run_rathcelon, daemon=True).start()
