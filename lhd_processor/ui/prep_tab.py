import gc
import os
import json
import threading
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

# CLEAN IMPORT: Importing directly from the package
from . import utils
from ..data_manager import DatabaseManager
from ..prep import LowHeadDam as PrepDam, create_reanalysis_file, condense_zarr

# Import Rathcelon carefully
try:
    from ..rathcelon.classes import RathCelonDam
except Exception as e:
    print(f"Warning: Could not import RathCelon. Error: {e}")
    RathCelonDam = None

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

    dd_var = add_combo(0, "DEM Resolution (m):", ("1", "10"), "1")
    flowline_var = add_combo(1, "Flowline Source:", ("NHDPlus", "TDX-Hydro"), "NHDPlus")
    streamflow_var = add_combo(2, "Streamflow Source:", ("National Water Model", "GEOGLOWS"), "National Water Model")
    baseflow_var = add_combo(3, "Baseflow Estimation:",
                             ("WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation"),
                             "WSE and LiDAR Date")
    baseflow_combo = hydro_frame.winfo_children()[-1]

    utils.ToolTip(baseflow_combo, "Choose how to estimate baseflow:\n"
                                  "- LiDAR Date: Uses flow on the specific day the LiDAR was flown.\n"
                                  "- Median: Uses long-term median daily flow.")

    # MODIFIED: Removed style="Accent.TButton" to ensure consistent font size
    prep_run_button = ttk.Button(prep_frame, text="1. Prepare Data & Create Input File", command=start_prep_thread)
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

    # MODIFIED: Removed style="Accent.TButton" to ensure consistent font size
    rath_run_button = ttk.Button(btn_frame, text="2. Run RathCelon", command=start_rath_thread)
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
        dam_i = RathCelonDam(**dam_dict)
        dam_i.process_dam()

        del dam_i
        gc.collect()

        return True, dam_name, None
    except Exception as e:
        gc.collect()
        return False, dam_name, str(e)


# --- Worker Functions for ThreadPoolExecutor ---

def worker_assign_flowlines(sid, db, flowline_source, streamflow_source, strm_folder, tdx_vpu_map):
    dam = PrepDam(sid, db)
    dam.set_flowline_source(flowline_source)
    dam.set_streamflow_source(streamflow_source)
    ids = dam.assign_flowlines(strm_folder, tdx_vpu_map)
    dam.save_changes()
    return ids

def worker_assign_dem(sid, db, dem_folder, dem_resolution):
    dam = PrepDam(sid, db)
    dam.assign_dem(dem_folder, dem_resolution)
    dam.save_changes()

def worker_assign_hydraulics(sid, db, results_folder, land_folder, baseflow_method, streamflow_source, reanalysis_df=None):
    dam = PrepDam(sid, db)
    dam.set_output_dir(results_folder)
    dam.assign_land(land_folder)
    
    # Pass reanalysis data if available
    if reanalysis_df is not None:
        # Filter for this dam's COMID
        if dam.nwm_id:
             dam_stats = reanalysis_df[reanalysis_df['comid'] == int(dam.nwm_id)]
             if not dam_stats.empty:
                 # You can now use these stats inside est_dem_baseflow or est_fatal_flows if you modify them
                 # For now, we just pass them or use them to set known values
                 pass

    dam.est_dem_baseflow(baseflow_method)
    
    lidar_date = None
    if dam.nwm_id:
         lidar_date = dam.site_data.get('lidar_date')
    
    dam.est_fatal_flows(streamflow_source)
    dam.save_changes()
    
    return (int(dam.nwm_id), lidar_date) if dam.nwm_id and lidar_date else None


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

        # Ensure all directories exist
        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        if land_folder:
            os.makedirs(land_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        db = DatabaseManager(xlsx_path)
        site_ids = db.sites['site_id'].tolist()
        total_dams = len(site_ids)

        # Path to the VPU map required for TDX/GEOGLOWS
        ui_dir = os.path.dirname(os.path.realpath(__file__))
        package_root = os.path.dirname(ui_dir)
        tdx_vpu_map = os.path.join(package_root, 'data', 'vpu-boundaries.gpkg')

        # --- STAGE 1: BATCH FLOWLINES (Parallel) ---
        utils.set_status(f"Stage 1/4: Assigning flowlines for {total_dams} sites (Parallel)...")
        all_comids = set()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(worker_assign_flowlines, sid, db, flowline_source, streamflow_source, strm_folder, tdx_vpu_map): sid for sid in site_ids}
            
            completed_count = 0
            for future in as_completed(futures):
                completed_count += 1
                utils.set_status(f"Flowlines: Processed {completed_count}/{total_dams}")
                try:
                    ids = future.result()
                    if ids:
                        all_comids.update(ids)
                except Exception as e:
                    print(f"Error in flowline worker: {e}")

        # --- STAGE 2: BATCH STREAMFLOW (Cloud Extraction) ---
        utils.set_status("Stage 2/4: Running Condense Zarr for all identified IDs...")
        
        # Use the internal function instead of external script
        nwm_zarr = os.path.join(package_root, 'data', 'nwm_v3_daily_retrospective.zarr')
        if all_comids:
            condense_zarr(list(all_comids), nwm_zarr)
        else:
            utils.set_status("No COMIDs found to extract streamflow for.")

        # --- STAGE 3: REANALYSIS (Moved Up) ---
        # Calculate stats NOW so they are available for Stage 4
        reanalysis_df = None
        if all_comids:
            utils.set_status(f"Generating reanalysis for {len(all_comids)} reaches...")
            # We pass None for comid_date_map initially because we haven't found lidar dates yet
            # But we can re-run or update it later if needed. 
            # Actually, create_reanalysis_file returns nothing, it saves to CSV.
            # Let's modify it to return the DF or read it back.
            create_reanalysis_file(list(all_comids), results_folder, streamflow_source, package_root, comid_date_map=None, db_manager=db)
            
            # Determine filename based on source
            filename = "nwm_reanalysis.csv" if streamflow_source == 'National Water Model' else "geoglows_reanalysis.csv"
            reanalysis_path = os.path.join(results_folder, filename)
            if os.path.exists(reanalysis_path):
                reanalysis_df = pd.read_csv(reanalysis_path)

        # --- STAGE 4: BATCH DEMs (Parallel) ---
        utils.set_status(f"Stage 3/4: Downloading DEMs for {total_dams} sites (Parallel)...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker_assign_dem, sid, db, dem_folder, dem_resolution) for sid in site_ids]
            
            completed_count = 0
            for future in as_completed(futures):
                completed_count += 1
                utils.set_status(f"DEMs: Processed {completed_count}/{total_dams}")
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in DEM worker: {e}")

        # --- STAGE 5: BATCH LAND USE & HYDRAULICS (Parallel) ---
        utils.set_status(f"Stage 4/4: Finalizing Land Use and Baseflows (Parallel)...")
        comid_date_map = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Pass reanalysis_df to the worker
            futures = [executor.submit(worker_assign_hydraulics, sid, db, results_folder, land_folder, baseflow_method, streamflow_source, reanalysis_df) for sid in site_ids]
            
            completed_count = 0
            for future in as_completed(futures):
                completed_count += 1
                utils.set_status(f"Hydraulics: Processed {completed_count}/{total_dams}")
                try:
                    res = future.result()
                    if res:
                        comid_date_map[res[0]] = res[1]
                except Exception as e:
                    print(f"Error in Hydraulics worker: {e}")

        # --- FINAL REANALYSIS UPDATE (Optional) ---
        # If we found new LiDAR dates, we might want to update the "known_baseflow" column in the reanalysis file
        if all_comids and comid_date_map:
             utils.set_status("Updating reanalysis with found LiDAR dates...")
             create_reanalysis_file(list(all_comids), results_folder, streamflow_source, package_root, comid_date_map, db_manager=db)

        # --- FINALIZATION ---
        utils.set_status("Saving database and creating RathCelon input file...")
        db.save()

        json_loc = json_entry.get()
        # nwm_zarr is already defined above

        # CHANGED: Using db.to_json instead of rathcelon_input
        db.to_json(
            json_loc,
            baseflow_method,
            nwm_zarr,
            flowline_source,
            streamflow_source
        )

        utils.set_status(f"Prep complete. {total_dams} dams processed across 4 stages.")
        messagebox.showinfo("Success", f"Batch Prep complete.\nDatabase and JSON file updated.")

    except Exception as e:
        utils.set_status(f"Error during batch prep: {e}")
        print(f"Detailed Error: {e}")
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
        worker_count = max(1, int(total_cores/2))

        utils.set_status(
            f"Skipping {skipped_count}. Initializing Dask (Workers: {worker_count}) for {count_to_run} dams...")

        with LocalCluster(processes=True, threads_per_worker=1, n_workers=worker_count) as cluster:
            with Client(cluster) as client:

                print(f"Dask Dashboard: {client.dashboard_link}")
                utils.set_status(f"Processing {count_to_run} dams... (Dashboard: {client.dashboard_link})")

                futures = client.map(process_single_dam_rathcelon, dams_to_process)

                success_count = 0
                processed_so_far = 0

                for future in as_completed(dask_as_completed(futures)):
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