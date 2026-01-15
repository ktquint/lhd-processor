import gc
import os
import threading
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

# CLEAN IMPORT: Importing directly from the package
from . import utils
from ..data_manager import DatabaseManager
from ..prep import LowHeadDam as PrepDam, condense_zarr, create_reanalysis_file

# Import Rathcelon carefully
try:
    from ..rathcelon.classes import RathCelonDam
except Exception as e:
    print(f"Warning: Could not import RathCelon. Error: {e}")
    RathCelonDam = None

# Module-level widgets
database_entry = None
dem_entry = None
strm_entry = None
land_use_entry = None
results_entry = None
flowline_var = None
dd_var = None
streamflow_var = None
prep_run_button = None
rath_xlsx_entry = None
rath_run_button = None
rath_stop_button = None
rath_results_entry = None
rath_flowline_var = None
rath_streamflow_var = None
rath_baseflow_var = None

# Global Threading Event for Stopping
stop_event = threading.Event()


def setup_prep_tab(parent_tab):
    """Constructs the UI for the Prep tab, using Excel as the sole input."""
    global database_entry, dem_entry, strm_entry, land_use_entry, results_entry
    global flowline_var, dd_var, streamflow_var
    global prep_run_button, rath_xlsx_entry, rath_run_button, rath_stop_button
    global rath_results_entry, rath_flowline_var, rath_streamflow_var, rath_baseflow_var

    # --- Step 1 Frame: Prepare Data ---
    prep_frame = ttk.LabelFrame(parent_tab, text="Step 1: Download and Prepare Data")
    prep_frame.pack(pady=10, padx=10, fill="x")

    # Unified frame for interleaved inputs
    input_frame = ttk.Frame(prep_frame)
    input_frame.pack(pady=5, padx=10, fill="x")
    input_frame.columnconfigure(1, weight=1)

    def add_path_row(row, label_text, cmd, is_file=False, must_exist=True):
        ttk.Label(input_frame, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        entry = ttk.Entry(input_frame)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Bind focus out event to check existence and show popup if needed
        if must_exist:
            def check_exists(event):
                path = entry.get()
                if path:
                    exists = os.path.isfile(path) if is_file else os.path.isdir(path)
                    if not exists:
                        messagebox.showwarning("Path Not Found", f"The specified path does not exist:\n{path}")
            entry.bind("<FocusOut>", check_exists)

        ttk.Button(input_frame, text="Select...", command=cmd).grid(row=row, column=2, padx=5, pady=5)
        return entry

    def add_combo(row, label, values, default):
        ttk.Label(input_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(input_frame, textvariable=var, state="readonly", values=values)
        cb.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
        return var

    # Row counter
    r = 0

    # Database
    database_entry = add_path_row(r, "Database (.xlsx):", select_database, is_file=True, must_exist=True); r+=1

    # DEM
    dem_entry = add_path_row(r, "DEM Folder:", select_dem_dir, is_file=False, must_exist=False); r+=1
    dd_var = add_combo(r, "   ↳ DEM Resolution (m):", ("1", "10"), "1"); r+=1

    # Hydrography
    strm_entry = add_path_row(r, "Hydrography Folder:", select_strm_dir, is_file=False, must_exist=False); r+=1
    flowline_var = add_combo(r, "   ↳ Flowline Source:", ("NHDPlus", "TDX-Hydro", "Both"), "NHDPlus"); r+=1
    streamflow_var = add_combo(r, "   ↳ Streamflow Source:", ("National Water Model", "GEOGLOWS", "Both"), "National Water Model"); r+=1

    # Land Use
    land_use_entry = add_path_row(r, "Land Use Folder:", select_land_use_dir, is_file=False, must_exist=False); r+=1

    # Results (Removed from Step 1 as requested, but variable kept for compatibility if needed, though unused in UI)
    # results_entry = add_path_row(r, "Results Folder:", select_results_dir, is_file=False, must_exist=False); r+=1

    prep_run_button = ttk.Button(prep_frame, text="1. Prepare Data & Update Database", command=start_prep_thread)
    prep_run_button.pack(pady=10, padx=10, fill="x")

    # --- Step 2 Frame: Run RathCelon ---
    rath_frame = ttk.LabelFrame(parent_tab, text="Step 2: Automated Rating Curves")
    rath_frame.pack(pady=10, padx=10, fill="x")
    rath_frame.columnconfigure(1, weight=1)

    # Row 0: Excel Database
    ttk.Label(rath_frame, text="Excel Database:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    rath_xlsx_entry = ttk.Entry(rath_frame)
    rath_xlsx_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    
    # Bind focus out event for RathCelon Excel entry
    def check_rath_xlsx_exists(event):
        path = rath_xlsx_entry.get()
        if path and not os.path.isfile(path):
            messagebox.showwarning("Path Not Found", f"The specified Excel file does not exist:\n{path}")
    rath_xlsx_entry.bind("<FocusOut>", check_rath_xlsx_exists)

    ttk.Button(rath_frame, text="Select...", command=select_rath_xlsx).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Results Folder (Moved here)
    ttk.Label(rath_frame, text="Results Folder:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    rath_results_entry = ttk.Entry(rath_frame)
    rath_results_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    # utils.bind_path_validation(rath_results_entry, is_file=False, must_exist=False)
    ttk.Button(rath_frame, text="Select...", command=select_rath_results_dir).grid(row=1, column=2, padx=5, pady=5)

    # Row 2: Flowline Source & Streamflow Source
    opts_frame = ttk.Frame(rath_frame)
    opts_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=5)
    opts_frame.columnconfigure(1, weight=1)
    opts_frame.columnconfigure(3, weight=1)

    ttk.Label(opts_frame, text="Flowline Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    rath_flowline_var = tk.StringVar(value="NHDPlus")
    ttk.Combobox(opts_frame, textvariable=rath_flowline_var, state="readonly", values=("NHDPlus", "TDX-Hydro")).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Label(opts_frame, text="Streamflow Source:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    rath_streamflow_var = tk.StringVar(value="National Water Model")
    ttk.Combobox(opts_frame, textvariable=rath_streamflow_var, state="readonly", values=("National Water Model", "GEOGLOWS")).grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

    # Row 3: Baseflow Estimation
    bf_frame = ttk.Frame(rath_frame)
    bf_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
    bf_frame.columnconfigure(1, weight=1)

    ttk.Label(bf_frame, text="Baseflow Estimation:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    rath_baseflow_var = tk.StringVar(value="WSE and LiDAR Date")
    ttk.Combobox(bf_frame, textvariable=rath_baseflow_var, state="readonly", values=("WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation")).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    # Row 4: Buttons
    btn_frame = ttk.Frame(rath_frame)
    btn_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10)
    btn_frame.columnconfigure(0, weight=1)
    btn_frame.columnconfigure(1, weight=1)

    rath_run_button = ttk.Button(btn_frame, text="2. Run ARC", command=start_rath_thread)
    rath_run_button.grid(row=0, column=0, padx=5, sticky=tk.EW)

    rath_stop_button = ttk.Button(btn_frame, text="STOP Processing", command=stop_rath_thread, state=tk.DISABLED)
    rath_stop_button.grid(row=0, column=1, padx=5, sticky=tk.EW)


# --- Event Handlers (File Dialogs) ---

def select_database():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f:
        database_entry.delete(0, tk.END)
        database_entry.insert(0, f)
        rath_xlsx_entry.delete(0, tk.END)
        rath_xlsx_entry.insert(0, f)
        
        # Auto-fill based on parent directory
        try:
            project_path = os.path.dirname(f)
            
            # Auto-fill suggested folders
            for entry, folder in [(dem_entry, "DEM"), (strm_entry, "STRM"), (land_use_entry, "LAND")]:
                entry.delete(0, tk.END)
                entry.insert(0, os.path.join(project_path, folder))
            
            # Also fill rath results
            rath_results_entry.delete(0, tk.END)
            rath_results_entry.insert(0, os.path.join(project_path, "Results"))
            
            utils.set_status("Project paths loaded from database location.")
        except Exception as e:
            print(f"Error auto-filling paths: {e}")


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
    if d: 
        # results_entry.delete(0, tk.END)
        # results_entry.insert(0, d)
        rath_results_entry.delete(0, tk.END)
        rath_results_entry.insert(0, d)

def select_rath_results_dir():
    d = filedialog.askdirectory()
    if d:
        rath_results_entry.delete(0, tk.END)
        rath_results_entry.insert(0, d)

def select_rath_xlsx():
    f = filedialog.askopenfilename(filetypes=[("Excel Database", "*.xlsx")])
    if f:
        rath_xlsx_entry.delete(0, tk.END)
        rath_xlsx_entry.insert(0, f)


# --- Logic and Worker Functions ---

def process_single_dam_rathcelon(dam_dict):
    """Dask worker processing a single dam dictionary from the Excel row."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['GDAL_CACHEMAX'] = '256'

    dam_name = dam_dict.get('name', "Unknown Dam")
    try:
        dam_i = RathCelonDam(dam_row=dam_dict)
        dam_i.process_dam()
        del dam_i
        gc.collect()
        return True, dam_name, None
    except Exception as e:
        gc.collect()
        return False, dam_name, str(e)


def threaded_prepare_data():
    """Execution loop for Step 1 preparation."""
    try:
        xlsx_path = database_entry.get()
        flowline_source = flowline_var.get()
        dem_resolution = dd_var.get()
        streamflow_source = streamflow_var.get()
        dem_folder = dem_entry.get()
        strm_folder = strm_entry.get()
        land_folder = land_use_entry.get() if land_use_entry else None
        results_folder = rath_results_entry.get() # Use the one from Step 2 or default

        if not os.path.exists(xlsx_path):
            messagebox.showerror("Error", "Database file not found.")
            return

        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        if land_folder: os.makedirs(land_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        db = DatabaseManager(xlsx_path)
        site_ids = db.sites['site_id'].tolist()
        # total_dams = len(site_ids)

        ui_dir = os.path.dirname(os.path.realpath(__file__))
        package_root = os.path.dirname(ui_dir)
        tdx_vpu_map = os.path.join(package_root, 'data', 'vpu-boundaries.gpkg')

        # Stage 1: Flowlines
        utils.set_status("Stage 1/4: Assigning flowlines...")
        all_comids = set()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(worker_assign_flowlines, sid, db, flowline_source, streamflow_source, strm_folder,
                                tdx_vpu_map): sid for sid in site_ids}
            for future in as_completed(futures):
                ids = future.result()
                if ids: all_comids.update(ids)

        # Stage 2: Streamflow (Zarr Condensation & Reanalysis)
        if all_comids:
            utils.set_status("Stage 2/4: Processing streamflow statistics...")
            
            if streamflow_source == 'National Water Model':
                nwm_zarr = os.path.join(package_root, 'data', 'nwm_v3_daily_retrospective.zarr')
                condense_zarr(list(all_comids), nwm_zarr)
            
            # Generate the reanalysis CSV required by RathCelon
            create_reanalysis_file(list(all_comids), streamflow_source, package_root, db_manager=db)

        # Stage 3: DEMs
        utils.set_status("Stage 3/4: Fetching DEMs...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            for sid in site_ids: executor.submit(worker_assign_dem, sid, db, dem_folder, dem_resolution)

        # Stage 4: Hydraulics & Land Use
        utils.set_status("Stage 4/4: Finalizing hydraulics...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            for sid in site_ids: executor.submit(worker_assign_hydraulics, sid, db, results_folder, land_folder,
                                                 "WSE and LiDAR Date", streamflow_source, flowline_source)

        # Save and point to Excel for Step 2
        utils.set_status("Saving database...")
        db.save()
        rath_xlsx_entry.delete(0, tk.END)
        rath_xlsx_entry.insert(0, xlsx_path)
        
        # Also update results folder in Step 2
        rath_results_entry.delete(0, tk.END)
        rath_results_entry.insert(0, results_folder)
        
        # Update dropdowns in Step 2
        rath_flowline_var.set(flowline_source)
        rath_streamflow_var.set(streamflow_source)

        utils.set_status("Prep complete. Ready for Step 2.")
        messagebox.showinfo("Success", "Excel database updated and set as input for RathCelon.")

    except Exception as e:
        utils.set_status(f"Error during preparation: {e}")
    finally:
        prep_run_button.config(state=tk.NORMAL)


def threaded_run_rathcelon():
    """Runs Step 2 RathCelon processing using Excel as the input source."""
    try:
        stop_event.clear()
        rath_run_button.config(state=tk.DISABLED)
        rath_stop_button.config(state=tk.NORMAL)

        excel_loc = rath_xlsx_entry.get()
        results_loc = rath_results_entry.get()
        
        if not os.path.exists(excel_loc):
            messagebox.showerror("Error", "Excel database not found.")
            return

        # Read Excel and run through Dask
        df = pd.read_excel(excel_loc)
        
        # Override output_dir in the dataframe if specified in UI
        if results_loc:
            df['output_dir'] = results_loc
            
        # Override sources if needed (though they should be in the excel from step 1)
        # But user might change them in Step 2 UI
        fl_source = rath_flowline_var.get()
        sf_source = rath_streamflow_var.get()
        bf_method = rath_baseflow_var.get()
        
        if fl_source:
            df['flowline_source'] = fl_source
        if sf_source:
            df['streamflow_source'] = sf_source
        if bf_method:
            df['baseflow_method'] = bf_method

        dams = df.to_dict(orient='records')
        
        # We process all dams, relying on RathCelonDam.process_dam to skip heavy ARC simulation
        # if Bathy.tif exists, but ALWAYS re-extract XS lines as requested.
        dams_to_process = dams
        count_to_run = len(dams_to_process)

        total_cores = os.cpu_count() or 1
        # worker_count = max(1, int(total_cores / 3))
        worker_count = 3

        utils.set_status(f"Initializing Dask Workers ({worker_count}) for {count_to_run} dams...")

        with LocalCluster(processes=True, threads_per_worker=1, n_workers=worker_count) as cluster:
            with Client(cluster) as client:
                utils.set_status(f"Processing... (Dashboard: {client.dashboard_link})")
                futures = client.map(process_single_dam_rathcelon, dams_to_process)

                processed_so_far = 0
                for future in as_completed(dask_as_completed(futures)):
                    if stop_event.is_set():
                        client.cancel(futures)
                        break

                    try:
                        success, name, err = future.result()
                        processed_so_far += 1
                        if success:
                            utils.set_status(f"Finished {name} ({processed_so_far}/{count_to_run})")
                        else:
                            utils.set_status(f"Failed {name} ({processed_so_far}/{count_to_run}): {err}")
                    except Exception as e:
                        # Handle worker crashes (e.g. KilledWorker) gracefully
                        processed_so_far += 1
                        utils.set_status(f"Worker crashed on a task ({processed_so_far}/{count_to_run}): {e}")

        if stop_event.is_set():
            messagebox.showwarning("Stopped", "Processing stopped by user.")
        else:
            utils.set_status("Batch complete.")
            messagebox.showinfo("Success", f"Finished processing {count_to_run} dams.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
    finally:
        rath_run_button.config(state=tk.NORMAL)
        rath_stop_button.config(state=tk.DISABLED)


# --- UI Thread Starters ---

def stop_rath_thread():
    if messagebox.askyesno("Stop", "Stop processing after current tasks finish?"):
        stop_event.set()


def start_prep_thread():
    prep_run_button.config(state=tk.DISABLED)
    threading.Thread(target=threaded_prepare_data, daemon=True).start()


def start_rath_thread():
    threading.Thread(target=threaded_run_rathcelon, daemon=True).start()


# --- Parallel Worker Helpers (Identical to original) ---

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


def worker_assign_hydraulics(sid, db, results_folder, land_folder, baseflow_method, streamflow_source, flowline_source):
    dam = PrepDam(sid, db)
    dam.set_output_dir(results_folder)
    dam.set_flowline_source(flowline_source)
    dam.set_streamflow_source(streamflow_source)
    dam.assign_land(land_folder)
    dam.est_dem_baseflow(baseflow_method)
    dam.est_fatal_flows(streamflow_source)
    dam.save_changes()
