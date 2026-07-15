import os
import threading
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import utils
from ..data_manager import DatabaseManager
from ..prep import LowHeadDam as PrepDam, create_reanalysis_file, prune_raw_dem_tiles

# Module-level widgets
database_entry = None
dem_entry = None
strm_entry = None
land_use_entry = None
flowline_var = None
dd_var = None
streamflow_var = None
delete_raw_tiles_var = None
prep_run_button = None

def setup_download_tab(parent_tab):
    global database_entry, dem_entry, strm_entry, land_use_entry
    global flowline_var, dd_var, streamflow_var, delete_raw_tiles_var, prep_run_button

    # Frame
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
        
        if must_exist:
            def check_exists(_):
                path = entry.get()
                if path:
                    exists = os.path.isfile(path) if is_file else os.path.isdir(path)
                    if not exists:
                        messagebox.showwarning("Path Not Found", f"The specified path does not exist:\n{path}")
            entry.bind("<FocusOut>", check_exists)

        ttk.Button(input_frame, text="Select...", command=cmd).grid(row=row, column=2, padx=5, pady=5)
        return entry

    def add_combo(row, label, values, default, state="readonly"):
        ttk.Label(input_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(input_frame, textvariable=var, state=state, values=values)
        cb.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
        return var, cb

    r = 0
    database_entry = add_path_row(r, "Database (.xlsx):", select_database, is_file=True, must_exist=True); r+=1
    dem_entry = add_path_row(r, "DEM Folder:", select_dem_dir, is_file=False, must_exist=False); r+=1
    dd_var, _ = add_combo(r, "   ↳ DEM Resolution (m):", ("1", "10"), "1"); r+=1

    delete_raw_tiles_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(input_frame, text="   ↳ Delete raw 3DEP tiles after merging (frees disk space)",
                    variable=delete_raw_tiles_var).grid(row=r, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
    r += 1

    strm_entry = add_path_row(r, "Hydrography Folder:", select_strm_dir, is_file=False, must_exist=False); r+=1
    flowline_var, _ = add_combo(r, "   ↳ Flowline Source:", ("NHDPlus", "TDX-Hydro", "Both"), "NHDPlus"); r+=1
    streamflow_var, _ = add_combo(
        r, "   ↳ Streamflow Source (locked to Flowline Source):",
        ("National Water Model", "GEOGLOWS", "Both"), "National Water Model", state="disabled"); r+=1
    land_use_entry = add_path_row(r, "Land Use Folder:", select_land_use_dir, is_file=False, must_exist=False); r+=1

    # NHDPlus reaches only have NWM reach IDs; TDX-Hydro reaches only have
    # GEOGLOWS linknos -- so the two aren't independent choices, and picking
    # them separately is how a dam ends up processed with mismatched
    # flowline/streamflow data (and DEM/land rasters built to the wrong
    # flowline's extent).
    FLOWLINE_TO_STREAMFLOW = {
        "NHDPlus": "National Water Model",
        "TDX-Hydro": "GEOGLOWS",
        "Both": "Both",
    }

    def _sync_streamflow_source(*args):
        streamflow_var.set(FLOWLINE_TO_STREAMFLOW.get(flowline_var.get(), streamflow_var.get()))

    flowline_var.trace_add("write", _sync_streamflow_source)
    _sync_streamflow_source()

    prep_run_button = ttk.Button(prep_frame, text="1. Prepare Data & Update Database", command=start_prep_thread)
    prep_run_button.pack(pady=10, padx=10, fill="x")

def select_database():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f:
        database_entry.delete(0, tk.END)
        database_entry.insert(0, f)
        try:
            project_path = os.path.dirname(f)
            for entry, folder in [(dem_entry, "DEM"), (strm_entry, "STRM"), (land_use_entry, "LAND")]:
                entry.delete(0, tk.END)
                entry.insert(0, os.path.join(project_path, folder))
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

def threaded_prepare_data():
    try:
        xlsx_path = database_entry.get()
        flowline_source = flowline_var.get()
        dem_resolution = dd_var.get()
        streamflow_source = streamflow_var.get()
        dem_folder = dem_entry.get()
        strm_folder = strm_entry.get()
        land_folder = land_use_entry.get() if land_use_entry else None
        delete_raw_tiles = delete_raw_tiles_var.get() if delete_raw_tiles_var else True
        
        # Infer results folder for Stage 4
        results_folder = os.path.join(os.path.dirname(xlsx_path), "Results")

        if not os.path.exists(xlsx_path):
            messagebox.showerror("Error", "Database file not found.")
            return

        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        if land_folder: os.makedirs(land_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        db = DatabaseManager(xlsx_path)
        site_ids = db.sites['site_id'].tolist()

        ui_dir = os.path.dirname(os.path.realpath(__file__))
        package_root = os.path.dirname(ui_dir)
        hydroshare_resource_id = '71277c621bb54e1d85094f779a1f9bd4'
        tdx_vpu_map = os.path.join(package_root, 'data', 'vpu-boundaries.gpkg')

        if not os.path.exists(tdx_vpu_map):
            utils.set_status("Downloading VPU boundaries from HydroShare...")
            try:
                os.makedirs(os.path.dirname(tdx_vpu_map), exist_ok=True)
                url = f"https://www.hydroshare.org/resource/{hydroshare_resource_id}/data/contents/vpu-boundaries.gpkg"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(tdx_vpu_map, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                utils.set_status("VPU boundaries downloaded.")
            except Exception as e:
                utils.set_status(f"Error downloading VPU map: {e}")
                print(f"Failed to download VPU boundaries: {e}")

        utils.set_status("Stage 1/4: Assigning flowlines...")
        all_comids = set()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(worker_assign_flowlines, sid, db, flowline_source, streamflow_source, strm_folder,
                                tdx_vpu_map): sid for sid in site_ids}
            for future in as_completed(futures): 
                try:
                    ids = future.result()
                    if ids: all_comids.update(ids)
                except Exception as e:
                    print(f"Error in Flowline worker: {e}")

        if all_comids:
            utils.set_status("Stage 2/4: Processing streamflow statistics...")
            create_reanalysis_file(list(all_comids), streamflow_source, package_root)

        utils.set_status("Stage 3/4: Fetching DEMs...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker_assign_dem, sid, db, dem_folder, dem_resolution, flowline_source) for sid in site_ids]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in DEM worker: {e}")

        if delete_raw_tiles:
            count, freed_bytes = prune_raw_dem_tiles(dem_folder)
            if count:
                utils.set_status(f"Deleted {count} raw 3DEP tile(s) ({freed_bytes / 1e6:.1f} MB freed).")

        utils.set_status("Stage 4/4: Finalizing hydraulics...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_assign_hydraulics, sid, db, results_folder, land_folder,
                                                 "WSE and LiDAR Date", streamflow_source, flowline_source) for sid in site_ids]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in Hydraulics worker: {e}")

        # Stage 4 is what actually resolves each dam's LiDAR acquisition date
        # (site_data['lidar_date']) -- too late to have known it back in Stage 2, so
        # re-run the reanalysis file now with a real comid->date map. This is what
        # makes known_baseflow reflect the dam's actual LiDAR-date flow instead of
        # just the long-term median.
        id_col = 'reach_id' if streamflow_source == 'National Water Model' else 'linkno'
        if all_comids and id_col in db.sites.columns and 'lidar_date' in db.sites.columns:
            dated = db.sites.dropna(subset=[id_col, 'lidar_date'])
            comid_date_map = {
                int(row[id_col]): str(row['lidar_date']) for _, row in dated.iterrows()
                if str(row['lidar_date']).strip() not in ('', 'None', 'nan')
            }
            if comid_date_map:
                utils.set_status(f"Refining baseflow with LiDAR-date flows for {len(comid_date_map)} dam(s)...")
                create_reanalysis_file(list(all_comids), streamflow_source, package_root, comid_date_map=comid_date_map)

        utils.set_status("Saving database...")
        db.save()
        utils.set_status("Prep complete. Ready for Step 2.")
        messagebox.showinfo("Success", "Excel database updated.")

    except Exception as e:
        utils.set_status(f"Error during preparation: {e}")
    finally:
        prep_run_button.config(state=tk.NORMAL)

def start_prep_thread():
    prep_run_button.config(state=tk.DISABLED)
    threading.Thread(target=threaded_prepare_data, daemon=True).start()

def worker_assign_flowlines(sid, db, flowline_source, streamflow_source, strm_folder, tdx_vpu_map):
    dam = PrepDam(sid, db)
    dam.set_flowline_source(flowline_source)
    dam.set_streamflow_source(streamflow_source)
    ids = dam.assign_flowlines(strm_folder, tdx_vpu_map)
    dam.save_changes()
    return ids

def worker_assign_dem(sid, db, dem_folder, dem_resolution, flowline_source):
    dam = PrepDam(sid, db)
    dam.set_flowline_source(flowline_source)
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
    dam.generate_stream_raster()
    dam.save_changes()
