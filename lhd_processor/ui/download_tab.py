import os
import threading
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import utils
from ..data_manager import DatabaseManager
from ..prep import LowHeadDam as PrepDam, condense_zarr, create_reanalysis_file

# Module-level widgets
database_entry = None
dem_entry = None
strm_entry = None
land_use_entry = None
flowline_var = None
dd_var = None
streamflow_var = None
prep_run_button = None

def setup_download_tab(parent_tab):
    global database_entry, dem_entry, strm_entry, land_use_entry
    global flowline_var, dd_var, streamflow_var, prep_run_button

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

    def add_combo(row, label, values, default):
        ttk.Label(input_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(input_frame, textvariable=var, state="readonly", values=values)
        cb.grid(row=row, column=1, padx=5, pady=5, sticky=tk.EW)
        return var

    r = 0
    database_entry = add_path_row(r, "Database (.xlsx):", select_database, is_file=True, must_exist=True); r+=1
    dem_entry = add_path_row(r, "DEM Folder:", select_dem_dir, is_file=False, must_exist=False); r+=1
    dd_var = add_combo(r, "   ↳ DEM Resolution (m):", ("1", "10"), "1"); r+=1
    strm_entry = add_path_row(r, "Hydrography Folder:", select_strm_dir, is_file=False, must_exist=False); r+=1
    flowline_var = add_combo(r, "   ↳ Flowline Source:", ("NHDPlus", "TDX-Hydro", "Both"), "NHDPlus"); r+=1
    streamflow_var = add_combo(r, "   ↳ Streamflow Source:", ("National Water Model", "GEOGLOWS", "Both"), "National Water Model"); r+=1
    land_use_entry = add_path_row(r, "Land Use Folder:", select_land_use_dir, is_file=False, must_exist=False); r+=1

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

def create_mannings_esa(manning_txt):
    with open(manning_txt, 'w') as f:
        f.write('LC_ID\tDescription\tManning_n\n')
        f.write('10\tTree Cover\t0.120\n')
        f.write('20\tShrubland\t0.050\n')
        f.write('30\tGrassland\t0.030\n')
        f.write('40\tCropland\t0.035\n')
        f.write('50\tBuiltup\t0.075\n')
        f.write('60\tBare\t0.030\n')
        f.write('70\tSnowIce\t0.030\n')
        f.write('80\tWater\t0.030\n')
        f.write('90\tHerbaceous Wetland\t0.100\n')
        f.write('95\tMangroves\t0.100\n')
        f.write('100\tMossLichen\t0.035\n')

def threaded_prepare_data():
    try:
        xlsx_path = database_entry.get()
        flowline_source = flowline_var.get()
        dem_resolution = dd_var.get()
        streamflow_source = streamflow_var.get()
        dem_folder = dem_entry.get()
        strm_folder = strm_entry.get()
        land_folder = land_use_entry.get() if land_use_entry else None
        
        # Infer results folder for Stage 4
        results_folder = os.path.join(os.path.dirname(xlsx_path), "Results")

        if not os.path.exists(xlsx_path):
            messagebox.showerror("Error", "Database file not found.")
            return

        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        if land_folder: os.makedirs(land_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        if land_folder:
            manning_txt = os.path.join(land_folder, 'Manning_n.txt')
            if not os.path.exists(manning_txt):
                create_mannings_esa(manning_txt)

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
            if streamflow_source == 'National Water Model':
                nwm_zarr = os.path.join(package_root, 'data', 'nwm_v3_daily_retrospective.zarr')
                condense_zarr(list(all_comids), nwm_zarr)
            create_reanalysis_file(list(all_comids), streamflow_source, package_root)

        utils.set_status("Stage 3/4: Fetching DEMs...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker_assign_dem, sid, db, dem_folder, dem_resolution, flowline_source) for sid in site_ids]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in DEM worker: {e}")

        utils.set_status("Stage 4/4: Finalizing hydraulics...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_assign_hydraulics, sid, db, results_folder, land_folder,
                                                 "WSE and LiDAR Date", streamflow_source, flowline_source) for sid in site_ids]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in Hydraulics worker: {e}")

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
