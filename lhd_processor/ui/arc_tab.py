import gc
import os
import threading
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed

from . import utils

try:
    from ..prep.lhd_arc import ArcDam
except Exception as e:
    print(f"Warning: Could not import ARC. Error: {e}")
    ArcDam = None

# Module-level widgets
arc_xlsx_entry = None
arc_results_entry = None
arc_flowline_var = None
arc_streamflow_var = None
arc_baseflow_var = None
arc_run_button = None
arc_stop_button = None

stop_event = threading.Event()

def setup_arc_tab(parent_tab):
    global arc_xlsx_entry, arc_results_entry, arc_flowline_var, arc_streamflow_var, arc_baseflow_var
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

    ttk.Label(opts_frame, text="Streamflow Source:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
    arc_streamflow_var = tk.StringVar(value="National Water Model")
    ttk.Combobox(opts_frame, textvariable=arc_streamflow_var, state="readonly", values=("National Water Model", "GEOGLOWS")).grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

    # Row 3: Baseflow Estimation
    bf_frame = ttk.Frame(arc_frame)
    bf_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
    bf_frame.columnconfigure(1, weight=1)

    ttk.Label(bf_frame, text="Baseflow Estimation:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    arc_baseflow_var = tk.StringVar(value="WSE and LiDAR Date")
    ttk.Combobox(bf_frame, textvariable=arc_baseflow_var, state="readonly", values=("WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation")).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    # Row 4: Buttons
    btn_frame = ttk.Frame(arc_frame)
    btn_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10)
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
        except:
            pass

def select_arc_results_dir():
    d = filedialog.askdirectory()
    if d:
        arc_results_entry.delete(0, tk.END)
        arc_results_entry.insert(0, d)

def process_single_dam_arc(dam_dict, results_dir, flowline_source, streamflow_source, baseflow_method):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['GDAL_CACHEMAX'] = '256'

    dam_name = dam_dict.get('name', "Unknown Dam")
    try:
        dam_i = ArcDam(dam_dict, results_dir, flowline_source, streamflow_source, baseflow_method)
        dam_i.process_dam()
        del dam_i
        gc.collect()
        return True, dam_name, None
    except Exception as e:
        gc.collect()
        return False, dam_name, str(e)

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
        
        if not os.path.exists(excel_loc):
            messagebox.showerror("Error", "Excel database not found.")
            return

        df = pd.read_excel(excel_loc)
        dams = df.to_dict(orient='records')
        count_to_run = len(dams)

        total_cores = os.cpu_count() or 1
        worker_count = max(1, int(total_cores / 3))

        utils.set_status(f"Initializing Dask Workers ({worker_count}) for {count_to_run} dams...")

        with LocalCluster(processes=True, threads_per_worker=1, n_workers=worker_count) as cluster:
            with Client(cluster) as client:
                utils.set_status(f"Processing... (Dashboard: {client.dashboard_link})")
                futures = client.map(process_single_dam_arc, dams, results_dir=results_loc, flowline_source=fl_source, streamflow_source=sf_source, baseflow_method=bf_method)

                processed_so_far = 0
                for future in dask_as_completed(futures):
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
        arc_run_button.config(state=tk.NORMAL)
        arc_stop_button.config(state=tk.DISABLED)

def stop_arc_thread():
    if messagebox.askyesno("Stop", "Stop processing after current tasks finish?"):
        stop_event.set()

def start_arc_thread():
    threading.Thread(target=threaded_run_arc, daemon=True).start()
