import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dask.distributed import Client, LocalCluster, as_completed

from . import utils
from ..data_manager import DatabaseManager
from ..analysis.classes import Dam as AnalysisDam

# Module-level widgets
db_entry = None
res_entry = None
run_btn = None
calc_mode_var = None
flowline_source_var = None
streamflow_source_var = None

def setup_calc_tab(parent_tab):
    global db_entry, res_entry, run_btn, calc_mode_var, flowline_source_var, streamflow_source_var

    path_frame = ttk.LabelFrame(parent_tab, text="Step 3: Analyze Results")
    path_frame.pack(pady=10, padx=10, fill="x")
    path_frame.columnconfigure(1, weight=1)

    # Row 0: Database
    ttk.Label(path_frame, text="Database (.xlsx):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    db_entry = ttk.Entry(path_frame)
    db_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(path_frame, text="Select...", command=select_db).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Flowline Source
    flowline_source_var = tk.StringVar(value="NHDPlus")
    ttk.Label(path_frame, text="Flowline Source:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    ttk.Combobox(path_frame, textvariable=flowline_source_var, values=["NHDPlus", "TDX-Hydro"],
                 state="readonly").grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

    # Row 2: Streamflow Source
    streamflow_source_var = tk.StringVar(value="National Water Model")
    ttk.Label(path_frame, text="Streamflow Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    ttk.Combobox(path_frame, textvariable=streamflow_source_var,
                 values=["National Water Model", "GEOGLOWS"],
                 state="readonly").grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

    # Row 3: Calculation Mode
    calc_mode_var = tk.StringVar(value="Advanced")
    ttk.Label(path_frame, text="Calculation Mode:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

    mode_frame = ttk.Frame(path_frame)
    mode_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W)

    ttk.Radiobutton(mode_frame, text="Advanced",
                    variable=calc_mode_var, value="Advanced").pack(side="left", padx=(0, 10))
    ttk.Radiobutton(mode_frame, text="Simplified",
                    variable=calc_mode_var, value="Simplified").pack(side="left")

    # Row 4: Results Folder
    ttk.Label(path_frame, text="Results Folder:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
    res_entry = ttk.Entry(path_frame)
    res_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(path_frame, text="Select...", command=select_res).grid(row=4, column=2, padx=5, pady=5)

    # Row 5: Run Button
    run_btn = ttk.Button(path_frame, text="3. Analyze & Save All Dam Data", command=start_analysis)
    run_btn.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)

def select_db():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f:
        db_entry.delete(0, tk.END)
        db_entry.insert(0, f)

def select_res():
    d = filedialog.askdirectory()
    if d:
        res_entry.delete(0, tk.END)
        res_entry.insert(0, d)

def process_single_dam_analysis(dam_id, xlsx_path, res_dir, calc_mode, flowline_source, streamflow_source):
    try:
        db = DatabaseManager(xlsx_path)
        dam = AnalysisDam(dam_id, db, base_results_dir=res_dir, calc_mode=calc_mode,
                          flowline_source=flowline_source, streamflow_source=streamflow_source)
        xs_data, res_data = dam.run_analysis()
        return True, dam_id, dam.site_data, xs_data, res_data, None
    except Exception as e:
        return False, dam_id, None, None, None, str(e)

def threaded_analysis(mode, params):
    try:
        xlsx_path = params["xlsx_path"]
        res_dir = params["res_dir"]
        f_source = params["f_source"]
        s_source = params["s_source"]

        db = DatabaseManager(xlsx_path)
        valid_ids = [site_id for site_id in db.sites['site_id']
                     if os.path.exists(os.path.join(res_dir, str(site_id), "VDT"))]

        total = len(valid_ids)
        if total == 0:
            utils.set_status("No processed dams found.")
            messagebox.showwarning("No Data", "No processed dams found in the specified Results folder.")
            return

        worker_count = 2
        with LocalCluster(n_workers=worker_count, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                utils.set_status(f"Parallelizing analysis across {worker_count} workers...")

                futures = client.map(process_single_dam_analysis, valid_ids,
                                     xlsx_path=xlsx_path, res_dir=res_dir,
                                     calc_mode=mode, 
                                     flowline_source=f_source,
                                     streamflow_source=s_source)

                processed_count = 0
                for future in as_completed(futures):
                    success, dam_id, site_data, xs_data, res_data, err = future.result()
                    processed_count += 1

                    if success:
                        db.update_site_data(dam_id, site_data)
                        db.update_analysis_results(dam_id, xs_data, res_data, f_source, s_source)
                        utils.set_status(f"Analyzed Dam {dam_id} ({processed_count}/{total})")
                    else:
                        print(f"Error on Dam {dam_id}: {err}")
                        utils.set_status(f"Error on Dam {dam_id}: {err}")

        utils.set_status("Saving all results to database...")
        db.save()
        utils.set_status("Results saved.")
        messagebox.showinfo("Success", "Parallel analysis complete.")
    except Exception as e:
        utils.set_status(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        run_btn.config(state=tk.NORMAL)

def start_analysis():
    run_btn.config(state=tk.DISABLED)
    mode = calc_mode_var.get()
    
    params = {
        "xlsx_path": db_entry.get(),
        "res_dir": res_entry.get(),
        "f_source": flowline_source_var.get(),
        "s_source": streamflow_source_var.get(),
    }

    threading.Thread(target=threaded_analysis, args=(mode, params), daemon=True).start()
