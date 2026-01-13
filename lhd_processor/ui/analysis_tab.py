import os
import threading
import pandas as pd
import tkinter as tk
from matplotlib.figure import Figure
from tkinter import ttk, filedialog, messagebox
from dask.distributed import Client, LocalCluster, as_completed

# Relative imports
from . import utils, carousel
from ..data_manager import DatabaseManager  # Import Manager
from ..analysis.classes import Dam as AnalysisDam

# Module-level widgets
db_entry = None
res_entry = None
res_entry_display = None
run_btn = None
dam_dropdown = None
display_btn = None
calc_mode_var = None
flowline_source_var = None
streamflow_source_var = None

# Checkbox Vars
chk_xs = None
chk_rc = None
chk_map = None
chk_wsp = None
chk_fdc = None
chk_bar = None


def setup_analysis_tab(parent_tab):
    global db_entry, res_entry, res_entry_display, run_btn, dam_dropdown, display_btn
    global chk_xs, chk_rc, chk_map, chk_wsp, chk_fdc, chk_bar
    global calc_mode_var, flowline_source_var, streamflow_source_var

    # --- Step 3 Frame ---
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

    # --- Figure Frame ---
    fig_frame = ttk.LabelFrame(parent_tab, text="4. Select Figures to Display")
    fig_frame.pack(pady=10, padx=10, fill="x")
    fig_frame.columnconfigure(1, weight=1)

    # Row 0: Results Folder (Display)
    ttk.Label(fig_frame, text="Results Folder:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    res_entry_display = ttk.Entry(fig_frame)
    res_entry_display.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(fig_frame, text="Select...", command=select_res_display).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Dropdown + Refresh Button
    ttk.Label(fig_frame, text="Dam to Display:").grid(row=1, column=0, padx=5, pady=5)

    dam_dropdown = ttk.Combobox(fig_frame, state="readonly")
    dam_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

    # Refresh Button
    ttk.Button(fig_frame, text="↻", command=update_dropdown).grid(row=1, column=2, padx=5, pady=5)

    # Checkboxes
    chk_xs = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Cross-Sections", variable=chk_xs).grid(row=2, column=0, sticky=tk.W)
    chk_rc = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Rating Curves", variable=chk_rc).grid(row=3, column=0, sticky=tk.W)
    chk_map = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Location Map", variable=chk_map).grid(row=4, column=0, sticky=tk.W)
    chk_wsp = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Water Surface Profile",
                    variable=chk_wsp).grid(row=2, column=1, sticky=tk.W)
    chk_fdc = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Flow Duration Curve",
                    variable=chk_fdc).grid(row=3, column=1, sticky=tk.W)
    chk_bar = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Generate Bar Chart (All)",
                    variable=chk_bar).grid(row=4, column=1, sticky=tk.W)

    display_btn = ttk.Button(parent_tab, text="4. Generate & Display Figures", command=start_display)
    display_btn.pack(fill="x", padx=10, pady=10)

    # Initialize Carousel (creates the hidden frame)
    carousel.setup_carousel_ui(parent_tab)


# --- Event Handlers ---
def select_db():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f:
        db_entry.delete(0, tk.END)
        db_entry.insert(0, f)
        # Auto-update if possible
        update_dropdown()


def select_res():
    d = filedialog.askdirectory()
    if d:
        res_entry.delete(0, tk.END)
        res_entry.insert(0, d)


def select_res_display():
    d = filedialog.askdirectory()
    if d:
        res_entry_display.delete(0, tk.END)
        res_entry_display.insert(0, d)
        update_dropdown()


def update_dropdown():
    """
    Refreshes the 'Dam to Display' list by scanning the Results Folder (Display)
    for valid site directories containing required GPKG files.
    """
    res_dir = res_entry_display.get()

    if not res_dir or not os.path.isdir(res_dir):
        return

    try:
        valid_ids = []
        
        # Iterate over items in the results directory
        for item in os.listdir(res_dir):
            site_path = os.path.join(res_dir, item)
            
            # Check if it's a directory and the name is an integer (site_id)
            if os.path.isdir(site_path) and item.isdigit():
                site_id = item
                
                # Define required file paths
                vdt_db = os.path.join(site_path, "VDT", f"{site_id}_Local_VDT_Database.gpkg")
                curve_db = os.path.join(site_path, "VDT", f"{site_id}_Local_Curve.gpkg")
                xs_db = os.path.join(site_path, "XS", f"{site_id}_Local_XS.gpkg")
                
                # Check if all exist
                if os.path.exists(vdt_db) and os.path.exists(curve_db) and os.path.exists(xs_db):
                    valid_ids.append(int(site_id))
        
        valid_ids.sort()

        # Update Dropdown
        dams = ["All Dams"] + [str(d) for d in valid_ids]
        # noinspection PyTypeHints
        dam_dropdown["values"] = dams

        # Preserve selection if it's still valid, otherwise default to "All Dams"
        current = dam_dropdown.get()
        if current in dams:
            dam_dropdown.set(current)
        elif dams:
            dam_dropdown.set(dams[0])

        utils.set_status(f"Refreshed list: Found {len(valid_ids)} dams with valid data.")

    except Exception as e:
        utils.set_status(f"Error updating dropdown: {e}")


def process_single_dam_analysis(dam_id, xlsx_path, res_dir, calc_mode, flowline_source, streamflow_source):
    try:
        db = DatabaseManager(xlsx_path)
        # 1. Initialize (Loads geometry only - Fast)
        dam = AnalysisDam(dam_id, db, base_results_dir=res_dir, calc_mode=calc_mode,
                          flowline_source=flowline_source, streamflow_source=streamflow_source)

        # 2. Run Analysis (Calculates P, H, Jumps - Expensive)
        xs_data, res_data = dam.run_analysis()

        return True, dam_id, dam.site_data, xs_data, res_data, None
    except Exception as e:
        return False, dam_id, None, None, None, str(e)


# --- Helpers ---
def generate_summary_charts(db_path, f_source, s_source, filter_id=None):
    figures_list = []
    try:
        db = DatabaseManager(db_path)
        
        # Determine which sheet to read based on current selection
        # (Assuming user wants charts for the currently selected sources)
        # f_source and s_source passed as arguments
        
        # Use the manager's helper (we can access it via a temporary instance or just replicate logic)
        # Since we already have 'db' instance, let's use the public getter logic manually or add a helper
        # But wait, generate_summary_charts is static-ish.
        
        # Let's use the public getter for a dummy ID to find the sheet name, or just iterate?
        # Actually, the user might want to see charts for ALL sources? 
        # For now, let's stick to the selected source to avoid clutter.
        
        # We need to access the specific dataframe from the dictionary
        # The keys are like 'ResultsNHDNWM'
        # Let's use the internal helper if possible, or just reconstruct the name
        
        abbr_map = {
            'NHDPlus': 'NHD',
            'TDX-Hydro': 'TDX',
            'National Water Model': 'NWM',
            'GEOGLOWS': 'GEO'
        }
        f_abbr = abbr_map.get(f_source, f_source)
        s_abbr = abbr_map.get(s_source, s_source)
        res_sheet = f"Results{f_abbr}{s_abbr}"
        xs_sheet = f"CrossSections{f_abbr}{s_abbr}"
        
        if res_sheet not in db.results or xs_sheet not in db.xsections:
            print(f"No data found for {f_source} / {s_source}")
            return []

        results_df = db.results[res_sheet]
        xsections_df = db.xsections[xs_sheet]

        # Loop through cross-sections 1 to 4
        for i in range(1, 5):
            try:
                # 1. Get Hydraulic Results for this XS index
                res_df = results_df[results_df['xs_index'] == i].copy()
                if res_df.empty:
                    continue

                # 2. Get Slopes from CrossSections tab to join
                xs_df = xsections_df[xsections_df['xs_index'] == i][['site_id', 'slope']]

                # 3. Merge results with slopes
                # This gives us: site_id, date, y_t, y_2, y_flip, slope
                plot_df = pd.merge(res_df, xs_df, on='site_id')

                if filter_id is not None:
                    plot_df = plot_df[plot_df['site_id'] == filter_id]

                if plot_df.empty:
                    continue

                # 4. Sort for the bar chart (by Site, then Slope)
                plot_df = plot_df.sort_values(['site_id', 'slope']).reset_index(drop=True)

                # 5. Plotting Logic
                fig = Figure(figsize=(11, 5))
                ax = fig.add_subplot(111)

                x_vals = range(len(plot_df))

                # Convert to feet for display
                to_ft = 3.281

                for idx, row in plot_df.iterrows():
                    y_t = row['y_t'] * to_ft
                    y_2 = row['y_2'] * to_ft
                    y_flip = row['y_flip'] * to_ft

                    # Color based on jump type logic
                    if y_2 < y_t < y_flip:
                        c = 'red'  # Dangerous Type C
                    elif y_t >= y_flip:
                        c = 'green'  # Type D
                    else:
                        c = 'blue'  # Type A/B

                    cap = 0.2
                    # noinspection PyTypeChecker
                    idx = float(idx)
                    ax.vlines(idx, y_2, y_flip, color='black', lw=1)
                    ax.hlines(y_2, idx - cap, idx + cap, color='black', lw=1)
                    ax.hlines(y_flip, idx - cap, idx + cap, color='black', lw=1)
                    ax.scatter(idx, y_t, color=c, marker='x', zorder=3)

                # Shading by Dam ID
                current_id = None
                start_idx = 0
                shade = True
                for idx, site_id in enumerate(plot_df['site_id']):
                    if site_id != current_id:
                        if current_id is not None and shade:
                            ax.axvspan(start_idx - 0.5, idx - 0.5, color='gray', alpha=0.1)
                        current_id = site_id
                        start_idx = idx
                        shade = not shade

                if shade:
                    ax.axvspan(start_idx - 0.5, len(plot_df) - 0.5, color='gray', alpha=0.1)

                ax.set_xticks(x_vals)
                # Label with slope
                ax.set_xticklabels(plot_df['slope'].round(4).astype(str), rotation=90)

                ax.set_title(f"Summary Results for Cross-Section {i} ({f_source}/{s_source})")
                ax.set_ylabel("Depth (ft)")
                ax.set_xlabel("Channel Slope")

                fig.tight_layout()
                figures_list.append((fig, f"Summary XS {i}"))

            except Exception as e:
                print(f"Error generating chart for XS {i}: {e}")

    except Exception as e:
        print(f"Error reading DB for charts: {e}")

    return figures_list


# --- Threaded Logic ---
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
            return

        # worker_count = (os.cpu_count() - 1) or 1
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
                    # UNPACK THE NEW RETURN VALUES HERE
                    success, dam_id, site_data, xs_data, res_data, err = future.result()
                    processed_count += 1

                    if success:
                        # Update the main thread's database manager with ALL results
                        db.update_site_data(dam_id, site_data)

                        # Use the existing manager method to merge the lists of dicts
                        db.update_analysis_results(dam_id, xs_data, res_data,
                                                   f_source, s_source)

                        utils.set_status(f"Analyzed Dam {dam_id} ({processed_count}/{total})")
                    else:
                        print(f"Error on Dam {dam_id}: {err}")

        utils.set_status("Saving all results to database...")
        db.save()
        utils.set_status("Results saved.")

        # AUTO-REFRESH LIST AFTER RUN
        utils.get_root().after(0, update_dropdown)

        messagebox.showinfo("Success", "Parallel analysis complete.")
    except Exception as e:
        utils.set_status(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        run_btn.config(state=tk.NORMAL)


def threaded_display(mode, params):
    figs = []
    try:
        xlsx_path = params["xlsx_path"]
        res_dir = params["res_dir"]
        dam_sel = params["dam_sel"]
        
        f_source = params["f_source"]
        s_source = params["s_source"]

        # 1. Determine which dams to process based on selection
        target_dams = []
        filter_id = None  # Default to None (implies "All" for summary charts)

        if dam_sel == "All Dams":
            all_values = params["dam_values"]
            if len(all_values) > 1:
                target_dams = [int(x) for x in all_values[1:]]
            else:
                target_dams = []

        elif dam_sel:
            # Single dam selection
            try:
                d_id = int(dam_sel)
                target_dams = [d_id]
                filter_id = d_id  # Set filter for summary charts
            except ValueError:
                pass

        # 2. Summary Charts
        if params["chk_bar"]:
            utils.set_status("Generating summary charts...")
            # Pass the filter_id to restrict the chart if a specific dam is selected
            figs.extend(generate_summary_charts(xlsx_path, f_source, s_source, filter_id=filter_id))

        # 3. Dam Specific Figures (Iterate through target_dams)
        if target_dams:
            # Re-initialize DB for reading
            db = DatabaseManager(xlsx_path)

            total = len(target_dams)
            for idx, d_id in enumerate(target_dams):
                utils.set_status(f"Generating plots for Dam {d_id} ({idx + 1}/{total})...")

                try:
                    # Check if VDT folder exists before trying to load
                    # (Even if results exist in DB, we need files for some plots like maps)
                    if not os.path.exists(os.path.join(res_dir, str(d_id), "VDT")):
                        print(f"Skipping plots for Dam {d_id}: Missing VDT folder.")
                        continue

                    # Initialize Dam Object
                    dam = AnalysisDam(d_id, db, base_results_dir=res_dir, calc_mode=mode,
                                      flowline_source=f_source, streamflow_source=s_source)
                    dam.load_results()

                    # Helper to add and save figures
                    # Note: The plot methods in classes.py already default to save=True

                    if params["chk_xs"]:
                        # This works purely on geometry loaded in __init__
                        for xs in dam.cross_sections:
                            figs.append((xs.plot_cross_section(), f"Dam {d_id} XS {xs.index}"))

                    if params["chk_rc"]:
                        # This uses the P values loaded by dam.load_results()
                        for xs in dam.cross_sections[1:]:
                            figs.append((xs.create_combined_fig(), f"Dam {d_id} RC {xs.index}"))

                    if params["chk_map"]:
                        figs.append((dam.plot_map(), f"Dam {d_id} Map"))

                    if params["chk_wsp"]:
                        figs.append((dam.plot_water_surface(), f"Dam {d_id} WSE"))

                    if params["chk_fdc"]:
                        for xs in dam.cross_sections[1:]:
                            figs.append((xs.create_combined_fdc(), f"Dam {d_id} FDC {xs.index}"))

                except Exception as e:
                    print(f"Skipping plots for Dam {d_id} due to error: {e}")

        # Pass to main thread to display in Carousel
        utils.get_root().after(0, carousel.load_figures, figs)

        # Notify user of completion (Scheduled on main thread)
        # utils.get_root().after(0, lambda: messagebox.showinfo("Success", f"Generated {len(figs)} figures."))

    except Exception as e:
        utils.set_status(f"Error displaying: {e}")
    finally:
        display_btn.config(state=tk.NORMAL)
        utils.set_status("Figure generation complete.")


# Thread Starters
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


def start_display():
    display_btn.config(state=tk.DISABLED)
    mode = calc_mode_var.get()
    
    params = {
        "xlsx_path": db_entry.get(),
        "res_dir": res_entry_display.get(),
        "dam_sel": dam_dropdown.get(),
        "dam_values": dam_dropdown["values"],
        "f_source": flowline_source_var.get(),
        "s_source": streamflow_source_var.get(),
        "chk_xs": chk_xs.get(),
        "chk_rc": chk_rc.get(),
        "chk_map": chk_map.get(),
        "chk_wsp": chk_wsp.get(),
        "chk_fdc": chk_fdc.get(),
        "chk_bar": chk_bar.get(),
    }
    
    threading.Thread(target=threaded_display, args=(mode, params), daemon=True).start()
