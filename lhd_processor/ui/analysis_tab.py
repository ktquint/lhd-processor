import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dask.distributed import Client, LocalCluster, as_completed

# Relative imports
from ..analysis.classes import Dam as AnalysisDam
from ..data_manager import DatabaseManager  # Import Manager
from . import utils, carousel

# Module-level widgets
db_entry = None
res_entry = None
run_btn = None
dam_dropdown = None
display_btn = None

# Checkbox Vars
chk_xs = None
chk_rc = None
chk_map = None
chk_wsp = None
chk_fdc = None
chk_bar = None


def setup_analysis_tab(parent_tab):
    global db_entry, res_entry, run_btn, dam_dropdown, display_btn
    global chk_xs, chk_rc, chk_map, chk_wsp, chk_fdc, chk_bar

    # --- Step 3 Frame ---
    path_frame = ttk.LabelFrame(parent_tab, text="Step 3: Analyze Results")
    path_frame.pack(pady=10, padx=10, fill="x")
    path_frame.columnconfigure(1, weight=1)

    # Row 0: Database
    ttk.Label(path_frame, text="Database (.xlsx):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    db_entry = ttk.Entry(path_frame)
    db_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(path_frame, text="Select...", command=select_db).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Results Folder
    ttk.Label(path_frame, text="Results Folder:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    res_entry = ttk.Entry(path_frame)
    res_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(path_frame, text="Select...", command=select_res).grid(row=1, column=2, padx=5, pady=5)

    # Row 2: Run Button
    run_btn = ttk.Button(path_frame, text="3. Analyze & Save All Dam Data", command=start_analysis)
    run_btn.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)

    # --- Figure Frame ---
    fig_frame = ttk.LabelFrame(parent_tab, text="Select Figures to Display")
    fig_frame.pack(pady=10, padx=10, fill="x")
    fig_frame.columnconfigure(1, weight=1)

    # Row 0: Dropdown + Refresh Button
    ttk.Label(fig_frame, text="Dam to Display:").grid(row=0, column=0, padx=5, pady=5)

    dam_dropdown = ttk.Combobox(fig_frame, state="readonly")
    dam_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    # Refresh Button
    ttk.Button(fig_frame, text="↻", command=update_dropdown).grid(row=0, column=2, padx=5, pady=5)

    # Checkboxes
    chk_xs = tk.BooleanVar(value=False);
    ttk.Checkbutton(fig_frame, text="Cross-Sections", variable=chk_xs).grid(row=1, column=0, sticky=tk.W)
    chk_rc = tk.BooleanVar(value=False);
    ttk.Checkbutton(fig_frame, text="Rating Curves", variable=chk_rc).grid(row=2, column=0, sticky=tk.W)
    chk_map = tk.BooleanVar(value=False);
    ttk.Checkbutton(fig_frame, text="Location Map", variable=chk_map).grid(row=3, column=0, sticky=tk.W)
    chk_wsp = tk.BooleanVar(value=False);
    ttk.Checkbutton(fig_frame, text="Water Surface Profile", variable=chk_wsp).grid(row=1, column=1, sticky=tk.W)
    chk_fdc = tk.BooleanVar(value=False);
    ttk.Checkbutton(fig_frame, text="Flow Duration Curve", variable=chk_fdc).grid(row=2, column=1, sticky=tk.W)
    chk_bar = tk.BooleanVar(value=False);
    ttk.Checkbutton(fig_frame, text="Generate Bar Chart (All)", variable=chk_bar).grid(row=3, column=1, sticky=tk.W)

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


def update_dropdown():
    """
    Refreshes the 'Dam to Display' list by querying the database for
    sites that have actual Hydraulic Results or Cross-Section data.
    """
    db_path = db_entry.get()

    if not os.path.isfile(db_path):
        return

    try:
        # 1. Load the Database (Force reload from disk)
        db = DatabaseManager(db_path)

        # 2. Find IDs that exist in either 'results' (Hydraulics) or 'xsections' (Geometry)
        # This fulfills the request to "look at rows that actually have results"
        ids_res = set(db.results['site_id'].dropna().unique())
        ids_xs = set(db.xsections['site_id'].dropna().unique())

        # Union them to be safe (in case a dam has geometry but no incident results yet)
        valid_ids_raw = ids_res.union(ids_xs)

        # Clean and Sort
        valid_ids = []
        for x in valid_ids_raw:
            try:
                valid_ids.append(int(x))
            except:
                pass
        valid_ids.sort()

        # 3. Update Dropdown
        dams = ["All Dams"] + [str(d) for d in valid_ids]
        dam_dropdown['values'] = dams

        # Preserve selection if it's still valid, otherwise default to "All Dams"
        current = dam_dropdown.get()
        if current in dams:
            dam_dropdown.set(current)
        elif dams:
            dam_dropdown.set(dams[0])

        utils.set_status(f"Refreshed list: Found {len(valid_ids)} dams with analysis data.")

    except Exception as e:
        utils.set_status(f"Error updating dropdown: {e}")


def process_single_dam_analysis(dam_id, xlsx_path, res_dir):
    try:
        from ..analysis.classes import Dam as AnalysisDam
        from ..data_manager import DatabaseManager

        db = DatabaseManager(xlsx_path)
        # 1. Initialize (Loads geometry only - Fast)
        dam = AnalysisDam(dam_id, db, base_results_dir=res_dir)

        # 2. Run Analysis (Calculates P, H, Jumps - Expensive)
        dam.run_analysis(est_dam=True)

        # 3. Extract results to return
        xs_data = db.xsections[db.xsections['site_id'] == dam_id].to_dict('records')
        res_data = db.results[db.results['site_id'] == dam_id].to_dict('records')

        return True, dam_id, dam.site_data, xs_data, res_data, None
    except Exception as e:
        return False, dam_id, None, None, None, str(e)


# --- Helpers ---
def generate_summary_charts(db_path, filter_id=None):
    figures_list = []
    try:
        db = DatabaseManager(db_path)

        # Loop through cross-sections 1 to 4
        for i in range(1, 5):
            try:
                # 1. Get Hydraulic Results for this XS index
                # We filter the 'HydraulicResults' tab
                res_df = db.results[db.results['xs_index'] == i].copy()
                if res_df.empty:
                    continue

                # 2. Get Slopes from CrossSections tab to join
                xs_df = db.xsections[db.xsections['xs_index'] == i][['site_id', 'slope']]

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

                ax.set_title(f"Summary Results for Cross-Section {i}")
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
def threaded_analysis():
    try:
        xlsx_path = db_entry.get()
        res_dir = res_entry.get()

        db = DatabaseManager(xlsx_path)
        valid_ids = [site_id for site_id in db.sites['site_id']
                     if os.path.exists(os.path.join(res_dir, str(site_id), "VDT"))]

        total = len(valid_ids)
        if total == 0:
            utils.set_status("No processed dams found.")
            return

        worker_count = (os.cpu_count() - 1) or 1
        with LocalCluster(n_workers=worker_count, threads_per_worker=1) as cluster:
            with Client(cluster) as client:
                utils.set_status(f"Parallelizing analysis across {worker_count} workers...")

                futures = client.map(process_single_dam_analysis, valid_ids,
                                     xlsx_path=xlsx_path, res_dir=res_dir)

                processed_count = 0
                for future in as_completed(futures):
                    # UNPACK THE NEW RETURN VALUES HERE
                    success, dam_id, site_data, xs_data, res_data, err = future.result()
                    processed_count += 1

                    if success:
                        # Update the main thread's database manager with ALL results
                        db.update_site_data(dam_id, site_data)

                        # Use the existing manager method to merge the lists of dicts
                        db.update_analysis_results(dam_id, xs_data, res_data)

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


def threaded_display():
    figs = []
    try:
        xlsx_path = db_entry.get()
        res_dir = res_entry.get()
        dam_sel = dam_dropdown.get()

        # 1. Determine which dams to process based on selection
        target_dams = []
        filter_id = None  # Default to None (implies "All" for summary charts)

        if dam_sel == "All Dams":
            # Fetch all valid IDs from the database (USING THE NEW LOGIC)
            db = DatabaseManager(xlsx_path)

            # Find IDs that have results or cross-section data in the DB
            # We use a set union to catch dams that might have one but not the other
            ids_res = set(db.results['site_id'].dropna().unique())
            ids_xs = set(db.xsections['site_id'].dropna().unique())

            valid_ids = sorted([int(x) for x in ids_res.union(ids_xs)])
            target_dams = valid_ids

        elif dam_sel:
            # Single dam selection
            try:
                d_id = int(dam_sel)
                target_dams = [d_id]
                filter_id = d_id  # Set filter for summary charts
            except ValueError:
                pass

        # 2. Summary Charts
        if chk_bar.get():
            utils.set_status("Generating summary charts...")
            # Pass the filter_id to restrict the chart if a specific dam is selected
            figs.extend(generate_summary_charts(xlsx_path, filter_id=filter_id))

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
                    dam = AnalysisDam(d_id, db, base_results_dir=res_dir)
                    dam.load_results()

                    # Helper to add and save figures
                    # Note: The plot methods in classes.py already default to save=True

                    if chk_xs.get():
                        # This works purely on geometry loaded in __init__
                        for xs in dam.cross_sections:
                            figs.append((xs.plot_cross_section(), f"Dam {d_id} XS {xs.index}"))

                    if chk_rc.get():
                        # This uses the P values loaded by dam.load_results()
                        for xs in dam.cross_sections[1:]:
                            figs.append((xs.create_combined_fig(), f"Dam {d_id} RC {xs.index}"))

                    if chk_map.get():
                        figs.append((dam.plot_map(), f"Dam {d_id} Map"))

                    if chk_wsp.get():
                        figs.append((dam.plot_water_surface(), f"Dam {d_id} WSE"))

                    if chk_fdc.get():
                        for xs in dam.cross_sections[1:]:
                            figs.append((xs.create_combined_fdc(), f"Dam {d_id} FDC {xs.index}"))

                except Exception as e:
                    print(f"Skipping plots for Dam {d_id} due to error: {e}")

        # Pass to main thread to display in Carousel
        utils.get_root().after(0, carousel.load_figures, figs)

        # Notify user of completion (Scheduled on main thread)
        utils.get_root().after(0, lambda: messagebox.showinfo("Success", f"Generated {len(figs)} figures."))

    except Exception as e:
        utils.set_status(f"Error displaying: {e}")
    finally:
        display_btn.config(state=tk.NORMAL)
        utils.set_status("Figure generation complete.")


# Thread Starters
def start_analysis():
    run_btn.config(state=tk.DISABLED)
    threading.Thread(target=threaded_analysis, daemon=True).start()


def start_display():
    display_btn.config(state=tk.DISABLED)
    threading.Thread(target=threaded_display, daemon=True).start()