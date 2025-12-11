import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Relative imports
from ..analysis.classes import Dam as AnalysisDam
from ..data_manager import DatabaseManager  # Import Manager
from . import utils, carousel

# Module-level widgets
db_entry = None
res_entry = None
model_var = None
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
    global db_entry, res_entry, model_var, run_btn, dam_dropdown, display_btn
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

    # Row 2: Streamflow Source
    ttk.Label(path_frame, text="Streamflow Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    model_var = tk.StringVar(value="National Water Model")
    # Columnspan=2 to span across the Entry and Button columns from rows above
    ttk.Combobox(path_frame, textvariable=model_var, values=("USGS", "GEOGLOWS", "National Water Model")).grid(
        row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

    # Row 3: Run Button
    run_btn = ttk.Button(path_frame, text="3. Analyze & Save All Dam Data", command=start_analysis,
                         style="Accent.TButton")
    run_btn.grid(row=3, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)

    # --- Figure Frame ---
    fig_frame = ttk.LabelFrame(parent_tab, text="Select Figures to Display")
    fig_frame.pack(pady=10, padx=10, fill="x")
    fig_frame.columnconfigure(1, weight=1)

    ttk.Label(fig_frame, text="Dam to Display:").grid(row=0, column=0, padx=5, pady=5)
    dam_dropdown = ttk.Combobox(fig_frame, state="readonly")
    dam_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

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

    display_btn = ttk.Button(parent_tab, text="4. Generate & Display Figures", command=start_display,
                             style="Accent.TButton")
    display_btn.pack(fill="x", padx=10, pady=10)

    # Initialize Carousel (creates the hidden frame)
    carousel.setup_carousel_ui(parent_tab)


# --- Event Handlers ---
def select_db():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f: db_entry.delete(0, tk.END); db_entry.insert(0, f)


def select_res():
    d = filedialog.askdirectory()
    if d:
        res_entry.delete(0, tk.END);
        res_entry.insert(0, d)
        update_dropdown()


def update_dropdown():
    results_dir = res_entry.get()
    db_path = db_entry.get()

    if not os.path.isdir(results_dir) or not os.path.isfile(db_path):
        return

    try:
        # Use DataManager to read IDs
        db = DatabaseManager(db_path)
        if db.sites.empty: return

        dam_ids = db.sites['site_id'].dropna().astype(int).astype(str).tolist()

        # Check which have results
        successes = []
        for lhd_id in dam_ids:
            run_dir = os.path.join(results_dir, lhd_id)
            if os.path.exists(os.path.join(run_dir, "VDT", f"{lhd_id}_Local_VDT_Database.gpkg")):
                successes.append(lhd_id)

        dams_sorted = sorted([int(x) for x in successes])
        dams = ["All Dams"] + [str(d) for d in dams_sorted]
        dam_dropdown['values'] = dams
        if dams: dam_dropdown.set(dams[0])
        utils.set_status(f"Found {len(dams) - 1} processed dams.")
    except Exception as e:
        utils.set_status(f"Error updating dropdown: {e}")


# --- Helpers ---
def generate_summary_charts(db_path):
    figures_list = []
    try:
        db = DatabaseManager(db_path)

        # Loop through cross-sections 1 to 4
        for i in range(1, 5):
            try:
                # 1. Get Hydraulic Results for this XS index
                # We filter the 'HydraulicResults' tab
                res_df = db.results[db.results['xs_index'] == i].copy()

                if res_df.empty: continue

                # 2. Get Slopes from CrossSections tab to join
                xs_df = db.xsections[db.xsections['xs_index'] == i][['site_id', 'slope']]

                # 3. Merge results with slopes
                # This gives us: site_id, date, y_t, y_2, y_flip, slope
                plot_df = pd.merge(res_df, xs_df, on='site_id')

                if plot_df.empty: continue

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
        model = model_var.get()

        # Initialize DB
        db = DatabaseManager(xlsx_path)

        # Filter for successful runs
        valid_ids = []
        for site_id in db.sites['site_id']:
            str_id = str(site_id)
            if os.path.exists(os.path.join(res_dir, str_id, "VDT", f"{str_id}_Local_VDT_Database.gpkg")):
                valid_ids.append(site_id)

        total = len(valid_ids)
        if total == 0:
            utils.set_status("No processed dams found.")
            return

        for i, dam_id in enumerate(valid_ids):
            utils.set_status(f"Analyzing Dam {dam_id} ({i + 1}/{total})...")
            try:
                # Pass DB Manager
                dam = AnalysisDam(dam_id, db, est_dam=True, base_results_dir=res_dir)

                # Generate plots (logic remains same)
                for xs in dam.cross_sections: plt.close(xs.plot_cross_section())
                for xs in dam.cross_sections[1:]: plt.close(xs.create_combined_fig()); plt.close(
                    xs.create_combined_fdc())
                plt.close(dam.plot_map())
                plt.close(dam.plot_water_surface())

            except Exception as e:
                print(f"Skipping Dam {dam_id}: {e}")

        # Save DB at the end
        utils.set_status("Saving analysis results...")
        db.save()

        utils.set_status("Analysis complete.")
        messagebox.showinfo("Success", "Analysis complete.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
    finally:
        run_btn.config(state=tk.NORMAL)


def threaded_display():
    figs = []
    try:
        xlsx_path = db_entry.get()
        res_dir = res_entry.get()
        model = model_var.get()
        dam_sel = dam_dropdown.get()

        # 1. Summary Charts (Placeholder - logic needs update for new schema)
        if chk_bar.get():
            utils.set_status("Generating summary charts...")
            figs.extend(generate_summary_charts(xlsx_path))

        # 2. Dam Specific
        if dam_sel != "All Dams" and dam_sel:
            utils.set_status(f"Loading Dam {dam_sel}...")
            db = DatabaseManager(xlsx_path)
            dam = AnalysisDam(int(dam_sel), db, est_dam=True, base_results_dir=res_dir)

            if chk_xs.get():
                for xs in dam.cross_sections:
                    figs.append((xs.plot_cross_section(), f"XS {xs.index}"))

            if chk_rc.get():
                for xs in dam.cross_sections[1:]:
                    figs.append((xs.create_combined_fig(), f"Rating Curve {xs.index}"))

            if chk_map.get():
                figs.append((dam.plot_map(), "Location Map"))

            if chk_wsp.get():
                figs.append((dam.plot_water_surface(), "WSE Profile"))

            if chk_fdc.get():
                for xs in dam.cross_sections[1:]:
                    figs.append((xs.create_combined_fdc(), f"FDC {xs.index}"))

        # Pass to main thread
        utils.get_root().after(0, carousel.load_figures, figs)

    except Exception as e:
        utils.set_status(f"Error displaying: {e}")
    finally:
        display_btn.config(state=tk.NORMAL)


# Thread Starters
def start_analysis():
    run_btn.config(state=tk.DISABLED)
    threading.Thread(target=threaded_analysis, daemon=True).start()


def start_display():
    display_btn.config(state=tk.DISABLED)
    threading.Thread(target=threaded_display, daemon=True).start()
