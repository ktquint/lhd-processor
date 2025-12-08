import os
import ast
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Relative imports
from ..analysis import Dam as AnalysisDam
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

    ttk.Button(path_frame, text="Select Database (.csv)", command=select_db).grid(row=0, column=0, padx=5, pady=5,
                                                                                  sticky=tk.W)
    db_entry = ttk.Entry(path_frame)
    db_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Button(path_frame, text="Select Results Folder", command=select_res).grid(row=1, column=0, padx=5, pady=5,
                                                                                  sticky=tk.W)
    res_entry = ttk.Entry(path_frame)
    res_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Label(path_frame, text="Streamflow Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    model_var = tk.StringVar(value="National Water Model")
    ttk.Combobox(path_frame, textvariable=model_var, values=("USGS", "GEOGLOWS", "National Water Model")).grid(row=2,
                                                                                                               column=1,
                                                                                                               padx=5,
                                                                                                               pady=5,
                                                                                                               sticky=tk.EW)

    run_btn = ttk.Button(path_frame, text="3. Analyze & Save All Dam Data", command=start_analysis,
                         style="Accent.TButton")
    run_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)

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
    f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if f: db_entry.delete(0, tk.END); db_entry.insert(0, f)


def select_res():
    d = filedialog.askdirectory()
    if d:
        res_entry.delete(0, tk.END);
        res_entry.insert(0, d)
        update_dropdown()


def update_dropdown():
    results_dir = res_entry.get()
    database_csv = db_entry.get()

    if not os.path.isdir(results_dir) or not os.path.isfile(database_csv):
        return

    try:
        dam_strs = analysis_successful_runs(results_dir, database_csv)
        dams_int = [int(d) for d in dam_strs if d.isdigit()]
        dams_sorted = sorted(dams_int)
        dams = ["All Dams"] + [str(d) for d in dams_sorted]

        dam_dropdown['values'] = dams
        if dams: dam_dropdown.set(dams[0])
        utils.set_status(f"Found {len(dams) - 1} processed dams.")
    except Exception as e:
        utils.set_status(f"Error updating dropdown: {e}")


# --- Helpers ---
def analysis_successful_runs(results_dir, database_csv):
    try:
        lhd_df = pd.read_csv(database_csv)
        if 'ID' not in lhd_df.columns: return []
        dam_nos = set(pd.to_numeric(lhd_df['ID'], errors='coerce').dropna().astype(int).astype(str).tolist())
    except:
        return []

    successes = []
    for lhd_id in dam_nos:
        run_dir = os.path.join(results_dir, lhd_id)
        gpkg = os.path.join(run_dir, "VDT", f"{lhd_id}_Local_VDT_Database.gpkg")
        if os.path.exists(gpkg):
            try:
                if not gpd.read_file(gpkg).empty:
                    successes.append(lhd_id)
            except:
                pass
    return successes


def generate_summary_charts(lhd_df_path):
    figures_list = []
    try:
        lhd_df = pd.read_csv(lhd_df_path)
    except Exception as e:
        print(f"Error reading DB: {e}")
        return []

    for i in range(1, 5):
        try:
            cols = [f'y_t_{i}', f'y_flip_{i}', f'y_2_{i}', f's_{i}']
            filtered = lhd_df.dropna(subset=cols).copy()
            if filtered.empty: continue

            # Helper for safe eval
            def safe_eval(x):
                try:
                    return ast.literal_eval(x) if isinstance(x, str) else []
                except:
                    return []

            # Lists for plotting
            slopes, y_ts, y_flips, y_2s, ids = [], [], [], [], []

            for _, row in filtered.iterrows():
                row_y_t = safe_eval(row[f'y_t_{i}'])
                row_y_flip = safe_eval(row[f'y_flip_{i}'])
                row_y_2 = safe_eval(row[f'y_2_{i}'])
                count = len(row_y_t)

                if count > 0:
                    y_ts.extend(row_y_t)
                    y_flips.extend(row_y_flip)
                    y_2s.extend(row_y_2)
                    slopes.extend([row[f's_{i}']] * count)
                    ids.extend([row['ID']] * count)

            if not slopes: continue

            df = pd.DataFrame({'slope': slopes, 'y_t': y_ts, 'y_flip': y_flips, 'y_2': y_2s, 'id': ids})
            df = df.sort_values(['id', 'slope']).reset_index(drop=True)

            # Plotting logic
            fig = Figure(figsize=(11, 5))
            ax = fig.add_subplot(111)
            x_vals = range(len(df))

            # Convert m to ft (3.281)
            to_ft = 3.281
            cap = 0.2

            for idx, r in df.iterrows():
                y, y2, yf = r['y_t'] * to_ft, r['y_2'] * to_ft, r['y_flip'] * to_ft
                c = 'red' if y2 < y < yf else ('green' if y >= yf else 'blue')

                ax.vlines(idx, y2, yf, color='black', lw=1)
                ax.hlines(y2, idx - cap, idx + cap, color='black', lw=1)
                ax.hlines(yf, idx - cap, idx + cap, color='black', lw=1)
                ax.scatter(idx, y, color=c, marker='x', zorder=3)

            # Shading by Dam ID
            current, start, shade = None, 0, True
            for idx, dam_id in enumerate(df['id']):
                if dam_id != current:
                    if current is not None and shade:
                        ax.axvspan(start - 0.5, idx - 0.5, color='gray', alpha=0.1)
                    current, start, shade = dam_id, idx, not shade
            if shade: ax.axvspan(start - 0.5, len(df) - 0.5, color='gray', alpha=0.1)

            ax.set_xticks(x_vals)
            ax.set_xticklabels(df['slope'].round(6).astype(str), rotation=90)
            ax.set_title(f"Summary Cross-Section {i}")
            ax.set_ylabel("Depth (ft)")
            fig.tight_layout()
            figures_list.append((fig, f"Summary XS {i}"))
        except Exception as e:
            print(f"Error summary chart {i}: {e}")

    return figures_list


# --- Threaded Logic ---

def threaded_analysis():
    try:
        csv = db_entry.get()
        res_dir = res_entry.get()
        model = model_var.get()

        dam_strs = analysis_successful_runs(res_dir, csv)
        dams_int = sorted([int(d) for d in dam_strs if d.isdigit()])

        total = len(dams_int)
        if total == 0:
            utils.set_status("No processed dams found.")
            return

        for i, dam_id in enumerate(dams_int):
            utils.set_status(f"Analyzing Dam {dam_id} ({i + 1}/{total})...")
            try:
                dam = AnalysisDam(dam_id, csv, model, True, res_dir)
                # Generate and save all standard plots
                for xs in dam.cross_sections:
                    plt.close(xs.plot_cross_section())
                for xs in dam.cross_sections[1:]:
                    plt.close(xs.create_combined_fig())
                    plt.close(xs.create_combined_fdc())
                plt.close(dam.plot_map())
                plt.close(dam.plot_water_surface())
            except Exception as e:
                print(f"Skipping Dam {dam_id}: {e}")

        utils.set_status("Analysis complete.")
        messagebox.showinfo("Success", "Analysis complete.")

    except Exception as e:
        utils.set_status(f"Error: {e}")
    finally:
        run_btn.config(state=tk.NORMAL)


def threaded_display():
    figs = []
    try:
        csv = db_entry.get()
        res_dir = res_entry.get()
        model = model_var.get()
        dam_sel = dam_dropdown.get()

        # 1. Summary Charts
        if chk_bar.get():
            utils.set_status("Generating summary charts...")
            figs.extend(generate_summary_charts(csv))

        # 2. Dam Specific
        if dam_sel != "All Dams" and dam_sel:
            utils.set_status(f"Loading Dam {dam_sel}...")
            dam = AnalysisDam(int(dam_sel), csv, model, True, res_dir)

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