import os
import threading
import pandas as pd
import tkinter as tk
from matplotlib.figure import Figure
from tkinter import ttk, filedialog, messagebox

from . import utils, carousel
from ..data_manager import DatabaseManager
from ..analysis.classes import Dam as AnalysisDam

# Module-level widgets
db_entry = None
res_entry_display = None
dam_dropdown = None
display_btn = None
flowline_source_var = None
streamflow_source_var = None
calc_mode_var = None

# Checkbox Vars
chk_xs = None
chk_rc = None
chk_map = None
chk_wsp = None
chk_fdc = None
chk_bar = None

def setup_vis_tab(parent_tab):
    global db_entry, res_entry_display, dam_dropdown, display_btn
    global chk_xs, chk_rc, chk_map, chk_wsp, chk_fdc, chk_bar
    global flowline_source_var, streamflow_source_var, calc_mode_var

    # --- Figure Frame ---
    fig_frame = ttk.LabelFrame(parent_tab, text="4. Select Figures to Display")
    fig_frame.pack(pady=10, padx=10, fill="x")
    fig_frame.columnconfigure(1, weight=1)

    # Row 0: Database (Needed for reading results)
    ttk.Label(fig_frame, text="Database (.xlsx):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    db_entry = ttk.Entry(fig_frame)
    db_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(fig_frame, text="Select...", command=select_db).grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Results Folder (Display)
    ttk.Label(fig_frame, text="Results Folder:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    res_entry_display = ttk.Entry(fig_frame)
    res_entry_display.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(fig_frame, text="Select...", command=select_res_display).grid(row=1, column=2, padx=5, pady=5)

    # Row 2: Sources (Needed to know which results to pull)
    src_frame = ttk.Frame(fig_frame)
    src_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW)
    src_frame.columnconfigure(1, weight=1)
    src_frame.columnconfigure(3, weight=1)

    ttk.Label(src_frame, text="Flowline Source:").grid(row=0, column=0, padx=5, pady=5)
    flowline_source_var = tk.StringVar(value="NHDPlus")
    ttk.Combobox(src_frame, textvariable=flowline_source_var, values=["NHDPlus", "TDX-Hydro"], state="readonly").grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Label(src_frame, text="Streamflow Source:").grid(row=0, column=2, padx=5, pady=5)
    streamflow_source_var = tk.StringVar(value="National Water Model")
    ttk.Combobox(src_frame, textvariable=streamflow_source_var, values=["National Water Model", "GEOGLOWS"], state="readonly").grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

    # Row 3: Dropdown + Refresh Button
    ttk.Label(fig_frame, text="Dam to Display:").grid(row=3, column=0, padx=5, pady=5)
    dam_dropdown = ttk.Combobox(fig_frame, state="readonly")
    dam_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(fig_frame, text="↻", command=update_dropdown).grid(row=3, column=2, padx=5, pady=5)

    # Checkboxes
    chk_xs = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Cross-Sections", variable=chk_xs).grid(row=4, column=0, sticky=tk.W)
    chk_rc = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Rating Curves", variable=chk_rc).grid(row=5, column=0, sticky=tk.W)
    chk_map = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Location Map", variable=chk_map).grid(row=6, column=0, sticky=tk.W)
    chk_wsp = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Water Surface Profile", variable=chk_wsp).grid(row=4, column=1, sticky=tk.W)
    chk_fdc = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Flow Duration Curve", variable=chk_fdc).grid(row=5, column=1, sticky=tk.W)
    chk_bar = tk.BooleanVar(value=False)
    ttk.Checkbutton(fig_frame, text="Generate Bar Chart (All)", variable=chk_bar).grid(row=6, column=1, sticky=tk.W)

    # Calculation Mode (Hidden but needed for AnalysisDam init)
    calc_mode_var = tk.StringVar(value="Advanced")

    display_btn = ttk.Button(parent_tab, text="4. Generate & Display Figures", command=start_display)
    display_btn.pack(fill="x", padx=10, pady=10)

    # Initialize Carousel
    carousel.setup_carousel_ui(parent_tab)

def select_db():
    f = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if f:
        db_entry.delete(0, tk.END)
        db_entry.insert(0, f)
        update_dropdown()

def select_res_display():
    d = filedialog.askdirectory()
    if d:
        res_entry_display.delete(0, tk.END)
        res_entry_display.insert(0, d)
        update_dropdown()

def update_dropdown():
    res_dir = res_entry_display.get()
    if not res_dir or not os.path.isdir(res_dir):
        return

    try:
        valid_ids = []
        for item in os.listdir(res_dir):
            site_path = os.path.join(res_dir, item)
            if os.path.isdir(site_path) and item.isdigit():
                site_id = item
                vdt_db = os.path.join(site_path, "VDT", f"{site_id}_Local_VDT_Database.gpkg")
                if os.path.exists(vdt_db):
                    valid_ids.append(int(site_id))
        
        valid_ids.sort()
        dams = ["All Dams"] + [str(d) for d in valid_ids]
        dam_dropdown["values"] = dams

        current = dam_dropdown.get()
        if current in dams:
            dam_dropdown.set(current)
        elif dams:
            dam_dropdown.set(dams[0])

        utils.set_status(f"Refreshed list: Found {len(valid_ids)} dams with valid data.")
    except Exception as e:
        utils.set_status(f"Error updating dropdown: {e}")

def generate_summary_charts(db_path, f_source, s_source, filter_id=None):
    figures_list = []
    try:
        db = DatabaseManager(db_path)
        abbr_map = {'NHDPlus': 'NHD', 'TDX-Hydro': 'TDX', 'National Water Model': 'NWM', 'GEOGLOWS': 'GEO'}
        f_abbr = abbr_map.get(f_source, f_source)
        s_abbr = abbr_map.get(s_source, s_source)
        res_sheet = f"Results{f_abbr}{s_abbr}"
        xs_sheet = f"CrossSections{f_abbr}{s_abbr}"
        
        if res_sheet not in db.results or xs_sheet not in db.xsections:
            print(f"No data found for {f_source} / {s_source}")
            return []

        results_df = db.results[res_sheet]
        xsections_df = db.xsections[xs_sheet]

        for i in range(1, 5):
            try:
                res_df = results_df[results_df['xs_index'] == i].copy()
                if res_df.empty: continue

                xs_df = xsections_df[xsections_df['xs_index'] == i][['site_id', 'slope']]
                plot_df = pd.merge(res_df, xs_df, on='site_id')

                if filter_id is not None:
                    plot_df = plot_df[plot_df['site_id'] == filter_id]

                if plot_df.empty: continue

                plot_df = plot_df.sort_values(['site_id', 'slope']).reset_index(drop=True)

                fig = Figure(figsize=(11, 5))
                ax = fig.add_subplot(111)
                x_vals = range(len(plot_df))
                to_ft = 3.281

                for idx, row in plot_df.iterrows():
                    y_t = row['y_t'] * to_ft
                    y_2 = row['y_2'] * to_ft
                    y_flip = row['y_flip'] * to_ft

                    if y_2 < y_t < y_flip: c = 'red'
                    elif y_t >= y_flip: c = 'green'
                    else: c = 'blue'

                    cap = 0.2
                    idx = float(idx)
                    ax.vlines(idx, y_2, y_flip, color='black', lw=1)
                    ax.hlines(y_2, idx - cap, idx + cap, color='black', lw=1)
                    ax.hlines(y_flip, idx - cap, idx + cap, color='black', lw=1)
                    ax.scatter(idx, y_t, color=c, marker='x', zorder=3)

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

def threaded_display(mode, params):
    figs = []
    try:
        xlsx_path = params["xlsx_path"]
        res_dir = params["res_dir"]
        dam_sel = params["dam_sel"]
        f_source = params["f_source"]
        s_source = params["s_source"]

        target_dams = []
        filter_id = None

        if dam_sel == "All Dams":
            all_values = params["dam_values"]
            if len(all_values) > 1:
                target_dams = [int(x) for x in all_values[1:]]
        elif dam_sel:
            try:
                d_id = int(dam_sel)
                target_dams = [d_id]
                filter_id = d_id
            except ValueError:
                pass

        if params["chk_bar"]:
            utils.set_status("Generating summary charts...")
            figs.extend(generate_summary_charts(xlsx_path, f_source, s_source, filter_id=filter_id))

        if target_dams:
            db = DatabaseManager(xlsx_path)
            total = len(target_dams)
            for idx, d_id in enumerate(target_dams):
                utils.set_status(f"Generating plots for Dam {d_id} ({idx + 1}/{total})...")
                try:
                    if not os.path.exists(os.path.join(res_dir, str(d_id), "VDT")):
                        print(f"Skipping plots for Dam {d_id}: Missing VDT folder.")
                        continue

                    dam = AnalysisDam(d_id, db, base_results_dir=res_dir, calc_mode=mode,
                                      flowline_source=f_source, streamflow_source=s_source)
                    dam.load_results()

                    if params["chk_xs"]:
                        for xs in dam.cross_sections:
                            figs.append((xs.plot_cross_section(), f"Dam {d_id} XS {xs.index}"))

                    if params["chk_rc"]:
                        for xs in dam.cross_sections[1:]:
                            figs.append((xs.create_combined_fig(), f"Dam {d_id} RC {xs.index}"))

                    if params["chk_map"]:
                        import matplotlib
                        original_backend = matplotlib.get_backend()
                        matplotlib.use('Agg')
                        try:
                            figs.append((dam.plot_map(), f"Dam {d_id} Map"))
                        finally:
                            matplotlib.use(original_backend)

                    if params["chk_wsp"]:
                        figs.append((dam.plot_water_surface(), f"Dam {d_id} WSE"))

                    if params["chk_fdc"]:
                        for xs in dam.cross_sections[1:]:
                            figs.append((xs.create_combined_fdc(), f"Dam {d_id} FDC {xs.index}"))

                except Exception as e:
                    print(f"Skipping plots for Dam {d_id} due to error: {e}")

        utils.get_root().after(0, carousel.load_figures, figs)

    except Exception as e:
        utils.set_status(f"Error displaying: {e}")
    finally:
        display_btn.config(state=tk.NORMAL)
        utils.set_status("Figure generation complete.")

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
