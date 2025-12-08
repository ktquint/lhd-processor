import os
import ast
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hsclient import HydroShare
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Main Application Imports ---
# Import helper modules from your project
try:
    # NOTE: These have been changed to relative imports (starting with '.')
    # This is required because the file is now part of the 'lhd_processor' package.
    # noinspection PyUnresolvedReferences
    from .prep.dam import Dam as PrepDam
    # noinspection PyUnresolvedReferences
    from .prep import create_json as cj
    # noinspection PyUnresolvedReferences
    from .analysis.classes import Dam as AnalysisDam
except ImportError as e:
    messagebox.showerror("Module Error",
                         f"Could not import a required module: {e}\n\n"
                         "Please ensure you are running the package as a module: "
                         "python -m lhd_processor")
    exit()

# Import rathcelon (must be installed in your environment)
try:
    # noinspection PyUnresolvedReferences
    from rathcelon.classes import Dam as RathcelonDam
except ImportError:
    messagebox.showerror("Import Error",
                         "Could not find 'rathcelon' package. \n"
                         "Please ensure it is installed in your Python environment.")
    exit()


# -------------------------------------------------------------------
# GLOBAL VARIABLES (Set to None and initialized in main())
# -------------------------------------------------------------------

# Tkinter root/app instance
root = None

# Variables for controlling GUI state and input
status_var = None
prep_project_entry = None
prep_database_entry = None
prep_dem_entry = None
prep_strm_entry = None
prep_results_entry = None
prep_json_entry = None
prep_flowline_var = None
prep_dd_var = None
prep_streamflow_var = None
prep_baseflow_var = None
prep_run_button = None
rathcelon_json_entry = None
rathcelon_run_button = None

analysis_database_entry = None
analysis_results_entry = None
analysis_model_var = None
analysis_run_button = None
analysis_dam_dropdown = None
analysis_display_cross_section = None
analysis_display_rating_curves = None
analysis_display_map = None
analysis_display_wsp = None
analysis_display_fdc = None
analysis_display_bar_chart = None
analysis_display_button = None
analysis_figure_viewer_frame = None
analysis_figure_canvas_frame = None
analysis_figure_label_var = None
prev_button = None
next_button = None

# --- Global variables for figure carousel (also initialized in main) ---
current_figure_list = []
current_figure_index = 0
current_figure_canvas = None


"""
===================================================================

           TAB 1: PREPARATION & PROCESSING FUNCTIONS

===================================================================
"""


# noinspection PyUnresolvedReferences
def select_prep_project_dir():
    """Selects the main project directory and auto-populates all paths for Tab 1."""
    project_path = filedialog.askdirectory()
    if not project_path:
        return  # User canceled

    prep_project_entry.delete(0, tk.END)
    prep_project_entry.insert(0, project_path)

    try:
        # Find the .xlsx database
        csv_files = [f for f in os.listdir(project_path) if f.endswith('.csv')]
        if not csv_files:
            messagebox.showwarning("No Database", f"No .csv database file found in:\n{project_path}")
            return

        database_path = os.path.join(project_path, csv_files[0])
        prep_database_entry.delete(0, tk.END)
        prep_database_entry.insert(0, database_path)

        # Auto-populate other paths
        results_path = os.path.join(project_path, "Results")

        prep_dem_entry.delete(0, tk.END)
        prep_dem_entry.insert(0, os.path.join(project_path, "DEM"))  # Changed to dedicated DEM folder

        prep_strm_entry.delete(0, tk.END)
        prep_strm_entry.insert(0, os.path.join(project_path, "STRM"))

        prep_results_entry.delete(0, tk.END)
        prep_results_entry.insert(0, results_path)

        # Auto-populate both the creation path and the run path for the JSON
        json_path = os.path.splitext(database_path)[0] + '.json'
        prep_json_entry.delete(0, tk.END)
        prep_json_entry.insert(0, json_path)
        rathcelon_json_entry.delete(0, tk.END)
        rathcelon_json_entry.insert(0, json_path)

        # Also populate analysis tab paths
        analysis_database_entry.delete(0, tk.END)
        analysis_database_entry.insert(0, database_path)
        analysis_results_entry.delete(0, tk.END)
        analysis_results_entry.insert(0, results_path)
        update_analysis_dropdown()  # Update analysis dropdown

        status_var.set("Project paths loaded.")

    except IndexError:
        messagebox.showerror("Error", f"No .csv database file found in:\n{project_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load project paths: {e}")

# noinspection PyUnresolvedReferences
def select_prep_database_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        prep_database_entry.delete(0, tk.END)
        prep_database_entry.insert(0, file_path)

        prep_json_entry.delete(0, tk.END)
        prep_json_entry.insert(0, os.path.splitext(file_path)[0] + '.json')

# noinspection PyUnresolvedReferences
def select_prep_dem_dir():
    dem_path = filedialog.askdirectory()
    if dem_path:
        prep_dem_entry.delete(0, tk.END)
        prep_dem_entry.insert(0, dem_path)

# noinspection PyUnresolvedReferences
def select_prep_strm_dir():
    strm_path = filedialog.askdirectory()
    if strm_path:
        prep_strm_entry.delete(0, tk.END)
        prep_strm_entry.insert(0, strm_path)

# noinspection PyUnresolvedReferences
def select_prep_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        prep_results_entry.delete(0, tk.END)
        prep_results_entry.insert(0, results_path)

# noinspection PyUnresolvedReferences
def select_prep_json_file():
    json_path = filedialog.asksaveasfilename(filetypes=[("JSON files", "*.json")], defaultextension=".json")
    if json_path:
        prep_json_entry.delete(0, tk.END)
        prep_json_entry.insert(0, json_path)

# noinspection PyUnresolvedReferences
def select_rathcelon_json_file():
    """NEW function to select the JSON file for the 'Run' step."""
    json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if json_path:
        rathcelon_json_entry.delete(0, tk.END)
        rathcelon_json_entry.insert(0, json_path)

# noinspection PyUnresolvedReferences
def threaded_prepare_data():
    """
    This function now prepares data and creates the input files.
    """
    try:
        # --- 1. Get all values from GUI ---
        lhd_csv = prep_database_entry.get()
        flowline_source = prep_flowline_var.get()
        dem_resolution = prep_dd_var.get()
        streamflow_source = prep_streamflow_var.get()
        dem_folder = prep_dem_entry.get()
        strm_folder = prep_strm_entry.get()
        results_folder = prep_results_entry.get()
        baseflow_method = prep_baseflow_var.get()

        # --- 2. Validate inputs ---
        if not os.path.exists(lhd_csv):
            messagebox.showerror("Error", f"Database file not found:\n{lhd_csv}")
            return
        if not all(
                [lhd_csv, flowline_source, dem_resolution, streamflow_source, dem_folder, strm_folder, results_folder]):
            messagebox.showwarning("Missing Info", "Please fill out all path and setting fields.")
            return

        status_var.set("Inputs validated. Starting data prep...")

        # --- 3. Create directories ---
        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        lhd_df = pd.read_csv(lhd_csv)
        final_df = lhd_df.copy()

        # We just use the first row as an example to get the attribute names
        try:
            sample_dam_dict = PrepDam(**lhd_df.iloc[0].to_dict()).__dict__
            cols_to_update = [key for key in sample_dam_dict.keys() if key in final_df.columns]

            # Change the type of these columns to 'object' in the DataFrame
            for col in cols_to_update:
                if final_df[col].dtype != 'object':
                    final_df[col] = final_df[col].astype(object)

            status_var.set("DataFrame types prepared.")
        except Exception as e:
            print(f"Warning: Could not pre-set DataFrame dtypes: {e}")

        # kenny's hydroshare ID
        hydroshare_id = "88759266f9c74df8b5bb5f52d142ba8e"

        # --- 4. Load NWM or GEOGLOWS data once --- #
        nwm_ds = None
        # Get the absolute path to the directory containing __main__.py (which is now in lhd_processor/)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        # Join that directory path with the relative path to your file
        nwm_parquet = os.path.join(data_dir, 'nwm_v3_daily_retrospective.parquet')
        # if we want the GEOGLOWS streamflow, we'll need to also download the GEOGLOWS flowlines
        vpu_filename = "vpu-boundaries.gpkg"
        tdx_vpu_map = os.path.join(data_dir, vpu_filename)

        if streamflow_source == 'National Water Model':
            # --- check if nwm parquet file exists --- #
            if not os.path.exists(nwm_parquet):
                try:
                    messagebox.showinfo("Downloading Data",
                                        "The NWM Parquet file is missing.\n"
                                        "The program will now download it from HydroShare.\n"
                                        "This may take a moment.")

                    # download the parquet
                    hs = HydroShare()
                    resource = hs.resource(hydroshare_id)
                    resource.file_download(path='nwm_v3_daily_retrospective.parquet',
                                           save_path=data_dir)
                    status_var.set("NWM Parquet download complete.")

                except Exception as e:
                    messagebox.showerror("Download Failed",
                                         f"Failed to automatically download the NWM Parquet file.\n\n"
                                         f"Error: {e}\n\n"
                                         "Please download the file manually and place it in the 'lhd_processor/data' folder.")
                    status_var.set("ERROR: NWM parquet file download failed.")

            status_var.set("Reading NWM Parquet...")
            # --- Try to load the dataset (if download didn't fail) ---
            if nwm_ds is None:  # Will be None if file existed OR download succeeded
                try:
                    status_var.set("Loading NWM dataset...")
                    nwm_df = pd.read_parquet(nwm_parquet)
                    nwm_ds = nwm_df.to_xarray()
                    status_var.set("NWM dataset loaded.")
                except FileNotFoundError:
                    print(f"ERROR: NWM Parquet file not found at {nwm_parquet}")
                    status_var.set("ERROR: NWM parquet file not found.")
                    nwm_ds = None
                except Exception as e:
                    print(f"Could not open NWM dataset. Error: {e}")
                    status_var.set("Error: Could not load NWM dataset.")
                    nwm_ds = None

        if streamflow_source == 'GEOGLOWS' or flowline_source == 'GEOGLOWS':
            if not os.path.exists(tdx_vpu_map):
                # File is missing, let's download it from HydroShare
                try:
                    # Inform the user this is a one-time download
                    messagebox.showinfo("Downloading Flowlines...",
                                        "The GEOGLOWS VPU map file is missing.\n"
                                        "The program will now download it from HydroShare.\n")
                    status_var.set("Downloading vpu-boundaries.gpkg...")

                    # Ensure the 'data' directory exists before saving to it
                    os.makedirs(data_dir, exist_ok=True)

                    # Download the file
                    hs = HydroShare()
                    resource = hs.resource(hydroshare_id)

                    # Pass the *directory* to save_path, not the full file path
                    resource.file_download(path=vpu_filename,
                                           save_path=data_dir)

                    status_var.set("VPU map download complete.")

                except Exception as e:
                    messagebox.showerror("Download Failed",
                                         f"Failed to automatically download the VPU map file \
                                         from: HydroShare\n\n"
                                         f"Error: {e}\n\n"
                                         "Please download the file manually and place it in the 'data' folder "
                                         "of your project, then try again.")

        # --- 5. Main Processing Loop ---
        total_dams = len(lhd_df)
        processed_dams_count = 0

        for i, row in lhd_df.iterrows():
            i = int(str(i))
            dam_id = row.get("ID", f"Row_{i + 2}")
            dam_id = int(dam_id)

            try:
                status_var.set(f"Prep: Dam {dam_id} ({i + 1} of {total_dams})...")

                dam = PrepDam(**row.to_dict())
                dam.set_streamflow_source(streamflow_source)
                dam.set_flowline_source(flowline_source)

                status_var.set(f"Dam {dam_id}: Assigning flowlines...")

                # 'strm_folder' is the LHD_STRMs path (where NHD/VPU downloads go)
                # 'tdx_vpu_map_path' is the path to our map file (it either existed or was just downloaded)
                dam.assign_flowlines(strm_folder, tdx_vpu_map)

                status_var.set(f"Dam {dam_id}: Assigning DEM...")
                dam.assign_dem(dem_folder, dem_resolution)

                if dam.dem_1m is None and dam.dem_3m is None and dam.dem_10m is None:
                    print(f"Skipping Dam No. {dam_id}: DEM assignment failed.")
                    status_var.set(f"Dam {dam_id}: DEM failed. Skipping.")
                    continue

                dam.set_output_dir(results_folder)

                needs_reach = False
                if streamflow_source == 'National Water Model':
                    if pd.isna(row.get('dem_baseflow_NWM')) or pd.isna(row.get('fatality_flows_NWM')):
                        needs_reach = True
                elif streamflow_source == 'GEOGLOWS':
                    if pd.isna(row.get('dem_baseflow_GEOGLOWS')) or pd.isna(row.get('fatality_flows_GEOGLOWS')):
                        needs_reach = True

                if needs_reach:
                    if streamflow_source == 'National Water Model' and nwm_ds is None:
                        print(f"Skipping flow estimation for Dam No. {dam_id}: NWM dataset not loaded.")
                    else:
                        status_var.set(f"Dam {dam_id}: Creating stream reach...")
                        dam.create_reach(nwm_ds, dam.flowline_TDX)

                        status_var.set(f"Dam {dam_id}: Estimating baseflow...")
                        dam.set_dem_baseflow(baseflow_method)

                        status_var.set(f"Dam {dam_id}: Estimating fatal flows...")
                        dam.set_fatal_flows()

                for key, value in dam.__dict__.items():
                    # Check if the value is a list or numpy array
                    if isinstance(value, (list, np.ndarray)):
                        # Convert it to a string representation before saving
                        final_df.loc[i, key] = str(value)
                    else:
                        # Otherwise, assign it directly
                        final_df.loc[i, key] = value

                processed_dams_count += 1
                print(f'Finished Prep for Dam No. {dam_id}')

            except Exception as e:
                print(f"---" * 20)
                print(f"CRITICAL ERROR preparing Dam No. {dam_id}: {e}")
                print(f"Skipping this dam and moving to the next one.")
                print(f"---" * 20)
                status_var.set(f"Error on Dam {dam_id}: {e}. Skipping.")
                continue

        # --- 6. Final Output Generation (CSV/Excel/JSON) ---
        if processed_dams_count > 0:
            status_var.set("Saving updated database file...")
            final_df.to_csv(lhd_csv, index=False)

            status_var.set("Creating rathcelon_input.json...")
            json_loc = prep_json_entry.get()

            # This function now also returns the dam dictionaries
            cj.rathcelon_input(lhd_csv, json_loc, baseflow_method, nwm_parquet)

            status_var.set(f"Data preparation complete. {processed_dams_count} dams prepped.")
            messagebox.showinfo("Success", f"Data preparation complete.\n{processed_dams_count} dams processed.\n"
                                           f"Input file created at:\n{json_loc}")
        else:
            status_var.set("No dams were pre-processed successfully.")
            messagebox.showwarning("Process Finished", "No new dam data was pre-processed successfully.")

    except Exception as e:
        status_var.set(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"The process failed:\n{e}")
    finally:
        # --- 7. Re-enable button ---
        prep_run_button.config(state=tk.NORMAL)

# noinspection PyUnresolvedReferences
def threaded_run_rathcelon():
    """
    This new function ONLY runs the Rathcelon analysis using a selected JSON file.
    """
    try:
        # --- 1. Get JSON file path ---
        json_loc = rathcelon_json_entry.get()
        if not os.path.exists(json_loc):
            messagebox.showerror("Error", f"RathCelon input file not found:\n{json_loc}")
            return

        status_var.set(f"Loading input file: {os.path.basename(json_loc)}...")

        # --- 2. Load and parse JSON ---
        with open(json_loc, 'r') as f:
            data = json.load(f)

        dam_dictionaries = data.get("dams", [])
        if not dam_dictionaries:
            status_var.set("No dams found in the selected JSON file.")
            messagebox.showwarning("Empty File", "No dams to process in the selected JSON file.")
            return

        # --- 3. RUN RATHCELON PROCESS ---
        status_var.set(f"Starting RathCelon analysis for {len(dam_dictionaries)} dams...")
        rathcelon_success_count = 0
        total_dams = len(dam_dictionaries)

        for i, dam_dict in enumerate(dam_dictionaries):
            dam_name = dam_dict.get('name', f"Dam {i + 1}")
            try:
                status_var.set(f"RathCelon: Processing Dam {dam_name} ({i + 1} of {total_dams})...")

                dam_i = RathcelonDam(**dam_dict)
                dam_i.process_dam()

                rathcelon_success_count += 1
            except Exception as e:
                print(f"RathCelon failed for dam {dam_name}: {e}")
                status_var.set(f"RathCelon Error on Dam {dam_name}. Skipping.")
                # Continue to the next dam

        status_var.set(f"RathCelon process complete. {rathcelon_success_count} dams processed.")
        messagebox.showinfo("Success", f"RathCelon process complete.\n{rathcelon_success_count} dams processed.")

    except Exception as e:
        status_var.set(f"Fatal error during Rathcelon run: {e}")
        messagebox.showerror("Fatal Error", f"The RathCelon process failed:\n{e}")
    finally:
        # --- 4. Re-enable button ---
        rathcelon_run_button.config(state=tk.NORMAL)

# noinspection PyUnresolvedReferences
def start_prep_thread():
    """Triggers the preparation thread."""
    prep_run_button.config(state=tk.DISABLED)
    status_var.set("Starting data preparation...")
    threading.Thread(target=threaded_prepare_data, daemon=True).start()

# noinspection PyUnresolvedReferences
def start_rathcelon_run_thread():
    """Triggers the Rathcelon analysis thread."""
    rathcelon_run_button.config(state=tk.DISABLED)
    status_var.set("Starting RathCelon analysis...")
    threading.Thread(target=threaded_run_rathcelon, daemon=True).start()


"""
===================================================================

            TAB 2: ANALYSIS & VISUALIZATION FUNCTIONS

===================================================================
"""


def analysis_successful_runs(results_dir, database_csv):
    """
    Finds all dams in the results_dir that have valid output,
    but ONLY checks for dams listed in the database_csv.
    """
    # --- 1. Get dams from CSV database ---
    try:
        lhd_df = pd.read_csv(database_csv)
        if 'ID' not in lhd_df.columns:
            print("Analysis tab: Database CSV must have an 'ID' column.")
            return []
        # Get a clean set of 'ID's as strings
        dam_nos_from_csv = set(pd.to_numeric(lhd_df['ID'], errors='coerce').dropna().astype(int).astype(str).tolist())
    except Exception as e:
        print(f"Error reading database CSV {database_csv}: {e}")
        return []

    if not dam_nos_from_csv:
        print("No dams found in the database CSV.")
        return []

    # --- 2. Check which of those dams have successful runs ---
    successes = []
    for lhd_id in dam_nos_from_csv:
        run_results_dir = os.path.join(results_dir, lhd_id)

        # Skip if the corresponding folder doesn't even exist
        if not os.path.isdir(run_results_dir):
            continue

        # Check for the specific success file
        local_vdt_gpkg = os.path.join(str(run_results_dir), "VDT", f"{lhd_id}_Local_VDT_Database.gpkg")
        if not os.path.exists(local_vdt_gpkg):
            continue

        # Finally, check if the file is valid and not empty
        try:
            local_vdt_gdf = gpd.read_file(local_vdt_gpkg)
            if not local_vdt_gdf.empty:
                successes.append(lhd_id)  # Add the string ID
        except Exception as e:
            print(f"Error reading GPKG for {lhd_id}: {e}")
            continue

    return successes

# noinspection PyUnresolvedReferences
def update_analysis_dropdown():
    """Updates the dam selection dropdown on the Analysis tab."""
    results_dir = analysis_results_entry.get()
    database_csv = analysis_database_entry.get()

    # --- 1. Validate paths ---
    if not os.path.isdir(results_dir):
        analysis_dam_dropdown['values'] = []
        analysis_dam_dropdown.set("Invalid results path")
        status_var.set("Invalid results path. Please select a valid folder.")
        return

    if not os.path.isfile(database_csv):
        analysis_dam_dropdown['values'] = []
        analysis_dam_dropdown.set("Invalid database path")
        status_var.set("Invalid database .csv. Please select a valid file.")
        return

    try:
        # --- 2. Get successfully run dams that are in the CSV ---
        # This call is now much more efficient
        dam_strs = analysis_successful_runs(results_dir, database_csv)

        # --- 3. Populate dropdown ---
        dams_int = []
        for d in dam_strs:
            try:
                dams_int.append(int(d))
            except ValueError:
                print(f"Analysis tab: Skipping non-numeric success ID '{d}'")

        dams_sorted = sorted(dams_int)
        dams = ["All Dams"] + [str(d) for d in dams_sorted]

        analysis_dam_dropdown['values'] = dams
        if dams:
            analysis_dam_dropdown.set(dams[0])

        status_var.set(f"Found {len(dams) - 1} processed dams from database for analysis.")

    except Exception as e:
        analysis_dam_dropdown['values'] = []
        analysis_dam_dropdown.set("Error")
        status_var.set(f"Error updating dropdown: {e}")

# noinspection PyUnresolvedReferences
def select_analysis_csv_file():
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        analysis_database_entry.delete(0, tk.END)
        analysis_database_entry.insert(0, file_path)

# noinspection PyUnresolvedReferences
def select_analysis_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        analysis_results_entry.delete(0, tk.END)
        analysis_results_entry.insert(0, results_path)
        update_analysis_dropdown()  # Update dropdown on folder select


# --- Figure Carousel Functions ---

# noinspection PyUnresolvedReferences
def clear_figure_carousel():
    """Hides the figure viewer and clears its contents."""
    global current_figure_list, current_figure_index, current_figure_canvas
    current_figure_list = []
    current_figure_index = 0

    # Destroy the canvas widget if it exists
    if current_figure_canvas:
        current_figure_canvas.get_tk_widget().destroy()
        current_figure_canvas = None

    # Hide the main viewer frame
    analysis_figure_viewer_frame.pack_forget()

# noinspection PyUnresolvedReferences
def display_figure(index):
    """Displays the figure at the given index in the carousel."""
    global current_figure_list, current_figure_index, current_figure_canvas

    if not current_figure_list:
        clear_figure_carousel()
        return

    # Ensure index is valid
    index = max(0, min(index, len(current_figure_list) - 1))
    current_figure_index = index

    # Clear old canvas
    if current_figure_canvas:
        current_figure_canvas.get_tk_widget().destroy()

    # Get new figure and title
    fig, title = current_figure_list[index]

    # Create new canvas
    current_figure_canvas = FigureCanvasTkAgg(fig, master=analysis_figure_canvas_frame)
    current_figure_canvas.draw()
    current_figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Update label and buttons
    analysis_figure_label_var.set(f"{title} ({index + 1} of {len(current_figure_list)})")
    prev_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if index < len(current_figure_list) - 1 else tk.DISABLED)


def on_prev_figure():
    display_figure(current_figure_index - 1)


def on_next_figure():
    display_figure(current_figure_index + 1)

# noinspection PyUnresolvedReferences
def setup_figure_carousel(figures_list):
    """Called by the thread to populate and show the figure carousel."""
    global current_figure_list, current_figure_index

    clear_figure_carousel()  # Clear any previous figures

    if figures_list:
        current_figure_list = figures_list
        current_figure_index = 0
        analysis_figure_viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)  # Show the viewer
        display_figure(0)  # Show the first figure
    else:
        status_var.set("Analysis complete. No figures were selected for display.")


# --- Core Logic Functions (for Tab 2) ---
# noinspection PyUnresolvedReferences
def threaded_process_ARC():
    """
    Runs the analysis for ALL dams in a thread.
    This function ONLY saves data and figures, it does not display them.
    """
    try:
        status_var.set("Starting data processing for ALL dams...")
        analysis_run_button.config(state=tk.DISABLED)

        database_csv = analysis_database_entry.get()
        results_dir = analysis_results_entry.get()
        selected_model = analysis_model_var.get()
        estimate_dam = True  # <-- Always estimate dam height

        if not os.path.exists(database_csv):
            messagebox.showerror("Error", f"Database file not found:\n{database_csv}")
            return
        if not os.path.isdir(results_dir):
            messagebox.showerror("Error", f"Results directory not found:\n{results_dir}")
            return

        # --- This function now ONLY runs for ALL DAMS ---
        dam_strs = analysis_successful_runs(results_dir, database_csv)

        dams_int = []
        for d in dam_strs:
            try:
                dams_int.append(int(d))
            except ValueError:
                pass  # Skip non-numeric folders

        dam_ints = sorted(dams_int)
        total_dams = len(dam_ints)
        if total_dams == 0:
            status_var.set("No processed dams found in the database to analyze.")
            messagebox.showwarning("No Dams", "No processed dams found in the database to analyze.")
            return

        for i, dam_id in enumerate(dam_ints):
            try:
                status_var.set(f"Analyzing Dam {dam_id} ({i + 1} of {total_dams})...")
                dam_i = AnalysisDam(int(dam_id), database_csv, selected_model, estimate_dam, results_dir)

                # When "All Dams" is selected, just save figs
                for xs in dam_i.cross_sections:
                    plt.close(xs.plot_cross_section())  # Plot and close to save
                for xs in dam_i.cross_sections[1:]:
                    plt.close(xs.create_combined_fig())  # Plot and close to save
                    plt.close(xs.create_combined_fdc())  # Plot and close to save
                plt.close(dam_i.plot_map())  # Plot and close to save
                plt.close(dam_i.plot_water_surface())  # Plot and close to save

            except ValueError as e:
                if "Invalid flow conditions" in str(e):
                    print(f"---" * 20)
                    print(f"SKIPPING Dam {dam_id}: {e}")
                    print(f"---" * 20)
                    status_var.set(f"Skipping Dam {dam_id} (Invalid flow)")
                    continue  # Move to the next dam
                else:
                    print(f"---" * 20)
                    print(f"CRITICAL ValueError on Dam {dam_id}: {e}")
                    print(f"---" * 20)
                    status_var.set(f"Error on Dam {dam_id}. Skipping.")
                    continue
            except Exception as e:
                print(f"---" * 20)
                print(f"CRITICAL FAILED processing Dam {dam_id}: {e}")
                print(f"Skipping this dam and moving to the next one.")
                print(f"---" * 20)
                status_var.set(f"Error on Dam {dam_id}. Skipping.")
                continue

        status_var.set("Analysis processing complete.")
        messagebox.showinfo("Success", f"Finished processing data for All Dams.")

    except Exception as e:
        status_var.set(f"Error during analysis: {e}")
        messagebox.showerror("Processing Error", f"An error occurred:\n{e}")
    finally:
        analysis_run_button.config(state=tk.NORMAL)

# noinspection PyUnresolvedReferences
def generate_summary_charts(lhd_df_path):
    """
    Generates the summary bar chart figures from the database.
    This is extracted from the original threaded_plot_shj.
    Returns a list of (Figure, title) tuples.
    """
    figures_list = []
    try:
        lhd_df = pd.read_csv(lhd_df_path)
    except Exception as e:
        print(f"Error reading database for summary chart: {e}")
        status_var.set(f"Error reading database for summary chart: {e}")
        return []

    plot_generated = False
    for i in range(1, 5):
        print(f"Processing Summary for Cross-Section {i}...")
        cols_to_check = [f'y_t_{i}', f'y_flip_{i}', f'y_2_{i}', f's_{i}']
        filtered_df = lhd_df.dropna(subset=cols_to_check).copy()

        if filtered_df.empty:
            print(f"No data available for cross-section {i}. Skipping plot.")
            continue

        def safe_literal_eval(item):
            try:
                if pd.notna(item) and isinstance(item, str) and item.strip().startswith('['):
                    return ast.literal_eval(item)
            except (ValueError, SyntaxError):
                return []
            return []

        y_t_strs = filtered_df[f'y_t_{i}'].to_list()
        y_flip_strs = filtered_df[f'y_flip_{i}'].to_list()
        y_2_strs = filtered_df[f'y_2_{i}'].to_list()
        slopes = filtered_df[f's_{i}'].to_list()
        dam_ids = filtered_df['ID'].tolist()

        y_t_list = [num for item in y_t_strs for num in safe_literal_eval(item)]
        y_flip_list = [num for item in y_flip_strs for num in safe_literal_eval(item)]
        y_2_list = [num for item in y_2_strs for num in safe_literal_eval(item)]
        nested_list = [safe_literal_eval(item) for item in y_t_strs]

        if not any(nested_list):
            print(f"All rows for cross-section {i} contained empty lists. Skipping plot.")
            continue

        expanded_slopes = [val for val, group in zip(slopes, nested_list) for _ in range(len(group))]
        expanded_ids = [val for val, group in zip(dam_ids, nested_list) for _ in range(len(group))]

        df = pd.DataFrame({
            'slope': expanded_slopes,
            'y_t': y_t_list,
            'y_flip': y_flip_list,
            'y_2': y_2_list,
            'dam_id': expanded_ids
        })

        if df.empty:
            print(f"DataFrame is empty for cross-section {i} after processing. Skipping plot.")
            continue

        df = df.sort_values(['dam_id', 'slope']).reset_index(drop=True)
        x_vals = np.arange(len(df))
        slope = df['slope']
        conjugate = df['y_2'] * 3.281
        flip = df['y_flip'] * 3.281
        tailwater = df['y_t'] * 3.281
        x_labels = slope.round(6).astype(str)

        fig = Figure(figsize=(11, 5))  # Use Figure instead of plt.subplots
        ax = fig.add_subplot(111)
        cap_width = 0.2

        for x, y, y2, y_flip in zip(x_vals, tailwater, conjugate, flip):
            if y2 < y < y_flip:
                c = 'red'
            elif y >= y_flip:
                c = 'green'
            else:
                c = 'blue'
            ax.vlines(x, y2, y_flip, color='black', linewidth=1)
            ax.hlines(y2, x - cap_width, x + cap_width, color='black', linewidth=1)
            ax.hlines(y_flip, x - cap_width, x + cap_width, color='black', linewidth=1)
            ax.scatter(x, y, color=c, marker='x', zorder=3)

        current_id = None
        start_idx = 0
        shade = True
        for idx, dam in enumerate(df['dam_id']):
            if dam != current_id:
                if current_id is not None and shade:
                    ax.axvspan(start_idx - 0.5, idx - 0.5, color='gray', alpha=0.1)
                current_id = dam
                start_idx = idx
                shade = not shade
        if shade and len(df) > 0:
            ax.axvspan(start_idx - 0.5, len(df) - 0.5, color='gray', alpha=0.1)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel('Slope')
        ax.set_ylabel('Depth (ft)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_title(f"Summary of Results from Cross-Section No. {i}")
        fig.tight_layout()

        figures_list.append((fig, f"Summary of Results from Cross-Section No. {i}"))
        plot_generated = True

    if not plot_generated:
        print("No data found to generate summary plots.")

    return figures_list

# noinspection PyUnresolvedReferences
def threaded_display_dam_figures():
    """
    NEW Function.
    Generates and displays figures for a SINGLE selected dam in the carousel
    AND/OR the summary bar chart.
    """
    figures_to_display = []
    try:
        status_var.set("Generating figures for display...")
        analysis_display_button.config(state=tk.DISABLED)

        database_csv = analysis_database_entry.get()
        results_dir = analysis_results_entry.get()
        selected_model = analysis_model_var.get()
        estimate_dam = True  # <-- Always estimate dam height
        selected_dam = analysis_dam_dropdown.get()

        if not os.path.exists(database_csv) or not os.path.isdir(results_dir):
            messagebox.showerror("Error", "Database or Results path is invalid.")
            return

        # --- 1. Generate Bar Chart if selected ---
        if analysis_display_bar_chart.get():
            status_var.set("Generating summary bar charts...")
            bar_chart_figures = generate_summary_charts(database_csv)
            figures_to_display.extend(bar_chart_figures)
            status_var.set(f"Generated {len(bar_chart_figures)} summary charts.")

        # --- 2. Check if dam-specific figures are requested ---
        dam_specific_figs_requested = (
                analysis_display_cross_section.get() or
                analysis_display_rating_curves.get() or
                analysis_display_map.get() or
                analysis_display_wsp.get() or
                analysis_display_fdc.get()
        )

        # --- 3. Handle dam-specific figures ---
        if dam_specific_figs_requested:
            if selected_dam == "All Dams":
                # Requested dam figs but didn't select a dam
                if not figures_to_display:  # i.e., bar chart NOT selected
                    messagebox.showinfo("Select Dam",
                                        "Please select a single dam from the dropdown to display dam-specific figures.")
                else:  # Bar chart WAS selected
                    status_var.set("Showing summary charts. Select a single dam to see other figures.")

                # Show what we have (bar charts or nothing) and stop
                root.after(0, setup_figure_carousel, figures_to_display)
                return

            # --- If we are here, a single dam IS selected, and figs are requested ---
            status_var.set(f"Loading Dam {selected_dam} for display...")
            dam_i = AnalysisDam(int(selected_dam), database_csv, selected_model, estimate_dam, results_dir)

            if analysis_display_cross_section.get():
                for xs in dam_i.cross_sections:
                    fig = xs.plot_cross_section()
                    title = f"Cross Section {xs.index} (Dam {dam_i.id})"
                    figures_to_display.append((fig, title))

            if analysis_display_rating_curves.get():
                for xs in dam_i.cross_sections[1:]:
                    fig = xs.create_combined_fig()
                    title = f"Rating Curve {xs.index} (Dam {dam_i.id})"
                    figures_to_display.append((fig, title))

            if analysis_display_map.get():
                fig = dam_i.plot_map()
                title = f"Dam Location (Dam {dam_i.id})"
                figures_to_display.append((fig, title))

            if analysis_display_wsp.get():
                fig = dam_i.plot_water_surface()
                title = f"Water Surface Profile (Dam {dam_i.id})"
                figures_to_display.append((fig, title))

            if analysis_display_fdc.get():
                for xs in dam_i.cross_sections[1:]:
                    fig = xs.create_combined_fdc()
                    title = f"Flow Duration Curve {xs.index} (Dam {dam_i.id})"
                    figures_to_display.append((fig, title))

        # --- 4. Final Display ---
        if not figures_to_display:
            status_var.set("No figures selected or generated.")
        else:
            status_var.set(f"Generated {len(figures_to_display)} total figures.")

        root.after(0, setup_figure_carousel, figures_to_display)

    except Exception as e:
        status_var.set(f"Error generating figures: {e}")
        messagebox.showerror("Figure Generation Error", f"An error occurred while generating figures:\n{e}")
        # noinspection PyTypeChecker
        root.after(0, clear_figure_carousel)  # Clear carousel on error
    finally:
        analysis_display_button.config(state=tk.NORMAL)

# noinspection PyUnresolvedReferences
def start_analysis_processing():
    """Triggers the analysis processing thread."""
    clear_figure_carousel()  # Clear any old figures
    analysis_run_button.config(state=tk.DISABLED)
    status_var.set("Starting analysis processing...")
    threading.Thread(target=threaded_process_ARC, daemon=True).start()

# noinspection PyUnresolvedReferences
def start_display_dam_figures_thread():
    """Triggers the NEW display figures thread."""
    clear_figure_carousel()
    analysis_display_button.config(state=tk.DISABLED)
    status_var.set("Starting to generate dam figures...")
    threading.Thread(target=threaded_display_dam_figures, daemon=True).start()


def main():
    """
    Main function to set up and run the Tkinter GUI application.
    This function is called when the script is run as a module (python -m lhd_processor).
    """
    # -------------------------------------------------------------------
    # GLOBAL VARIABLE DEFINITIONS (MUST BE HERE TO BE ACCESSIBLE BY ABOVE FUNCTIONS)
    # -------------------------------------------------------------------
    global root, status_var
    global prep_project_entry, prep_database_entry, prep_dem_entry, prep_strm_entry, prep_results_entry, prep_json_entry
    global prep_flowline_var, prep_dd_var, prep_streamflow_var, prep_baseflow_var, prep_run_button
    global rathcelon_json_entry, rathcelon_run_button
    global analysis_database_entry, analysis_results_entry, analysis_model_var, analysis_run_button, analysis_dam_dropdown
    global analysis_display_cross_section, analysis_display_rating_curves, analysis_display_map, analysis_display_wsp
    global analysis_display_fdc, analysis_display_bar_chart, analysis_display_button
    global analysis_figure_viewer_frame, analysis_figure_canvas_frame, analysis_figure_label_var
    global prev_button, next_button
    global current_figure_list, current_figure_index, current_figure_canvas

    root = tk.Tk()
    root.title("LHD Control Center")
    root.geometry("700x1000")

    # --- Style ---
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    style.configure("TNotebook.Tab", font=("Arial", 10, "bold"))

    # --- Notebook (Tabs) ---
    notebook = ttk.Notebook(root)
    prep_tab = ttk.Frame(notebook)
    analysis_tab = ttk.Frame(notebook)

    notebook.add(prep_tab, text="  Preparation & Processing  ")
    notebook.add(analysis_tab, text="  Analysis & Visualization  ")
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    """
    ===================================================================
                    --- GUI: PREPARATION TAB ---
    ===================================================================
    """

    # --- Frame for Step 1: Data Preparation ---
    prep_data_frame = ttk.LabelFrame(prep_tab, text="Step 1: Prepare Data and Create Input File")
    prep_data_frame.pack(pady=10, padx=10, fill="x")

    # --- Unified Paths Frame (Holds ALL file inputs) ---
    # We create one frame to hold all 6 rows so spacing is identical
    prep_paths_frame = ttk.Frame(prep_data_frame)
    prep_paths_frame.pack(pady=5, padx=10, fill="x")
    prep_paths_frame.columnconfigure(1, weight=1)

    # 1. Project Folder
    ttk.Button(prep_paths_frame, text="Select Project Folder", command=select_prep_project_dir).grid(row=0, column=0,
                                                                                                     padx=5, pady=5,
                                                                                                     sticky=tk.W)
    prep_project_entry = ttk.Entry(prep_paths_frame)
    prep_project_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    # 2. Database File
    ttk.Button(prep_paths_frame, text="Select Database File (.csv)", command=select_prep_database_file).grid(row=1,
                                                                                                             column=0,
                                                                                                             padx=5,
                                                                                                             pady=5,
                                                                                                             sticky=tk.W)
    prep_database_entry = ttk.Entry(prep_paths_frame)
    prep_database_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

    # 3. DEM Folder
    ttk.Button(prep_paths_frame, text="Select DEM Folder", command=select_prep_dem_dir).grid(row=2, column=0, padx=5,
                                                                                             pady=5, sticky=tk.W)
    prep_dem_entry = ttk.Entry(prep_paths_frame)
    prep_dem_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

    # 4. Hydrography Folder
    ttk.Button(prep_paths_frame, text="Select Hydrography Folder", command=select_prep_strm_dir).grid(row=3, column=0,
                                                                                                      padx=5, pady=5,
                                                                                                      sticky=tk.W)
    prep_strm_entry = ttk.Entry(prep_paths_frame)
    prep_strm_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

    # 5. Results Folder
    ttk.Button(prep_paths_frame, text="Select Results Folder", command=select_prep_results_dir).grid(row=4, column=0,
                                                                                                     padx=5, pady=5,
                                                                                                     sticky=tk.W)
    prep_results_entry = ttk.Entry(prep_paths_frame)
    prep_results_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)

    # 6. RathCelon Input
    ttk.Button(prep_paths_frame, text="RathCelon Input File (.json)", command=select_prep_json_file).grid(row=5,
                                                                                                          column=0,
                                                                                                          padx=5,
                                                                                                          pady=5,
                                                                                                          sticky=tk.W)
    prep_json_entry = ttk.Entry(prep_paths_frame)
    prep_json_entry.grid(row=5, column=1, padx=5, pady=5, sticky=tk.EW)

    # --- Hydraulics and Hydrology Settings ---
    prep_hydro_frame = ttk.Frame(prep_data_frame)
    prep_hydro_frame.pack(pady=5, padx=10, fill="x")
    prep_hydro_frame.columnconfigure(1, weight=1)
    ttk.Label(prep_hydro_frame, text="Flowline Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    prep_flowline_var = tk.StringVar(value="NHDPlus")
    prep_flowline_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_flowline_var, state="readonly",
                                       values=("NHDPlus", "GEOGLOWS"))
    prep_flowline_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Label(prep_hydro_frame, text="DEM Resolution:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    prep_dd_var = tk.StringVar(value="1 m")
    prep_dd_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_dd_var, state="readonly",
                                    values=("1 m", "1/9 arc-second (~3 m)", "1/3 arc-second (~10 m)"))
    prep_dd_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Label(prep_hydro_frame, text="Streamflow Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    prep_streamflow_var = tk.StringVar(value="National Water Model")
    prep_streamflow_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_streamflow_var, state="readonly",
                                      values=("National Water Model", "GEOGLOWS"))
    prep_streamflow_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Label(prep_hydro_frame, text="Baseflow Estimation:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
    prep_baseflow_var = tk.StringVar(value="WSE and LiDAR Date")
    prep_baseflow_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_baseflow_var, state="readonly", values=(
        "WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation"))
    prep_baseflow_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

    # --- Run function button (Step 1) ---
    prep_run_button = ttk.Button(prep_data_frame, text="1. Prepare Data & Create Input File", command=start_prep_thread,
                                 style="Accent.TButton")
    prep_run_button.pack(pady=10, padx=10, fill="x", ipady=5)

    # --- Frame for Step 2: Run Rathcelon ---
    run_rathcelon_frame = ttk.LabelFrame(prep_tab, text="Step 2: Run Rathcelon Processing")
    run_rathcelon_frame.pack(pady=10, padx=10, fill="x")
    run_rathcelon_frame.columnconfigure(1, weight=1)

    ttk.Button(run_rathcelon_frame, text="Select Input File (.json)", command=select_rathcelon_json_file).grid(row=0,
                                                                                                               column=0,
                                                                                                               padx=5,
                                                                                                               pady=5,
                                                                                                               sticky=tk.W)
    rathcelon_json_entry = ttk.Entry(run_rathcelon_frame)
    rathcelon_json_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    rathcelon_run_button = ttk.Button(run_rathcelon_frame, text="2. Run RathCelon",
                                      command=start_rathcelon_run_thread,
                                      style="Accent.TButton")
    rathcelon_run_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW, ipady=5)

    """
    ===================================================================
                     --- GUI: ANALYSIS TAB ---
    ===================================================================
    """

    # --- Analysis: Database and Results Paths ---
    analysis_path_frame = ttk.LabelFrame(analysis_tab, text="3. Select Results to Analyze")
    analysis_path_frame.pack(pady=10, padx=10, fill="x", side="top")
    analysis_path_frame.columnconfigure(1, weight=1)
    ttk.Button(analysis_path_frame, text="Select Database File (.csv)", command=select_analysis_csv_file).grid(row=0,
                                                                                                               column=0,
                                                                                                               padx=5,
                                                                                                               pady=5,
                                                                                                               sticky=tk.W)
    analysis_database_entry = ttk.Entry(analysis_path_frame)
    analysis_database_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    ttk.Button(analysis_path_frame, text="Select Results Folder", command=select_analysis_results_dir).grid(row=1, column=0,
                                                                                                            padx=5, pady=5,
                                                                                                            sticky=tk.W)
    analysis_results_entry = ttk.Entry(analysis_path_frame)
    analysis_results_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

    ttk.Label(analysis_path_frame, text="Streamflow Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    analysis_model_var = tk.StringVar(value="National Water Model")
    analysis_model_dropdown = ttk.Combobox(analysis_path_frame, textvariable=analysis_model_var, state="readonly",
                                           values=("USGS", "GEOGLOWS", "National Water Model"))
    analysis_model_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)


    # --- Analysis: Run Buttons ---
    analysis_button_frame = ttk.Frame(analysis_tab)
    analysis_button_frame.pack(pady=10, fill="x", padx=10, side="top")
    analysis_button_frame.columnconfigure(0, weight=1)
    # --- Column 1 configure REMOVED ---
    analysis_run_button = ttk.Button(analysis_button_frame, text="3. Analyze & Save All Dam Data",
                                     command=start_analysis_processing, style="Accent.TButton")
    analysis_run_button.grid(row=0, column=0, padx=5, ipady=5, sticky=tk.EW)
    # --- Summary Button REMOVED (moved to checkbox) ---


    # --- Analysis: Figure Display Options ---
    analysis_figure_frame = ttk.LabelFrame(analysis_tab, text="Select Figures to Display")
    analysis_figure_frame.pack(pady=10, padx=10, fill="x", side="top")
    analysis_figure_frame.columnconfigure(0, weight=1)
    analysis_figure_frame.columnconfigure(1, weight=1)

    # --- Dam(s) to Analyze Dropdown MOVED HERE ---
    ttk.Label(analysis_figure_frame, text="Dam to Display:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    analysis_dam_dropdown = ttk.Combobox(analysis_figure_frame, state="readonly")
    analysis_dam_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

    # --- Checkboxes for figures ---
    analysis_display_cross_section = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_figure_frame, text="Cross-Sections", variable=analysis_display_cross_section).grid(row=1,
                                                                                                                column=0,
                                                                                                                padx=5,
                                                                                                                pady=2,
                                                                                                                sticky=tk.W)
    analysis_display_rating_curves = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_figure_frame, text="Rating Curves", variable=analysis_display_rating_curves).grid(row=2,
                                                                                                               column=0,
                                                                                                               padx=5,
                                                                                                               pady=2,
                                                                                                               sticky=tk.W)
    analysis_display_map = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_figure_frame, text="Dam Location", variable=analysis_display_map).grid(row=3, column=0, padx=5,
                                                                                                    pady=2, sticky=tk.W)
    analysis_display_wsp = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_figure_frame, text="Water Surface Profile", variable=analysis_display_wsp).grid(row=1,
                                                                                                             column=1,
                                                                                                             padx=5, pady=2,
                                                                                                             sticky=tk.W)
    analysis_display_fdc = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_figure_frame, text="Flow-Duration Curve", variable=analysis_display_fdc).grid(row=2, column=1,
                                                                                                           padx=5, pady=2,
                                                                                                           sticky=tk.W)

    # --- Bar Chart Checkbox ---
    analysis_display_bar_chart = tk.BooleanVar(value=False)
    ttk.Checkbutton(analysis_figure_frame, text="Generate Bar Chart (all dams)",
                    variable=analysis_display_bar_chart).grid(row=3,
                                                              column=1,
                                                              padx=5, pady=2,
                                                              sticky=tk.W)

    # --- Analysis: Display Button ---
    analysis_display_button_frame = ttk.Frame(analysis_tab)
    analysis_display_button_frame.pack(pady=10, fill="x", padx=10, side="top")
    analysis_display_button = ttk.Button(analysis_display_button_frame, text="4. Generate & Display Dam Figures",
                                         command=start_display_dam_figures_thread, style="Accent.TButton")
    analysis_display_button.pack(fill="x", ipady=5)

    # --- Analysis: Figure Viewer Frame ---
    analysis_figure_viewer_frame = ttk.LabelFrame(analysis_tab, text="Figure Viewer")
    # This frame is packed at the end, but its content is filled by functions

    # This frame will hold the controls (buttons, label) - PACKED FIRST
    analysis_figure_controls_frame = ttk.Frame(analysis_figure_viewer_frame)
    analysis_figure_controls_frame.pack(fill="x", pady=5, side="top")  # Explicitly pack at the top

    prev_button = ttk.Button(analysis_figure_controls_frame, text="< Previous", command=on_prev_figure)
    prev_button.pack(side="left", padx=10)

    next_button = ttk.Button(analysis_figure_controls_frame, text="Next >", command=on_next_figure)
    next_button.pack(side="right", padx=10)

    analysis_figure_label_var = tk.StringVar(value="No figure loaded.")
    analysis_figure_label = ttk.Label(analysis_figure_controls_frame, textvariable=analysis_figure_label_var,
                                      anchor="center")
    analysis_figure_label.pack(side="left", fill="x", expand=True)

    # This frame will hold the matplotlib canvas - PACKED SECOND
    analysis_figure_canvas_frame = ttk.Frame(analysis_figure_viewer_frame)
    analysis_figure_canvas_frame.pack(fill="both", expand=True)

    # Pack the main viewer frame (it will be un-packed by clear_figure_carousel)
    analysis_figure_viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)
    # Hide the figure viewer frame initially
    analysis_figure_viewer_frame.pack_forget()

    """
    ===================================================================
                            --- STATUS BAR ---
    ===================================================================
    """
    status_var = tk.StringVar()
    status_var.set("Ready. Please select a Project Folder in the 'Preparation' tab to begin.")
    status_label = ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w", padding=5)
    status_label.pack(side="bottom", fill="x", ipady=2)

    root.mainloop()


if __name__ == '__main__':
    main()
