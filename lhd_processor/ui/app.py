import sys
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from . import utils, download_tab, arc_tab, calc_tab, vis_tab


# --- Helper Classes ---

class PrintLogger:
    """
    Redirects stdout/stderr to a Tkinter Text widget so logs appear in the GUI.
    """

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.terminal = sys.stdout  # Keep reference to original stdout

    def write(self, message):
        self.terminal.write(message)  # Write to terminal (optional, good for debugging)
        # Update UI in a thread-safe way (mostly safe for simple prints)
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Auto-scroll to bottom
        self.text_widget.configure(state="disabled")

    def flush(self):
        self.terminal.flush()


class ScrollableFrame(ttk.Frame):
    """
    A scrollable container. Useful for small screens since the app is tall.
    """

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)

        # This frame sits inside the canvas
        self.scrollable_frame = ttk.Frame(canvas)

        # Update scroll region when the inner frame changes size
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Create the window inside the canvas
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Layout
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel scrolling (Optional but nice)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind mousewheel to canvas
        canvas.bind_all("<MouseWheel>", _on_mousewheel)


# --- Main Application ---

def main():
    root = tk.Tk()
    root.title("LHD Control Center")
    # Geometry can be smaller now because we have a scrollbar!
    root.geometry("750x800")

    # Initialize shared utils (root and status var)
    utils.init_utils(root)

    # Styles
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    style.configure("TNotebook.Tab", font=("Arial", 10, "bold"))

    # 1. Create Scrollable Container
    main_scroll_container = ScrollableFrame(root)
    main_scroll_container.pack(fill="both", expand=True, padx=5, pady=5)

    # We add content to 'main_scroll_container.scrollable_frame' instead of 'root'
    content_frame = main_scroll_container.scrollable_frame

    # 2. Create Notebook (Tabs)
    notebook = ttk.Notebook(content_frame)
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    tab4 = ttk.Frame(notebook)
    tab5 = ttk.Frame(notebook)  # New Log Tab

    notebook.add(tab1, text="  Download  ")
    notebook.add(tab2, text="  ARC  ")
    notebook.add(tab3, text="  Calculation  ")
    notebook.add(tab4, text="  Visualization  ")
    notebook.add(tab5, text="  Logs  ")
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # 3. Setup Tab Content
    download_tab.setup_download_tab(tab1)
    arc_tab.setup_arc_tab(tab2)
    calc_tab.setup_calc_tab(tab3)
    vis_tab.setup_vis_tab(tab4)

    # 4. Setup Log Tab
    log_text = ScrolledText(tab5, state="disabled", height=20, width=80)
    log_text.pack(fill="both", expand=True, padx=5, pady=5)

    # Redirect print() to this tab
    sys.stdout = PrintLogger(log_text)
    sys.stderr = PrintLogger(log_text)  # Catch errors too

    # 5. Status Bar (Outside scroll frame so it's always visible at bottom)
    status_frame = ttk.Frame(root, relief="sunken", padding=2)
    status_frame.pack(side="bottom", fill="x")

    # Progress Bar (Small, determinate)
    progress_bar = ttk.Progressbar(status_frame, variable=utils.progress_var, maximum=100)
    progress_bar.pack(side="right", padx=5, fill="y")

    # Status Label
    status_label = ttk.Label(status_frame, textvariable=utils.status_var, anchor="w")
    status_label.pack(side="left", fill="x", expand=True)

    # --- Persistence Logic (Save/Load Settings) ---

    def on_close():
        """Gather settings from widgets and save to JSON before closing."""
        # Use try/except to prevent crashing if widgets aren't fully initialized
        try:
            settings = {
                # Download Tab - Paths
                "database_path": download_tab.database_entry.get() if download_tab.database_entry else "",
                "dem_dir": download_tab.dem_entry.get() if download_tab.dem_entry else "",
                "strm_dir": download_tab.strm_entry.get() if download_tab.strm_entry else "",
                "land_dir": download_tab.land_use_entry.get() if download_tab.land_use_entry else "",
                
                # ARC Tab Paths
                "arc_xlsx": arc_tab.arc_xlsx_entry.get() if arc_tab.arc_xlsx_entry else "",
                "arc_results": arc_tab.arc_results_entry.get() if arc_tab.arc_results_entry else "",

                # Download Tab - Dropdowns
                "flowline_source": download_tab.flowline_var.get() if download_tab.flowline_var else "",
                "dem_res": download_tab.dd_var.get() if download_tab.dd_var else "",
                "streamflow_source": download_tab.streamflow_var.get() if download_tab.streamflow_var else "",
                
                # ARC Tab Dropdowns
                "arc_flowline": arc_tab.arc_flowline_var.get() if arc_tab.arc_flowline_var else "",
                "arc_streamflow": arc_tab.arc_streamflow_var.get() if arc_tab.arc_streamflow_var else "",
                "arc_baseflow": arc_tab.arc_baseflow_var.get() if arc_tab.arc_baseflow_var else "",

                # Calc Tab
                "calc_db": calc_tab.db_entry.get() if calc_tab.db_entry else "",
                "calc_res": calc_tab.res_entry.get() if calc_tab.res_entry else "",
                "calc_mode": calc_tab.calc_mode_var.get() if calc_tab.calc_mode_var else "",
                "calc_flowline": calc_tab.flowline_source_var.get() if calc_tab.flowline_source_var else "",
                "calc_streamflow": calc_tab.streamflow_source_var.get() if calc_tab.streamflow_source_var else "",

                # Vis Tab
                "vis_db": vis_tab.db_entry.get() if vis_tab.db_entry else "",
                "vis_res": vis_tab.res_entry_display.get() if vis_tab.res_entry_display else "",
                "vis_flowline": vis_tab.flowline_source_var.get() if vis_tab.flowline_source_var else "",
                "vis_streamflow": vis_tab.streamflow_source_var.get() if vis_tab.streamflow_source_var else "",
                "vis_dam_sel": vis_tab.dam_dropdown.get() if vis_tab.dam_dropdown else "",
                "vis_chk_xs": vis_tab.chk_xs.get() if vis_tab.chk_xs else False,
                "vis_chk_rc": vis_tab.chk_rc.get() if vis_tab.chk_rc else False,
                "vis_chk_map": vis_tab.chk_map.get() if vis_tab.chk_map else False,
                "vis_chk_wsp": vis_tab.chk_wsp.get() if vis_tab.chk_wsp else False,
                "vis_chk_fdc": vis_tab.chk_fdc.get() if vis_tab.chk_fdc else False,
                "vis_chk_bar": vis_tab.chk_bar.get() if vis_tab.chk_bar else False,
            }

            # Save using utils (ensure you added save_settings to utils.py!)
            utils.save_settings(settings)
            print("Settings saved.")
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

        root.destroy()

    def load_startup_settings():
        """Load settings from JSON and populate widgets."""
        try:
            settings = utils.load_settings()
            if not settings:
                return

            # Helper to set Entry widgets
            def set_entry(widget, key):
                if widget and key in settings:
                    widget.delete(0, tk.END)
                    widget.insert(0, settings[key])

            # Helper to set Variables/Dropdowns
            def set_var(tk_var, key):
                if tk_var and key in settings:
                    tk_var.set(settings[key])

            # Download Tab
            set_entry(download_tab.database_entry, "database_path")
            set_entry(download_tab.dem_entry, "dem_dir")
            set_entry(download_tab.strm_entry, "strm_dir")
            set_entry(download_tab.land_use_entry, "land_dir")
            
            set_entry(arc_tab.arc_xlsx_entry, "arc_xlsx")
            set_entry(arc_tab.arc_results_entry, "arc_results")

            set_var(download_tab.flowline_var, "flowline_source")
            set_var(download_tab.dd_var, "dem_res")
            set_var(download_tab.streamflow_var, "streamflow_source")
            
            set_var(arc_tab.arc_flowline_var, "arc_flowline")
            set_var(arc_tab.arc_streamflow_var, "arc_streamflow")
            set_var(arc_tab.arc_baseflow_var, "arc_baseflow")

            # Calc Tab
            set_entry(calc_tab.db_entry, "calc_db")
            set_entry(calc_tab.res_entry, "calc_res")
            set_var(calc_tab.calc_mode_var, "calc_mode")
            set_var(calc_tab.flowline_source_var, "calc_flowline")
            set_var(calc_tab.streamflow_source_var, "calc_streamflow")

            # Vis Tab
            set_entry(vis_tab.db_entry, "vis_db")
            set_entry(vis_tab.res_entry_display, "vis_res")
            set_var(vis_tab.flowline_source_var, "vis_flowline")
            set_var(vis_tab.streamflow_source_var, "vis_streamflow")
            
            set_var(vis_tab.chk_xs, "vis_chk_xs")
            set_var(vis_tab.chk_rc, "vis_chk_rc")
            set_var(vis_tab.chk_map, "vis_chk_map")
            set_var(vis_tab.chk_wsp, "vis_chk_wsp")
            set_var(vis_tab.chk_fdc, "vis_chk_fdc")
            set_var(vis_tab.chk_bar, "vis_chk_bar")

            # Trigger update for dropdown in Vis Tab
            if vis_tab.res_entry_display.get():
                # Pre-set the dropdown value so update_dropdown tries to preserve it
                if "vis_dam_sel" in settings and vis_tab.dam_dropdown:
                     vis_tab.dam_dropdown.set(settings["vis_dam_sel"])

                vis_tab.update_dropdown()

            print("Startup settings loaded successfully.")

        except Exception as e:
            print(f"Error loading settings: {e}")

    # Load settings immediately before starting loop
    load_startup_settings()

    # Register the close handler
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
