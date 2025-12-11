import sys
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from . import utils, prep_tab, analysis_tab


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
    tab3 = ttk.Frame(notebook)  # New Log Tab

    notebook.add(tab1, text="  Preparation & Processing  ")
    notebook.add(tab2, text="  Analysis & Visualization  ")
    notebook.add(tab3, text="  Logs  ")
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # 3. Setup Tab Content
    prep_tab.setup_prep_tab(tab1)
    analysis_tab.setup_analysis_tab(tab2)

    # 4. Setup Log Tab
    log_text = ScrolledText(tab3, state="disabled", height=20, width=80)
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
                # Prep Tab - Paths
                "project_dir": prep_tab.project_entry.get(),
                "database_path": prep_tab.database_entry.get(),
                "dem_dir": prep_tab.dem_entry.get(),
                "strm_dir": prep_tab.strm_entry.get(),
                "land_dir": prep_tab.land_use_entry.get(),
                "results_dir": prep_tab.results_entry.get(),
                "json_path": prep_tab.json_entry.get(),

                # Prep Tab - Dropdowns
                "flowline_source": prep_tab.flowline_var.get(),
                "dem_res": prep_tab.dd_var.get(),
                "streamflow_source": prep_tab.streamflow_var.get(),
                "baseflow_method": prep_tab.baseflow_var.get(),

                # Analysis Tab
                "analysis_db": analysis_tab.db_entry.get(),
                "analysis_res": analysis_tab.res_entry.get(),
                "analysis_model": analysis_tab.model_var.get()
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

            # Prep Tab
            set_entry(prep_tab.project_entry, "project_dir")
            set_entry(prep_tab.database_entry, "database_path")
            set_entry(prep_tab.dem_entry, "dem_dir")
            set_entry(prep_tab.strm_entry, "strm_dir")
            set_entry(prep_tab.land_use_entry, "land_dir")
            set_entry(prep_tab.results_entry, "results_dir")
            set_entry(prep_tab.json_entry, "json_path")

            set_var(prep_tab.flowline_var, "flowline_source")
            set_var(prep_tab.dd_var, "dem_res")
            set_var(prep_tab.streamflow_var, "streamflow_source")
            set_var(prep_tab.baseflow_var, "baseflow_method")

            # Analysis Tab
            set_entry(analysis_tab.db_entry, "analysis_db")
            set_entry(analysis_tab.res_entry, "analysis_res")
            set_var(analysis_tab.model_var, "analysis_model")

            print("Startup settings loaded successfully.")

        except Exception as e:
            print(f"Error loading settings: {e}")

    # Load settings immediately before starting loop
    load_startup_settings()

    # Register the close handler
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()