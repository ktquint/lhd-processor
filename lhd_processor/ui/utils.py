import os
import json
import tkinter as tk

# Global variables shared across modules
root = None
status_var = None
progress_var = None  # Controls the progress bar
CONFIG_FILE = "user_settings.json"


def init_utils(app_root):
    """Initialize the shared root, status, and progress variables."""
    global root, status_var, progress_var
    root = app_root

    # Status Message
    status_var = tk.StringVar()
    status_var.set("Ready. Please select a Database File to begin.")

    # Progress Bar (0.0 to 100.0)
    progress_var = tk.DoubleVar(value=0.0)


def set_status(message):
    """Thread-safe status update."""
    if status_var:
        status_var.set(message)


def set_progress(value):
    """Sets the progress bar value (0-100)."""
    if progress_var:
        progress_var.set(value)


def get_root():
    """Returns the main Tk instance."""
    return root


# --- Persistence Logic (Save/Load Settings) ---

def save_settings(settings_dict):
    """Saves a dictionary of settings to disk."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings_dict, f, indent=4)
        set_status("Settings saved.")
    except Exception as e:
        print(f"Failed to save settings: {e}")


def load_settings():
    """Returns a dictionary of settings or an empty dict if none exist."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load settings: {e}")
    return {}


# --- GUI Helpers (Validation & Tooltips) ---

def bind_path_validation(entry_widget, is_file=True, must_exist=True):
    """
    Binds an event to an Entry widget.

    NOTE: Color validation logic has been removed to support system themes (Light/Dark mode).
    This function is kept for compatibility but performs no actions.
    """
    pass


class ToolTip(object):
    """
    Create a tooltip for a given widget.
    """

    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # milliseconds
        self.wraplength = 180  # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window chrome
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))

        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()
