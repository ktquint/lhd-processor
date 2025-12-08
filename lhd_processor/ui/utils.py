import tkinter as tk

# Global variables shared across modules
root = None
status_var = None

def init_utils(app_root):
    """Initialize the shared root and status variable."""
    global root, status_var
    root = app_root
    status_var = tk.StringVar()
    status_var.set("Ready. Please select a Project Folder in the 'Preparation' tab to begin.")

def set_status(message):
    """Thread-safe status update."""
    if status_var:
        status_var.set(message)

def get_root():
    """Returns the main Tk instance."""
    return root