import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from . import utils

# Module-level state for the carousel
current_figure_list = []
current_figure_index = 0
current_figure_canvas = None

# UI Widgets
viewer_frame = None
canvas_frame = None
label_var = None
prev_btn = None
next_btn = None


def setup_carousel_ui(parent_frame):
    """Creates the hidden carousel UI components."""
    global viewer_frame, canvas_frame, label_var, prev_btn, next_btn

    viewer_frame = ttk.LabelFrame(parent_frame, text="Figure Viewer")

    # Controls
    controls_frame = ttk.Frame(viewer_frame)
    controls_frame.pack(fill="x", pady=5, side="top")

    prev_btn = ttk.Button(controls_frame, text="< Previous", command=on_prev_figure)
    prev_btn.pack(side="left", padx=10)

    next_btn = ttk.Button(controls_frame, text="Next >", command=on_next_figure)
    next_btn.pack(side="right", padx=10)

    label_var = tk.StringVar(value="No figure loaded.")
    lbl = ttk.Label(controls_frame, textvariable=label_var, anchor="center")
    lbl.pack(side="left", fill="x", expand=True)

    # Canvas Frame
    canvas_frame = ttk.Frame(viewer_frame)
    canvas_frame.pack(fill="both", expand=True)

    # Note: We don't pack viewer_frame yet; it appears only when needed.


def clear_figure_carousel():
    """Hides the viewer and clears content."""
    global current_figure_list, current_figure_index, current_figure_canvas
    current_figure_list = []
    current_figure_index = 0

    if current_figure_canvas:
        current_figure_canvas.get_tk_widget().destroy()
        current_figure_canvas = None

    if viewer_frame:
        viewer_frame.pack_forget()


def display_figure(index):
    """Displays figure at specific index."""
    global current_figure_list, current_figure_index, current_figure_canvas

    if not current_figure_list:
        clear_figure_carousel()
        return

    index = max(0, min(index, len(current_figure_list) - 1))
    current_figure_index = index

    if current_figure_canvas:
        current_figure_canvas.get_tk_widget().destroy()

    fig, title = current_figure_list[index]

    # Create new canvas
    current_figure_canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    current_figure_canvas.draw()
    current_figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    label_var.set(f"{title} ({index + 1} of {len(current_figure_list)})")

    # Update button states
    prev_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
    next_btn.config(state=tk.NORMAL if index < len(current_figure_list) - 1 else tk.DISABLED)


def on_prev_figure():
    display_figure(current_figure_index - 1)


def on_next_figure():
    display_figure(current_figure_index + 1)


def load_figures(figures_list):
    """Public entry point to load and show figures."""
    global current_figure_list
    clear_figure_carousel()

    if figures_list:
        current_figure_list = figures_list
        viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)
        display_figure(0)
    else:
        utils.set_status("Analysis complete. No figures were selected for display.")