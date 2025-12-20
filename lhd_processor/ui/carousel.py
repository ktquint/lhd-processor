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

    # === BIND RESIZE EVENT HERE ===
    canvas_frame.bind("<Configure>", on_resize)

    # Note: We don't pack viewer_frame yet; it appears only when needed.


def on_resize(event):
    """Dynamically resizes the figure when the window size changes."""
    if current_figure_canvas and current_figure_canvas.figure:
        w = event.width
        h = event.height

        # Prevent errors if the window is being destroyed or extremely small
        if w > 50 and h > 50:
            fig = current_figure_canvas.figure
            dpi = fig.get_dpi()
            # Update figure size in inches
            fig.set_size_inches(w / dpi, h / dpi)

            # Optional: Re-tighten layout to prevent clipped labels
            # fig.tight_layout()

            # Redraw efficiently
            current_figure_canvas.draw_idle()


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

    # === FIX FOR GUI STRETCHING ===
    # Matplotlib figures have a default size (often 640x480).
    # If we pack this directly, it forces the Tkinter window to expand.
    # By setting a small initial size, we ensure it fits in the *current*
    # space. The 'on_resize' handler or sync block below will immediately
    # scale it UP to fill the frame properly.
    fig.set_size_inches(2, 2)
    # ==============================

    # Create new canvas
    current_figure_canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    current_figure_canvas.draw()
    current_figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    # === FORCE INITIAL SIZE SYNC ===
    # This ensures the new figure adopts the current window size immediately
    if canvas_frame.winfo_width() > 50:
        w = canvas_frame.winfo_width()
        h = canvas_frame.winfo_height()
        dpi = fig.get_dpi()
        fig.set_size_inches(w / dpi, h / dpi)
        current_figure_canvas.draw_idle()

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