import tkinter as tk
from tkinter import ttk
from . import utils, prep_tab, analysis_tab


def main():
    root = tk.Tk()
    root.title("LHD Control Center")
    root.geometry("700x1000")

    # Initialize shared utils
    utils.init_utils(root)

    # Styles
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    style.configure("TNotebook.Tab", font=("Arial", 10, "bold"))

    # Notebook
    notebook = ttk.Notebook(root)
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)

    notebook.add(tab1, text="  Preparation & Processing  ")
    notebook.add(tab2, text="  Analysis & Visualization  ")
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # Setup Content
    prep_tab.setup_prep_tab(tab1)
    analysis_tab.setup_analysis_tab(tab2)

    # Status Bar
    status_label = ttk.Label(root, textvariable=utils.status_var, relief="sunken", anchor="w", padding=5)
    status_label.pack(side="bottom", fill="x", ipady=2)

    root.mainloop()