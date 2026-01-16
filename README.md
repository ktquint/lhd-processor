# Low-Head Dam Processor

This project analyzes the downstream flow conditions at low-head dams using location and width data.
It integrates functionality from the [ARC](https://github.com/MikeFHS/automated-rating-curve) package into a GUI application for low-head dam analysis.

The application is structured as a Python package (`lhd_processor`) and is run directly from your terminal.

---

## 1. Setup Instructions (Do this only once)

1.  **Install Miniconda:**
    * Go to the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system (Windows or macOS).
    * Run the installer and accept all the default settings.

2.  **Download the Project Code:**
    * Download this repository as a ZIP file (or use `git clone`).
    * Download the [ARC](https://github.com/MikeFHS/automated-rating-curve) repository.
    * Unzip the folders.

3.  **Create the Conda Environment:**
    * **Open Terminal/Anaconda Prompt** and navigate to the **LHD Processor** folder (where `environment.yaml` is located).
    * Run the following commands to create and activate the base environment. This step will take 5-10 minutes.
        ```bash
        conda env create -f environment.yaml
        conda activate lhd-environment
        ```
    * **Install ARC:** Navigate to the `automated-rating-curve` folder you downloaded in Step 2.
        *(Note: The `..` in the command below assumes the two project folders are next to each other. Adjust the path if yours are different.)*
        ```bash
        cd ../automated-rating-curve
        pip install -e .
        ```
    * **⚠️ Important:** Do not install `arc` using standard commands like `pip install arc` or by adding it to `environment.yaml`. That will install an unrelated archive utility that conflicts with this project.

---

## 2. Project Directory Organization

To use the LHD Processor effectively, it is recommended to set up a dedicated folder for your project. The application uses an Excel database (`.xlsx`) to manage site data and results.

### Recommended Structure
Create a folder (e.g., `My_Dam_Project`) and place your input Excel file inside it. The application will help you organize the data into standard subdirectories.

```text
My_Dam_Project/
│
├── my_dams_database.xlsx  <-- Your required input file (Must contain a 'Sites' sheet)
│
├── DEM/                   <-- Stores downloaded DEMs (USGS 3DEP)
│
├── STRM/                  <-- Stores downloaded flowline data (NHD/GEOGLOWS)
│
├── LAND/                  <-- Stores Land Use rasters (ESA WorldCover)
│
└── Results/               <-- Stores analysis outputs, figures, and ARC intermediate files
    ├── [Dam_ID]/
    │   ├── ARC_InputFiles/
    │   ├── Bathymetry/
    │   ├── FIGS/
    │   ├── STRM/
    │   ├── VDT/
    │   └── XS/
```

### Input Database Format
The input file must be an Excel workbook (`.xlsx`) with a sheet named **`Sites`**.
Required columns in the `Sites` sheet:
*   `site_id` (Integer)
*   `latitude` (Decimal Degrees)
*   `longitude` (Decimal Degrees)
*   `weir_length` (Meters)
*   `name` (Optional, String)

---

## 3. How to Run the Program

1.  **Activate the Environment:**
    * Open your terminal (Anaconda Prompt on Windows, Terminal on macOS).
    * Type the following command:
        ```bash
        conda activate lhd-environment
        ```

2.  **Run the Processor:**
    * Navigate to the **root project directory** where the `lhd_processor` folder is located.
        ```bash
        cd Users/username/Developer/lhd-processor
        ```
    * Launch the GUI using the module command:
        ```bash
        python -m lhd_processor
        ```

---

## 4. Using the GUI

The application is divided into two main tabs:

### Tab 1: Preparation & Processing
This tab handles data acquisition and the hydraulic simulation.

1.  **Step 1: Download and Prepare Data**
    *   Select your **Excel Database**.
    *   Select or create folders for **DEM**, **Hydrography**, and **Land Use**.
    *   Choose your data sources (NHDPlus/TDX-Hydro, National Water Model/GEOGLOWS).
    *   Click **"1. Prepare Data & Update Database"**. This will download necessary geospatial data and update your Excel file with flowline IDs and baseflow estimates.

2.  **Step 2: Automated Rating Curves (ARC)**
    *   Ensure the Excel file and Results folder are selected.
    *   Click **"2. Run ARC"**. This runs the Automated Rating Curve process to estimate bathymetry and hydraulic properties for each dam.

### Tab 2: Analysis & Visualization
This tab analyzes the hydraulic results and generates figures.

1.  **Step 3: Analyze Results**
    *   Select your **Excel Database** and **Results Folder**.
    *   Choose the **Calculation Mode** (Advanced vs Simplified).
    *   Click **"3. Analyze & Save All Dam Data"**. This calculates hydraulic jump types and dangerous flow ranges.

2.  **Step 4: Generate & Display Figures**
    *   Select the dam you want to visualize from the dropdown.
    *   Check the boxes for the desired figures (Cross-Sections, Rating Curves, Maps, etc.).
    *   Click **"4. Generate & Display Figures"**.
