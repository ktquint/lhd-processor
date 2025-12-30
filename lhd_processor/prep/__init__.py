# Expose the main Dam class for easier importing
from .classes import LowHeadDam

# Expose key helper functions if needed (optional)
from .streamflow_processing import create_reanalysis_file, condense_zarr
