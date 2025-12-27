# import xarray as xr
# import os
#
# # --- 1. Set the path to your new Zarr store ---
# zarr_path = '/Users/kennyquintana/Developer/lhd-processor/lhd_processor/data/nwm_v3_daily_retrospective.zarr'
#
#
# def test_zarr_integrity(path):
#     if not os.path.exists(path):
#         print(f"Error: Path does not exist: {path}")
#         return
#
#     print(f"Opening Zarr store at: {path}")
#     try:
#         # Attempt to open with consolidated metadata first
#         try:
#             ds = xr.open_zarr(path, consolidated=True)
#             print("✅ Successfully opened with consolidated metadata.")
#         except Exception:
#             print("Consolidated metadata not found, opening with consolidated=False...")
#             ds = xr.open_zarr(path, consolidated=False)
#             print("✅ Successfully opened Zarr store.")
#
#         # --- 2. Check Dimensions and Variables ---
#         print("\n--- Dataset Summary ---")
#         print(ds)
#
#         # --- 3. Verify COMIDs (feature_id) ---
#         num_ids = len(ds.feature_id)
#         print(f"\nTotal COMIDs (feature_id) found: {num_ids}")
#         if num_ids > 0:
#             print(f"First 5 IDs: {ds.feature_id.values[:5]}")
#
#         # --- 4. Verify Time Range ---
#         time_min = ds.time.min().values
#         time_max = ds.time.max().values
#         print(f"Time Range: {time_min} to {time_max}")
#
#         # --- 5. Data Validation (Check for NaNs or 0s) ---
#         # Select the first COMID and check its streamflow values
#         sample_id = ds.feature_id.values[0]
#         sample_data = ds.streamflow.sel(feature_id=sample_id)
#
#         non_null_count = sample_data.count().compute().item()
#         print(f"\nChecking data for COMID {sample_id}:")
#         print(f"Number of valid (non-NaN) daily records: {non_null_count}")
#
#         if non_null_count > 0:
#             # --- 6. Quick Visualization ---
#             print("Generating sample plot for the first COMID...")
#             sample_data.plot()
#             import matplotlib.pyplot as plt
#             plt.title(f"40-Year Daily Streamflow: COMID {sample_id}")
#             plt.ylabel("Streamflow (cms)")
#             plt.show()
#             print("✅ Visualization complete.")
#         else:
#             print("⚠️ Warning: The first COMID contains no valid data points.")
#
#     except Exception as e:
#         print(f"❌ Failed to test Zarr store: {e}")
#
#
# if __name__ == "__main__":
#     test_zarr_integrity(zarr_path)
import xarray as xr
import os
import pandas as pd

# Path to your local Zarr store
zarr_path = '/Users/kennyquintana/Developer/lhd-processor/lhd_processor/data/nwm_v3_daily_retrospective.zarr'

def check_zarr_manually(comid, target_date):
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr store not found at {zarr_path}")
        return

    try:
        # Open the dataset
        try:
            ds = xr.open_zarr(zarr_path, consolidated=True)
        except Exception:
            ds = xr.open_zarr(zarr_path, consolidated=False)

        print(f"--- Searching for COMID: {comid} on Date: {target_date} ---")

        # 1. Check if COMID exists
        if comid not in ds.feature_id.values:
            print(f"❌ COMID {comid} is NOT in the local Zarr store.")
            return

        # 2. Check Time Range
        time_min = pd.to_datetime(ds.time.min().values).strftime('%Y-%m-%d')
        time_max = pd.to_datetime(ds.time.max().values).strftime('%Y-%m-%d')
        print(f"Dataset Time Range: {time_min} to {time_max}")

        # 3. Attempt to select specific data point
        try:
            # .sel() performs the coordinate lookup
            val = ds['streamflow'].sel(feature_id=comid, time=target_date).values
            print(f"✅ Success! Streamflow for {comid} on {target_date}: {float(val):.4f} cms")
        except KeyError:
            print(f"❌ Date {target_date} exists in range but is missing for COMID {comid}.")
        except Exception as e:
            print(f"❌ Error during selection: {e}")

    except Exception as e:
        print(f"❌ Failed to open/read Zarr: {e}")

if __name__ == "__main__":
    # Test with your specific problematic data
    check_zarr_manually(comid=368123, target_date='2006-06-06')
    check_zarr_manually(comid=399514, target_date='2022-06-22')
