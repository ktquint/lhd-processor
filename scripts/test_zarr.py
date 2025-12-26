import xarray as xr
import os

# --- 1. Set the path to your new Zarr store ---
zarr_path = '/Users/kennyquintana/Developer/lhd-processor/lhd_processor/data/nwm_v3_daily_retrospective.zarr'


def test_zarr_integrity(path):
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}")
        return

    print(f"Opening Zarr store at: {path}")
    try:
        # Attempt to open with consolidated metadata first
        try:
            ds = xr.open_zarr(path, consolidated=True)
            print("✅ Successfully opened with consolidated metadata.")
        except Exception:
            print("Consolidated metadata not found, opening with consolidated=False...")
            ds = xr.open_zarr(path, consolidated=False)
            print("✅ Successfully opened Zarr store.")

        # --- 2. Check Dimensions and Variables ---
        print("\n--- Dataset Summary ---")
        print(ds)

        # --- 3. Verify COMIDs (feature_id) ---
        num_ids = len(ds.feature_id)
        print(f"\nTotal COMIDs (feature_id) found: {num_ids}")
        if num_ids > 0:
            print(f"First 5 IDs: {ds.feature_id.values[:5]}")

        # --- 4. Verify Time Range ---
        time_min = ds.time.min().values
        time_max = ds.time.max().values
        print(f"Time Range: {time_min} to {time_max}")

        # --- 5. Data Validation (Check for NaNs or 0s) ---
        # Select the first COMID and check its streamflow values
        sample_id = ds.feature_id.values[0]
        sample_data = ds.streamflow.sel(feature_id=sample_id)

        non_null_count = sample_data.count().compute().item()
        print(f"\nChecking data for COMID {sample_id}:")
        print(f"Number of valid (non-NaN) daily records: {non_null_count}")

        if non_null_count > 0:
            # --- 6. Quick Visualization ---
            print("Generating sample plot for the first COMID...")
            sample_data.plot()
            import matplotlib.pyplot as plt
            plt.title(f"40-Year Daily Streamflow: COMID {sample_id}")
            plt.ylabel("Streamflow (cms)")
            plt.show()
            print("✅ Visualization complete.")
        else:
            print("⚠️ Warning: The first COMID contains no valid data points.")

    except Exception as e:
        print(f"❌ Failed to test Zarr store: {e}")


if __name__ == "__main__":
    test_zarr_integrity(zarr_path)