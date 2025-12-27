# Code to download GEOGLOWS retrospective datasets
# GEOGLOWS data can be downloaded from http://geoglows-v2.s3-website-us-west-2.amazonaws.com/

# built-in imports
import os
import gc

# third-party imports
import s3fs
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.stats import gumbel_r
from shapely.geometry import Point

# custom imports
from .classes import RathCelonDam


def get_stream_coords(stream_gdf: gpd.GeoDataFrame, rivid_field: str, rivids: list[int | str], method='centroid'):
    """

    Parameters
    ----------
    stream_gdf -
    rivid_field - either 'LINKNO' or 'hydroseq'
    rivids - list of all rivids...
    method - where we pull the coords from

    Returns
    -------
    results - each rivid and its coords {rivid: [lat, lon]}
    """
    # Ensure original CRS is EPSG:4326 for lat/lon output
    if stream_gdf.crs != "EPSG:4269":
        # Using 4269 as it's a common default for lat/lon
        try:
            stream_gdf = stream_gdf.to_crs("EPSG:4269")
        except Exception as e:
            print(f"Warning: Could not reproject stream GDF to EPSG:4269. Assuming WGS84. Error: {e}")

    result = {}

    for rivid in rivids:
        match = stream_gdf[stream_gdf[rivid_field] == rivid]

        if match.empty:
            print(f"No match found for {rivid_field} = {rivid}")
            continue

        geom = match.geometry.iloc[0]

        # extract the point from the geom
        if method == "centroid":
            point = geom.centroid
        elif method == "start":
            point = Point(geom.coords[0])
        elif method == "end":
            point = Point(geom.coords[-1])
        else:
            raise ValueError(f"Unknown method '{method}'")

        result[rivid] = [point.y, point.x]

    return result


def nwm_return_periods(ts_df: pd.DataFrame):
    """
    Calculates return periods from a time-series dataframe using Annual Maximum Series (AMS)
    and a Gumbel distribution.

    Parameters
    ----------
    ts_df: A DataFrame with at least 'river_id', 'time', and 'streamflow' columns.

    Returns
    -------
    summary_df: A DataFrame indexed by 'river_id' with summary stats
                (qout_median, qout_max) and return periods (rp2, rp5, etc.).
    """
    print("Calculating return periods from time-series data...")
    if 'time' not in ts_df.columns or 'streamflow' not in ts_df.columns:
        raise ValueError("Parquet file must contain 'time' and 'streamflow' columns.")
    if 'river_id' not in ts_df.columns:
        raise ValueError("Internal error: 'river_id' not found in joined dataframe.")

    # Ensure time is datetime
    ts_df['time'] = pd.to_datetime(ts_df['time'])

    grouped = ts_df.groupby('river_id')
    summary_list = []

    return_periods_T = [2, 5, 10, 25, 50, 100]
    # Non-exceedance probability
    return_periods_P = [1 - (1 / t) for t in return_periods_T]
    rp_cols = [f'rp{t}' for t in return_periods_T]

    for river_id, group_df in grouped:
        # 1. Calculate simple stats
        q_median = group_df['streamflow'].median()
        q_max = group_df['streamflow'].max()

        # 2. Resample for Annual Maximum Series (AMS)
        # We need to set time as index for resampling
        ams = group_df.set_index('time')['streamflow'].resample('YE').max().dropna()

        # 3. Fit Gumbel distribution
        rp_values = [np.nan] * len(return_periods_T)
        # Require at least 5 years of data for a stable fit
        if len(ams) >= 5:
            try:
                # Fit Gumbel Type 1 (right-skewed)
                loc, scale = gumbel_r.fit(ams)
                # Calculate flow for each return period probability
                rp_values = gumbel_r.ppf(return_periods_P, loc=loc, scale=scale)
            except Exception as e:
                print(f"Warning: Gumbel fit failed for river_id {river_id}: {e}")
        else:
            print(f"Warning: Not enough data ({len(ams)} years) for Gumbel fit on river_id {river_id}.")

        # 4. Store results
        data = {
            'river_id': river_id,
            'qout_median': q_median,
            'qout_max': q_max,
        }
        for col, val in zip(rp_cols, rp_values):
            data[col] = val

        summary_list.append(data)

    if not summary_list:
        print("Warning: No rivers were processed for summary statistics.")
        return pd.DataFrame()

    # 5. Combine and set index
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.set_index('river_id')

    # 6. Rounding for clean output
    cols_to_round = ['qout_median', 'qout_max'] + rp_cols
    for col in cols_to_round:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)

    return summary_df


def create_reanalysis(dam: RathCelonDam, rivids_int: list, utm_crs, StrmShp_filtered_gdf):
    """
    This function handles fetching or calculating streamflow statistics.
    It supports:
    1. Calculating stats from a local NWM parquet file (if provided).
    2. Fetching pre-computed stats from GEOGLOWS S3 (if no parquet and rivid_field is 'LINKNO').
    """

    # Define final_df outside the blocks
    final_df = None

    # flag to track if we need to run the s3 fetcher
    run_standard_s3 = False

    # change the path to a string
    raw_path = dam.streamflow
    file_path = str(raw_path) if raw_path is not None else None

    # national water model parquet logic
    if dam.streamflow and file_path.endswith('.parquet') and os.path.exists(dam.streamflow):
        """
            this section is for when NWM hydrology is selected...
            we will access the parquet for median and max Q
            then calculate return periods and make a reanalysis file
        """
        print(f"Reading streamflow data from local parquet file: {dam.streamflow}")
        try:
            # 1. Load the parquet file
            flow_df = pd.read_parquet(dam.streamflow)

            # 2. Check for required columns in parquet
            if 'latitude' not in flow_df.columns or 'longitude' not in flow_df.columns:
                raise ValueError("Parquet file must contain 'latitude' and 'longitude' columns.")

            # 3. Convert parquet data to a GeoDataFrame
            flow_gdf = gpd.GeoDataFrame(
                flow_df,
                geometry=gpd.points_from_xy(flow_df.longitude, flow_df.latitude),
                crs="EPSG:4269"  # Assume WGS84 for lat/lon
            )

            del flow_df
            gc.collect()

            # 4. Get coordinates for the flowline segments we care about (from dam object)
            #    rivids_int is already defined (e.g., [LINKNO_1, LINKNO_2, ...])
            #    dam.flowline_gdf is the full GDF for the area, still in its original CRS (flowline_crs)
            strm_coords = get_stream_coords(dam.flowline_gdf, dam.rivid_field, rivids_int, method='centroid')

            if not strm_coords:
                raise ValueError("Could not extract stream coordinates for the dam's flowlines.")

            # 5. Convert stream coordinates to a GeoDataFrame
            strm_df = pd.DataFrame.from_dict(strm_coords, orient='index', columns=['latitude', 'longitude'])
            strm_df['original_rivid'] = strm_df.index

            strm_gdf = gpd.GeoDataFrame(
                strm_df,
                geometry=gpd.points_from_xy(strm_df.longitude, strm_df.latitude),
                crs="EPSG:4269"  # CRS from get_stream_coords
            )

            # 6. Reproject both to the same UTM CRS for accurate distance joining
            #    utm_crs was defined earlier when finding the closest stream
            flow_gdf = flow_gdf.to_crs(utm_crs)
            strm_gdf = strm_gdf.to_crs(utm_crs)

            # 7. Perform the spatial join
            #    For each stream segment (strm_gdf), find the nearest NWM point (flow_gdf)
            joined_gdf = gpd.sjoin_nearest(strm_gdf, flow_gdf, how='left')

            # 8. Set up the time-series DataFrame
            ts_df = joined_gdf.rename(columns={'original_rivid': 'river_id'})

            # Drop unnecessary columns
            cols_to_drop = ['geometry', 'latitude', 'longitude', 'index_right']
            # Drop columns that might have been duplicated from the right GDF
            cols_to_drop.extend([col for col in ts_df.columns if col.endswith('_right')])
            ts_df = ts_df.drop(columns=[col for col in cols_to_drop if col in ts_df.columns], errors='ignore')

            # 9. Calculate summary statistics (including return periods)
            # This calls the new function you just added
            final_df = nwm_return_periods(ts_df)
            final_df.index.name = 'river_id'  # Ensure index name matches other methods

            if final_df.empty:
                print(f"Warning: Statistical analysis yielded no results.")
                final_df = None  # Fallback
            else:
                print(f"Successfully calculated summary statistics for {len(final_df)} reaches from parquet file.")

        except Exception as e:
            print(f"Error reading or processing local parquet file: {e}. Falling back to remote data sources.")
            final_df = None  # Ensure fallback on error

    elif dam.streamflow and file_path.endswith('.gpkg') and os.path.exists(dam.streamflow):
        print("Mixed models detected: using NHDPlus flowlines with GEOGLOWS streamflow")

        try:
            # 1. find geoglows linkno
            print(f"Loading GEOGLOWS stream network from: {dam.streamflow}")
            geoglows_gdf = gpd.read_file(dam.streamflow)

            dam_loc_df = pd.read_csv(dam.csv_path)
            dam_row = dam_loc_df[dam_loc_df[dam.id_field] == dam.dam_id].iloc[0]
            dam_point = Point(dam_row['longitude'], dam_row['latitude'])

            if geoglows_gdf.crs != "EPSG:4326":
                geoglows_gdf.to_crs("EPSG:4326")

            geoglows_proj = geoglows_gdf.to_crs(utm_crs)
            dam_point_proj = gpd.GeoSeries([dam_point], crs="EPSG:4269").to_crs(utm_crs).iloc[0]


            geoglows_proj['distance'] = geoglows_proj.distance(dam_point_proj)
            nearest_stream = geoglows_proj.sort_values('distance').iloc[0]
            found_linkno = int(nearest_stream['LINKNO'])

            geoglows_id = found_linkno
            print(f" identified nearest GEOGLOWS LINKNO: {geoglows_id}")

            # B. Fetch S3 Data for JUST that GEOGLOWS ID
            print(f"Fetching GEOGLOWS data from S3 for ID: {geoglows_id}")
            ODP_S3_BUCKET_REGION = 'us-west-2'
            s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=ODP_S3_BUCKET_REGION))

            s3_fetch_rivids = [geoglows_id]

            # -- Fetch FDC --
            fdc_store = s3fs.S3Map(root='s3://geoglows-v2/retrospective/fdc.zarr', s3=s3, check=False)
            fdc_ds = xr.open_zarr(fdc_store).sel(p_exceed=[50.0, 0.0], river_id=s3_fetch_rivids)
            fdc_df = fdc_ds.to_dataframe().reset_index()

            del fdc_ds
            gc.collect()

            if fdc_df.empty:
                raise ValueError("FDC data empty for GEOGLOWS ID")

            q_median = fdc_df.loc[fdc_df['p_exceed'] == 50.0, 'hourly_annual'].max()
            q_max = fdc_df.loc[fdc_df['p_exceed'] == 0.0, 'hourly_annual'].max()

            # -- Fetch Return Periods --
            rp_store = s3fs.S3Map(root='s3://geoglows-v2/retrospective/return-periods.zarr', s3=s3, check=False)
            rp_ds = xr.open_zarr(rp_store).sel(river_id=[geoglows_id])
            rp_df = rp_ds.to_dataframe().reset_index()

            del rp_ds
            gc.collect()

            if rp_df.empty:
                raise ValueError("Return Period data empty for GEOGLOWS ID")

            rp_df['flow'] = rp_df[['gumbel', 'logpearson3']].mean(axis=1).round(3)
            rp_vals = rp_df.pivot(index='river_id', columns='return_period', values='flow')
            rp_vals.columns = [f'rp{int(c)}' for c in rp_vals.columns]

            # C. Broadcast to NHDPlus IDs
            # We create a DataFrame where the INDEX matches 'rivids_int' (the NHDPlus IDs)
            # but the DATA comes from the single GEOGLOWS ID.
            print(f"Broadcasting flows to {len(rivids_int)} NHDPlus segments.")

            # Create a single row of data
            data_row = {
                'qout_median': round(q_median, 3),
                'qout_max': round(q_max, 3)
            }
            # Add return periods to the row
            data_row.update(rp_vals.iloc[0].to_dict())

            # Replicate this row for every NHDPlus ID
            final_df = pd.DataFrame([data_row] * len(rivids_int), index=rivids_int)
            final_df.index.name = 'river_id'


        except Exception as e:
            print(f"Error in mixed model processing: {e}")

    if final_df is None:
        if dam.rivid_field == 'LINKNO':
            run_standard_s3 = True
        else:
            print(f"Error: rivid_field is {dam.rivid_field} but no valid flow source provided.")
            return dam.reanalysis_csv, dam.dam_shp, rivids_int, StrmShp_filtered_gdf

    if run_standard_s3:
        print(f"Fetching GEOGLOWS data from s3 for river_ids: {rivids_int}")
        # Set up the S3 connection
        ODP_S3_BUCKET_REGION = 'us-west-2'
        s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=ODP_S3_BUCKET_REGION))

        # # Load FDC data from S3 using Dask
        # # Convert to a list of integers
        fdc_s3_uri = 's3://geoglows-v2/retrospective/fdc.zarr'
        fdc_s3store = s3fs.S3Map(root=fdc_s3_uri, s3=s3, check=False)
        p_exceedance = [float(50.0), float(0.0)]
        fdc_ds = xr.open_zarr(fdc_s3store).sel(p_exceed=p_exceedance, river_id=rivids_int)

        # Convert Xarray to Dask DataFrame
        fdc_df = fdc_ds.to_dataframe().reset_index()

        # Check if fdc_df is empty
        if fdc_df.empty:
            # print(f"Skipping processing for {DEM_Tile} because fdc_df is empty.")
            dam.reanalysis_csv = None
            dam.dam_shp = None
            rivids_int = None
            # Note: StrmShp_filtered_gdf is defined in the parent function
            return dam.reanalysis_csv, dam.dam_shp, rivids_int, StrmShp_filtered_gdf

        # Create 'qout_median' column where 'p_exceed' is 50.0
        fdc_df.loc[fdc_df['p_exceed'] == 50.0, 'qout_median'] = fdc_df['hourly_annual']
        # Create 'qout_max' column where 'p_exceed' is 100.0
        fdc_df.loc[fdc_df['p_exceed'] == 0.0, 'qout_max'] = fdc_df['hourly_annual']
        # Group by 'river_id' and aggregate 'qout_median' and 'qout_max' by taking the non-null value
        fdc_df = fdc_df.groupby('river_id').agg({
            'qout_median': 'max',  # or use 'max' as both approaches would work
            'qout_max': 'max'
        }).reset_index()

        # set the dataframe index
        fdc_df = fdc_df.set_index('river_id')

        # round the values
        fdc_df['qout_median'] = fdc_df['qout_median'].round(3)
        fdc_df['qout_max'] = fdc_df['qout_max'].round(3)

        # drop all the columns except for the river_id, qout_median, and qout_max
        fdc_df = fdc_df[['qout_median', 'qout_max']]

        # Load return periods data from S3 using Dask
        rp_s3_uri = 's3://geoglows-v2/retrospective/return-periods.zarr'
        rp_s3store = s3fs.S3Map(root=rp_s3_uri, s3=s3, check=False)
        rp_ds = xr.open_zarr(rp_s3store).sel(river_id=rivids_int)

        # Convert Xarray to Dask DataFrame and pivot
        rp_df = rp_ds.to_dataframe().reset_index()

        # find the maximum between the gumbel and logpearson3 return periods and label this new column 'return_period_flow'
        rp_df['return_period_flow'] = rp_df[['gumbel', 'logpearson3']].mean(axis=1).round(3)

        # keep just the column 'return_period_flow'
        rp_df = rp_df[['river_id', 'return_period', 'return_period_flow']]

        # Check if rp_df is empty
        if rp_df.empty:
            # print(f"Skipping processing for {DEM_Tile} because rp_df is empty.")
            dam.reanalysis_csv = None
            dam.dam_shp = None
            rivids_int = None
            # Note: StrmShp_filtered_gdf is defined in the parent function
            return dam.reanalysis_csv, dam.dam_shp, rivids_int, StrmShp_filtered_gdf

        # Convert 'return_period' to category dtype
        rp_df['return_period'] = rp_df['return_period'].astype('category')

        # Pivot the table
        rp_pivot_df = rp_df.pivot_table(index='river_id', columns='return_period', values='return_period_flow',
                                        aggfunc='mean', observed=True)

        # Rename columns to indicate return periods
        rp_pivot_df = rp_pivot_df.rename(columns={col: f'rp{int(col)}' for col in rp_pivot_df.columns})

        # Combine the results from retrospective and return periods data
        # final_df = pd.concat([combined_df, rp_pivot_df], axis=1)
        final_df = pd.concat([fdc_df, rp_pivot_df], axis=1)

    # --- 5. FINAL CSV WRITING ---
    if final_df is not None:
        final_df['COMID'] = final_df.index
        target_column = 'COMID'
        columns = [target_column] + [col for col in final_df.columns if col != target_column]
        final_df = final_df[columns]

        # Add Safety Factors / Baseflow overrides
        for col in final_df.columns:
            if col in ['qout_max', 'rp100']:
                if f'{col}_premium' not in final_df.columns:
                    final_df[f'{col}_premium'] = round(final_df[col] * 1.5, 3)

        if dam.known_baseflow is not None:
            final_df['known_baseflow'] = dam.known_baseflow
        if dam.known_channel_forming_discharge is not None:
            final_df['known_channel_forming_discharge'] = dam.known_channel_forming_discharge

        print('Writing final DataFrame to CSV')
        final_df.to_csv(dam.reanalysis_csv, index=False)

    return rivids_int, StrmShp_filtered_gdf


def Process_and_Write_Retrospective_Data_for_Dam(dam: RathCelonDam):
    """
    Finds the dam location, identifies upstream/downstream river segments,
    and then calls create_reanalysis to handle flow data processing.

    Parameters
    ----------
    dam: A Dam object... i'll write more later
    -------

    """
    # Load the dam data in as a geodataframe
    print('Process_and_Write_Retrospective_Data_for_Dam: Load the dam data in as a geodataframe')
    dam_gdf = pd.read_csv(dam.csv_path)
    dam_gdf = gpd.GeoDataFrame(dam_gdf, geometry=gpd.points_from_xy(dam_gdf['longitude'], dam_gdf['latitude']),
                               crs="EPSG:4269")

    # Filter the dam data to the dam of interest
    print('Process_and_Write_Retrospective_Data_for_Dam: Filter the dam data to the dam of interest')
    print(dam_gdf.tail())
    print(f'dam_id = {dam.dam_id}')
    dam_gdf = dam_gdf[dam_gdf[dam.id_field] == dam.dam_id]

    # Ensure there is at least one row remaining
    print('Process_and_Write_Retrospective_Data_for_Dam: Ensure there is at least one row remaining')
    if dam_gdf.empty:
        raise ValueError("No matching dam found for the given dam_id.")

    # Reset index to avoid index errors
    print('Process_and_Write_Retrospective_Data_for_Dam: Reset index to avoid index errors')
    dam_gdf = dam_gdf.reset_index(drop=True)

    # Print stage info
    print('Process_and_Write_Retrospective_Data_for_Dam: Convert both GeoDataFrames to a common projected CRS')

    # save the StrmShp CRS to convert StrmShp_filtered_gdf back to after the distance calculation
    flowline_crs = dam.flowline_gdf.crs

    # Determine an appropriate UTM zone using GeoPandas
    dam.flowline_gdf = dam.flowline_gdf[dam.flowline_gdf.geometry.notnull()]
    dam.flowline_gdf = dam.flowline_gdf[~dam.flowline_gdf.geometry.is_empty]

    if not dam.flowline_gdf.empty:
        utm_crs = dam.flowline_gdf.estimate_utm_crs()
    else:
        raise ValueError(f"{dam.flowline_gdf} has no valid geometries for UTM estimation.")

    # Reproject both GeoDataFrames to the UTM CRS
    dam_gdf = dam_gdf.to_crs(utm_crs)
    dam.flowline_gdf = dam.flowline_gdf.to_crs(utm_crs)

    # Distance calculation
    print('Process_and_Write_Retrospective_Data_for_Dam: Find the closest stream using distance calculation')
    dam.flowline_gdf['distance'] = dam.flowline_gdf.distance(dam_gdf.geometry.iloc[0])
    dam.flowline_gdf = dam.flowline_gdf.sort_values('distance')
    filtered_flowline_gdf = dam.flowline_gdf.head(1)

    # convert flowline_gdf and filtered_flowline_gdf back to its original CRS for ARC to use
    dam.flowline_gdf = dam.flowline_gdf.to_crs(flowline_crs)
    filtered_flowline_gdf = filtered_flowline_gdf.to_crs(flowline_crs)

    # # Use the 'LINKNO' and 'DSLINKNO' fields to find the stream upstream and downstream of the dam

    # current_rivid = StrmShp_filtered_gdf['LINKNO'].values[0]
    # downstream_rivid = StrmShp_filtered_gdf['DSLINKNO'].values[0]
    # upstream_gdf = dam.flowline_gdf[dam.flowline_gdf['DSLINKNO'] == current_rivid]
    # downstream_gdf = dam.flowline_gdf[dam.flowline_gdf['LINKNO'] == downstream_rivid]

    # set the field names equal to the values as they appear in GEOGLOWS or NHDPlus
    # This is already set in dam.rivid_field by the classes.py logic
    if dam.rivid_field == 'LINKNO':
        ds_rivid_field = 'DSLINKNO'
        stream_order_field = 'strmOrder'

    else:  # rivid_field == 'hydroseq'
        ds_rivid_field = 'dnhydroseq'
        stream_order_field = 'streamorde'

    # Use the 'LINKNO' and 'DSLINKNO' fields to find the stream upstream and downstream of the dam
    print('Process_and_Write_Retrospective_Data_for_Dam: Use the LINKNO and DSLINKNO (hydroseq and dnhydroseq) '
          'fields to find the stream upstream and downstream of the dam')
    current_rivid = filtered_flowline_gdf[dam.rivid_field].values[0]
    downstream_rivid = filtered_flowline_gdf[ds_rivid_field].values[0]

    # Find the upstream segment (if needed)
    print('Process_and_Write_Retrospective_Data_for_Dam: Find the upstream segment (if needed)')
    upstream_gdf = dam.flowline_gdf[dam.flowline_gdf[ds_rivid_field] == current_rivid]
    # Select the upstream segment with the highest Stream Order, if needed

    if not upstream_gdf.empty:
        if stream_order_field in upstream_gdf.columns:
            # Use the highest stream order if available
            upstream_gdf = upstream_gdf.loc[[upstream_gdf[stream_order_field].idxmax()]]
        else:
            # Use all matching upstream segments
            print(f"Optional field '{stream_order_field}' not found — using all upstream matches.")
    else:
        print("No upstream segments found.")

    # Initialize a list to store the downstream segments.
    downstream_segments = []

    # Start with the dam's downstream segment.
    current_downstream_rivid = downstream_rivid

    # Loop to find up to 2 downstream segments. lol
    print('Process_and_Write_Retrospective_Data_for_Dam: Loop to find up to ... downstream segments.')
    for i in range(8):
        # Find the stream segment whose LINKNO matches the current downstream rivid.
        segment = dam.flowline_gdf[dam.flowline_gdf[dam.rivid_field] == current_downstream_rivid]

        # If no segment is found, break the loop.
        if segment.empty:
            print(f"No downstream segment found after {i} segments.")
            break

        # Append the found segment to our list.
        downstream_segments.append(segment)

        # Update the current_downstream_rivid to the DSLINKNO of the found segment.
        # This will be used to find the next downstream segment.
        current_downstream_rivid = segment[ds_rivid_field].values[0]

    # Combine the downstream segments into one GeoDataFrame.
    print('Process_and_Write_Retrospective_Data_for_Dam: Combine the downstream segments into one GeoDataFrame.')
    if downstream_segments:
        downstream_gdf = pd.concat(downstream_segments, ignore_index=True)
    else:
        # If no downstream segments were found, create an empty GeoDataFrame.
        downstream_gdf = gpd.GeoDataFrame()

    # merge the StrmShp_filtered_gdf, upstream_gdf, and downstream_gdf into a single geodataframe
    print(
        'Process_and_Write_Retrospective_Data_for_Dam: merge the filtered_flowline_gdf, upstream_gdf, and downstream_gdf into a single geodataframe')
    StrmShp_filtered_gdf = pd.concat([filtered_flowline_gdf, upstream_gdf, downstream_gdf])

    StrmShp_filtered_gdf.to_file(dam.dam_shp)
    StrmShp_filtered_gdf[dam.rivid_field] = StrmShp_filtered_gdf[dam.rivid_field].astype(int)

    # create a list of river IDs
    rivids_int = StrmShp_filtered_gdf[dam.rivid_field].astype(int).to_list()

    print(f'Rivids for dam {dam.rivid_field}: {rivids_int}')

    create_reanalysis(dam, rivids_int, utm_crs, StrmShp_filtered_gdf)

    # Return the combined DataFrame as a Dask DataFrame
    return rivids_int, StrmShp_filtered_gdf
