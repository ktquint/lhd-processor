import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cross_sections(combined_gdf, output_dir, lhd_id):
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=len(combined_gdf) - 1)

    plt.figure()
    for i in range(len(combined_gdf)):
        y_1 = combined_gdf['XS1_Profile'].iloc[i]
        y_2 = combined_gdf['XS2_Profile'].iloc[i]
        x_1 = [0 - j * combined_gdf['Ordinate_Dist'].iloc[i] for j in range(len(y_1))]
        x_2 = [0 + j * combined_gdf['Ordinate_Dist'].iloc[i] for j in range(len(y_2))]

        INVALID_THRESHOLD = -1e5
        x = x_1[::-1] + x_2
        y = y_1[::-1] + y_2

        x_clean = []
        y_clean = []
        for xi, yi in zip(x, y):
            if yi > INVALID_THRESHOLD:
                x_clean.append(xi)
                y_clean.append(yi)

        color = cmap(norm(i))
        label = f'Downstream Cross-section {i}' if i > 0 else 'Upstream Cross-section'
        plt.plot(x_clean, y_clean, label=label, color=color)

    plt.legend(title="Cross-Sections", loc='best', fontsize='small')
    plt.ylabel('Elevation (m)')
    plt.xlabel('Lateral Distance (m)')

    png_output = os.path.join(output_dir, f'Reach Cross-Sections at LHD No. {lhd_id}.png')
    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rating_curves(curve_file, output_dir, lhd_id):
    x = np.linspace(1, 1000, 100)
    plt.figure(figsize=(10, 6))

    for index, row in curve_file.iterrows():
        a = row['depth_a']
        b = row['depth_b']
        y = a * x ** b
        label = f'Upstream Rating Curve {index}' if index == 0 else f'Downstream Rating Curve No. {index}'
        plt.plot(x, y, label=f'{label}: $y = {a:.3f} x^{{{b:.3f}}}$')

    plt.xlabel('Flow (m$^{3}$/s)')
    plt.ylabel('Depth (m)')
    plt.title(f'Downstream Rating Curves at LHD No. {lhd_id}')
    plt.legend(title="Rating Curve Equations", loc='best', fontsize='small')
    plt.grid(True)

    png_output = os.path.join(output_dir, f'Downstream Rating Curves at LHD No. {lhd_id}.png')
    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.close()


def plot_water_profiles(combined_gdf, full_database_df, output_dir, lhd_id, save=True):
    plt.figure()
    plt.plot(full_database_df.index, full_database_df['DEM_Elev'], color='dodgerblue', label='DEM Elevation')

    upstream_xs = combined_gdf.iloc[0]
    upstream_idx = full_database_df[
        (full_database_df["Row"] == upstream_xs['Row']) & (full_database_df["Col"] == upstream_xs['Col'])].index[0]
    plt.scatter(upstream_idx, upstream_xs['DEM_Elev'], label=f'Upstream Elevation')

    for i in range(1, len(combined_gdf)):
        ds_xs = combined_gdf.iloc[i]
        ds_idx = \
        full_database_df[(full_database_df["Row"] == ds_xs['Row']) & (full_database_df["Col"] == ds_xs['Col'])].index[0]
        plt.scatter(ds_idx, ds_xs['DEM_Elev'], label=f'Downstream Elevation No. {i}')

    plt.legend()
    plt.title(label=f'DEM WSE at LHD No. {lhd_id}')

    if save:
        png_output = os.path.join(output_dir, f'Longitudinal Water Surface Profile at LHD No. {lhd_id}.png')
        plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.close()
