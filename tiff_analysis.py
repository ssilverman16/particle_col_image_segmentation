#!/usr/bin/env python3

'''
Created on Wed Mar 12 19:00:25 2025

@author: vercelli

Change from hardcoding cells to 1,2,3 to actual cell types
If channels are split, there will be multiple h5 files in the same folder.
Compute the particle area for ONLY the RFP (3D05) channel, do cell counts for each channel, sum and divide by particle area to get cell density.
Output these to a csv file with a cell type column

Will know if it's split based on how many strains are in the file name.
If there's one strain in file name, that's the only cell type.
If there's two-three strains in file name and only one h5 file, they are both in the same file. Then 1 = strain 1, 2 = strain 2, 3 = strain 3 | particle.
    This order will always be in the order [3D05, 6B07, C3M10]
If there's two-three strains in file name and more than one h5 file. Then 1 strain in each file. Have to do
    combination of densities as mentioned above.
    If split:
        RFP -> 3D05 -> Magenta
        DAPI -> 6B07 -> Cyan
        GFP -> C3M10 -> Yellow

    If split, also have to check DAPI and RFP after doing cell positions. If DAPI (6B07) + RFP (3D05) at approx the same position -> 3D05, only DAPI -> 6B07.
    Do this by looking at regions of DAPI (6B07) and RFP (3D05) and seeing if there is overlap. If overlap above a certain threshold, then it's 3D05, otherwise it's 6B07.

Think if there's a way to add cell area to particle area if the cell is on the particle.
Is this already done by regions? Need to check if it is. Otherwise, ideally fill in the particle area.
    It is not, need to find overlap between particles and cells and count that as particle area.

# Comment out raw position csv and only write the combined one
# Have three quadrant pngs with 1 being raw, 2 being denoised, 3 being filtered with cell positions and clusters, 4 being area of particle that's filled in.
# For split channels, 3 and overlap only for the RFP.
# One png for the DAPI-RFP overlap.
# One png for all three channels overlaid, with DAPI-RFP overlap already done.
# CSV is just combined, raw commented out.
# Create density with updated DAPI counts and area ratios


'''

# Load libraries
import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, median_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

# Define constants
# cmap = colors.ListedColormap(['#c0a0c0', '#1f607f', 'black']) # Cells, Diatoms, Background
CMAP = {"3D05": "magenta", "6B07": "cyan", "C3M10": "yellow", "Particle": "#1f607f", "Background": "black", "Overlap": "red"}
BASE_TYPE_MAP = {1: "3D05", 2: "6B07", 3: "C3M10", 4: "Particle", 5: "Background", 6: "Overlap"}
CELL_TYPES = ["3D05", "6B07", "C3M10"]
CHANNELS = ["RFP", "DAPI", "GFP"]
CHANNEL_MAP = {"RFP": "3D05", "DAPI": "6B07", "GFP": "C3M10"}

TOP_LEVEL_FOLDER = '3D05_6B07_test/24h/Tp_3D05_6B07_C3M10_1_24h_60X_9' # Change this to the folder you want to process but you have to include the top level strain folder
# TOP_LEVEL_FOLDER = '3D05_6B07_test/24h/Tp_3D05_C3M10_1_24h_60X_15' # Change this to the folder you want to process but you have to include the top level strain folder
MIN_CELL_AREA = 20 # Change this to the minimum area of a cell
MIN_CLUSTER_AREA = 100 # Change this to the minimum area of a cluster
DENOISE_SIZE = 5 # Change this to the size of the denoising kernel
DILATION_RADIUS = 20
DISTANCE_THRESHOLD = 2
DAPI_RFP_OVERLAP_THRESHOLD = 0.1
def find_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"- {name}")

def process_h5_folder(cur_folder, h5_files):
    if len(h5_files) == 1:
        process_single_h5_file(cur_folder, h5_files[0])
    else:
        process_multiple_h5_files(cur_folder, h5_files)

def process_multiple_h5_files(cur_folder, h5_files):
    density_info_file_path, cell_pos_file_name = get_pos_and_density_file_names(cur_folder)
    # cell_pos_raw_file_name = cell_pos_file_name.replace("_cell_pos.csv", "_cell_pos_raw.csv")
    cell_pos_combined_file_name = cell_pos_file_name.replace("_cell_pos.csv", "_cell_pos_combined.csv")
    processed_folder = cur_folder.split("/")[-1]
    rfp_particle_area = None
    master_cell_area = {}
    master_cell_pos = {}
    master_cell_clusters = {}
    channels_to_combine = {"3D05": None, "6B07": None}
    dapi_cell_types = None
    # Process the file very similar to single file case but don't calculate densities right away
    # Wait until all files are processed to calculate densities based on the particle density for RFP (3D05)
    # To do this, we create a master cell area which is updated with each cell's area (background and particle are included but don't really matter and aren't used)
    for file in h5_files:
        full_file_path = os.path.join(cur_folder, file)
        cell_types = get_cell_types(file)
        strain_type = cell_types[1]
        channel = None
        for channel, strain in CHANNEL_MAP.items():
            if strain == strain_type:
                channel = channel
                break
        figure_name = f"{processed_folder}_{channel}"
        print("Processing channel:", channel)
        if len(cell_types) == 0:
            raise ValueError("Cell type not found in file path")
        cmap, norm = get_color_map(cell_types)
        base_name = full_file_path.replace('.h5', '') # Make sure this is correct
        with h5py.File(full_file_path, "r") as f:
            a_group_key = next(iter(f.keys())) # retrieve first key in the HDF5 file
            ds_arr = f[a_group_key][()]  # returns as a numpy array
        ds_arr = normalize_ds_arr(ds_arr)
        ds_arr_denoised = median_filter(ds_arr, size=DENOISE_SIZE)

        print("Getting cell positions and areas")
        # If channel is DAPI, don't get cell positions and areas. Wait until after RFP is processed to get DAPI positions
        cell_positions, cell_clusters, cell_area, particle_area = get_cell_positions_and_areas(ds_arr_denoised, cell_types)
        channels_to_combine[channel] = ds_arr_denoised
        if channel == "RFP": # Check if the first cell type (key 1) is 3D05
            rfp_particle_area = particle_area
            ds_arr_overlap, rfp_particle_area = recreate_particle_area(ds_arr_denoised, cell_types, particle_area)
        elif channel == "DAPI":
            dapi_cell_types = cell_types # Needed for update later
            ds_arr_overlap = None
        else:
            ds_arr_overlap = None

        create_channel_plots(ds_arr, cmap, norm, figure_name, base_name, ds_arr_denoised, ds_arr_overlap, cell_positions=cell_positions, cell_clusters=cell_clusters)
        if channel != "DAPI": # Don't update master for DAPI. This is done after the DAPI-RFP overlap is combined.
            master_cell_area.update(cell_area)
            master_cell_pos.update(cell_positions)
            master_cell_clusters.update(cell_clusters)

    if rfp_particle_area is None:
        raise ValueError("RFP particle area not found")
    # Combine DAPI and RFP, then get new DAPI cell positions and clusters. Finally, update master cell positions, clusters, and areas.
    # Write raw cell position info to csv file
    # write_cell_position_info(master_cell_pos, master_cell_clusters, cell_pos_raw_file_name) # Uncomment to write raw cell position info to csv file
    dapi_updated = combine_cell_positions_and_clusters(channels_to_combine)
    dapi_cell_positions, dapi_cell_clusters, dapi_cell_area, _ = get_cell_positions_and_areas(dapi_updated, dapi_cell_types)
    master_cell_pos.update(dapi_cell_positions)
    master_cell_clusters.update(dapi_cell_clusters)
    master_cell_area.update(dapi_cell_area)
    dapi_cmap, dapi_norm = get_color_map(dapi_cell_types)
    cmap, norm = get_color_map(BASE_TYPE_MAP)

    print("Visualizing DAPI-RFP overlap")
    # RFP will be the base channel, so we need to update the RFP image to have the required cell type numbers to the CELL_TYPES numbers
    # (so RFP will still be 1 but the particle will be 4, background will be 5, etc. So those need to have 2 added to them)
    rfp_updated = channels_to_combine["RFP"].copy()
    rfp_updated[rfp_updated == 2] = 4 # Particle
    rfp_updated[rfp_updated == 3] = 5 # Background
    final_rfp = rfp_updated.copy()
    visualize_dapi_overlap_results(channels_to_combine["DAPI"], rfp_updated, dapi_updated, cmap, norm, dapi_cmap, dapi_norm, processed_folder, base_name)

    # Get cell counts, densities, and area ratios
    cell_counts, cell_densities, cell_area_ratios = get_cell_counts_and_densities(master_cell_pos, master_cell_clusters, master_cell_area, rfp_particle_area)
    # Write density info to csv file
    write_density_info(density_info_file_path, processed_folder, cell_densities, cell_area_ratios, cell_counts)
    combined_channels = combine_channels(final_rfp, channels_to_combine["DAPI"], channels_to_combine["GFP"])
    output_name = f"{base_name}_combined_channels.png"
    create_plot(combined_channels, cmap, norm, output_name, cell_positions=master_cell_pos, cell_clusters=master_cell_clusters, title=f"{processed_folder} Combined Channels")
    # Write combined cell position csv file and write to CSV file
    write_cell_position_info(master_cell_pos, master_cell_clusters, cell_pos_combined_file_name)


def combine_channels(rfp_updated, dapi, gfp):
    rfp_updated[dapi == 1] = 2
    rfp_updated[gfp == 1] = 3
    return rfp_updated

def combine_cell_positions_and_clusters(channels_to_combine):
    print("Checking DAPI and RFP overlap")
    # Since these are single channels, 1 = cell and 2 = particle
    cell_to_be_removed = 4
    dapi_mask = (channels_to_combine["DAPI"] == 1)
    rfp_mask = (channels_to_combine["RFP"] == 1)

    # Label connected components in image1
    labeled_dapi = label(dapi_mask)

    # Get properties of each cell in image1
    regions_dapi = regionprops(labeled_dapi)

    # Create a mask to track which cells to remove
    cells_to_remove = np.zeros_like(dapi_mask, dtype=bool)
    # Check each DAPI cell
    for region in regions_dapi:
        # Get a mask for just this DAPI cell
        cell_mask = (labeled_dapi == region.label)

        # Calculate overlap with cells in RFP
        overlap = np.logical_and(cell_mask, rfp_mask)
        # Calculate fraction of cell area that overlaps with image2 cells
        overlap_fraction = np.sum(overlap) / region.area

        # If overlap exceeds threshold, mark this cell for removal
        if overlap_fraction > DAPI_RFP_OVERLAP_THRESHOLD:
            cells_to_remove = np.logical_or(cells_to_remove, cell_mask)

    # Create the merged image
    dapi_combined = channels_to_combine["DAPI"].copy()

    # Remove the cells that are significantly overlapped
    # Set them to the same value as in image2
    dapi_combined[cells_to_remove] = cell_to_be_removed
    return dapi_combined

def visualize_dapi_overlap_results(original_dapi, original_rfp, updated_dapi, cmap, norm, dapi_cmap, dapi_norm, base_name, output_name):
    """
    Visualize the overlap detection results.

    Parameters:
    -----------
    original_dapi : numpy.ndarray
        Original DAPI image (with 1 = cell)
    original_rfp : numpy.ndarray
        Original RFP image (with 1 = cell)
    modified_dapi : numpy.ndarray
        DAPI image after removing overlapped cells
    cells_to_remove : numpy.ndarray
        Boolean mask indicating which cells were removed
    """

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f"{base_name} DAPI-RFP Overlap", fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.8)

    # Original DAPI
    axes[0, 0].imshow(original_dapi, cmap=dapi_cmap, norm=dapi_norm)
    axes[0, 0].set_title('Original DAPI')

    # Original RFP
    axes[0, 1].imshow(original_rfp, cmap=cmap, norm=norm)
    axes[0, 1].set_title('Original RFP')

    # Set DAPI=1 pixels to 2
    original_rfp[original_dapi == 1] = 2

    axes[1, 0].imshow(original_rfp, cmap=cmap, norm=norm)
    axes[1, 0].set_title('DAPI overlaid with RFP')

    # Result with removed cells highlighted
    axes[1, 1].imshow(updated_dapi, cmap=dapi_cmap, norm=dapi_norm)
    axes[1, 1].set_title('Updated DAPI (Red=removed cells)')
    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background color
            continue
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=cell_type))

    # Add the legend below the subplots
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=len(legend_elements), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{output_name}_dapi_rfp_overlap.png")
    plt.close()


def create_channel_plots(raw_arr, cmap, norm, base_name, output_name, denoised_arr, overlap_arr=None, cell_positions=None, cell_clusters=None):
    # Create figure
    fig = plt.figure(figsize=(16, 16))

    if overlap_arr is None:
        # Create 2x2 grid
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1])

        # Create the three axes
        ax1 = fig.add_subplot(gs[0, 0])  # Top left
        ax2 = fig.add_subplot(gs[0, 1])  # Top right
        ax3 = fig.add_subplot(gs[1, :])  # Bottom spanning both columns

        # Convert to numpy array format for consistent indexing
        axes = np.array([[ax1, ax2], [ax3, None]])
    else:
        # Create regular 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    fig.suptitle(base_name, fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.9)

    axes[0, 0].imshow(raw_arr, cmap=cmap, norm=norm)
    axes[0, 0].set_title('Raw')

    axes[0, 1].imshow(denoised_arr, cmap=cmap, norm=norm)
    axes[0, 1].set_title('De-Noised')


    axes[1, 0].imshow(denoised_arr, cmap=cmap, norm=norm)
    axes[1, 0].set_title('Cell Positions')


    # Plots cell positions if cell_positions is not None and there are any cell positions
    if cell_positions is not None and any(cell_positions.values()):
        all_positions = np.concatenate([np.array(positions) for positions in cell_positions.values()])
        axes[1, 0].scatter(all_positions[:, 1], all_positions[:, 0], s=1, c='red', marker='.')

    # Plots cell clusters if cell_clusters is not None and there are any cell clusters
    if cell_clusters is not None and any(cell_clusters.values()):
        all_clusters = np.concatenate([np.array(clusters) for clusters in cell_clusters.values()])
        axes[1, 0].scatter(all_clusters[:, 1], all_clusters[:, 0], s=3, c='blue', marker='.')

    if overlap_arr is not None:
        axes[1, 1].imshow(overlap_arr, cmap=cmap, norm=norm)
        axes[1, 1].set_title('Particle Overlap')

    # Create legend patches for each color in the colormap
    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background color
            continue
        if cell_type in ["Overlap"] and overlap_arr is None:
            continue
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=cell_type))

    # Add red dot for cell positions and blue dot for clusters
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='red',
                                    label='Cell Positions', markersize=10))
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='blue',
                                    label='Cell Clusters', markersize=10))

    # Add the legend below the subplots
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=len(legend_elements), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{output_name}_plots.png")
    plt.close()


def get_pos_and_density_file_names(cur_folder):
    cur_folder_split = cur_folder.split("/")
    density_info_file_name = f"{cur_folder_split[-3]}_{cur_folder_split[-2]}_cell_density_info.csv"
    density_info_file_path = os.path.join(cur_folder, "..", density_info_file_name)
    cell_pos_file_name = os.path.join(cur_folder, f"{cur_folder_split[-1]}_cell_pos.csv")
    return density_info_file_path, cell_pos_file_name

def process_single_h5_file(cur_folder, file_path):
    print("Processing file: ", file_path)
    full_file_path = os.path.join(cur_folder, file_path)
    density_info_file_path, cell_pos_file_name = get_pos_and_density_file_names(cur_folder)
    base_name = full_file_path.replace('.h5', '')
    processed_folder = cur_folder.split("/")[-1]

    cell_types = get_cell_types(file_path)
    if len(cell_types) == 0:
        raise ValueError("Cell type not found in file path")
    cmap, norm = get_color_map(cell_types)

    with h5py.File(full_file_path, "r") as f:
        a_group_key = next(iter(f.keys())) # retrieve first key in the HDF5 file
        ds_arr = f[a_group_key][()]  # returns as a numpy array
    ds_arr = normalize_ds_arr(ds_arr)
    ds_arr_denoised = median_filter(ds_arr, size=DENOISE_SIZE)

    # Get cell positions and densities, note that these are dictionaries mapping cell type to an array of values, or single value for densities
    print("Getting cell positions and densities")
    cell_positions, cell_clusters, cell_area, particle_area = get_cell_positions_and_areas(ds_arr_denoised, cell_types)
    print("Getting cell counts and densities")
    cell_count, cell_density, cell_area_ratio = get_cell_counts_and_densities(cell_positions, cell_clusters, cell_area, particle_area)
    ds_arr_recreated, particle_area = recreate_particle_area(ds_arr_denoised, cell_types, particle_area)

    # Create plots, write position and density info to csv
    print("Creating plots")
    create_single_plots(ds_arr, cmap, norm, processed_folder, base_name,ds_arr_denoised, ds_arr_recreated, cell_positions=cell_positions, cell_clusters=cell_clusters)
    print("Writing position and density info to csv")
    write_cell_position_info(cell_positions, cell_clusters, cell_pos_file_name)
    write_density_info(density_info_file_path, processed_folder, cell_density, cell_area_ratio, cell_count)

# Checks file path for cell types and channels
# If just cell_types are found, returns a list of cell types found
# If a channel is found, returns a list with the single cell type that corresponds to that channel
# If more than one channel is found, raises an error
# Returns dictionary mapping cell value to cell type i.e. {1: "3D05", 2: "6B07", 3: "C3M10", 4: "particle", 5: "background"}
def get_cell_types(file_path):
    cell_types = []
    channels = []
    for cell_type in CELL_TYPES:
        if cell_type in file_path.upper():
            cell_types.append(cell_type)
    for channel in CHANNELS:
        if channel in file_path.upper():
            channels.append(channel)
    if len(channels) > 1:
        raise ValueError("More than one channel found in file path")
    if len(channels) == 1:
        cell_types = [CHANNEL_MAP[channels[0]]]
    cell_type_map = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_map[i+1] = cell_type
    cell_type_map[i+2] = "Particle"
    cell_type_map[i+3] = "Background"
    cell_type_map[i+4] = "Overlap"
    return cell_type_map

def get_color_map(cell_type_map):
    cell_colors = []
    bounds = []
    for cell_num, cell_type in cell_type_map.items():
        cell_colors.append(CMAP[cell_type])
        bounds.append(cell_num - .5)
    bounds.append(len(cell_type_map) + .5)
    cmap = colors.ListedColormap(cell_colors)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def normalize_ds_arr(ds_arr):
    # If shape is (2048,2048,1)
    if ds_arr.shape[-1] == 1:
        return np.squeeze(ds_arr)  # Removes single-dimensional entries
    # If shape is (1,2048,2048)
    elif ds_arr.shape[0] == 1:
        return ds_arr[0]
    elif ds_arr.shape[0] == 2048 and ds_arr.shape[1] == 2048:
        return ds_arr
    else:
        raise ValueError(f"DS arr shape is not (2048,2048,1) or (1,2048,2048) or (2048,2048). Shape: {ds_arr.shape}")

def create_plot(ds_arr, cmap, norm, file_name, cell_positions=None, cell_clusters=None, title=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(ds_arr, cmap=cmap, norm=norm, interpolation='None')
    if title is not None:
        plt.title(title, fontsize=20, y=0.98)

    # Plots cell positions if cell_positions is not None and there are any cell positions
    if cell_positions is not None and any(cell_positions.values()):
        all_positions = np.concatenate([np.array(positions) for positions in cell_positions.values() if len(positions) > 0])
        plt.scatter(all_positions[:, 1], all_positions[:, 0], s=1, c='red', marker='.')

    # Plots cell clusters if cell_clusters is not None and there are any cell clusters
    if cell_clusters is not None and any(cell_clusters.values()):
        all_clusters = np.concatenate([np.array(clusters) for clusters in cell_clusters.values() if len(clusters) > 0])
        plt.scatter(all_clusters[:, 1], all_clusters[:, 0], s=3, c='blue', marker='.')

    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background and overlap colors
            continue
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=cell_type))

    # Add red dot for cell positions and blue dot for clusters
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='red',
                                    label='Cell Positions', markersize=10))
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='blue',
                                    label='Cell Clusters', markersize=10))

    # Add the legend below the subplots
    plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=len(legend_elements), frameon=False)
    # Adjust figure layout to make room for title and legend
    plt.subplots_adjust(top=0.85, bottom=0.20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def create_single_plots(raw_arr, cmap, norm, base_name, output_name, denoised_arr, overlap_arr, cell_positions=None, cell_clusters=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(base_name, fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.9)

    axes[0, 0].imshow(raw_arr, cmap=cmap, norm=norm)
    axes[0, 0].set_title('Raw')

    axes[0, 1].imshow(denoised_arr, cmap=cmap, norm=norm)
    axes[0, 1].set_title('De-Noised')


    axes[1, 0].imshow(denoised_arr, cmap=cmap, norm=norm)
    axes[1, 0].set_title('Cell Positions')
    # Plots cell positions if cell_positions is not None and there are any cell positions
    if cell_positions is not None and any(cell_positions.values()):
        all_positions = np.concatenate([np.array(positions) for positions in cell_positions.values()])
        axes[1, 0].scatter(all_positions[:, 1], all_positions[:, 0], s=1, c='red', marker='.')

    # Plots cell clusters if cell_clusters is not None and there are any cell clusters
    if cell_clusters is not None and any(cell_clusters.values()):
        all_clusters = np.concatenate([np.array(clusters) for clusters in cell_clusters.values()])
        axes[1, 0].scatter(all_clusters[:, 1], all_clusters[:, 0], s=3, c='blue', marker='.')

    axes[1, 1].imshow(overlap_arr, cmap=cmap, norm=norm)
    axes[1, 1].set_title('Particle Overlap')
    # Create legend patches for each color in the colormap
    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background and overlap colors
            continue
        legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor=color, label=cell_type))

    # Add red dot for cell positions and blue dot for clusters
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='red',
                                    label='Cell Positions', markersize=10))
    legend_elements.append(plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='blue',
                                    label='Cell Clusters', markersize=10))

    # Add the legend below the subplots
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=len(legend_elements), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{output_name}_plots.png")
    plt.close()



def get_cell_positions_and_areas(z_slice, cell_types):
    label_im = label(z_slice) # converts ds_arr into a labeled image where each connected region gets a unique integer label
    regions = regionprops(label_im) # finds connected regions in label_im, where each detected region becomes a regionprops
    # object with properties (region.centroid, region.area, region.coords)
    cell_pos = {}
    cell_clusters = {}
    cell_area = {}

    for region in regions:
        region_type = get_type(region, z_slice) # Returns 1,2,3,4,5 etc.
        cell_type = cell_types[region_type] # Returns "3D05", "6B07", "C3M10", "particle", "background"
        # Handle background and particle cases
        if cell_type not in CELL_TYPES:
            if cell_type not in cell_area:
                cell_area[cell_type] = region.area
            else:
                cell_area[cell_type] += region.area
            continue
        # Handle cell cases
        if cell_type not in cell_area:
            # Initialize cell area, positions, and clusters for this cell type
            cell_area[cell_type] = 0
            cell_pos[cell_type] = []
            cell_clusters[cell_type] = []

        if region.area >= MIN_CELL_AREA and region.area < MIN_CLUSTER_AREA:
            cell_area[cell_type] += region.area
            cell_pos[cell_type].append(region.centroid) # if true, stores the region's centroid (region.centroid) in cell_pos
        if region.area >= MIN_CLUSTER_AREA:
            cell_area[cell_type] += region.area
            cell_clusters[cell_type].append(region.centroid)
    return cell_pos, cell_clusters, cell_area, cell_area["Particle"]


def recreate_particle_area(ds_arr, cell_types, particle_area):
    """
    Calculates "real" particle area by filling in the overlap area between particles and cells
    Returns the updated ds_arr and the new particle area
    """
    # Find the key corresponding to the "particle" value in cell_types
    particle_label = None
    overlap_label = None
    for key, value in cell_types.items():
        if value == "Particle":
            particle_label = key
        if value == "Overlap":
            overlap_label = key
    for cell_type_label, cell_type in cell_types.items():
        if cell_type not in CELL_TYPES:
            continue
        updated_ds_arr, overlap_area = fill_particle_area(ds_arr, particle_label, cell_type_label, overlap_label)
        particle_area += overlap_area
        ds_arr = updated_ds_arr
    return ds_arr, particle_area

# Binary way of filling particle area. Only includes cells that are completely enclosed by particles.
# def fill_particle_area(ds_arr, particle_label, cell_label, overlap_label):
#     # Create a boolean mask where True indicates the presence of particles
#     particle_mask = ds_arr == particle_label

#     # Create a boolean mask where True indicates the presence of cells
#     cell_mask = ds_arr == cell_label

#     # Fill holes in the particle mask using scipy's binary_fill_holes
#     # This creates a new mask where any holes (False regions) completely
#     # surrounded by True values are filled in
#     filled_particle = binary_fill_holes(particle_mask)

#     # Find the overlap between the filled particle area and the cell mask
#     # This identifies cells that are completely enclosed by particles
#     enclosed_cell_mask = np.logical_and(filled_particle, cell_mask)

#     # Count how many pixels are in the enclosed area
#     overlap_area = np.sum(enclosed_cell_mask)

#     # Create a copy of the original array to modify
#     updated_ds_arr = ds_arr.copy()

#     # Change the values in the enclosed areas to the particle label
#     updated_ds_arr[enclosed_cell_mask] = overlap_label
#     print("Overlap area: ", overlap_area)
#     return updated_ds_arr, overlap_area

def fill_particle_area(ds_arr, particle_label, cell_label, overlap_label):

    # Create a boolean mask where True indicates the presence of particles
    particle_mask = ds_arr == particle_label

    # Create a boolean mask where True indicates the presence of cells
    cell_mask = ds_arr == cell_label

    # Dilates (expands) the particle mask by a disk basically make the particle larger (fill in potential gaps)
    dilated_particle = binary_dilation(particle_mask, disk(DILATION_RADIUS))

    # Find the distance transform from particle boundaries
    # ~particle mask is the inverse of the particle mask (i.e. where the particle is not, makes non-particle pixels True)
    # distance_transform_edt then calculates the Euclidean distance from each True pixel to the nearest False pixel
    # This gives a distance map where pixels get higher values the farther they are from any particle.
    dist_transform = ndimage.distance_transform_edt(~particle_mask)

    # Find potential overlap based on distance
    # Looks for cells that are within a certain distance threshold of the particle boundary
    potential_overlap = cell_mask & (dist_transform < DISTANCE_THRESHOLD)

    # Find overlap based on dilation.
    # Finds cells that are within dilated particle regions (stronger evidence of overlap)
    overlap_regions = cell_mask & dilated_particle

    # Combine the dilation approach and distance approach
    combined_overlap = potential_overlap | overlap_regions

    # Create a copy of the original array to modify
    updated_ds_arr = ds_arr.copy()

    # Change the values in the combined overlap areas to the overlap label
    updated_ds_arr[combined_overlap] = overlap_label

    return updated_ds_arr, np.sum(combined_overlap)



def get_cell_counts_and_densities(cell_pos, cell_clusters, cell_area, particle_area):
    # Calculate cell counts, density, and area ratios
    cell_count = {}
    cell_density = {}
    cell_area_ratio = {}
    for cell_type, area in cell_area.items():
        if cell_type not in CELL_TYPES:
            continue
        cell_count[cell_type] = len(cell_pos[cell_type]) + len(cell_clusters[cell_type])
        cell_density[cell_type] = round(cell_count[cell_type] / particle_area, 5)
        cell_area_ratio[cell_type] = round(area / particle_area, 5)
    return cell_count, cell_density, cell_area_ratio

def get_type(region, data):
    point = region.coords[0] # retrieves one coordinate (first pixel of the region)
    point_val = data[point[0], point[1]]
    return point_val

def write_cell_position_info(cell_positions, cell_clusters, csv_output_file):
    with open(csv_output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["x_pos", "y_pos", "strain_type", "cell_type"])
        for strain_type, pos in cell_positions.items():
            for p in pos:
                writer.writerow([round(p[1], 2), round(p[0], 2), strain_type, "cell"])
        for strain_type, cluster in cell_clusters.items():
            for c in cluster:
                writer.writerow([round(c[1], 2), round(c[0], 2), strain_type, "cluster"])

def write_density_info(csv_output_file, h5_folder, cell_density, cell_area_ratio, cell_count):
    header = ["folder", "strain", "cell_density", "cell_area_ratio", "cell_count"]
    # Read existing data if file exists
    existing_data = []
    path_exists = os.path.exists(csv_output_file)
    data_exists = False
    if path_exists:
        with open(csv_output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if row[0] == h5_folder:
                    data_exists = True
                else:
                    existing_data.append(row)

    # If folder exists in data, write out previous data
    if data_exists:
        # Rewrite entire file with updated data
        with open(csv_output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(existing_data)
        # Just append if folder not found
    with open(csv_output_file, 'a') as f:
        writer = csv.writer(f)
        if not path_exists:
            writer.writerow(header)
        for strain in cell_density:
            writer.writerow([h5_folder, strain, cell_density[strain], cell_area_ratio[strain], cell_count[strain]])






# Retrieves a dictionary mapping each folder to a list of h5 files in that folder
# This way, we can see if there are multiple h5 files in the same folder and use
# That information to determine if it has been split into multiple channels
def get_h5_files_recursively(folder_path):
    h5_files = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                h5_file = os.path.join(root, file)
                h5_folder = os.path.dirname(h5_file)
                if h5_folder not in h5_files:
                    h5_files[h5_folder] = []
                h5_files[h5_folder].append(file)
    return h5_files


def main():
    print("Processing folder: ", TOP_LEVEL_FOLDER)
    h5_files = get_h5_files_recursively(TOP_LEVEL_FOLDER)

    for folder, files in h5_files.items():
        print("Processing folder: ", folder)
        process_h5_folder(folder, files)

    print("Processing complete")

if __name__ == "__main__":
    main()
