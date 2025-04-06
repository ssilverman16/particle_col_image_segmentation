#!/usr/bin/env python3

"""
Created on Wed Mar 12 19:00:25 2025
@author: marshallyale

# Of layers is variable depending on the strain condition anything with 6B07 and that's split into channels
To update: how script parses different .h5 files and determines which strain is which.
- 6B07 condition: .h5 files split into DAPI channel (6B07) & RFP channel (particle & background).
    6B07: If multiple h5 files but only 6B07 strain, the RFP channel will be 1 = particle, 2=background, NO CELL! However, would also not need to look at overlap conditions.
- 3D05 + 6B07 condition: .h5 files split into DAPI channel (= 6B07 & 3D05) & RFP channel (= 3D05, particle, & background).
- 6B07 + C3M10 condition: .h5 files split into DAPI channel (= 6B07 & C3M10), GFP channel (= C3M10), & RFP channel (= particle, & background).
    RFP channel has NO cell_type. Also have to do the same thing as when RFP goes through the channel. Have to look at overlap between GFP and DAPI and then
    if overlap, take DAPi.

    These are only for the case where there are multiple H5 files, but there are only a couple of channels in it.

Cells and channels:
Extract strain types from folder name (Happens at top level)
Extract channel from file name (Happens for each file)

Add ability to see which cells are near other cells and if they are, add them as part of a cluster.
AKA if a cell from a certain strain is within a certain distance of a cell from another strain, then include both of them as part of a cluster.
Start with non inter-cell comparisons
Want output to be a csv with with: centroid_x, centroid_y, type (Cell_type or combined), area (that cell's area or for combined, the area of each cell type+combined)
Get the centroids for all of the regions and then can combine if they are close enough for cell types comparison. Then do the same for combined.
When channels are broken up, want to take the combined image and run it through this.
Add column for cell area ratio which is the area of the cell divided by the area of the particle.
Check combined cell areas
Maybe try to get ride of the stuff that's not cell ()
"""

import csv
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, median_filter
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import binary_dilation, disk

# Define constants
CMAP = {
    "3D05": "#c0a0c0",
    "6B07": "cyan",
    "C3M10": "yellow",
    "Particle": "#1f607f",
    "Background": "black",
    # "Overlap": "#1f607f",
}
BASE_TYPE_MAP = {1: "3D05", 2: "6B07", 3: "C3M10", 4: "Particle", 5: "Background"}
CELL_TYPES = ["3D05", "6B07", "C3M10"]
CHANNELS = ["RFP", "DAPI", "GFP"]
CHANNEL_MAP = {"RFP": "3D05", "DAPI": "6B07", "GFP": "C3M10"}
STRAIN_MAP = {"3D05": "RFP", "6B07": "DAPI", "C3M10": "GFP"}

# TOP_LEVEL_FOLDER = "/Volumes/WD_Elements/3D05_C3M10/120h"  # Change this to the folder you want to process but you have to include the top level strain folder
# TOP_LEVEL_FOLDER = "3D05_6B07_test/24h/Tp_3D05_C3M10_1_24h_60X_15"
TOP_LEVEL_FOLDER = "files_w_6B07"
MIN_CELL_AREA = {"3D05": 20, "6B07": 20, "C3M10": 20}  # Change this to the minimum area of a cell (in sq. pixels)

MIN_CLUSTER_AREA = {
    "3D05": 200,
    "6B07": 200,
    "C3M10": 370,
}  # Change this to the minimum area of a cluster (in sq. pixels)

DENOISE_SIZE = 5  # Change this to the size of the denoising kernel (in pixels)
DILATION_RADIUS = (
    20  # Change this to the radius you want to dilate the particle by. This helps find cells on the particle.
)
DISTANCE_THRESHOLD = (
    2  # Change this to the distance threshold you want to use for the distance transform. This does same as above
)
CELL_CLUSTER_DISTANCE_THRESHOLD = 10
DAPI_RFP_OVERLAP_THRESHOLD = 0.1  # Change this to the threshold you want to use for the DAPI-RFP overlap.
PX_TO_UM_CONV = 9.95  # Change this to the conversion factor for pixels --> microns


def process_h5_folder(cur_folder, h5_files):
    if len(h5_files) == 1:
        process_single_h5_file(cur_folder, h5_files[0])
    else:
        process_multiple_h5_files(cur_folder, h5_files)


def process_multiple_h5_files(cur_folder, h5_files):
    density_info_file_path, cell_pos_file_name = get_pos_and_density_file_names(cur_folder)
    cell_pos_raw_file_name = cell_pos_file_name.replace("_cell_pos.csv", "_cell_pos_raw.csv")
    cell_pos_combined_file_name = cell_pos_file_name.replace("_cell_pos.csv", "_cell_pos_combined.csv")
    processed_folder = cur_folder.split("/")[-1]
    rfp_particle_area = None
    master_cell_pos = {}
    master_cell_clusters = {}
    channel_ds_arrs = {}
    dapi_cell_types = None
    cell_strains = get_strains_from_file(cur_folder)
    print("Cell strains found in file", cell_strains)
    # Process the file very similar to single file case but don't calculate densities right away
    # Wait until all files are processed to calculate densities based on the particle density for RFP (3D05)
    # To do this, we create a master cell area which is updated with each cell's area (background and particle are included but don't really matter and aren't used)
    for file in h5_files:
        full_file_path = os.path.join(cur_folder, file)
        channel = get_channel_from_file(file)
        cell_types = get_cell_type_map_from_channel(cell_strains, channel)
        strain_type = cell_types[1] # This can be a particle instead of strain but it should continue before needing the variable.
        figure_name = f"{processed_folder}_{channel}"
        print("Processing channel:", channel)
        if len(cell_types) == 0:
            raise ValueError("Cell type not found in file path")
        cmap, norm = get_color_map(cell_types)
        base_name = full_file_path.replace(".h5", "")  # Make sure this is correct
        with h5py.File(full_file_path, "r") as f:
            a_group_key = next(iter(f.keys()))  # retrieve first key in the HDF5 file
            ds_arr = f[a_group_key][()]  # returns as a numpy array
        ds_arr = normalize_ds_arr(ds_arr)
        ds_arr_denoised: np.ndarray = median_filter(ds_arr, size=DENOISE_SIZE)

        print("Getting cell positions and areas")
        cell_positions, cell_clusters, particle_area, _ = get_cell_positions_and_areas(ds_arr_denoised, cell_types)
        print("Finished getting cell positions")
        channel_ds_arrs[channel] = ds_arr_denoised
        if channel == "RFP":  # Check if the first cell type (key 1) is 3D05
            rfp_particle_area = particle_area
            ds_arr_overlap, rfp_particle_area = recreate_particle_area(ds_arr_denoised, cell_types, particle_area)
            if strain_type == "Particle": # Only the case if no cell on it
                continue
        elif channel == "DAPI":
            dapi_cell_types = cell_types  # Needed for update later
            ds_arr_overlap = None
        else:
            ds_arr_overlap = None
        if strain_type not in CELL_TYPES:
            raise ValueError(f"Strain type not in cell types. {strain_type}")
        create_channel_plots(
            ds_arr,
            strain_type,
            cmap,
            norm,
            figure_name,
            base_name,
            ds_arr_denoised,
            ds_arr_overlap,
            cell_positions=cell_positions,
            cell_clusters=cell_clusters,
        )
        master_cell_pos.update(cell_positions)
        master_cell_clusters.update(cell_clusters)

    if rfp_particle_area is None:
        raise ValueError("RFP particle area not found")
    # Combine DAPI and RFP, then get new DAPI cell positions and clusters. Finally, update master cell positions, clusters, and areas.
    # Write raw cell position info to csv file
    write_cell_position_info(
        master_cell_pos, master_cell_clusters, cell_pos_raw_file_name, rfp_particle_area
    )  # Uncomment to write raw cell position info to csv file
    cmap, norm = get_color_map(BASE_TYPE_MAP)
    if len(cell_strains) > 1:
        other_channel = channel_ds_arrs["GFP"] if cell_strains == ["6B07", "C3M10"] else channel_ds_arrs["RFP"]
        other_channel_name = "GFP" if cell_strains == ["6B07", "C3M10"] else "RFP"

        dapi_updated = combine_cell_positions_and_clusters(channel_ds_arrs["DAPI"], other_channel)
        dapi_cell_positions, dapi_cell_clusters, _, _ = get_cell_positions_and_areas(dapi_updated, dapi_cell_types)
        master_cell_pos["6B07"] = dapi_cell_positions["6B07"]
        master_cell_clusters["6B07"] = dapi_cell_clusters["6B07"]
        dapi_cmap, dapi_norm = get_color_map(dapi_cell_types)

        print(f"Visualizing DAPI-{other_channel_name} overlap")
        # The other channel (RFP/GFP) will be the base channel, so we need to update the other image to have the required cell type numbers to the CELL_TYPES numbers
        #

        other_updated = other_channel.copy()
        other_updated[other_updated == 3] = 5  # Background
        other_updated[other_updated == 2] = 4  # Particle
        if other_channel_name == "GFP":
            other_updated[other_updated == 1] = 3

        visualize_dapi_overlap_results(
            channel_ds_arrs["DAPI"],
            other_updated,
            dapi_updated,
            cmap,
            norm,
            dapi_cmap,
            dapi_norm,
            processed_folder,
            base_name,
            other_channel_name
        )

    # Get cell counts, densities, and area ratios
    cell_counts, cell_densities, cell_area_ratios = get_cell_counts_and_densities(
        master_cell_pos, master_cell_clusters, rfp_particle_area
    )
    # Write density info to csv file
    write_density_info(density_info_file_path, processed_folder, cell_densities, cell_area_ratios, cell_counts)
    rfp_base_arr = channel_ds_arrs["RFP"].copy()
    get_rfp_base_arr(rfp_base_arr, cell_strains)
    combined_channels = combine_channels(rfp_base_arr, channel_ds_arrs, cell_strains)
    print("Creating merged plots")
    _, _, _, merged_clusters = get_cell_positions_and_areas(combined_channels, BASE_TYPE_MAP, merged=True)
    plot_original_vs_merged(combined_channels, merged_clusters, master_cell_clusters, BASE_TYPE_MAP, processed_folder, base_name)
    output_name = f"{base_name}_combined_channels.png"
    create_plot(
        combined_channels,
        cmap,
        norm,
        output_name,
        cell_positions=master_cell_pos,
        cell_clusters=master_cell_clusters,
        title=f"{processed_folder} Combined Channels",
    )

    # Write combined cell position csv file and write to CSV file
    write_cell_position_info(master_cell_pos, master_cell_clusters, cell_pos_combined_file_name, rfp_particle_area)
    merged_file_name = cell_pos_combined_file_name.replace("_cell_pos_combined.csv", "_merged_cell_pos.csv")
    write_merged_cell_position_info(merged_clusters, merged_file_name, rfp_particle_area)

def get_rfp_base_arr(rfp_arr, cell_strains):
    if cell_strains == ["6B07"] or cell_strains == ["6B07", "C3M10"]:
        rfp_arr[rfp_arr == 1] = 4  # Particle
        rfp_arr[rfp_arr == 2] = 5  # Background
    else:
        rfp_arr[rfp_arr == 2] = 4  # Particle
        rfp_arr[rfp_arr == 3] = 5  # Background
    return rfp_arr

def combine_channels(rfp_base, channel_ds_arrs, cell_strains):
    for strain in cell_strains:
        if strain == "3D05":
            continue
        channel_name = STRAIN_MAP[strain]
        for val, strain_name in BASE_TYPE_MAP.items():
            if strain_name == strain:
                rfp_base[channel_ds_arrs[channel_name] == 1] = val
        # rfp_base[channel_ds_arrs["DAPI"] == 1] = 2
        # if rfp_base_name == "RFP":
        #     rfp_base[channel_ds_arrs["GFP"] == 1] = 3
        # elif rfp_base_name == "GFP":
        #     rfp_base[channel_ds_arrs["RFP"] == 1] = 3
        # else:
        #     raise ValueError(f"Other channel name is something besides RFP, GFP. Name: {other_channel_name}")

    return rfp_base


def combine_cell_positions_and_clusters(dapi_channel, other_channel):
    print("Checking DAPI and RFP overlap")
    # Since these are single channels, 1 = cell and 2 = particle (set to be equal to particle)
    cell_to_be_removed = 2
    dapi_mask = dapi_channel == 1
    rfp_mask = other_channel == 1

    # Label connected components in image1
    labeled_dapi = label(dapi_mask)

    # Get properties of each cell in image1
    regions_dapi = regionprops(labeled_dapi)

    # Create a mask to track which cells to remove
    cells_to_remove = np.zeros_like(dapi_mask, dtype=bool)
    # Check each DAPI cell
    for region in regions_dapi:
        # Get a mask for just this DAPI cell
        cell_mask = labeled_dapi == region.label

        # Calculate overlap with cells in RFP
        overlap = np.logical_and(cell_mask, rfp_mask)
        # Calculate fraction of cell area that overlaps with image2 cells
        overlap_fraction = np.sum(overlap) / region.area

        # If overlap exceeds threshold, mark this cell for removal
        if overlap_fraction > DAPI_RFP_OVERLAP_THRESHOLD:
            cells_to_remove = np.logical_or(cells_to_remove, cell_mask)

    # Create the merged image
    dapi_combined = dapi_channel.copy()

    # Remove the cells that are significantly overlapped
    # Set them to the same value as in image2
    dapi_combined[cells_to_remove] = cell_to_be_removed
    return dapi_combined


def visualize_dapi_overlap_results(
    original_dapi, original_rfp, updated_dapi, cmap, norm, dapi_cmap, dapi_norm, base_name, output_name, other_channel_name
):
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
    axes[0, 0].set_title("Original DAPI")

    # Original RFP
    axes[0, 1].imshow(original_rfp, cmap=cmap, norm=norm)
    axes[0, 1].set_title(f"Original {other_channel_name}")

    # Set DAPI=1 pixels to 2
    original_rfp[original_dapi == 1] = 2

    axes[1, 0].imshow(original_rfp, cmap=cmap, norm=norm)
    axes[1, 0].set_title(f"DAPI overlaid with {other_channel_name}")

    # Result with removed cells highlighted
    axes[1, 1].imshow(updated_dapi, cmap=dapi_cmap, norm=dapi_norm)
    axes[1, 1].set_title("Updated DAPI")
    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background color
            continue
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cell_type))

    # Add the legend below the subplots
    fig.legend(
        handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements), frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{output_name}_dapi_{other_channel_name}_overlap.png", dpi=300)
    plt.close()


def create_channel_plots(
    raw_arr,
    strain,
    cmap,
    norm,
    base_name,
    output_name,
    denoised_arr,
    overlap_arr=None,
    cell_positions=None,
    cell_clusters=None,
):
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
    axes[0, 0].set_title("Raw segmentation")

    min_cell_area = MIN_CELL_AREA[strain] / (PX_TO_UM_CONV**2)
    axes[0, 1].imshow(denoised_arr, cmap=cmap, norm=norm)
    # 3D05 title
    # axes[0, 1].set_title(f"Filtered w/denoise threshold={DENOISE_SIZE} and cell area >{MIN_CELL_AREA_3D05 / (PX_TO_UM_CONV**2):.2f}")
    # C3M10 title
    # axes[0, 1].set_title(f"Filtered w/denoise threshold={DENOISE_SIZE} and cell area >{MIN_CELL_AREA_C3M10 / (PX_TO_UM_CONV**2):.2f}")
    # 3D05 + C3M10 title
    axes[0, 1].set_title(
        f"Filtered w/denoise threshold={DENOISE_SIZE} and cell area >{MIN_CELL_AREA['3D05'] / (PX_TO_UM_CONV**2):.2f} (3D05) "  
        f"and >{MIN_CELL_AREA['C3M10'] / (PX_TO_UM_CONV**2):.2f} $\mu$m$^2$ (C3M10)")
    min_cluster_area = MIN_CLUSTER_AREA[strain] / (PX_TO_UM_CONV**2)
    axes[1, 0].imshow(denoised_arr, cmap=cmap, norm=norm)
    # 3D05 title
    # axes[1, 0].set_title(f"Cell Positions (where aggregates >{MIN_CLUSTER_AREA_3D05 / (PX_TO_UM_CONV**2):.2f}") 
    # C3M10 title
    # axes[1, 0].set_title(f"Cell Positions (where aggregates >{MIN_CLUSTER_AREA_C3M10 / (PX_TO_UM_CONV**2):.2f}")
    # 3D05 + C3M10 title
    axes[1, 0].set_title(f"Cell Positions (where aggregates >{MIN_CLUSTER_AREA['3D05'] / (PX_TO_UM_CONV**2):.2f} (3D05) " 
                         f"and >{MIN_CLUSTER_AREA['C3M10'] / (PX_TO_UM_CONV**2):.2f} $\mu$m$^2$ (C3M10)")

    # Plots cell positions if cell_positions is not None and there are any cell positions
    if cell_positions is not None and any(cell_positions.values()):
        all_positions = np.array([pos.centroid for positions in cell_positions.values() for pos in positions])
        axes[1, 0].scatter(all_positions[:, 1], all_positions[:, 0], s=3, c="white", marker=".")

    # Plots cell clusters if cell_clusters is not None and there are any cell clusters
    if cell_clusters is not None and any(cell_clusters.values()):
        all_clusters = np.array([pos.centroid for positions in cell_clusters.values() for pos in positions])

        axes[1, 0].scatter(all_clusters[:, 1], all_clusters[:, 0], s=10, c="red", marker=".")

    if overlap_arr is not None:
        axes[1, 1].imshow(overlap_arr, cmap=cmap, norm=norm)
        axes[1, 1].set_title("Particle Area")

    # Create legend patches for each color in the colormap
    legend_elements = []
    for cell_type, color in CMAP.items():
        if (cell_type in ["Background"]) or (cell_type in ["Overlap"] and overlap_arr is None):  # Skip background color
            continue
        # if cell_type in ["Overlap"] and overlap_arr is None:
        #     continue
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cell_type))

    # Add red dot for cell positions and blue dot for clusters
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            label="single cells",
            markersize=10,
        )
    )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            label="aggregates",
            markersize=10,
        )
    )

    # Add the legend below the subplots
    fig.legend(
        handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements), frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{output_name}_plots.png", dpi=300)
    plt.close()

def create_plot(ds_arr, cmap, norm, file_name, cell_positions=None, cell_clusters=None, title=None):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(ds_arr, cmap=cmap, norm=norm, interpolation="None")
    if title is not None:
        ax.set_title(title, fontsize=20, pad=20)

    # Plots cell positions if cell_positions is not None and there are any cell positions
    if cell_positions is not None and any(cell_positions.values()):
        all_positions = np.array(
            [pos.centroid for positions in cell_positions.values() if len(cell_positions) > 0 for pos in positions]
        )
        plt.scatter(all_positions[:, 1], all_positions[:, 0], s=3, c="white", marker=".")

    # Plots cell clusters if cell_clusters is not None and there are any cell clusters
    if cell_clusters is not None and any(cell_clusters.values()):
        all_clusters = np.array(
            [pos.centroid for positions in cell_clusters.values() if len(positions) > 0 for pos in positions]
        )
        plt.scatter(all_clusters[:, 1], all_clusters[:, 0], s=10, c="red", marker=".")

    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background and overlap colors
            continue
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cell_type))

    # Add red dot for cell positions and blue dot for clusters
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            label="single cells",
            markersize=10,
        )
    )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            label="aggregates",
            markersize=10,
        )
    )

    # Add the legend below the subplots
    fig.legend(
        handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.08), ncol=len(legend_elements), frameon=False
    )

    fig.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close()


def create_single_plots(
    raw_arr, cmap, norm, base_name, output_name, denoised_arr, overlap_arr, cell_positions=None, cell_clusters=None
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(base_name, fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.9)

    axes[0, 0].imshow(raw_arr, cmap=cmap, norm=norm)
    axes[0, 0].set_title("Raw segmentation")

    axes[0, 1].imshow(denoised_arr, cmap=cmap, norm=norm)
    # 3D05 title
    # axes[0, 1].set_title(f"Filtered w/denoise threshold={DENOISE_SIZE} and cell area >{MIN_CELL_AREA_3D05 / (PX_TO_UM_CONV**2):.2f}")  
    # C3M10 title
    # axes[0, 1].set_title(f"Filtered w/denoise threshold={DENOISE_SIZE} and cell area >{MIN_CELL_AREA_C3M10 / (PX_TO_UM_CONV**2):.2f}")
    # 3D05 + C3M10 title
    axes[0, 1].set_title(
        f"Filtered w/denoise threshold={DENOISE_SIZE} and cell area >{MIN_CELL_AREA['3D05'] / (PX_TO_UM_CONV**2):.2f} (3D05) "  
        f"and >{MIN_CELL_AREA['C3M10'] / (PX_TO_UM_CONV**2):.2f} $\mu$m$^2$ (C3M10)")

    axes[1, 0].imshow(denoised_arr, cmap=cmap, norm=norm)
    # 3D05 title
    # axes[1, 0].set_title(f"Cell Positions (where aggregates >{MIN_CLUSTER_AREA_3D05 / (PX_TO_UM_CONV**2):.2f}") 
    # C3M10 title
    # axes[1, 0].set_title(f"Cell Positions (where aggregates >{MIN_CLUSTER_AREA_C3M10 / (PX_TO_UM_CONV**2):.2f}")
    # 3D05 + C3M10 title
    axes[1, 0].set_title(f"Cell Positions (where aggregates >{MIN_CLUSTER_AREA['3D05'] / (PX_TO_UM_CONV**2):.2f} (3D05) " 
                         f"and >{MIN_CLUSTER_AREA['C3M10'] / (PX_TO_UM_CONV**2):.2f} $\mu$m$^2$ (C3M10)")

    # Plots cell positions if cell_positions is not None and there are any cell positions
    if cell_positions is not None and any(cell_positions.values()):
        all_positions = np.array(
            [pos.centroid for positions in cell_positions.values() if len(positions) > 0 for pos in positions]
        )
        if len(all_positions) > 0:
            axes[1, 0].scatter(all_positions[:, 1], all_positions[:, 0], s=3, c="white", marker=".")

    # Plots cell clusters if cell_clusters is not None and there are any cell clusters
    if cell_clusters is not None and any(cell_clusters.values()):
        all_clusters = np.array(
            [pos.centroid for positions in cell_clusters.values() if len(positions) > 0 for pos in positions]
        )
        if len(all_clusters) > 0:
            axes[1, 0].scatter(all_clusters[:, 1], all_clusters[:, 0], s=10, c="red", marker=".")

    axes[1, 1].imshow(overlap_arr, cmap=cmap, norm=norm)
    axes[1, 1].set_title("Particle Area")
    # Create legend patches for each color in the colormap
    legend_elements = []
    for cell_type, color in CMAP.items():
        if cell_type in ["Background"]:  # Skip background and overlap colors
            continue
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cell_type))

    # Add red dot for cell positions and blue dot for clusters
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            label="single cells",
            markersize=10,
        )
    )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            label="aggregates",
            markersize=10,
        )
    )

    # Add the legend below the subplots
    fig.legend(
        handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements), frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    print("Saving plots to", f"{output_name}_plots.png")
    plt.savefig(f"{output_name}_plots.png", dpi=300)
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
    base_name = full_file_path.replace(".h5", "")
    processed_folder = cur_folder.split("/")[-1]

    cell_types = get_cell_type_map(file_path)
    if len(cell_types) == 0:
        raise ValueError("Cell type not found in file path")
    cmap, norm = get_color_map(cell_types)

    with h5py.File(full_file_path, "r") as f:
        a_group_key = next(iter(f.keys()))  # retrieve first key in the HDF5 file
        ds_arr = f[a_group_key][()]  # returns as a numpy array
    ds_arr = normalize_ds_arr(ds_arr)
    ds_arr_denoised = median_filter(ds_arr, size=DENOISE_SIZE)

    # Get cell positions and densities, note that these are dictionaries mapping cell type to an array of values, or single value for densities
    print("Getting cell positions and densities")
    cell_positions, cell_clusters, particle_area, merged_clusters = get_cell_positions_and_areas(ds_arr_denoised, cell_types, merged=True)
    cell_count, cell_density, cell_area_ratio = get_cell_counts_and_densities(
        cell_positions, cell_clusters, particle_area
    )
    ds_arr_recreated, particle_area = recreate_particle_area(ds_arr_denoised, cell_types, particle_area)

    # Create plots, write position and density info to csv
    print("Creating plots")
    create_single_plots(
        ds_arr,
        cmap,
        norm,
        processed_folder,
        base_name,
        ds_arr_denoised,
        ds_arr_recreated,
        cell_positions=cell_positions,
        cell_clusters=cell_clusters,
    )
    plot_original_vs_merged(ds_arr_denoised, merged_clusters, cell_clusters, cell_types, processed_folder, base_name)
    print("Writing position and density info to csv")
    write_cell_position_info(cell_positions, cell_clusters, cell_pos_file_name, particle_area)
    merged_file_name = cell_pos_file_name.replace("_cell_pos.csv", "_merged_cell_pos.csv")
    write_merged_cell_position_info(merged_clusters, merged_file_name, particle_area)
    write_density_info(density_info_file_path, processed_folder, cell_density, cell_area_ratio, cell_count)

def get_strains_from_file(file_name):
    cell_types = []
    for cell_type in CELL_TYPES:
        if cell_type in file_name.upper():
            cell_types.append(cell_type)
    return cell_types

def get_channel_from_file(file_name):
    channels = []
    for channel in CHANNELS:
        if channel in file_name.upper():
            channels.append(channel)
    if len(channels) > 1:
        raise ValueError("More than one channel found in file path")
    return channels[0]

# Checks file path for cell types and channels
# If just cell_types are found, returns a list of cell types found
# If a channel is found, returns a list with the single cell type that corresponds to that channel
# If more than one channel is found, raises an error
# Returns dictionary mapping cell value to cell type i.e. {1: "3D05", 2: "6B07", 3: "C3M10", 4: "particle", 5: "background"}
def get_cell_type_map(file_path: str) -> dict[int, str]:
    cell_types = get_strains_from_file(file_path)
    cell_type_map = {}
    for i, cell_type in enumerate(cell_types):
        cell_type_map[i + 1] = cell_type
    cell_type_map[i + 2] = "Particle"
    cell_type_map[i + 3] = "Background"
    # cell_type_map[i + 4] = "Overlap"
    return cell_type_map

# Checks file path for cell types and channels
# If just cell_types are found, returns a list of cell types found
# If a channel is found, returns a list with the single cell type that corresponds to that channel
# If more than one channel is found, raises an error
# Returns dictionary mapping cell value to cell type i.e. {1: "3D05", 2: "6B07", 3: "C3M10", 4: "particle", 5: "background"}
def get_cell_type_map_from_channel(strain_types, channel):
    if (strain_types == ["6B07"] and channel == "RFP") or (strain_types == ["6B07", "C3M10"] and channel == "RFP"):
        return {1: "Particle", 2: "Background"}
    return {1: CHANNEL_MAP[channel], 2: "Particle", 3: "Background"}


def get_color_map(cell_type_map):
    cell_colors = []
    bounds = []
    for cell_num, cell_type in cell_type_map.items():
        cell_colors.append(CMAP[cell_type])
        bounds.append(cell_num - 0.5)
    bounds.append(len(cell_type_map) + 0.5)
    cmap = colors.ListedColormap(cell_colors)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def normalize_ds_arr(ds_arr) -> np.ndarray:
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




def get_cell_positions_and_areas(z_slice: np.ndarray, cell_types: dict[int, str], merged: bool = False):
    label_im = label(
        z_slice
    )  # converts ds_arr into a labeled image where each connected region gets a unique integer label
    regions = regionprops(
        label_im
    )  # finds connected regions in label_im, where each detected region becomes a regionprops
    # object with properties (region.centroid, region.area, region.coords)
    cell_pos = {}
    cell_clusters = {}
    particle_area = 0

    for region in regions:
        region_type = get_type(region, z_slice)  # Returns 1,2,3,4,5 etc.
        cell_type = cell_types[region_type]  # Returns "3D05", "6B07", "C3M10", "particle", "background"
        # Handle background and particle cases
        if cell_type not in CELL_TYPES:
            if cell_type == "Particle":
                particle_area += region.area
            continue
        # Handle cell cases
        min_cell_area = MIN_CELL_AREA[cell_type]
        min_cluster_area = MIN_CLUSTER_AREA[cell_type]
        if cell_type not in cell_pos:
            # Initialize cell area, positions, and clusters for this cell type
            cell_pos[cell_type] = []
            cell_clusters[cell_type] = []
        if region.area >= min_cell_area and region.area < min_cluster_area:
            # cell_images[cell_type]
            cell_pos[cell_type].append(region)  # if true, stores the region
        if region.area >= min_cluster_area:
            cell_clusters[cell_type].append(region)

    # Get average area of cells
    cell_area_averages = {}
    for cell_type, cell_array in cell_pos.items():
        cell_area_averages[cell_type] = np.average([cell.area for cell in cell_array])
    for cell_type, cluster_array in cell_clusters.items():
        for cluster in cluster_array:
            cluster.cells = int(cluster.area // cell_area_averages[cell_type])

    # Can create a binary image of just that cell type at a time. For the combined will have to set anything that is equal to a cell type to be true.
    if merged:
        merged_clusters, _ = get_cell_clusters_from_distances(z_slice, cell_pos, cell_clusters, cell_types)
    else:
        merged_clusters = {}

    return cell_pos, cell_clusters, particle_area, merged_clusters

def get_cell_clusters_from_distances(z_slice: np.ndarray, cell_pos: list[RegionProperties], cell_clusters: list[RegionProperties], cell_types: dict[int, str]):
    # Combine both together
    combined = {}
    all_keys = set(cell_pos) | set(cell_clusters)
    for key in all_keys:
        combined[key] = cell_pos.get(key, []) + cell_clusters.get(key, [])

    # Combine images and get the regions for only the same cell types first
    merged_regions = {}
    merged_images = {}
    img_vals = []
    combined_regions = []
    for cell_type, cell_regions in combined.items():
        cell_img_val = 0
        binary_image = np.zeros_like(z_slice, dtype=bool)
        for cell_val, cell_temp_type in cell_types.items():
            if cell_temp_type == cell_type:
                cell_img_val = cell_val
                break
        img_vals.append(cell_img_val)
        combined_regions.extend(cell_regions)
        binary_image = z_slice == cell_img_val
        merged_regions[cell_type], merged_images[cell_type] = get_merged_regions(binary_image, cell_regions)

    # Now processing the combined case
    combined_image = np.zeros_like(z_slice, dtype=bool)
    for img_val in img_vals:
        combined_image |= (z_slice == img_val)
    # combined_labels = label(combined_image)
    # initial_combined_regions = regionprops(combined_labels)

    merged_regions["combined"], merged_images["combined"] = get_merged_regions(combined_image, combined_regions)

    return merged_regions, merged_images

def get_merged_regions(binary_image: np.ndarray, og_cell_regions: list) -> np.ndarray:
    struct_elem = disk(CELL_CLUSTER_DISTANCE_THRESHOLD //2)
    dilated = binary_dilation(binary_image, struct_elem)
    dilated_labels = label(dilated)
    processed_dilated_labels = set()
    merged_regions = []
    merged_image = np.zeros_like(binary_image, dtype=bool)
    # for region in og_cell_regions:
    #     y, x = region.centroid
    #     y, x = int(y), int(x)
    #     if 0 <= y < dilated_labels.shape[0] and 0 <= x < dilated_labels.shape[1]:
    #         dilated_label_value = dilated_labels[y, x]
    #         if dilated_label_value > 0 and dilated_label_value not in processed_dilated_labels:
    #             # Add this dilated region to the final image
    #             merged_image |= (dilated_labels == dilated_label_value)
    #             processed_dilated_labels.add(dilated_label_value)

    for region in og_cell_regions:
        y, x = region.centroid
        y, x = int(y), int(x)
        if 0 <= y < dilated_labels.shape[0] and 0 <= x < dilated_labels.shape[1]:
            dilated_label_value = dilated_labels[y, x]
            if dilated_label_value > 0 and dilated_label_value not in processed_dilated_labels:
                # Find all regions that share the same dilated label
                touching_regions = [
                    r for r in og_cell_regions if dilated_labels[int(r.centroid[0]), int(r.centroid[1])] == dilated_label_value
                ]

                # Combine properties of touching regions
                combined_area = sum(r.area for r in touching_regions)
                combined_centroid = np.average(
                    [r.centroid for r in touching_regions], axis=0, weights=[r.area for r in touching_regions]
                )
                # Calculate the bounding box for the merged region
                minr = min(r.bbox[0] for r in touching_regions)
                minc = min(r.bbox[1] for r in touching_regions)
                maxr = max(r.bbox[2] for r in touching_regions)
                maxc = max(r.bbox[3] for r in touching_regions)
                bbox = (minr, minc, maxr, maxc)

                # Create a new merged region-like object
                merged_regions.append({
                    "area": combined_area,
                    "centroid": combined_centroid,
                    "regions": touching_regions,
                    "bbox": bbox,
                })

                # Mark this dilated label as processed
                processed_dilated_labels.add(dilated_label_value)

                # Update the merged image
                merged_image |= (dilated_labels == dilated_label_value)
        # Fill holes in the merged regions
    merged_image = binary_fill_holes(merged_image)
    # merged_labels = label(merged_image)
    # merged_regions = regionprops(merged_labels)
    return merged_regions, merged_image

def plot_original_vs_merged(original_image: np.ndarray, merged_regions: dict[str, list[RegionProperties]], cell_clusters: dict[str, list[RegionProperties]], cell_types: list[int, str], title_name: str, base_name: str):
    rows = 2 if len(merged_regions) > 2 else 1
    fig = plt.figure(figsize=(16, 16))
    if len(merged_regions) <= 2:
        fig, axes = plt.subplots(1, len(merged_regions), figsize=(16, 16))
    elif len(merged_regions) == 3:
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Top left
        ax2 = fig.add_subplot(gs[0, 1])  # Top right
        ax3 = fig.add_subplot(gs[1, :])  # Bottom spanning both columns

        # Convert to numpy array format for consistent indexing
        axes = np.array([[ax1, ax2], [ax3, None]])
    else:
        # Create regular 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    fig.suptitle(f"{title_name} Merged Cell Positions", fontsize=20, y=0.98)

    cmap, norm = get_color_map(cell_types)
    # Plot each cell_type separately
    for i, (cell_type, regions) in enumerate(merged_regions.items()):
        ax = axes[i // 2, i % 2] if rows > 1 else axes[i]
        ax.imshow(original_image, cmap=cmap, norm=norm)
        ax.set_title(cell_type.title())
        clusters = cell_clusters.get(cell_type, [])
        for cluster in clusters:
            minr, minc, maxr, maxc = cluster.bbox
            rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor="orange", linewidth=.5)
            ax.add_patch(rect)

        for region in regions:
            if len(region["regions"]) == 1:
                continue
            minr, minc, maxr, maxc = region["bbox"]
            rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                            fill=False, edgecolor="green", linewidth=1)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.savefig(f"{base_name}_cell_cluster_pos.png", dpi=300)
    plt.close()


def recreate_particle_area(ds_arr, cell_types, particle_area):
    """
    Calculates "real" particle area by filling in the overlap area between particles and cells
    Returns the updated ds_arr and the new particle area
    """
    # Find the key corresponding to the "particle" value in cell_types
    particle_label = None
    # overlap_label = None
    for key, value in cell_types.items():
        if value == "Particle":
            particle_label = key
        # if value == "Overlap":
        #     overlap_label = key
    for cell_type_label, cell_type in cell_types.items():
        if cell_type not in CELL_TYPES:
            continue
        updated_ds_arr, overlap_area = fill_particle_area(ds_arr, particle_label, cell_type_label, overlap_label=particle_label)
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

    # Change the values in the combined overlap areas to the overlap label (just the particle label)
    updated_ds_arr[combined_overlap] = overlap_label

    return updated_ds_arr, np.sum(combined_overlap)


def get_cell_counts_and_densities(cell_pos: dict[str, list[RegionProperties]], cell_clusters: dict[str, list[RegionProperties]], particle_area: float):
    # Calculate cell counts, density, and area ratios
    cell_count = {}
    cell_density = {}
    cell_area_ratio = {}
    particle_area = particle_area / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
    for cell_type, cell_array in cell_pos.items():
        if cell_type not in CELL_TYPES:
            continue
        cluster_cells = 0
        for cluster in cell_clusters[cell_type]:
            cluster_cells += cluster.cells
        cell_count[cell_type] = len(cell_array) + cluster_cells
        cell_area = np.sum([cell.area for cell in cell_array])
        for cluster in cell_clusters[cell_type]:
            cell_area += cluster["area"]

        area = cell_area / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
        cell_density[cell_type] = round(cell_count[cell_type] / particle_area, 5)
        cell_area_ratio[cell_type] = round(area / particle_area, 5)
    return cell_count, cell_density, cell_area_ratio


def get_type(region: RegionProperties, data: np.ndarray) -> int:
    point = region.coords[0]  # retrieves one coordinate (first pixel of the region)
    point_val = data[point[0], point[1]]
    return point_val


def write_cell_position_info(cell_positions, cell_clusters, csv_output_file, particle_area: float):
    particle_area = particle_area / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
    with open(csv_output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["strain", "cell_type", "x_pos", "y_pos", "cell_area", "cell_area_ratio", "cell_count"])
        for strain_type, pos in cell_positions.items():
            for p in pos:
                cell_pos = p.centroid
                area = p.area / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
                area_ratio = area / particle_area
                writer.writerow([strain_type, "cell", round(cell_pos[1], 2), round(cell_pos[0], 2), round(area, 5), round(area_ratio, 8), 1])
        for strain_type, cluster in cell_clusters.items():
            for c in cluster:
                pos = c.centroid
                area = c.area / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
                area_ratio = area / particle_area
                writer.writerow([strain_type, "cluster", round(pos[1], 2), round(pos[0], 2), area, round(area_ratio, 8), c.cells])

def write_merged_cell_position_info(merged_clusters: dict[str, RegionProperties], csv_output_file, particle_area: float):
    particle_area = particle_area / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
    with open(csv_output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["strain_type", "x_pos", "y_pos", "cell_area", "cell_area_ratio", "cell_num"])
        for strain_type, pos in merged_clusters.items():
            for p in pos:
                cell_pos = p["centroid"]
                area = p["area"] / (PX_TO_UM_CONV**2)  # convert pixels^2 --> microns^2
                area_ratio = area / particle_area
                writer.writerow([strain_type, round(cell_pos[1], 2), round(cell_pos[0], 2), round(area, 5), round(area_ratio, 8), len(p["regions"])])


def write_density_info(csv_output_file, h5_folder, cell_density, cell_area_ratio, cell_count):
    header = ["folder", "strain", "cell_density", "cell_area_ratio", "cell_count"]
    # Read existing data if file exists
    existing_data = []
    path_exists = os.path.exists(csv_output_file)
    data_exists = False
    if path_exists:
        with open(csv_output_file, "r") as f:
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
        with open(csv_output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(existing_data)
        # Just append if folder not found
    with open(csv_output_file, "a") as f:
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
            if file.endswith(".h5"):
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
