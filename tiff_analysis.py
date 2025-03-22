#!/usr/bin/env python3

'''
Created on Wed Mar 12 19:00:25 2025

@author: vercelli

1. H5 file is how it will look
2. Currently code is only extracting first zslice, want to extract all zslices and create pngs from them.
    These pngs go under the respective channel folders created in split_zstack.py
3. All tiffs and pngs will be in the same folder
'''

# Load libraries
import csv
import os

import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.measure import label, regionprops

# Visualize segmentation
file_name = 'Tp_3D05_1_24h_60X_12_z_Simple Segmentation.h5'
cmap = colors.ListedColormap(['#c0a0c0', '#1f607f', 'black']) # Cells, Diatoms, Background

def find_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"- {name}")

def process_h5_file(file_path):
    bounds = [.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    with h5py.File(file_path, "r") as f:
        a_group_key = next(iter(f.keys())) # retrieve first key in the HDF5 file
        ds_arr = f[a_group_key][()]  # returns as a numpy array

    output_folder = file_path.replace('_segmentation.h5', '')
    os.makedirs(output_folder, exist_ok=True) # Will probably need to change this to be more specific
    for i, z_slice in enumerate(ds_arr):
        print("Processing z-slice: ", i)
        # Save basic image
        plt.figure(figsize=(10, 10))
        plt.imshow(z_slice, cmap=cmap, norm=norm, interpolation='None')
        # plt.savefig(f"{output_folder}/{file_name}.png", bbox_inches='tight')
        cell_positions, cell_clusters = get_cell_positions(z_slice)
        plot_cell_positions(cell_positions, cell_clusters)
        output_location = os.path.join(output_folder, f"{output_folder.split('/')[-1]}_cell_positions_{i}.png")
        print("Saving to: ", output_location)
        plt.savefig(output_location, bbox_inches='tight')
        plt.close()
    quit()

def get_cell_positions(z_slice):
    label_im = label(z_slice) # converts ds_arr into a labeled image where each connected region gets a unique integer label
    regions = regionprops(label_im) # finds connected regions in label_im, where each detected region becomes a regionprops
    # object with properties (region.centroid, region.area, region.coords)
    cell_pos = []
    sizes = []
    cell_clusters = []
    for region in regions:
        if get_type(region, z_slice) == 1: # check if the region is labeled as 1 (cells)
            sizes.append(region.area)
            if region.area >= 20 and region.area < 100:
                cell_pos.append(region.centroid) # if true, stores the region's centroid (region.centroid) in cell_pos
            if region.area >=100:
                cell_clusters.append(region.centroid)
    return cell_pos, cell_clusters

def get_type(region, data):
    point = region.coords[0] # retrieves one coordinate (first pixel of the region)
    return data[point[0], point[1]] # looks up the segmentation label at that pixel

def plot_cell_positions(cell_pos, cell_clusters):
    for pos in cell_pos:
        plt.scatter(pos[1], pos[0], s=1, edgecolors='red')
    for cluster in cell_clusters:
        plt.scatter(cluster[1], cluster[0], s=3, edgecolors='blue')

# plt.hist(sizes, bins=50, range=[0,200])
# plt.savefig("area_distribution.png")
def get_h5_files_recursively(folder_path):
    h5_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def main():
    folder_name = '3D05_example'
    print("Processing folder: ", folder_name)

    h5_files = get_h5_files_recursively(folder_name)
    for h5_file in h5_files:
        print("Processing file: ", h5_file)
        process_h5_file(h5_file)
    print("Processing complete")

if __name__ == "__main__":
    main()
