#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 19:00:25 2025

@author: vercelli
"""

# Load libraries
import tifffile
import os
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors


# Visualize segmentation
data_folder = '/Volumes/WD_Elements/3D05/24h'
file_name = 'Tp_3D05_1_24h_60X_12_z_Simple Segmentation.h5'
def print_structure(name, obj):
    print(name, ":", "Group" if isinstance(obj, h5py.Group) else "Dataset")

def find_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"- {name}")

with h5py.File(f'{data_folder}/{file_name}', "r") as h5f:
    # Walk through the file and print its structure
    h5f.visititems(find_datasets)
    dataset_name = "exported_data"  # Change this to an actual dataset name
    if dataset_name in h5f:
        dataset = h5f[dataset_name]
        print(f"Dataset: {dataset_name}")
        print("Shape:", dataset.shape)
        print("Datatype:", dataset.dtype)
        print("First few values:", dataset[:5])

with h5py.File(os.path.join(data_folder, file_name), "r") as f:
    a_group_key = list(f.keys())[0] # retrieve first key in the HDF5 file
    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key])) 

    # If a_group_key is a dataset name, 
    # this gets the dataset values and returns as a list
    data = list(f[a_group_key])
    
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array

# Crop image
#ds_arr_cropped = ds_arr[0][500:1500, 500:1500]
ds_arr = ds_arr[0]
#print(ds_arr[0].shape)

# Color labels
cmap = colors.ListedColormap(['#c0a0c0', '#1f607f', 'black']) # Cells, Diatoms, Background
bounds = [.5, 1.5, 2.5, 3.5]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.figure(figsize=(10, 10)) 
plt.imshow(ds_arr, cmap=cmap, norm=norm, interpolation='None')
plt.savefig(os.path.join(data_folder, f"{file_name}.png"), bbox_inches='tight')


# Get x,y position of cells
# from skimage.measure import label, regionprops, regionprops_table
# import csv
# def get_type(region, data):
#     point = region.coords[0] # retrieves one coordinate (first pixel of the region)
#     return data[point[0], point[1]] # looks up the segmentation label at that pixel

# label_im = label(ds_arr) # converts ds_arr into a labeled image where each connected region gets a unique integer label
# regions = regionprops(label_im) # finds connected regions in label_im, where each detected region becomes a regionprops 
# # object with properties (region.centroid, region.area, region.coords)

# cell_pos = []
# sizes = []
# cell_clusters = []
# for region in regions:
#     if get_type(region, ds_arr) == 1: # check if the region is labeled as 1 (cells)
#         sizes.append(region.area)
#         if region.area >= 20 and region.area < 100:
#             cell_pos.append(region.centroid) # if true, stores the region's centroid (region.centroid) in cell_pos
#         if region.area >=100:
#             cell_clusters.append(region.centroid)


# with open("output.csv", "w", encoding='utf-8') as wf:
#     writer =  csv.writer(wf)
#     for pos in cell_pos:
#         plt.scatter(pos[1], pos[0], s=1, edgecolors='red')
#         writer.writerow(pos)
# for cluster in cell_clusters:
#     plt.scatter(cluster[1], cluster[0], s=3, edgecolors='blue')

# plt.savefig('cell_pos_plot.png')
# # plt.hist(sizes, bins=50, range=[0,200])
# # plt.savefig("area_distribution.png")
# # %%
