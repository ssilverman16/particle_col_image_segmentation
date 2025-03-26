"""
Goals: 
1. Use watershed segmentation to refine segmentations of cell boundaries. 
2. Integrate this into the tiff_analysis.py file and re-calculate the areas and cell positions of individual cells. 
   Hopefully this will resolve many of the cells in the clusters, but probably many will still not be perfectly segmented,
   so for residual clusters that don't get segmented, let's still label them and compute the # of cells they represent 
   (i.e. the code you already wrote).
3. Compute all cell-cell distances: (a) as the distance between each cell of a given strain and its nearest neighbor of the 
   same strain, and as (b) as the distance between each cell of a given strain and its nearest neighbor of a different strain.
   For example, if 3D05 and C3M10 are present, the distances between every pair of 3D05-3D05 cells, and between every pair of
   3D05-C3M10 cells.  [this is my ultimate analytical goal with my images- I think one of my lab mates might already have similar
   code for this, I will check with him and push the code up if he gives it to me.]

"""

import csv
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.segmentation import watershed
from scipy import ndimage as ndi


# Load the HDF5 file
file_path = "working_folder/Tp_C3M10_1_120h_60X_RFP_GFP_1_MIP_probabilities.h5"  # Adjust filename
with h5py.File(file_path, "r") as f:
    print(list(f.keys()))  # Show dataset keys inside the HDF5 file
    probabilities = np.array(f["exported_data"])  # Load dataset

# Extract only the "Boundaries" channel (usually channel index 1)
boundary_map = probabilities[3]  # Adjust index if needed

# Display the boundary probability map
plt.figure(figsize=(6,6))
plt.imshow(boundary_map, cmap="gray")
plt.title("Boundary Probability Map from Ilastik")
plt.colorbar()
# plt.show()

# Convert boundaries to binary mask
threshold = 0.5  # Adjust based on your data
binary_mask = boundary_map < threshold  # Objects (True), Boundaries (False)

plt.figure(figsize=(6,6))
plt.imshow(binary_mask, cmap="gray")
plt.title("Binary Foreground Mask")
plt.show()



# CODE WORKS UNTIL THIS POINT...I didn't get very far, lol



# Compute markers for watershed segmentation
## Compute the Euclidean distance transform
distance = ndi.distance_transform_edt(binary_mask)

## Identify local maxima as markers for watershed
local_max = morphology.local_maxima(distance)
markers = measure.label(local_max)  # Label each object

## Display markers
plt.figure(figsize=(6,6))
plt.imshow(markers, cmap="nipy_spectral")
plt.title("Watershed Markers")
# plt.show()

# Apply watershed using boundary map as the "barrier"
labels = watershed(boundary_map, markers, mask=binary_mask)

# Display final segmented objects
plt.figure(figsize=(6,6))
plt.imshow(labels, cmap="nipy_spectral")
plt.title("Final Segmentation")
plt.show()