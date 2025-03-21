import os

import tifffile as tiff


def process_tif(input_file, output_folder, channel_indices):
    file = tiff.TiffReader(input_file)
    zstack = file.asarray()
    for i, z_slice in enumerate(zstack):
        if z_slice.shape[0] != 4:
            channel_indices = [0,1]
        selected_channels = z_slice[channel_indices]  # Extract channels 2 and 3
        base_name = input_file.split('/')[-1].split('.')[0]
        clean_file_name = base_name.replace('CY5_RFP_GFP_DAPI_', 'RFP_GFP_').replace('_zstack', '')
        output_file = os.path.join(output_folder, f"{clean_file_name}_z{i}.tif")
        with tiff.TiffWriter(output_file, bigtiff=False) as tif:
            tif.write(selected_channels)

def create_output_folder(file):
    folder_name = file.split(".")[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def process_folder(top_level_folder, channel_indices):
    # Go through immediate subdirectories only
    for folder in os.listdir(top_level_folder):
        folder_path = os.path.join(top_level_folder, folder)

        # Skip if not a directory or starts with .
        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Process tif files in this directory
        for file in os.listdir(folder_path):
            if file.lower().endswith('_zstack.tif'):
                output_folder = create_output_folder(f"{folder_path}/{file}")
                input_file = os.path.join(folder_path, file)
                
                # Process the file and save result
                process_tif(input_file, output_folder, channel_indices)


def main():
    channel_indices = [1, 2] # 1=RFP, 2=GFP
    folder_name = '/Volumes/WD_Elements/3D05'
    print("Processing folder: ", folder_name)
    process_folder(folder_name, channel_indices)
    print("Processing complete")

if __name__ == "__main__":
    main()
