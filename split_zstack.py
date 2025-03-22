import os

import tifffile as tiff



def process_tif(input_file, output_folder, channel_indices):
    channel_map = {0:"CY5", 1:"RFP", 2:"GFP", 3:"DAPI"}
    file = tiff.TiffReader(input_file)
    zstack = file.asarray()
    print('parsing file', input_file)

    for i, z_slice in enumerate(zstack):
        if z_slice.shape[0] != 4:
            channel_map = {0:"RFP", 1:"GFP"}
            channel_indices = [0,1]

        channel_names = [channel_map[channel_idx] for channel_idx in channel_indices]
        selected_channels = z_slice[channel_indices]  # Extract channels 2 and 3
        base_name = input_file.split('/')[-1].split('.')[0]

        # conditional renaming
        if 'CY5_RFP_GFP_DAPI_' in base_name:
            clean_file_name = base_name.replace('CY5_RFP_GFP_DAPI_', '').replace('_zstack', '')
        elif 'RFP_GFP_' in base_name:
            clean_file_name = base_name.replace('RFP_GFP_', '').replace('_zstack', '')
        else:
            clean_file_name = base_name

        for idx, channel in enumerate(selected_channels):
            channel_name = channel_names[idx]
            output_file = os.path.join(output_folder, f"{clean_file_name}_z{i}_{channel_name}.tif")
            with tiff.TiffWriter(output_file, bigtiff=False) as tif:
                tif.write(channel)
    quit()

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
