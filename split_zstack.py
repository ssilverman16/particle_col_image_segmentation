'''
1. Create folder with filename but no channel details (RFP, GFP, DAPI, CY5) and no zstack or mip (keep the number though)
    2. 2 file structures: MIP and zstack which will be in last position before extension (.tif or .jpg)
        Every zstack is just a tif, mip has both tif and jpg
2. Move zstack/MIPs into the newly created folder
3. Create folders in this folder, one for GFP and one for RFP


'''
import os

import tifffile as tiff


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_clean_file_name(input_file):
    base_name = input_file.split('.')[0]
    if 'CY5_RFP_GFP_DAPI_' in base_name:
        channels = "_CY5_RFP_GFP_DAPI"
        clean_file_name = base_name.replace(channels, '').replace('_zstack', '').replace('_mip', '')
    elif 'RFP_GFP_' in base_name:
        channels = "_RFP_GFP"
        clean_file_name = base_name.replace(channels, '').replace('_zstack', '').replace('_mip', '')
    else:
        channels = ""
        clean_file_name = base_name
    return (channels, clean_file_name)

def create_channel_folder(destination, used_channels, channel_name):
    clean_name = destination.replace(".tif", "").replace("_mip", "").replace(used_channels, "")
    clean_name = clean_name + "_" + channel_name
    create_folder(clean_name)
    return clean_name

def process_tif(input_file, channel_indices):
    channel_map = {0:"CY5", 1:"RFP", 2:"GFP", 3:"DAPI"}
    print('parsing file', input_file)
    input_file_end = input_file.split('/')[-1].split('.')[0]
    used_channels, clean_file_name = get_clean_file_name(input_file)
    # new_tif_folder = f"{output_folder}/{clean_file_name}"
    create_folder(clean_file_name)
    # Move the original zstack file into the new folder
    destination = os.path.join(clean_file_name, os.path.basename(input_file))
    os.rename(input_file, destination)
    if not input_file.endswith('.tif'):
        return
    file = tiff.TiffReader(destination)
    zstack = file.asarray()
    for i, z_slice in enumerate(zstack):
        if z_slice.shape[0] != 4:
            channel_map = {0:"RFP", 1:"GFP"}
            channel_indices = [0,1]

        channel_names = [channel_map[channel_idx] for channel_idx in channel_indices]
        selected_channels = z_slice[channel_indices]
        for idx, channel in enumerate(selected_channels):
            channel_name = channel_names[idx]
            channel_folder = create_channel_folder(destination, used_channels, channel_name)
            channel_file_name = input_file_end.replace(used_channels, "")
            output_file = os.path.join(channel_folder, f"{channel_file_name}_z{i}_{channel_name}.tif")
            with tiff.TiffWriter(output_file, bigtiff=False) as tif:
                tif.write(channel)

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
            if file.lower().endswith('_zstack.tif') or file.lower().endswith('_mip.tif') or file.lower().endswith('_mip.jpg'):
                # output_folder = create_output_folder(f"{folder_path}/{file}")
                input_file = os.path.join(folder_path, file)

                # Process the file and save result
                process_tif(input_file, channel_indices)


def main():
    channel_indices = [1, 2] # 1=RFP, 2=GFP
    folder_name = '3D05_6B07'
    print("Processing folder: ", folder_name)
    process_folder(folder_name, channel_indices)
    print("Processing complete")

if __name__ == "__main__":
    main()
