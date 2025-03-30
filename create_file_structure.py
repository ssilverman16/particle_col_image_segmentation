'''
1. Create folder with filename but no channel details (RFP, GFP, DAPI, CY5) and no zstack or mip (keep the number though)
    2. 2 file structures: MIP and zstack which will be in last position before extension (.tif or .jpg)
        Every zstack is just a tif, mip has both tif and jpg. These will share the same prefix.
2. Move zstack/MIPs into the newly created folder
3. No subfolders for extracted mips, just extract into same folder
4. Channel names how they are, not appended to the end
5. Channel 0 = red, channel 1 = magenta, channel 2 = green, channel 3 = cyan

'''
import os

channels = [{"name": "CY5", "color": "red"},
            {"name": "RFP", "color": "magenta"},
            {"name": "GFP", "color": "green"},
            {"name": "DAPI", "color": "cyan"}]

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def remove_channels(filename):
    for channel in channels:
        filename = filename.replace(f"_{channel['name']}_", "_")
    return filename

def create_folder_from_tif(input_file):
    clean_name = input_file.split('.tif')[0]
    clean_name = remove_channels(clean_name)
    clean_name = clean_name.replace("_zstack", "")
    return clean_name

def create_channel_folder(destination, used_channels, channel_name):
    clean_name = destination.replace(".tif", "").replace("_mip", "").replace(used_channels, "")
    clean_name = clean_name + "_" + channel_name
    create_folder(clean_name)
    return clean_name

def get_similar_files(file_name, folder):
    similar_files = [os.path.join(folder, file_name)]
    clean_file_name = remove_channels(file_name)
    clean_file_name = clean_file_name.replace("_zstack", "").replace(".tif", "")
    for file in os.listdir(folder):
        check_file_name = remove_channels(file)
        check_file_name = check_file_name.replace("_zstack", "").replace(".tif", "")
        if clean_file_name in check_file_name and ("_mip.tif" in file.lower() or ".jpg" in file.lower()):
            similar_files.append(os.path.join(folder, file))
    return similar_files


def process_tif(input_file):
    input_file_name = input_file.split('/')[-1]
    input_folder = os.path.dirname(input_file)
    clean_folder_name = create_folder_from_tif(input_file)
    create_folder(clean_folder_name)
    similar_files = get_similar_files(input_file_name, input_folder)
    for file in similar_files:
        destination = os.path.join(clean_folder_name, os.path.basename(file))
        os.rename(file, destination)


def create_output_folder(file):
    folder_name = file.split(".")[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def get_tiff_files(top_level_folder_path):
    tiff_files = []
    for folder in os.listdir(top_level_folder_path):
        folder_path = os.path.join(top_level_folder_path, folder)

        # Skip if not a directory or starts with .
        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Process tif files in this directory
        for file in os.listdir(folder_path):
            if file.lower().endswith('.tif') and "mip" not in file.lower():
                tiff_files.append(os.path.join(folder_path, file))
    return tiff_files

def process_folder(top_level_folder):
    # Go through immediate subdirectories only
    tiff_files = get_tiff_files(top_level_folder)
    for file in tiff_files:
        process_tif(file)

def main():
    folder_name = '/Volumes/WD_Elements/6B07_C3M10'
    print("Processing folder: ", folder_name)
    process_folder(folder_name)
    print("Processing complete")

if __name__ == "__main__":
    main()
