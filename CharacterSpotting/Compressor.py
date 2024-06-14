# Creator of wordsCompressed that includes all images from the downloaded "words" but not in a listed directory.
import os
import shutil
import cv2
import GlobalConstants as paths

# Define the source and destination directories
source_dir = paths.words_source_RIMES  # From RIMES dataset
destination_dir = paths.source + "sentencesCompressedOriginal"  # Destination

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

for root, dirs, files in os.walk(source_dir):  # List over all files in the IamHandwriting dataset.
    for file in files:
        if file.endswith('.jpg'):
            file_path = os.path.join(root, file)  # Construct the full file path
            image = cv2.imread(file_path)  # Try to read the image
            if image is not None:  # Check if the image was read correctly
                try:
                    shutil.copy(file_path, destination_dir)  # Copy the file to the destination directory
                    # print(f"Copied: {file_path}")
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")
            else:
                print(f"Error reading image: {file_path}")

print('Success!')
