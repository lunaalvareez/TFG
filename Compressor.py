
#Creator of wordsCompressed that includes all images from the downloaded "words" but not in a listed directory.
import os
import shutil
import GlobalConstants as paths

# Define the source and destination directories
source_dir = paths.words_source #From IAMHandwriting website
destination_dir = paths.source + "wordsCompressed" #Destination



os.makedirs(destination_dir, exist_ok=True) # Ensure the destination directory exists
for root, dirs, files in os.walk(source_dir): # List over all files in the IamHandwriting dataset.
    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(root, file)  # Construct the full file path
            shutil.copy(file_path, destination_dir) # Copy the file to the destination directory
            #For error handling
            #print(file_path)
        
print('Sucess!')





