
import os
import random
import shutil
import GlobalConstants as paths

# Paths
root_path = paths.source
wordsCompressedPath = root_path + "wordsCompressed"
subset_path = wordsCompressedPath + "Subset40K" #It's not inside the folder "wordsCompressed" it becomes "/Users/ludvig/LTU/characterspotting5/wordsCompressedSubset40K"



os.makedirs(subset_path, exist_ok=True)  # Ensure destination exists
image_files = [f for f in os.listdir(wordsCompressedPath) if f.endswith('.jpg')]

# Creating img/annotation file pairs
pairs = [(img, img.replace('.jpg', '.txt')) for img in image_files if os.path.exists(os.path.join(wordsCompressedPath, img.replace('.jpg', '.txt')))]

# Randomly select 40,000 pairs (or the total number if fewer). Can be adjusted manually.
selected_pairs = random.sample(pairs, min(40000, len(pairs)))

# Copy the selected pairs
for img_file, anno_file in selected_pairs:
    shutil.copy(os.path.join(wordsCompressedPath, img_file), os.path.join(subset_path, img_file))  # Copy image
    shutil.copy(os.path.join(wordsCompressedPath, anno_file), os.path.join(subset_path, anno_file)) # Copy annotation
    #print(img_file)

print(f"Total {len(selected_pairs)} image-annotation pairs have been copied.") #There can be a difference with this output and the file "SubsetSizeVerify.py" so please run SubsetSizeVerify.py after this.
