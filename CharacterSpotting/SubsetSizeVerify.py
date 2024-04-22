import os
import shutil
import random
import GlobalConstants as paths

root = paths.source
wordsCompressed_path = root + "wordsCompressedSubset40k/"
count_img = 0
count_ann = 0

# Creating Image/Annotation pairs
image_files = [f for f in os.listdir(wordsCompressed_path) if f.endswith('.jpg')]
pairs = [(img, img.replace('.jpg', '.txt')) for img in image_files if os.path.exists(os.path.join(wordsCompressed_path, img.replace('.jpg', '.txt')))]


for img_file in pairs:
    count_img += 1

for anno_file in pairs:
    count_ann += 1

print(f"{count_ann} annotations in the folder.")
print(f"{count_img} images in the folder")
