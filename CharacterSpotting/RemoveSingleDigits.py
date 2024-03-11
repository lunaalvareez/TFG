import os
import GlobalConstants as paths

root = paths.source
wordsCompressed_path = root + "wordsCompressed/"
count_img = 0
count_ann = 0




annotation_files = [f for f in os.listdir(wordsCompressed_path) if f.endswith('.txt')]

for anno_file in annotation_files:
    anno_path = os.path.join(wordsCompressed_path, anno_file)
    try:
        with open(anno_path, 'r') as file:
            lines = file.readlines()
            if len(lines) == 1:
                os.remove(anno_path) #Remove annotation file
                count_ann += 1
                print(f"Removed annotation: {anno_file}")
                #Removal of corresponding image file
                img_file = anno_file.replace('.txt', '.png')
                img_path = os.path.join(wordsCompressed_path, img_file)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    count_img += 1
                    print(f"Removed image: {img_file}")
    except Exception as e:
        print(f"Error processing file {anno_file}: {e}")



print(f"Removed annotation count: {count_ann}")
print(f"Removed image count: {count_img}")
print(f"Removed annotationfiles with errors count: {count_ann - count_img}")
