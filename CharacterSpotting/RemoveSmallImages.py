from PIL import Image
import os
import GlobalConstants as paths

wordsCompressed_path = paths.source + "wordsCompressed/"

count_img = 0
count_ann = 0
minsize = 10


image_files = [f for f in os.listdir(wordsCompressed_path) if f.endswith('.jpg')] # List all image files
for img_file in image_files:
    img_path = os.path.join(wordsCompressed_path, img_file)
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            
            
            if width < minsize or height < minsize: # Check if either dimension is less than 10 pixels. For Yolov6, maybe also Yolov8, this is the minimum input image size.
                os.remove(img_path) # Remove the image file
                #print(f"Removed image: {img_file}")
                count_img += 1
                
                # Identify and remove the corresponding annotation file
                anno_file = img_file.replace('.png', '.txt')
                anno_path = os.path.join(wordsCompressed_path, anno_file)
                if os.path.exists(anno_path):
                    os.remove(anno_path)
                    #print(f"Removed annotation: {anno_file}")
                    count_ann += 1
        
    except Exception as e:
        print(f"Error processing file {img_file}: {e}") #Remove files that are not able to be read
        os.remove(img_path)
        #print(f"Removed image: {img_file}")
        count_img += 1
        

print(f"Removed annotation count: {count_ann}")
print(f"Removed image count: {count_img}")
print(f"Removed images with errors count: {count_img - count_ann}")