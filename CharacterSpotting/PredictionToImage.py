import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import math
import random
import GlobalConstants as paths

# Change folder to the folder with the images and the annotation files you want to superimpose.
folder = paths.source + "testSet"

folder_name = "testSet"  # this needs to be changed as well. File structure like above. !!!!!
test_folder_name = "testSet_predicted"  # this is where the images will be saved to

weights_path = paths.best_model  # Doesn't have to be the same weights as the predictions were made with as long as the class_id's are the same. See "data_info.yaml"

model = YOLO(weights_path)

def draw_boxes(image_path, annotation_path, output_path, target_width):
    
    image = Image.open(image_path)
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_height = int(target_width / aspect_ratio)
    image = image.resize((target_width, target_height), Image.LANCZOS)

    # Convert to numpy array
    image = np.array(image)
    width, height = image.shape[1], image.shape[0]

    # Reading annotations
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')  # cmap gray, otherwise the image gets a yellow hue.
    
    # Line by line for loop
    for line in lines:
        colors = []
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Convert normalized coordinates to pixel coordinates
        box_x = (x_center - box_width / 2) * width
        box_y = (y_center - box_height / 2) * height
        box_width *= width
        box_height *= height

        # Drawing the boxes
        rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2,
                                 edgecolor=(random.random(),random.random(),random.random()), facecolor='none')
        
        ax.add_patch(rect)

        # Adding class label at the top left corner of each box
        ax.text(box_x, box_y, model.names[int(class_id)], color='black', ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

        # Adding confidence value at the bottom left corner of each box
        if len(parts) == 6:
            confidence = float(parts[5])
            ax.text(box_x, box_y+(box_height), confidence, color='black', ha='left', va='bottom', fontsize=6,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Saving figure
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Main for looping through each image in the folder
annotation_files = [f for f in os.listdir(folder) if f.endswith('.txt')]

for anno_file in annotation_files:
    anno_path = os.path.join(folder, anno_file)
    if "*.txt" in anno_path:  # *.txt is prediction file and .txt is ground_truth
        img_path = anno_path.replace("*.txt", ".jpg")
    else:
        img_path = anno_path.replace(".txt", ".jpg")

    output_path = anno_path.replace(folder_name, test_folder_name).replace(".txt", ".jpg")
    draw_boxes(img_path, anno_path, output_path, 1080)
    
    print(output_path)
