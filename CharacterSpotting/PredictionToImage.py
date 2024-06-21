import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import math
import random

#change folder to the folder with the images and the annotation files you want to superimpose.
folder = "/Users/ludvig/LTU/Characterspotting11/testSetx_bigimg"

folder_name = "testSetx_bigimg" #this needs to be changed as well. File structure like above. !!!!!
test_folder_name = "test_x" #this is where the images will be saved to

weights_path = "/Users/ludvig/LTU/Characterspotting8/best_hypbigimg.pt" #Doesn't have to be the same weights as the predicitons were made with as long as the class_id's are the same. See "data_info.yaml"


model = YOLO(weights_path)

def draw_boxes(image_path, annotation_path, output_path):
    #convert to numpyarray
    image = Image.open(image_path)
    image = np.array(image)
    width, height = image.shape[1], image.shape[0]

    #reading annotations
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray') #cmap gray, otherwise the image gets a yellow hue.
    
    #line by line for loop
    for line in lines:
        colors = []
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        #convert normalized coordinates to pixel coordinates
        box_x = (x_center - box_width / 2) * width
        box_y = (y_center - box_height / 2) * height
        box_width *= width
        box_height *= height

        #drawing the boxes
        rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2,
                                 edgecolor=(random.random(),random.random(),random.random()), facecolor='none')
        
        ax.add_patch(rect)

        #adding class label at the top left corner of each box
        ax.text(box_x, box_y, model.names[int(class_id)], color='black', ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

        #adding confidence value at the bottom left corner of each box
        if len(parts) == 6:
            confidence = float(parts[5])
            ax.text(box_x, box_y+(box_height), confidence, color='black', ha='left', va='bottom',fontsize=6,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))


    #saving figure
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

#main for looping through each image in the folder
annotation_files = [f for f in os.listdir(folder) if f.endswith('.txt')]

for anno_file in annotation_files:
    anno_path = os.path.join(folder, anno_file)
    if "*.txt" in anno_path:#*.txt is prediction file and .txt is ground_truth
        img_path = anno_path.replace("*.txt",".png")

    else:
        img_path = anno_path.replace(".txt",".png")

    draw_boxes(img_path, anno_path, anno_path.replace(folder_name,test_folder_name).replace(".txt",".png"))
    
    print(anno_path.replace(folder_name,test_folder_name).replace(".txt",".png"))