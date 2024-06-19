##Sample code
import os
from ultralytics import YOLO
from PIL import Image
import GlobalConstants as paths

testSet = paths.source + "testSet"
weights_path = paths.best_model


model = YOLO(weights_path)


annotation_files = [f for f in os.listdir(testSet) if f.endswith('.txt')]

for anno_file in annotation_files:
    anno_path = os.path.join(testSet, anno_file)
    pred_path_temp = anno_path.split(".")
    pred_path = pred_path_temp[0] + "*." + pred_path_temp[1]

    img_path = anno_path.replace(".txt",".jpg")

    results = model(img_path) #Batched can be made via results = model([img_path],[img_path2]) etc...
    results_string = ""
    with Image.open(img_path) as img:
        width, height = img.size

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes: 
            cls = int(box.cls[0])
            # class_name = model.names[cls]
            conf = box.conf[0]
            bx = box.xywh.tolist()
            results_string += f"{cls} {bx[0][0]/width} {bx[0][1]/height} {bx[0][2]/width} {bx[0][3]/height} {conf}\n" ##Must normalise according to the size of the image
            # print(conf)
        with open(pred_path, 'w') as file:
            file.write(results_string)

