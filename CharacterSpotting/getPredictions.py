##Sample code
import os
from ultralytics import YOLO
from PIL import Image
import GlobalConstants as paths

def list_format(predictions):
    predictions_list = []
    for prediction in predictions:
        prediction_dict ={}
        prediction_data = prediction.split(" ")
        prediction_dict["class_num"] = prediction_data[0]
        prediction_dict["x"] = prediction_data[1]
        prediction_dict["y"] = prediction_data[2]
        prediction_dict["w"] = prediction_data[3]
        prediction_dict["h"] = prediction_data[4].replace("\n","")
        if len(prediction_data) == 6:
            prediction_dict["conf"] = prediction_data[5].replace("\n","")
        predictions_list.append(prediction_dict)
    return predictions_list

def dict_to_str(prediction):
    return f"{prediction['class_num']} {prediction['x']} {prediction['y']} {prediction['w']} {prediction['h']} {prediction['conf']}\n"


def sort_by_x(prediction_list):
    return sorted(prediction_list, key=lambda box: float(box['x']))

def remove_low_confidence(predictions_list):
    for i in range(len(predictions_list)-1,0,-1):
        predictions = predictions_list[i]
        if float(predictions["conf"]) < confidence_threshold:
            predictions_list.remove(predictions_list[i])
    return predictions_list


def nms(predictions_list, iou_threshold):
    predictions_list = sorted(predictions_list, key=lambda x: float(x['conf']), reverse=True)
    keep = []
    
    while predictions_list:
        current = predictions_list.pop(0)
        keep.append(current)
        predictions_list = [box for box in predictions_list if calculate_IOU(current, box) < iou_threshold]

    return keep

def xywh_to_x1y1x2y2(x,y,w,h): #Is it 1 in the top for y or at the bottom? This proposes at the top.
    x1 = float(x) - float(w)/2
    x2 = x1 + float(w)
    y1 = float(y) + float(h)/2
    y2 = y1 - float(h)
    return [x1,y1,x2,y2]


def calculate_IOU(prediction1, prediction2): #Not the greatest calculation but it should work
    box1 = xywh_to_x1y1x2y2(prediction1["x"],prediction1["y"],prediction1["w"],prediction1["h"])
    box2 = xywh_to_x1y1x2y2(prediction2["x"],prediction2["y"],prediction2["w"],prediction2["h"])
    #union
    leftmost_x = min(box1[0],box2[0])
    rightmost_x = max(box1[2],box2[2])
    upper_y = max(box1[1],box2[1])
    lower_y = min(box1[3],box2[3])
    union_area = (rightmost_x - leftmost_x) * (upper_y - lower_y)
    #intersection
    intersection_x1 = max(box1[0],box2[0]) #left side of intersection area
    intersection_y1 = min(box1[1],box2[1]) #upperside of intersection area
    intersection_x2 = min(box1[2],box2[2]) #right side of intersection area
    intersection_y2 = max(box1[0],box2[0]) #lower side of intersection area
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y1 - intersection_y2)
    # print(str(prediction1["class_num"]) + "     " + str(prediction2["class_num"]))
    # print(intersection_area / union_area)

    return intersection_area / union_area


testSet = paths.source + "testSet"
weights_path = paths.best_model
model = YOLO(weights_path)

confidence_threshold = 0.3
IOU_threshold = 0.55

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
        for box in boxes: ##Not quite sure if this is the right structure. I will test with Kubernetes later.
            cls = int(box.cls[0])
            # class_name = model.names[cls]
            conf = box.conf[0]
            bx = box.xywh.tolist()
            results_string += f"{cls} {bx[0][0]/width} {bx[0][1]/height} {bx[0][2]/width} {bx[0][3]/height} {conf}\n" ##Must normalise according to the size of the image
            # print(conf)
        with open(pred_path, 'w') as file:
            file.write(results_string)


prediction_files = [f for f in os.listdir(testSet) if f.endswith('*.txt')]
for pred_file in prediction_files:
    pred_path = os.path.join(testSet, pred_file)
    
    with open(pred_path, 'r') as file:
        predictions = file.readlines()
    predictions_list = list_format(predictions)

    predictions_list = sort_by_x(predictions_list)
    predictions_list = remove_low_confidence(predictions_list)
    predictions_list = sort_by_x(predictions_list)
    predictions_list = nms(predictions_list,IOU_threshold)
    predictions_list = sort_by_x(predictions_list)


    with open(pred_path, 'w') as file:
        for prediction in predictions_list:
            file.write(dict_to_str(prediction))

