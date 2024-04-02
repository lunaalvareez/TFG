import os
from ultralytics import YOLO
import editdistance
##Boxes to text + WER

def compare_words(prediction_word,label_word):
    if prediction_word == label_word:
        return True
    else:
        return False

def calc_cer(prediction_word,label_word):
    distance = editdistance.eval(label_word,prediction_word)
    cer = distance / len(label_word)

    return cer



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


def sort_by_x(prediction_list):
    return sorted(prediction_list, key=lambda box: float(box['x']))

def x_to_text(predictions_list):
    text = ""
    for predictions in predictions_list:
        text += " " + model.names[int(predictions["class_num"])]
    return text

def remove_low_confidence(predictions_list):
    for i in range(len(predictions_list)-1,0,-1):
        predictions = predictions_list[i]
        if float(predictions["conf"]) < confidence_threshold:
            predictions_list.remove(predictions_list[i])
    return predictions_list

# def remove_high_IOU(predictions_list): #Compares each box with eachother. However right now it adds them to this list even if they are deemed not good enough.
#     new_predictions_list = []
#     for i in range(len(predictions_list)):
#         for k in range(len(predictions_list)):
#             if calculate_IOU(predictions_list[i],predictions_list[k]) < IOU_threshold:
#                 if(predictions_list[i]["conf"] < predictions_list[k]["conf"]):
#                     if(predictions_list[k] not in new_predictions_list):
#                         new_predictions_list.append(predictions_list[k])
#                 else:
#                     if(predictions_list[i] not in new_predictions_list):
#                         new_predictions_list.append(predictions_list[i])
#     return new_predictions_list

def remove_high_IOU(predictions_list): #Compares each box with the box to the right
    new_predictions_list = []
    for i in range(len(predictions_list)-1):
        if calculate_IOU(predictions_list[i],predictions_list[i+1]) < IOU_threshold:
            if(predictions_list[i]["conf"] < predictions_list[i+1]["conf"]):
                if(predictions_list[i+1] not in new_predictions_list):
                    new_predictions_list.append(predictions_list[i+1])
            else:
                if(predictions_list[i] not in new_predictions_list):
                    new_predictions_list.append(predictions_list[i])
    return new_predictions_list




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






        
        
confidence_threshold=0.45
IOU_threshold = 0.2

comparisonFiles_path = "/Users/ludvig/LTU/Characterspotting9/trainEvalSet"
weights_path = "/Users/ludvig/LTU/Characterspotting9/best_1.pt" #To get "model.names" Change this to the list of indexes instead.
model = YOLO(weights_path)

correct_words = 0
incorrect_words = 0
correct_words_list = []
total_cer = 0
total_character_count = 0

annotation_files = [f for f in os.listdir(comparisonFiles_path) if f.endswith('*.txt')]
for anno_file in annotation_files:
    pred_path = os.path.join(comparisonFiles_path, anno_file)
    anno_path = pred_path.replace("*","")


    with open(pred_path, 'r') as file:
        predictions = file.readlines()
    predictions_list = list_format(predictions)
    


    with open(anno_path, 'r') as file:
        labels = file.readlines()
    label_list = list_format(labels)

    predictions_list = sort_by_x(predictions_list)
    predictions_list = remove_low_confidence(predictions_list)
    predictions_list = sort_by_x(predictions_list)
    # predictions_list = remove_high_IOU(predictions_list)
    # predictions_list = sort_by_x(predictions_list)

    pred_word = x_to_text(predictions_list).replace(" ","")
    ground_word = x_to_text(label_list).replace(" ","")



    #cer
    cer_individual = calc_cer(pred_word,ground_word)
    # print(f"Ground: {ground_word}    Pred: {pred_word}    Cer:{cer_individual}")
    total_cer += cer_individual
    total_character_count += len(x_to_text(label_list))


    #wer    
    if compare_words(pred_word,ground_word):
        correct_words +=1
        correct_words_list.append(pred_path)
    else:
        incorrect_words +=1
    if(incorrect_words%200 == 0):
        print(f"Correct words: {correct_words}")
        print(f"Incorrect words: {incorrect_words}")
    
    

print(f"WER: {incorrect_words/(correct_words+incorrect_words)}")
print(f"CER: {total_cer /total_character_count}")
       

# print(f"After remove_high_IOU: {x_to_text(predictions_list)}                       IOU_threshold = {IOU_threshold}")


