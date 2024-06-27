import os
from ultralytics import YOLO
import editdistance
import GlobalConstants as paths
##Boxes to text + WER
#AnvÃ¤nd denna!!!!

def compare_words(prediction_word,label_word):
    if prediction_word == label_word:
        return True
    else:
        return False

def calc_cer(prediction_word,label_word):
    distance = editdistance.eval(label_word,prediction_word)
    cer = distance

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


def x_to_text(predictions_list):
    text = ""
    for predictions in predictions_list:
        text += " " + model.names[int(predictions["class_num"])]
    return text


comparisonFiles_path = paths.source + "testSet"
weights_path = paths.best_model #To get "model.names" Change this to the list of indexes instead.
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
    
    ##Prediction result
    pred_word = x_to_text(predictions_list).replace(" ","")
    ground_word = x_to_text(label_list).replace(" ","")



    #cer
    cer_individual = calc_cer(pred_word,ground_word)
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
    
    

print(f"WER: {incorrect_words/(correct_words+incorrect_words)}")
print(f"CER: {total_cer /total_character_count}")