import os
import random
import GlobalConstants as paths

#Paths
root = paths.source
trainEvalSet_original_path = root + "trainEvalSet/"
testSet_original_path = root + "testset/"
txts_path = root + "txts/"
#For google drive. This is to be changed if you get errors in the Google Colab Notebook.
trainEvalSet_new_base_path = "/content/drive/MyDrive/Characterspotting/data/trainEvalSet/"
testSet_new_base_path = "/content/drive/MyDrive/Characterspotting/data/testSet/"


os.makedirs(txts_path, exist_ok=True) #Check if dir exists


def list_image_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.png')]

def format_paths_cloud(file_list, new_base):
    return [os.path.join(new_base, f) for f in file_list]

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write("\n".join(content))


trainEvalSet_files = list_image_files(trainEvalSet_original_path)
testSet_files = list_image_files(testSet_original_path)

random.shuffle(trainEvalSet_files)  #Randomize selection

#splits files into train and val
train_files = trainEvalSet_files[:1200]
val_files = trainEvalSet_files[1200:1500]



#formats file paths for drive and local
train_files_cloud = format_paths_cloud(train_files, trainEvalSet_new_base_path)
val_files_cloud = format_paths_cloud(val_files, trainEvalSet_new_base_path)
test_files_cloud = format_paths_cloud(testSet_files, testSet_new_base_path)

#drive path files
write_to_file(os.path.join(txts_path, "train.txt"), train_files_cloud)
write_to_file(os.path.join(txts_path, "val.txt"), val_files_cloud)
write_to_file(os.path.join(txts_path, "test.txt"), test_files_cloud)

#local path files
write_to_file(os.path.join(txts_path, "train_local.txt"), [os.path.join(trainEvalSet_original_path, f) for f in train_files])
write_to_file(os.path.join(txts_path, "val_local.txt"), [os.path.join(trainEvalSet_original_path, f) for f in val_files])
write_to_file(os.path.join(txts_path, "test_local.txt"), [os.path.join(testSet_original_path, f) for f in testSet_files])

#name-only files
write_to_file(os.path.join(txts_path, "train_names.txt"), train_files)
write_to_file(os.path.join(txts_path, "val_names.txt"), val_files)
write_to_file(os.path.join(txts_path, "test_names.txt"), testSet_files)

print("Files written")
