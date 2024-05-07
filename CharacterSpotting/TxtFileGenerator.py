import os
import random
import GlobalConstants as paths

#Paths
root = paths.source
trainEvalSet_original_path = root + "trainEvalSet/"
testSet_original_path = root + "testSet/"
txts_path = root + "txts/"

os.makedirs(txts_path, exist_ok=True) #Check if dir exists


def list_image_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.jpg')]

# def format_paths_cloud(file_list, new_base):
#    return [os.path.join(new_base, f) for f in file_list]

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write("\n".join(content))


trainEvalSet_files = list_image_files(trainEvalSet_original_path)
testSet_files = list_image_files(testSet_original_path)

random.shuffle(trainEvalSet_files)  #Randomize selection

#splits files into train and val
train_files = trainEvalSet_files[:1600]
val_files = trainEvalSet_files[1600:2000]

#local path files
write_to_file(os.path.join(txts_path, "train.txt"), [os.path.join(trainEvalSet_original_path, f) for f in train_files])
write_to_file(os.path.join(txts_path, "val.txt"), [os.path.join(trainEvalSet_original_path, f) for f in val_files])
write_to_file(os.path.join(txts_path, "test.txt"), [os.path.join(testSet_original_path, f) for f in testSet_files])

#name-only files
write_to_file(os.path.join(txts_path, "train_names.txt"), train_files)
write_to_file(os.path.join(txts_path, "val_names.txt"), val_files)
write_to_file(os.path.join(txts_path, "test_names.txt"), testSet_files)

print("Files written")
