import os
import shutil
import random
import GlobalConstants as paths
root = paths.source
wordsCompressedSubset_path = root + "wordsCompressedSubset40K/"
trainEvalSet_path = root + "trainEvalSet"
testSet_path = root + "testSet"
trainset_list_path = os.path.join(root, "trainset.txt")
testset_list_path = os.path.join(root, "testset.txt")

#Check if the directories exists
os.makedirs(trainEvalSet_path, exist_ok=True)
os.makedirs(testSet_path, exist_ok=True)

#load the list of identifiers to be included in the trainEvalSet. Identifiers are the ones provided in "trainset.txt". 
with open(trainset_list_path, 'r') as file:
    trainset_identifiers = file.read().splitlines()

def matches_identifier(filename, identifiers):
    return any(identifier in filename for identifier in identifiers)

#list all image/annotation pairs
all_image_files = [f for f in os.listdir(wordsCompressedSubset_path) if f.endswith('.jpg')]
all_pairs = [(img, img.replace('.jpg', '.txt')) for img in all_image_files if os.path.exists(os.path.join(wordsCompressedSubset_path, img.replace('.jpg', '.txt')))]

trainset_pairs = [(img, img.replace('.jpg', '.txt')) for img in all_image_files if matches_identifier(img, trainset_identifiers) and os.path.exists(os.path.join(wordsCompressedSubset_path, img.replace('.jpg', '.txt')))]#filter pairs based on identifiers in trainset.txt for potential inclusion in trainEvalSet

selected_pairs_for_trainEvalSet = random.sample(trainset_pairs, min(2000, len(trainset_pairs))) #randomly select up to 2000 pairs from those matching the identifiers

for img_file, anno_file in selected_pairs_for_trainEvalSet: #copy the selected pairs to trainEvalSet
    shutil.copy(os.path.join(wordsCompressedSubset_path, img_file), os.path.join(trainEvalSet_path, img_file))
    shutil.copy(os.path.join(wordsCompressedSubset_path, anno_file), os.path.join(trainEvalSet_path, anno_file))

#load the list of identifiers to be included in the testSet. Identifiers are the ones provided in "testset.txt". 
with open(testset_list_path, 'r') as file:
    testset_identifiers = file.read().splitlines()

testset_pairs = [(img, img.replace('.jpg', '.txt')) for img in all_image_files if matches_identifier(img, testset_identifiers) and os.path.exists(os.path.join(wordsCompressedSubset_path, img.replace('.jpg', '.txt')))]#filter pairs based on identifiers in testSet.txt for potential inclusion in testSet

selected_pairs_for_testSet = random.sample(testset_pairs, min(700, len(testset_pairs))) #randomly select up to 700 pairs from those matching the identifiers

for img_file, anno_file in selected_pairs_for_testSet: #copy the selected pairs to testSet
    shutil.copy(os.path.join(wordsCompressedSubset_path, img_file), os.path.join(testSet_path, img_file))
    shutil.copy(os.path.join(wordsCompressedSubset_path, anno_file), os.path.join(testSet_path, anno_file))

print(f"Total {len(selected_pairs_for_trainEvalSet)} image-annotation pairs have been copied to trainEvalSet.")
print(f"Total {len(selected_pairs_for_testSet)} image-annotation pairs have been copied to testset.")