import GlobalConstants as paths

#Paths
root = paths.source
txts_path = root + "txts/"
val_path = txts_path + "val_names.txt"
train_path = txts_path + "train_names.txt"
test_path = txts_path + "test_names.txt"

with open(val_path, 'r') as val_file:
    val_names = set(val_file.read().splitlines())

with open(train_path, 'r') as train_file:
    train_names = set(train_file.read().splitlines())

with open(test_path, 'r') as test_file:
    test_names = set(test_file.read().splitlines())


if not val_names.intersection(train_names) and not val_names.intersection(test_names) and not train_names.intersection(test_names): #Check for common elements between sets
    print("No common names found between the files.")
else:
    print("There are common names between the files.")
