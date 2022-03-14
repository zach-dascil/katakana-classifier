from package import glob
import random

train_path = 'images/train' 
test_path = 'images/test'

train_image_paths = []
test_image_paths = []
classes = []

# Get files in training folder
for data_path in glob.glob(train_path + '/*'):
    classes.append(data_path.split('/')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))

# Split files in training list for a validation list
train_image_paths = [val for sublist in train_image_paths for val in sublist]
random.shuffle(train_image_paths)
train_image_paths, valid_image_paths = train_image_paths[:int(0.9*len(train_image_paths))], train_image_paths[int(0.9*len(train_image_paths)):] 

# Get test files
for data_path in glob.glob(test_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = [val for sublist in test_image_paths for val in sublist]

#print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))