from package import Dataset

from train_test import train_image_paths
from train_test import test_image_paths
from train_test import valid_image_paths
from class_idx_conversion import class_to_idx
from matplotlib import pyplot

class KatakanaDataset(Dataset):
       
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = pyplot.imread(image_filepath)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        
        return image, label

train_dataset = KatakanaDataset(train_image_paths)
test_dataset = KatakanaDataset(test_image_paths)
valid_dataset = KatakanaDataset(valid_image_paths)