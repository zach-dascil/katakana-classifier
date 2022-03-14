from package import DataLoader
from katakana_dataset import train_dataset
from katakana_dataset import test_dataset
from katakana_dataset import valid_dataset

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=32, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=True
)