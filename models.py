from torch import nn

def cnn_basic() -> nn.Module:    
    return nn.Sequential(
        nn.Conv2d(1,32,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(27840,46),
        nn.LogSoftmax(dim=1)
        )

def cnn_complex() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1,32,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,3),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(11648,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128,46),
        nn.LogSoftmax(dim=1)
        )

def cnn_complex_denser() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1,32,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(9984,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(64,46),
        nn.LogSoftmax(dim=1)
    )


def cnn_complex_densest() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1,32,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Conv2d(64,128,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2048,512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128,46),
        nn.LogSoftmax(dim=1)
        )

def cnn_double_conv() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1,32,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.Conv2d(32,32,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.Conv2d(64,64,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(5760,512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(128,46),
        nn.LogSoftmax(dim=1)
        )

def lenet() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1,6,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2,2),
        nn.Conv2d(6,16,5),
        nn.ReLU(),
        nn.Dropout(p=.15),
        nn.MaxPool2d(2,2),
        nn.Flatten(),
        nn.Linear(2496,120),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Dropout(),    
        nn.Linear(84,46),
    )