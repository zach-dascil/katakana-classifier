from torch import nn
from torch import optim
import torch

from package import DataLoader


def train(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, criterion:nn.CrossEntropyLoss, optimizer: optim.Adam, device: torch.device, epochs: int):
    
    training_acc = []
    valid_acc = []

    for i in range(epochs):
        print("Epoch:" + str(i))

        acc = 0
        correct = 0
        total = 0

        for batch in train_loader:

            # Prepares data for training model
            train, label = batch
            train, label = train.to(device), label.to(device)
            train = torch.reshape(train, [train.shape[0], 1, train.shape[1], train.shape[2]])

            # Calculate loss
            pred = model(train)
            loss = criterion(pred, label)

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running Total 
            pred_max = torch.argmax(pred, dim=1)
            total += label.shape[0]
            correct += torch.eq(pred_max,label).sum().item()
        
        # Calculates accuracy of training
        acc = correct/total
        training_acc.append(acc)
        print("Training accuracy: " + str(acc))

        # Validation
        acc = accuracy_test(model, valid_loader, device)
        valid_acc.append(acc)
        print("Validation accuracy: " + str(acc))

    return (training_acc, valid_acc)

def accuracy_test(model: nn.Module, loader: DataLoader, device: torch.device):
    with torch.no_grad():    
        correct = 0
        total = 0
    
        for batch in loader:
            # Prepares data for model
            image, label = batch
            image, label = image.to(device), label.to(device)
            image = torch.reshape(image, [image.shape[0], 1, image.shape[1], image.shape[2]])
            pred = model(image)

            # Running total
            pred_max = torch.argmax(pred, dim=1)
            total += label.shape[0]
            correct += torch.eq(pred_max,label).sum().item()

        return correct/total