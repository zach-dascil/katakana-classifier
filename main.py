from dataloaders import *
from train import *
from plot_accuracy import *
from models import *



def main():
    # Training computation on gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    
    else:
        # cpu
        device = torch.device("cpu")

    model = cnn_complex_denser()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW
    epochs = 100
    
    train_acc, valid_acc = train(model, train_loader, valid_loader, criterion, optimizer(model.parameters(), lr=.0001), device, epochs)
    plot_model("Test ", train_acc, valid_acc, epochs)
    
    torch.save(model.state_dict(), 'models/test.pth')
    
    # Uncomment if and only if it it time to test the model
    #test_acc = accuracy_test(model, test_loader, device)
    #print("Testing Accuracy: " + str(test_acc))

if __name__ == "__main__":
    main()