import matplotlib.pyplot as plt

def plot_model(model_name: str, training_acc: list, valid_acc: list, epochs: int):
    x_axis = list(range(1,epochs+1))
    plt.figure()
    plt.plot(x_axis, training_acc, label="Training Accuracy")
    plt.plot(x_axis, valid_acc, label="Validation Accuracy")
    plt.title(model_name+"Model: Epochs vs Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

