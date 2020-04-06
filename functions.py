import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout


def plot_history(score_history):
    acc_history = score_history["accuracy"]
    val_acc_history = score_history["val_accuracy"]
    loss_history = score_history["loss"]
    val_loss_history = score_history["val_loss"]
    x = range(len(acc_history))

    plt.plot(x, acc_history, label="train_accuracy")
    plt.plot(x, val_acc_history, label="test_accuracy")
    plt.legend(loc="best")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(x, loss_history, label="train_loss")
    plt.plot(x, val_loss_history, label="test_loss")
    plt.legend(loc="best")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def define_model():
    model = Sequential()
    image_size = get_image_size()
    words = get_words()
    model.add(Conv2D(120, kernel_size=(4, 4), activation="relu", input_shape=image_size))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(len(words), activation="softmax"))
    print(model.summary())

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def scaling_data(x_train, x_test):
    x_train_scaled = x_train / 255.
    x_test_scaled = x_test / 255.
    return x_train_scaled, x_test_scaled

def get_words():
    with open("./word.txt", "r") as f:
        words = f.read().split(",")
        words = [word.strip("\n") for word in words]
    return words

def get_image_size(reversed=False):
    with open("./image_size.txt") as f:
        sizes = f.read().split(",")
        sizes = [int(size.strip("\n")) for size in sizes]

    if reversed:
        return (sizes[1], sizes[0], sizes[2])
    else:
        return tuple(sizes)

def categorical_data(images, labels):
    x_train, x_test, y_train, y_test = train_test_split(images, labels)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test
