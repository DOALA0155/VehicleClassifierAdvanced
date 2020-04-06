from keras.models import Sequential
from image_processing import get_image_data
import numpy as np
import matplotlib.pyplot as plt
from functions import *

images, labels = get_image_data()
print(images.shape)
x_train, x_test, y_train, y_test = categorical_data(images, labels)
x_train, x_test = scaling_data(x_train, x_test)
words = get_words()

model = define_model()

train_model = model.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=8)

score_history = train_model.history
plot_history(score_history)

score = model.evaluate(x_test, y_test)
print("Test loss: {:.3f}".format(score[0]))
print("Score: {:.3f}".format(score[1]))
model.save("./Models/CnnModel2.h5")
