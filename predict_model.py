from keras.models import load_model
from PIL import Image
import numpy as np
from functions import *
import sys

def predict_single(image_name):
    image_path = "./PredictImages/{}".format(image_name)
    model = load_model("./Models/VehivleCnnModel.h5")
    try:
        image_data = Image.open(image_path)
    except:
        print("I can't get image.")
        print("Only 'jpg' images can be judged, and please check your image_name and set in 'PredictImages' folder.")
        print("× './PredictImages/sample.jpg' ○ 'sample.jpg'")
        return

    image_size = get_image_size(reversed=True)
    resized_size = [image_size[0], image_size[1]]
    resized_image = image_data.resize(resized_size, Image.ANTIALIAS)
    resized_image_data = np.asarray(resized_image) / 255.
    image = np.expand_dims(resized_image_data, axis=0)

    class_probability = model.predict(image)[0]
    class_label = np.argmax(class_probability)

    words = get_words()
    class_word = words[class_label]

    probability = class_probability[class_label] * 100

    print("This image is {}, probability: {:2f}.".format(class_word, probability))

def get_model_summary():
    model = load_model("./Models/VehivleCnnModel.h5")
    print(model.summary())

if __name__ == "__main__":
    try:
        image_name = sys.argv[1]
        predict_single(image_name)
    except:
        print("Please set image name after 'python predict_model.py'")
        print("Example: 'python predict_model.py sample.jpg'")
