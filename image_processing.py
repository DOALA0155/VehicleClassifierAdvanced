from PIL import Image
import os
import numpy as np
import pyprind
from functions import get_image_size

def get_image_data():
    with open("./word.txt", "r") as f:
        words = f.read().split(",")
        words = [word.strip("\n") for word in words]

    images = []
    labels = []

    for index, word in enumerate(words):
        dir_path = "./images/{}".format(word)
        image_paths = os.listdir(dir_path)

        bar = pyprind.ProgBar(len(image_paths), track_time=True, title="Image {}".format(word))

        for j, image_path in enumerate(image_paths):
            bar.update()
            if "jpg" not in image_path:
                continue

            image = np.asarray(Image.open(dir_path + "/" + image_path))
            image_size = get_image_size()

            if image.shape != image_size:
                continue

            images.append(image)
            labels.append(index)

    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels
