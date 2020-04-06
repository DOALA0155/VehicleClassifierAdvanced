from PIL import Image
import os
import numpy as np
import pyprind
from functions import get_image_size

with open("./word.txt", "r") as f:
    words = f.read().split(",")
    words = [word.strip("\n") for word in words]

    for index, word in enumerate(words):
        dir_path = "./Images/{}".format(word)
        image_names = os.listdir(dir_path)

        bar = pyprind.ProgBar(len(image_names), track_time=True, title="Image {}".format(word))

        for j, image_name in enumerate(image_names):
            bar.update()
            if "jpg" not in image_name:
                continue

            image_path = dir_path + "/" + image_name
            image = Image.open(image_path)
            image_size = get_image_size(reversed=True)
            resized_size = [image_size[0], image_size[1]]
            resized_image = image.resize(resized_size, Image.ANTIALIAS)
            resized_image.save("{}/{}.jpg".format(dir_path, j))
            os.remove(image_path)
