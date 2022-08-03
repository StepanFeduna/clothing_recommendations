"""
With the already trained model, extract characteristics array of the crawl images.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import json
import numpy as np


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


dataset = {"item": {}, "info": []}


def get_embedding(model, imagename):
    """Use the trained model to generate a vector representation of the image"""
    img = image.load_img(imagename, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(x).reshape(-1)


restored_model = tf.keras.models.load_model("model/bestmodel.h5")
restored_model.compile(
    loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
)

# Reuse last activation layer to extract image characteristics 
secondmodel = Model(
    inputs=restored_model.input, outputs=restored_model.layers[-4].output
)
secondmodel.compile(
    loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
)

with open("datasets/crawlfile.json", "r") as f:
    temp = json.loads(f.read())
    for i in range(len(temp)):
        category = str(temp[i]["Category"])
        name = str(temp[i]["Name"])
        url = str(temp[i]["URL"])
        price = str(temp[i]["Price"])
        imagelink = str(temp[i]["ImageLink"])
        img = "datasets/crawldata/" + category + "/" + str(i).zfill(6) + ".jpg"
        notedarray = get_embedding(secondmodel, img).tolist()
        dataset["info"].append(
            {
                "no": i,
                "categories": category,
                "name": name,
                "url": url,
                "imagelink": img,
                "notedarray": notedarray,
            }
        )

# print(dataset)
with open("datasets/notedarray.json", "w") as f:
    json.dump(dataset, f)
