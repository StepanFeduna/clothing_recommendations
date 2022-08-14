"""
With the already trained model, extract characteristics array of the crawl images.
"""
from urllib import request
from io import BytesIO
from PIL import Image

import tensorflow as tf
from keras.models import Model
from keras.preprocessing import image
import numpy as np

from sqlmodel import Session, select
from database.database import create_db_and_tables, fill_table, engine
from database.db_tables import ResNet50v2Model, CrawlData


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


def get_embedding(model, image_url):
    """Use the trained model to generate a vector representation of the image"""

    with request.urlopen(image_url) as url:
        img = Image.open(BytesIO(url.read())).resize(IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # print(x)
    return model.predict(x).reshape(-1)


def load_model():
    """Load model file from it's path link in DB"""
    with Session(engine) as session:
        statement = select(ResNet50v2Model.best_model)
        best_model = session.exec(statement).first()

    print(best_model)

    restored_model = tf.keras.models.load_model(best_model)
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

    return secondmodel


def num_array_gen():
    """Generate array of images characteristics"""

    with Session(engine) as session:
        statement = select(CrawlData.id, CrawlData.image_link)
        ids_and_images_links = session.exec(statement).all()

    secondmodel = load_model()
    for _id, image_link in ids_and_images_links:
        try:
            notedarray = get_embedding(secondmodel, image_link).tolist()
        except ValueError as val_er:
            print(f"Image {image_link} in row {_id} is damaged! {val_er.__class__.__name__}")
            continue
        except Exception as exception:
            print(f"Oops! {exception.__class__.__name__}")
            continue
        
        yield {"id": _id, "notedarray": notedarray}
    # print(get_embedding(secondmodel, images_links[0]).tolist())


if __name__ == "__main__":
    create_db_and_tables()
    fill_table(CrawlData, num_array_gen(), update=True)
