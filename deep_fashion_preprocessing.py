"""
Used to transfer DeepFashion dataset from json files to PostgreSql
"""

import os
import json
import fnmatch
import uuid

from sqlmodel import Session

import cv2

from .database.database import create_db_and_tables, engine
from .database.db_tables import ClothesCategory, TrainDeepFashion, ValidationDeepFashion


TRAIN_JSON_PATH = r"datasets/train/annos/"
VALIDATION_JSON_PATH = r"datasets/validation/annos/"

TRAIN_IMAGES_PATH = r"datasets/train/image/"
VALIDATION_IMAGES_PATH = r"datasets/validation/image/"

TRAIN_IMAGES_NUMBER = len(fnmatch.filter(os.listdir(TRAIN_IMAGES_PATH), "*.jpg"))
VALIDATION_IMAGES_NUMBER = len(
    fnmatch.filter(os.listdir(VALIDATION_IMAGES_PATH), "*.jpg")
)

TRAIN_CROPPED_IMAGES_PATH = r"datasets/train/cropped_image/"
VALIDATION_CROPPED_IMAGES_PATH = r"datasets/validation/cropped_image/"
os.makedirs(TRAIN_CROPPED_IMAGES_PATH, exist_ok=True)
os.makedirs(VALIDATION_CROPPED_IMAGES_PATH, exist_ok=True)


def preproces_data(json_path, image_path, num_of_images, cropped_image_path):
    """Extract data from json files"""

    for num in range(1, num_of_images + 1):
        json_name = json_path + str(num).zfill(6) + ".json"
        image_name = image_path + str(num).zfill(6) + ".jpg"
        with open(json_name, "r", encoding="UTF-8") as f:
            temp = json.loads(f.read())
            for i in temp:
                if i in ("source", "pair_id"):
                    continue

                box = temp[i]["bounding_box"]
                bbox = [box[0], box[1], box[2], box[3]]
                cat = temp[i]["category_name"]

                image = cv2.imread(image_name)
                cropped_image_name = cropped_image_path + f"{uuid.uuid4()}" + ".jpg"

                try:
                    cv2.imwrite(
                        cropped_image_name, image[box[1] : box[3], box[0] : box[2]]
                    )
                except AssertionError:
                    print(
                        f"Can't write from {image_name}, check bounding boxes correctness!"
                    )
                    continue

                yield {
                    "categories": cat,
                    "boundingbox": bbox,
                    "image": image_name,
                    "cropped_image": cropped_image_name,
                }

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clothes_category():
    """Alias extendable clothes category names with generic ones"""

    cat_dict = {
        "shirt": {
            "short sleeve top",
            "long sleeve top",
            "vest",
            "sling",
        },
        "dress": {
            "short sleeve dress",
            "long sleeve dress",
            "vest dress",
            "sling dress",
        },
        "outwear": {"short sleeve outwear", "long sleeve outwear"},
        "short": {"shorts", "trousers"},
        "skirt": {"skirt"},
    }
    for key, value in cat_dict.items():
        for i in value:
            yield {"generic_category": key, "extended_category": i}


def fill_table(table_name, data_dict):
    """Fill DB table with data"""

    with Session(engine) as session:
        session.bulk_insert_mappings(
            table_name,
            data_dict,
        )

        session.commit()


if __name__ == "__main__":
    create_db_and_tables()

    fill_table(ClothesCategory, clothes_category())

    fill_table(
        TrainDeepFashion,
        preproces_data(
            TRAIN_JSON_PATH,
            TRAIN_IMAGES_PATH,
            TRAIN_IMAGES_NUMBER,
            TRAIN_CROPPED_IMAGES_PATH,
        ),
    )

    fill_table(
        ValidationDeepFashion,
        preproces_data(
            VALIDATION_JSON_PATH,
            VALIDATION_IMAGES_PATH,
            VALIDATION_IMAGES_NUMBER,
            VALIDATION_CROPPED_IMAGES_PATH,
        ),
    )
