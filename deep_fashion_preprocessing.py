"""
Used to transfer DeepFashion dataset from json files to PostgreSql
"""

import os
import json
import fnmatch


import cv2

from database.database import create_db_and_tables, fill_table
from database.db_tables import TrainDeepFashion, ValidationDeepFashion


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

    counter = 1
    for num in range(1, num_of_images + 1):
        json_name = json_path + str(num).zfill(6) + ".json"
        image_name = image_path + str(num).zfill(6) + ".jpg"
        with open(json_name, "r", encoding="UTF-8") as f:
            temp = json.loads(f.read())
            for i in temp:
                if i in ("source", "pair_id"):
                    continue

                bbox = temp[i]["bounding_box"]
                cat = temp[i]["category_name"]

                image = cv2.imread(image_name)
                cropped_image_name = cropped_image_path + str(counter).zfill(6) + ".jpg"

                try:
                    cv2.imwrite(
                        cropped_image_name, image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                    )
                except cv2.error:
                    print(
                        f"Can't write from {image_name}, check bounding boxes correctness!"
                    )
                    continue

                counter += 1

                yield {
                    "categories": cat,
                    "boundingbox": bbox,
                    "image": image_name,
                    "cropped_image": cropped_image_name,
                }

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    create_db_and_tables()

    fill_table(
        TrainDeepFashion,
        preproces_data(
            TRAIN_JSON_PATH,
            TRAIN_IMAGES_PATH,
            TRAIN_IMAGES_NUMBER,
            TRAIN_CROPPED_IMAGES_PATH,
        ),
        truncate=True,
    )

    fill_table(
        ValidationDeepFashion,
        preproces_data(
            VALIDATION_JSON_PATH,
            VALIDATION_IMAGES_PATH,
            VALIDATION_IMAGES_NUMBER,
            VALIDATION_CROPPED_IMAGES_PATH,
        ),
        truncate=True,
    )
