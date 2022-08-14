"""
Prepare the data for training the yolov5 model
"""

import os
import pandas as pd
from sqlmodel import Session, select

import cv2

from database.database import engine
from database.db_tables import ClothesCategory, TrainDeepFashion, ValidationDeepFashion


YOLO_TRAIN_PATH = r"datasets/train/yolo/"
YOLO_VALIDATION_PATH = r"datasets/validation/yolo/"
os.makedirs(YOLO_TRAIN_PATH, exist_ok=True)
os.makedirs(YOLO_VALIDATION_PATH, exist_ok=True)


def prepare_yolo_dataset(dataset, dataset_size, path):
    "Prepare the data for training the yolov5 model"

    for index, row in dataset.iterrows():
        if index >= dataset_size:
            break
        # print(row["image"])
        img = cv2.imread(row["image"])
        cv2.imwrite(path + row["image"].split("/")[-1], img)

        imgheight, imgwidth, _ = img.shape

        box = row["boundingbox"]
        xcenter = ((box[0] + box[2]) / 2) / imgwidth
        ycenter = ((box[1] + box[3]) / 2) / imgheight
        width = (box[2] - box[0]) / imgwidth
        height = (box[3] - box[1]) / imgheight

        with open(
            path + (row["image"].split("/")[-1]).split(".")[0] + ".txt",
            "a+",
            encoding="UTF-8",
        ) as f:
            f.write(
                str(row["cat_id"])
                + " "
                + str(xcenter)
                + " "
                + str(ycenter)
                + " "
                + str(width)
                + " "
                + str(height)
                + "\n"
            )
    print("DONE")


def read_db_table(table_name):
    """Read DeepFashion dataset from DB"""

    with Session(engine) as session:
        statement = select(
            table_name.image,
            table_name.boundingbox,
            ClothesCategory.generic_category,
        ).join_from(
            table_name,
            ClothesCategory,
            table_name.categories == ClothesCategory.extended_category,
        )
        data_table = session.exec(statement).all()

    dataset = pd.DataFrame.from_records(
        data_table, columns=["image", "boundingbox", "category"]
    )
    dataset["cat_id"], _ = pd.factorize(dataset["category"])

    return dataset


if __name__ == "__main__":
    data = read_db_table(TrainDeepFashion)
    prepare_yolo_dataset(data, 8000, YOLO_TRAIN_PATH)
    data = read_db_table(ValidationDeepFashion)
    prepare_yolo_dataset(data, 2000, YOLO_VALIDATION_PATH)
