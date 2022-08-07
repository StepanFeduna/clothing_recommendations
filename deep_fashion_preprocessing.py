"""
Used to transfer DeepFashion dataset from json files to PostgreSql
"""

import os
import json
import fnmatch

from typing import Optional, List

from sqlmodel import Column, Field, Session, SQLModel, create_engine, ARRAY, INTEGER


TRAIN_JSON_PATH = r"datasets/train/annos/"
VALIDATION_JSON_PATH = r"datasets/validation/annos/"

TRAIN_IMAGES_PATH = r"datasets/train/image/"
VALIDATION_IMAGES_PATH = r"datasets/validation/image/"

TRAIN_IMAGES_NUMBER = len(fnmatch.filter(os.listdir(TRAIN_IMAGES_PATH), "*.jpg"))
VALIDATION_IMAGES_NUMBER = len(
    fnmatch.filter(os.listdir(VALIDATION_IMAGES_PATH), "*.jpg")
)


class ClothesCategory(SQLModel, table=True):
    """Table that alias extendable clothes category names with generic ones"""

    extended_category: str = Field(default=None, primary_key=True)
    generic_category: str = None


class DeepFashionBase(SQLModel):
    """Template for DeepFashion dataset tables"""

    boundingbox: List = Field(sa_column=Column(ARRAY(INTEGER)))
    image: str
    categories: str = Field(foreign_key="clothescategory.extended_category")


class TrainDeepFashion(DeepFashionBase, table=True):
    """Table with train data of DeepFashion dataset"""

    id: Optional[int] = Field(default=None, primary_key=True)


class ValidationDeepFashion(DeepFashionBase, table=True):
    """Table with validation data of DeepFashion dataset"""

    id: Optional[int] = Field(default=None, primary_key=True)


engine = create_engine(
    "postgresql://postgres:umimuv27@localhost:5432/clothing_db", echo=True
)


def create_db_and_tables():
    """Set connection with DB and create tables"""
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def preproces_data(json_path, image_path, num_of_images):
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

                yield {"categories": cat, "boundingbox": bbox, "image": image_name}


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
        preproces_data(TRAIN_JSON_PATH, TRAIN_IMAGES_PATH, TRAIN_IMAGES_NUMBER),
    )

    fill_table(
        ValidationDeepFashion,
        preproces_data(
            VALIDATION_JSON_PATH, VALIDATION_IMAGES_PATH, VALIDATION_IMAGES_NUMBER
        ),
    )
