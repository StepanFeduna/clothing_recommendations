"""
Contain all Database Models
"""

from typing import Optional, List

from sqlmodel import Column, Field, SQLModel, ARRAY, INTEGER


class ClothesCategory(SQLModel, table=True):
    """Table that alias extendable clothes category names with generic ones"""

    extended_category: str = Field(default=None, primary_key=True)
    generic_category: str = None


class DeepFashionBase(SQLModel):
    """Template for DeepFashion dataset tables"""

    boundingbox: List = Field(sa_column=Column(ARRAY(INTEGER)))
    image: str
    cropped_image: str
    categories: str = Field(foreign_key="clothescategory.extended_category")


class TrainDeepFashion(DeepFashionBase, table=True):
    """Table with train data of DeepFashion dataset"""

    id: Optional[int] = Field(default=None, primary_key=True)


class ValidationDeepFashion(DeepFashionBase, table=True):
    """Table with validation data of DeepFashion dataset"""

    id: Optional[int] = Field(default=None, primary_key=True)


class ResNet50v2Model(SQLModel, table=True):
    """Table that contains ResNet50 model training results"""

    id: Optional[int] = Field(default=None, primary_key=True)
    epoch: int
    model: str
    best_model: str
    results_image: str
    accuracy: float
    val_accuracy: float
    loss: float
    val_loss: float


class CrawlData(SQLModel, table=True):
    """Table with crawled data"""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    url: str
    price: Optional[str]
    image_link: str
    category: str = Field(foreign_key="clothescategory.extended_category")
    notedarray: List = Field(default=None, sa_column=Column(ARRAY(INTEGER)))


class KnnModel(SQLModel, table=True):
    """Table that contains Knn model training results"""

    id: Optional[int] = Field(default=None, primary_key=True)
    category: str
    model: str


class Yolov5Model(SQLModel, table=True):
    """Table that contains Yolo model training results"""

    id: Optional[int] = Field(default=None, primary_key=True)
    epoch: int
    train_box_loss: float
    train_obj_loss: float
    train_cls_loss: float
    metrics_precision: float
    metrics_recall: float
    metrics_map_05: float
    metrics_map_05_095: float
    val_box_loss: float
    val_obj_loss: float
    val_cls_loss: float
    x_lr0: float
    x_lr1: float
    x_lr2: float
    model: str


class UserAPIBase(SQLModel):
    image_link: str
    crop_image_link: str
    category: str
    category_id: int
    boundingbox: List = Field(sa_column=Column(ARRAY(INTEGER)))
    notedarray: Optional[List] = Field(default=None, sa_column=Column(ARRAY(INTEGER)))


class UserAPI(UserAPIBase, table=True):
    id: int = Field(default=None, primary_key=True)


class UserAPICreate(UserAPIBase):
    pass

class UserAPIRead(UserAPIBase):
    id: int

class UserAPIUpdate(SQLModel):
    image_link: Optional[str] = None
    crop_image_link: Optional[str] = None
    category: Optional[str] = None
    category_id: Optional[int] = None
    boundingbox: Optional[List] = Field(default=None, sa_column=Column(ARRAY(INTEGER)))
    notedarray: Optional[List] = Field(default=None, sa_column=Column(ARRAY(INTEGER)))
