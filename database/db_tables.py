"""
Contain all Database Models
"""

from typing import Optional, List

from sqlmodel import Column, Field, SQLModel, Relationship, ARRAY, INTEGER


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


class YoloTrain(SQLModel, table=True):
    """Table with train data for Yolo model"""

    id: Optional[int] = Field(default=None, primary_key=True)
    category_id = int
    xcenter = float
    ycenter = float
    width = float
    height = float

    train_df_id: int = Field(foreign_key="traindeepfashion.id")
    train_df: Optional[TrainDeepFashion] = Relationship()


class YoloValidation(SQLModel, table=True):
    """Table with train data for Yolo model"""

    id: Optional[int] = Field(default=None, primary_key=True)
    category_id = int
    xcenter = float
    ycenter = float
    width = float
    height = float

    validation_df_id: int = Field(foreign_key="validationdeepfashion.id")
    validation_df: Optional[ValidationDeepFashion] = Relationship()


class Yolov5Model(SQLModel, table=True):
    """Table that contains Yolov5 model training results"""

    id: Optional[int] = Field(default=None, primary_key=True)
    epoch: int
    best_model: str
    train_box_loss: float
    train_obj_loss: float
    train_cls_loss: float
    metrics_precision: float
    metrics_recall: float
    metrics_mAP_05: float
    metrics_mAP_05_095: float
    val_box_loss: float
    val_obj_loss: float
    val_cls_loss: float
    x_lr0: float
    x_lr1: float
    x_lr2: float
    confusion_matrix: str
    f1_curve: str
    labels_correlogram: str
    labels: str
    p_curve: str
    pr_curve: str
    r_curve: str
    results: str
    val_batch0_labels: str
    val_batch0_pred: str
    val_batch1_labels: str
    val_batch1_pred: str
    val_batch2_labels: str
    val_batch2_pred: str


class CrawlData(SQLModel, table=True):
    """Table with crawled data"""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    url: str
    price: Optional[float]
    image: str
    category: str = Field(foreign_key="clothescategory.extended_category")
    notedarray: List = Field(default=None, sa_column=Column(ARRAY(INTEGER)))

class KnnModel(SQLModel, table=True):
    """Table that contains Knn model training results"""
    id: Optional[int] = Field(default=None, primary_key=True)
    category: str = Field(foreign_key="clothescategory.extended_category")
    model: str

class API(SQLModel, table=True):
    """Table with user provided data"""

    id: Optional[int] = Field(default=None, primary_key=True)
    image: str
    cropped_image: str
    category: str = Field(foreign_key="clothescategory.extended_category")
    boundingbox: List = Field(sa_column=Column(ARRAY(INTEGER)))
    notedarray: List = Field(default=None, sa_column=Column(ARRAY(INTEGER)))
    crawl_id: int = Field(foreign_key="crawldata.id")
