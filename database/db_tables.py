"""
Contain all Database Models
"""

from typing import Optional, List

from sqlmodel import Column, Field, SQLModel, ARRAY, INTEGER, FLOAT


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
    model: str = None
    best_model: str = None
    results_image: str = None
    accuracy: List = Field(default=None, sa_column=Column(ARRAY(FLOAT)))
    val_accuracy: List = Field(default=None, sa_column=Column(ARRAY(FLOAT)))
    loss: List = Field(default=None, sa_column=Column(ARRAY(FLOAT)))
    val_loss: List = Field(default=None, sa_column=Column(ARRAY(FLOAT)))
    
