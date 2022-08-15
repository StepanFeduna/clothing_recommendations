"""
This module holds the back-end logic on how to deploy a TensorFlow model as a RESTful API service.
"""

import os
import shutil
from typing import List, Union
import uuid
import pickle
import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import numpy as np
import torch
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model

from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.async_database import get_session, init_db
from database.db_tables import (
    ResNet50v2Model,
    Yolov5Model,
    KnnModel,
    CrawlData,
    ClothesCategory,
    UserAPI,
)

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,  # access the API in a different host
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)

IMAGEDIR = r"fastapi_images/"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


def get_embedding(model, image_name):
    """Use the trained model to generate a vector representation of the image"""

    img = image.load_img(image_name, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(x).reshape(-1)


@app.on_event("startup")
async def on_startup():
    """Create DB tables at startup"""
    await init_db()


@app.get("/", tags=["recommend-clothes"])
async def root():
    """Return a simple json message"""
    return {"message": "Welcome to the Clothing Recommendation API!"}


@app.get("/images/", tags=["recommend-clothes"], response_model=List[UserAPI])
async def get_image(session: AsyncSession = Depends(get_session)):
    """Read image file"""

    result = await session.execute(select(UserAPI))
    images = result.scalars().all()
    return [
        UserAPI(
            id=image.id,
            image_link=image.image_link,
            crop_image_link=image.crop_image_link,
            category=image.category,
            category_id=image.category_id,
            boundingbox=image.boundingbox,
        )
        for image in images
    ]


@app.post("/images/", tags=["recommend-clothes"])
async def add_image(
    session: AsyncSession = Depends(get_session),
    upload_image: Union[UploadFile, None] = None,
):
    """Upload image file"""
    if not upload_image:
        return HTTPException(404, detail="No upload file sent")
    if upload_image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(400, detail="Invalid document type")

    if os.path.exists(IMAGEDIR):
        shutil.rmtree(IMAGEDIR)
    os.makedirs(IMAGEDIR, exist_ok=True)
    await session.execute("""TRUNCATE TABLE UserAPI RESTART IDENTITY CASCADE""")
    await session.commit()

    upload_image.name = f"{IMAGEDIR}{uuid.uuid4().hex}.jpg"
    content = await upload_image.read()

    async with aiofiles.open(upload_image.name, "wb") as f:
        await f.write(content)

    statement = select(Yolov5Model.model)
    result = await session.execute(statement)
    model = result.scalars().first()

    yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path=model)

    image_array = cv2.imread(upload_image.name)
    detections = yolo_model(image_array)
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    x = np.array(results)

    if len(x) == 0:
        return HTTPException(404, detail="Clothes not found!")

    for i in range(len(x)):
        category = results[i]["name"]
        category_id = results[i]["class"] + 1
        xmin = int(results[i]["xmin"])
        ymin = int(results[i]["ymin"])
        xmax = int(results[i]["xmax"])
        ymax = int(results[i]["ymax"])
        crop_image = image_array[ymin:ymax, xmin:xmax]

        crop_image_name = IMAGEDIR + f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(crop_image_name, crop_image)

        image = UserAPI(
            image_link=upload_image.name,
            crop_image_link=crop_image_name,
            category=category,
            category_id=category_id,
            boundingbox=[xmin, ymin, xmax, ymax],
        )
        session.add(image)

    await session.commit()
    await session.refresh(image)

    return {"message": "Found " + str(len(x)) + " results in your picture"}


@app.patch("/images/{image_id}/", tags=["recommend-clothes"])
async def generate_notedarray(
    image_id: int = 1, session: AsyncSession = Depends(get_session)
):
    """Choose clothes element from image"""

    result = await session.get(UserAPI, image_id)
    if not result:
        return HTTPException(404, detail="Image not found")
    image_data = result.dict()

    statement = select(ResNet50v2Model.best_model)
    best_model = await session.execute(statement)
    best_model = best_model.scalars().first()

    restored_model = tf.keras.models.load_model(best_model)

    second_model = Model(
        inputs=restored_model.input, outputs=restored_model.layers[-4].output
    )
    second_model.compile(
        loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
    )
    imagearray = get_embedding(second_model, image_data["crop_image_link"])

    image_data["notedarray"] = imagearray
    for key, value in image_data.items():
        setattr(result, key, value)
    session.add(result)
    await session.commit()
    await session.refresh(result)

    return result


@app.get(
    "/images/{image_id}/", tags=["recommend-clothes"], response_model=List[CrawlData]
)
async def get_recommendations(
    image_id: int = 1, session: AsyncSession = Depends(get_session)
):
    """Recommend clothes"""

    result = await session.get(UserAPI, image_id)
    if not result:
        return HTTPException(404, detail="Image not found")
    image_data = result.dict()

    category = image_data["category"]
    statement = select(KnnModel.model).where(KnnModel.category == category)
    knn_model = await session.execute(statement)
    knn_model = knn_model.scalars().first()
    knn_model = pickle.load(open(knn_model, "rb"))

    _, indexes = knn_model.kneighbors([image_data["notedarray"]])
    indexes = np.reshape(indexes, -1)

    crawl_statement = (
        select(CrawlData)
        .join_from(
            CrawlData,
            ClothesCategory,
            CrawlData.category == ClothesCategory.extended_category,
        )
        .where(ClothesCategory.generic_category == category)
    )

    crawl_results = await session.execute(crawl_statement)
    crawl_results = crawl_results.scalars().all()

    return [
        CrawlData(
            id=crawl_result.id,
            name=crawl_result.name,
            category=crawl_result.category,
            url=crawl_result.url,
            price=crawl_result.price,
            image_link=crawl_result.image_link,
        )
        for index, crawl_result in enumerate(crawl_results)
        if index in indexes + 1
    ]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)

# python -m uvicorn main:app --reload
