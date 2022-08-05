"""
This module holds the back-end logic on how to deploy a TensorFlow model as a RESTful API service.
"""

import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import uuid

import numpy as np
import torch
import cv2
import os
import pickle
import numpy as np
import json
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v3 import preprocess_input
from keras.models import Model

IMAGEDIR = "fastapi-images/"

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


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


def get_embedding(model, imagename):
    img = image.load_img(imagename, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)


restored_model = tf.keras.models.load_model("model/bestmodel.h5")
secondmodel = Model(
    inputs=restored_model.input, outputs=restored_model.layers[-4].output
)
# print(secondmodel.summary())
restored_model.compile(
    loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
)
secondmodel.compile(
    loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
)
yolomodel = torch.hub.load(
    "ultralytics/yolov5", "custom", path="yolov5/runs/train/Model/weights/best.pt"
)


@app.get("/")
async def root():
    """Return a simple json message"""
    return {"message": "Welcome to the Clothing Recommendation API!"}


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    """Upload image file"""
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!

    # save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


@app.get("/images/")
async def read_upload_file():
    """Display uploaded filey"""
    files = os.listdir(IMAGEDIR)

    path = f"{IMAGEDIR}{files[-1]}"

    return FileResponse(path)


@app.delete("/images/")
async def delete_upload_file():
    """delete uploaded image"""
    pass


@app.get("/images/")
async def clothes_detection():
    """Search for clothes on image"""
    pass


@app.post("/images/")
async def choose_clothes():
    """Choose which clothes user want to find"""
    pass


@app.get("/images/")
async def recommend_clothes():
    """Recomend similar clothes"""
    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)
