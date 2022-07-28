"""
This module holds the back-end logic on how to deploy a TensorFlow model as a RESTful API service.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

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

# model_dir = "clothing-recommendation-model.h5"
# model = load_model(model_dir)


@app.get("/")
async def root():
    """Return a simple json message"""
    return {"message": "Welcome to the Clothing Recommendation API!"}


@app.post("/net/image/recommendation/")
async def get_net_image_prediction(image_link: str = ""):
    """Main API functionality"""
    if image_link == "":
        return {"message": "No image link provided"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)
