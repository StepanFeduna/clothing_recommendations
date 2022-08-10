"""
Train ResNet50 based CNN model for image classification
"""

import tensorflow as tf
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
    Conv2D,
    MaxPool2D,
)
from keras import callbacks
from keras.applications.resnet_v2 import ResNet50V2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from database.database import create_db_and_tables, engine_read, fill_table
from database.db_tables import ResNet50v2Model

if not tf.config.list_physical_devices("GPU"):
    print("No GPU was detected. CNNs can be very slow without a GPU.")

tf.random.set_seed(42)

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


def sql_reader(engine, query):
    """Read SQL query into a DataFrame."""

    return pd.read_sql_query(query, con=engine)


def read_sql():
    """Returns a DataFrame corresponding to the result set of the query string."""

    train_set = sql_reader(
        engine_read,
        "SELECT cropped_image AS image, generic_category AS category, boundingbox \
        FROM TrainDeepFashion JOIN ClothesCategory ON categories = extended_category",
    )
    validation_set = sql_reader(
        engine_read,
        "SELECT cropped_image AS image, generic_category AS category, boundingbox \
        FROM ValidationDeepFashion JOIN ClothesCategory ON categories = extended_category",
    )

    # Reduce dataset size for testing
    train_set = train_set.groupby("category").head(14000)
    validation_set = validation_set.groupby("category").head(2150)

    print(train_set.head(5))
    print(validation_set.head(5))

    return train_set, validation_set


def create_output_model(model):
    """Create functional CNN-model for image classification"""

    outmodel = model.output
    outmodel = Conv2D(128, kernel_size=3, padding="same", activation="relu")(outmodel)
    outmodel = Conv2D(256, kernel_size=3, padding="same", activation="relu")(outmodel)
    outmodel = MaxPool2D()(outmodel)
    outmodel = Flatten(name="flatten")(outmodel)
    outmodel = BatchNormalization()(outmodel)
    outmodel = Dense(1024)(outmodel)
    outmodel = Activation("relu")(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(512)(outmodel)
    outmodel = Activation("relu")(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(5)(outmodel)
    outmodel = Activation("softmax")(outmodel)
    return outmodel


def plot_image(*args):
    """Plot model training results."""

    epochs, accuracy, val_accuracy, loss, val_loss = args
    x = np.linspace(1, epochs, epochs)
    # Plot model accuracy and losses during training
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x, accuracy, label="Training Accuracy")
    plt.plot(x, val_accuracy, label="Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x, loss, label="Training Loss")
    plt.plot(x, val_loss, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()

    plt.show()
    plt.savefig("resnet50_model/resnet_model.png")


def prepare_tf_dataset():
    """Prepare detasets for model training"""

    # Augment images to increase training performance
    traingene = ImageDataGenerator(
        rotation_range=20,
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
    )
    validgene = ImageDataGenerator(rescale=1.0 / 255)

    train_set, validation_set = read_sql()

    # Load train images dataset
    train_set = traingene.flow_from_dataframe(
        train_set,
        x_col="image",
        y_col="category",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=123,
    )

    # Load validation images dataset
    validation_set = validgene.flow_from_dataframe(
        validation_set,
        x_col="image",
        y_col="category",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=32,
        shuffle=False,
        seed=123,
    )

    return train_set, validation_set


def model_training():
    """Train model"""

    # Use ResNet50V2 model pretrained on ImageNet dataset as layer in our model
    model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    )

    # Fix weights of pretrained layers during training
    for layer in model.layers:
        layer.trainable = False

    output_model = create_output_model(model)
    resnet_model = Model(inputs=model.input, outputs=output_model)

    # Create callback to save best model version to file
    mc = callbacks.ModelCheckpoint(
        "resnet50_model/bestmodel.h5",
        monitor="accuracy",
        mode="max",
        save_best_only=True,
    )
    # Create callback to break training if the model stoped learning
    es = callbacks.EarlyStopping(
        monitor="accuracy", verbose=1, mode="max", patience=3, restore_best_weights=True
    )

    # Compile model
    resnet_model.compile(
        loss="categorical_crossentropy",
        optimizer="nadam",
        metrics=["accuracy"],
    )

    epochs = 10

    train_set, validation_set = prepare_tf_dataset()

    history = resnet_model.fit(
        train_set,
        epochs=epochs,
        validation_data=validation_set,
        callbacks=[mc, es],
    )

    accuracy = resnet_model.history.history["accuracy"]
    val_accuracy = resnet_model.history.history["val_accuracy"]
    loss = resnet_model.history.history["loss"]
    val_loss = resnet_model.history.history["val_loss"]

    resnet_model.save(
        "resnet50_model/model.h5"
    )  # Save model to file after full training

    return epochs, accuracy, val_accuracy, loss, val_loss


def data_dict_gen():
    """Generate training results dictionary"""

    epochs, accuracy, val_accuracy, loss, val_loss = model_training()
    plot_image(epochs, accuracy, val_accuracy, loss, val_loss)
    model = r"resnet50_model/model.h5"
    best_model = r"resnet50_model/bestmodel.h5"
    results_image = r"resnet50_model/resnet_model.png"

    for epoch in range(epochs):
        yield {
            "epoch": epoch + 1,
            "model": model,
            "best_model": best_model,
            "results_image": results_image,
            "accuracy": accuracy[epoch],
            "val_accuracy": val_accuracy[epoch],
            "loss": loss[epoch],
            "val_loss": val_loss[epoch],
        }


if __name__ == "__main__":
    create_db_and_tables()
    fill_table(ResNet50v2Model, data_dict_gen(), truncate=True)
