import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
    Conv2D,
    MaxPool2D,
)
from tensorflow.keras import callbacks
from tensorflow.keras.applications import ResNet50V2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_io as tfio
from sqlmodel import create_engine

if not tf.config.list_physical_devices("GPU"):
    print("No GPU was detected. CNNs can be very slow without a GPU.")

tf.random.set_seed(42)

TRAIN_SAVE_FOLDER = "datasets/traindata/"
VALID_SAVE_FOLDER = "datasets/validdata/"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

engine = create_engine("postgresql://postgres:umimuv27@localhost:5432/clothing_db")
train_set = tfio.experimental.IODataset.from_sql(
    query="SELECT categories, image FROM TrainDeepFashion;", endpoint=engine
)

validation_set = tfio.experimental.IODataset.from_sql(
    query="SELECT categories, image FROM ValidationDeepFashion;", endpoint=engine
)


print(train_set.element_spec)


def output_model(model):
    """Create functional CNN-model for image classification"""
    outmodel = model.output
    outmodel = Conv2D(128, kernel_size=3, padding="same", activation="relu")(outmodel)
    outmodel = Conv2D(256, kernel_size=3, padding="same", activation="relu")(outmodel)
    outmodel = MaxPool2D()(outmodel)
    outmodel = Flatten(name="flatten")(outmodel)
    outmodel = BatchNormalization()(outmodel)
    # outmodel = Dense(512)(outmodel)
    # outmodel = Activation("relu")(outmodel)
    # outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(1024)(outmodel)
    outmodel = Activation("relu")(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(512)(outmodel)
    outmodel = Activation("relu")(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(5)(outmodel)
    outmodel = Activation("softmax")(outmodel)
    return outmodel


# Use ResNet50V2 model pretrained on ImageNet dataset as layer in our model
model = ResNet50V2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
)

# Fix weights of pretrained layers during training
for layer in model.layers:
    layer.trainable = False

output_model = output_model(model)
resnet_model = Model(inputs=model.input, outputs=output_model)

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

# Load train images dataset
train_set = traingene.flow_from_directory(
    TRAIN_SAVE_FOLDER,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
)

# Load validation images dataset
test_set = validgene.flow_from_directory(
    VALID_SAVE_FOLDER,
    target_size=IMAGE_SIZE,
    batch_size=16,
    class_mode="categorical",
    shuffle=False,
)

# Create callback to save best model version to file
mc = callbacks.ModelCheckpoint(
    "bestmodel.h5", monitor="accuracy", mode="max", save_best_only=True
)
# Create callback to break training if model stoped learning
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
history = resnet_model.fit(
    train_set, epochs=epochs, validation_data=test_set, verbose=1, callbacks=[mc, es]
)

resnet_model.save("model.h5")  # Save model to file after full training

accuracy = resnet_model.history.history["accuracy"]
val_accuracy = resnet_model.history.history["val_accuracy"]
loss = resnet_model.history.history["loss"]
val_loss = resnet_model.history.history["val_loss"]

# Plot model accuracy and losses during training
x = np.linspace(1, epochs, epochs)
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
plt.savefig("resnet_model.png")
