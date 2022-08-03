"""
Prepare the data for training the yolov5 model
Take images from DeepLearning dataset so that each categories has 14_000 training and 2_153 validation components
"""

import os
import json
import cv2
from collections import namedtuple

TRAIN_NO = 312_184
VALID_NO = 52_489

TRAIN_FOLDER = "datasets/train/"
VALID_FOLDER = "datasets/validation/"

TRAIN_JSON_PATH = "datasets/sum.json"
VALID_JSON_PATH = "datasets/validsum.json"

YOLO_TRAIN_FOLDER = "datasets/yolo_train_dataset/"
os.makedirs(YOLO_TRAIN_FOLDER, exist_ok=True)

YOLO_VALID_FOLDER = "datasets/yolo_valid_dataset/"
os.makedirs(YOLO_VALID_FOLDER, exist_ok=True)

DatasetInfo = namedtuple("DatasetInfo", ["data_no", "dataset_folder", "json_path", "yolo_dataset_folder", "num_of_category_images"])

train_dataset_info = DatasetInfo(TRAIN_NO, TRAIN_FOLDER, TRAIN_JSON_PATH, YOLO_TRAIN_FOLDER, 14_000)
valid_dataset_info = DatasetInfo(VALID_NO, VALID_FOLDER, VALID_JSON_PATH, YOLO_VALID_FOLDER, 2_153)

for dataset_info in (train_dataset_info, valid_dataset_info):
    data = []
    dataset = []
    with open(dataset_info.json_path, "r", encoding='UTF-8') as f:
        temp = json.loads(f.read())
        for i in temp:
            if i == "item":
                continue
            for s in range(dataset_info.data_no):
                image = temp[i][s]["image"]
                category = temp[i][s]["categories"]
                record = {
                    "Image": image,
                    "Category": category,
                }
                data.append(record)

    clothes_categories = ["shirt", "outwear", "short", "skirt", "dress"]
    for i in range(len(clothes_categories)):
        num = 0
        for j in range(dataset_info.data_no):
            if num < dataset_info.num_of_category_images:
                if data[j]["Category"] == clothes_categories[i]:
                    num += 1
                    if data[j]["Image"] not in dataset:
                        dataset.append(data[j]["Image"])

    print(len(dataset))
    
    for k in range(len(dataset)):
        jsondataset = {"item": {}, "info": []}
        filename = dataset_info.dataset_folder + "image/" + str(dataset[k])
        jsonname = dataset_info.dataset_folder + "annos/" + str(dataset[k][0:6]) + ".json"
        img = cv2.imread(filename)
        cv2.imwrite(dataset_info.yolo_dataset_folder + str(dataset[k]), img)
        imgheight, imgwidth, imgchannels = img.shape
        with open(jsonname, "r") as f:
            temp = json.loads(f.read())
            for i in temp:
                if i == "source" or i == "pair_id":
                    continue
                else:
                    box = temp[i]["bounding_box"]
                    bbox = [box[0], box[1], box[2], box[3]]
                    xcenter = ((box[0] + box[2]) / 2) / imgwidth
                    ycenter = ((box[1] + box[3]) / 2) / imgheight
                    width = (box[2] - box[0]) / imgwidth
                    height = (box[3] - box[1]) / imgheight
                    cat = temp[i]["category_id"]
                    if (cat == 1) | (cat == 2) | (cat == 5) | (cat == 6):
                        cat = 0
                    elif (cat == 3) | (cat == 4):
                        cat = 1
                    elif (cat == 7) | (cat == 8):
                        cat = 2
                    elif cat == 9:
                        cat = 3
                    else:
                        cat = 4
                    jsondataset["info"].append(
                        {
                            "categories": cat,
                            "xcenter": xcenter,
                            "ycenter": ycenter,
                            "width": width,
                            "height": height,
                        }
                    )
                    with open(dataset_info.yolo_dataset_folder + str(dataset[k][0:6]) + ".txt", "w+", encoding='UTF-8') as f:
                        for i in range(len(jsondataset["info"])):
                            f.write(
                                str(jsondataset["info"][i]["categories"])
                                + " "
                                + str(jsondataset["info"][i]["xcenter"])
                                + " "
                                + str(jsondataset["info"][i]["ycenter"])
                                + " "
                                + str(jsondataset["info"][i]["width"])
                                + " "
                                + str(jsondataset["info"][i]["height"])
                                + "\n"
                            )
                        f.close()
        jsondataset.clear()

    print("DONE")
