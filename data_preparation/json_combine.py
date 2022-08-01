"""
Use to combine all the json in the DeepLearning Dataset to one main json file
Take 32_153 images from validation file and 191_961 images from train file
"""


import json

TRAIN_FOLDER = "datasets/train/annos/"
VALID_FOLDER = "datasets/validation/annos/"
TRAIN_JSON_PATH = "datasets/sum.json"
VALID_JSON_PATH = "datasets/validsum.json"


train_dataset = {"item": {}, "info": []}
valid_dataset = {"item": {}, "info": []}


def combine_json(folder_path, num_of_images, data_dict):
    """Extract data from json files"""
    j = 1
    for num in range(num_of_images):
        name = folder_path + str(num).zfill(6) + ".json"
        image_name = str(num).zfill(6) + ".jpg"
        if num > 0:
            with open(name, "r", encoding="UTF-8") as f:
                temp = json.loads(f.read())
                # print(temp)
                for i in temp:
                    if i in ("source", "pair_id"):
                        continue
                    box = temp[i]["bounding_box"]
                    bbox = [box[0], box[1], box[2], box[3]]
                    cat = temp[i]["category_id"]
                    data_dict["info"].append(
                        {
                            "no": j,
                            "categories": cat,
                            "boundingbox": bbox,
                            "image": image_name,
                        }
                    )
                    j += 1
    print(len(data_dict["info"]))


def export_json(json_path, data_dict):
    """Export data to json file"""
    with open(json_path, "w", encoding="UTF-8") as f:
        json.dump(data_dict, f)
        print("Data extracted")


def main():
    combine_json(TRAIN_FOLDER, 191_961, train_dataset)
    combine_json(VALID_FOLDER, 32_153, valid_dataset)
    export_json(TRAIN_JSON_PATH, train_dataset)
    export_json(VALID_JSON_PATH, valid_dataset)


if __name__ == "__main__":
    main()
