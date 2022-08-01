"""
Since the dataset contains 13 categories, we need to narrow down to 5 categories: shirt, outwear, short, skirt and dress
"""

import json

TRAIN_NO = 312_184
VALID_NO = 52_489
TRAIN_FOLDER = "datasets/train/annos/"
VALID_FOLDER = "datasets/validation/annos/"
TRAIN_JSON_PATH = "datasets/sum.json"
VALID_JSON_PATH = "datasets/validsum.json"


def change_category(folder_path, json_path, num_of_images):
    """Narrow down the number of categories in dataset"""
    with open(folder_path, "r", encoding="UTF-8") as f:
        temp = json.loads(f.read())
        for i in temp:
            if i == "item":
                continue
            for s in range(num_of_images):
                if (
                    (temp[i][s]["categories"] == 1)
                    | (temp[i][s]["categories"] == 2)
                    | (temp[i][s]["categories"] == 5)
                    | (temp[i][s]["categories"] == 6)
                ):
                    temp[i][s]["categories"] = "shirt"
                elif (temp[i][s]["categories"] == 3) | (temp[i][s]["categories"] == 4):
                    temp[i][s]["categories"] = "outwear"
                elif (temp[i][s]["categories"] == 7) | (temp[i][s]["categories"] == 8):
                    temp[i][s]["categories"] = "short"
                elif temp[i][s]["categories"] == 9:
                    temp[i][s]["categories"] = "skirt"
                else:
                    temp[i][s]["categories"] = "dress"

    with open(json_path, "w", encoding="UTF-8") as f:
        json.dump(temp, f)
        print("Done")


def main():
    change_category(TRAIN_JSON_PATH, TRAIN_JSON_PATH, TRAIN_NO)
    change_category(VALID_JSON_PATH, VALID_JSON_PATH, VALID_NO)


if __name__ == "__main__":
    main()
