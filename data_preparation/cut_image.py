"""
Deploy images from the DeepFashion dataset for our training
Take 10_765 images, each categories has 2_153(size of the smallest category) images for validation
Take 70_000 images, each categories has 14_000(size of the smallest category) images for training
"""

import os
import json
import cv2


TRAIN_NO = 312_184
VALID_NO = 52_489
TRAIN_FOLDER = "datasets/train/image/"
VALID_FOLDER = "datasets/validation/image/"
TRAIN_JSON_PATH = "datasets/sum.json"
VALID_JSON_PATH = "datasets/validsum.json"
TRAIN_SAVE_FOLDER = "datasets/traindata/"
VALID_SAVE_FOLDER = "datasets/validdata/"


def cut_image(
    images_path,
    json_path,
    save_to_path,
    num_of_images,
    num_of_category_images,
):
    """
    Deploy images from the DeepFashion dataset for our training
    """
    with open(json_path, "r", encoding="UTF-8") as f:
        temp = json.loads(f.read())

        for i in temp:
            if i == "item":
                continue
            for s in range(num_of_images):
                image_save_to_path = save_to_path + str(temp[i][s]["categories"])
                os.makedirs(image_save_to_path, exist_ok=True)

                if (
                    len(os.listdir(image_save_to_path)) < num_of_category_images
                ):  # all categories should have same size
                    image = cv2.imread(images_path + "/" + temp[i][s]["image"])
                    cv2.imwrite(
                        image_save_to_path
                        + "/"
                        + str(temp[i][s]["no"]).zfill(6)
                        + ".jpg",  # generate image file path
                        image[
                            temp[i][s]["boundingbox"][1] : temp[i][s]["boundingbox"][
                                3
                            ],  # cut image by bounding boxes
                            temp[i][s]["boundingbox"][0] : temp[i][s]["boundingbox"][2],
                        ],
                    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    cut_image(TRAIN_FOLDER, TRAIN_JSON_PATH, TRAIN_SAVE_FOLDER, TRAIN_NO, 14_000)
    cut_image(VALID_FOLDER, VALID_JSON_PATH, VALID_SAVE_FOLDER, VALID_NO, 2_153)


if __name__ == "__main__":
    main()
