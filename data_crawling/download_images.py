"""
From the imageurl in crawl json file, download and store them to the database
"""

import os
from unicodedata import category
import urllib.request
import json


with open("datasets/crawlfile.json", "r") as f:
    temp = json.loads(f.read())
    for i in range(len(temp)):
        category = str(temp[i]["Category"])
        url = str(temp[i]["ImageLink"])
        dir_path = "datasets/crawldata/" + category
        os.makedirs(dir_path, exist_ok=True)
        full_path = dir_path + "/" + str(i).zfill(6) + ".jpg"
        urllib.request.urlretrieve(url, full_path)
        # print(temp[i]["ImageLink"])
