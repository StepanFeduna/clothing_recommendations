"""Write Yolo model results to DB table"""

import pandas as pd
from database.database import create_db_and_tables, engine

PATH = r"yolov5/runs/train/Model/"

if __name__ == "__main__":
    create_db_and_tables()
    results = pd.read_csv(PATH + "results.csv")
    results["model"] = PATH + "weights/best.pt"
    results.to_sql("yolov5model", con=engine)
