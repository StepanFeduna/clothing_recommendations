"""Write Yolo model results to DB table"""

import pandas as pd
from database.database import create_db_and_tables, fill_table
from database.db_tables import Yolov5Model

PATH = r"yolov5/runs/train/Model/"


def data_dict():
    """Generate dict of Yolo model results"""

    results = pd.read_csv(PATH + "results.csv",header=0)
    results["model"] = PATH + "weights/best.pt"

    for _, row in results.iterrows():
        yield {
            "epoch": row.iloc[0],
            "train_box_loss": row.iloc[1],
            "train_obj_loss": row.iloc[2],
            "train_cls_loss": row.iloc[3],
            "metrics_precision": row.iloc[4],
            "metrics_recall": row.iloc[5],
            "metrics_map_05": row.iloc[6],
            "metrics_map_05_095": row.iloc[7],
            "val_box_loss": row.iloc[8],
            "val_obj_loss": row.iloc[9],
            "val_cls_loss": row.iloc[10],
            "x_lr0": row.iloc[11],
            "x_lr1": row.iloc[12],
            "x_lr2": row.iloc[13],
            "model": row.iloc[14],
        }


if __name__ == "__main__":
    create_db_and_tables()
    fill_table(Yolov5Model, data_dict(), truncate=True)
    data_dict()
