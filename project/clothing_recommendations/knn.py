"""
Run the similarity calculation algorithm (K-NN) on numarray of crawled images 
"""

import pickle
import tensorflow as tf
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from keras.models import Model

from sqlmodel import Session, select
from database.database import create_db_and_tables, fill_table, engine
from database.db_tables import ResNet50v2Model, CrawlData, KnnModel, ClothesCategory


def read_sql():
    """Returns a DataFrame corresponding to the result set of the query string."""

    with Session(engine) as session:
        statement = select(
            CrawlData.notedarray, ClothesCategory.generic_category
        ).join_from(
            CrawlData,
            ClothesCategory,
            CrawlData.category == ClothesCategory.extended_category,
        )
        data_table = session.exec(statement).all()

    crawl_data = pd.DataFrame.from_records(
        data_table, columns=["notedarray", "category"]
    )

    print(crawl_data.head(5))

    return crawl_data


def gen_knn_model():
    """Generate results of KNN model calculations"""

    crawl_data = read_sql()
    categories_set = crawl_data["category"].unique()
    for category in categories_set:
        df_select = crawl_data.loc[crawl_data["category"] == category]
        # print(df_select.head(1))
        map_embeddings = df_select["notedarray"]
        # print(map_embeddings.head(1))
        df_embs = map_embeddings.apply(pd.Series)
        # print(df_embs.shape)
        neighbors = NearestNeighbors(
            n_neighbors=len(categories_set), algorithm="brute", metric="euclidean"
        )
        neighbors.fit(df_embs)
        knn_path = f"knn/{category}_knn.pkl"
        with open(knn_path, "wb") as knn_pickle:
            pickle.dump(neighbors, knn_pickle)
        yield {"category": category, "model": knn_path}


if __name__ == "__main__":
    create_db_and_tables()
    fill_table(KnnModel, gen_knn_model(), truncate=True)
