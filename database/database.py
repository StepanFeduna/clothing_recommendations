"""
Contain code creating the engine and the function to create all the tables
"""

import re
from sqlmodel import SQLModel, create_engine, Session

engine = create_engine(
    "postgresql://postgres:umimuv27@localhost:5432/clothing_db", echo=True
)

engine_read = create_engine(
    "postgresql://postgres:umimuv27@localhost:5432/clothing_db",
    echo=False,
    future=False,
)  # pandas.read_sql_query compatibility issue


def create_db_and_tables():
    """Set connection with DB and create tables"""

    # SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def fill_table(table_name, data_dict, *, truncate=False, update=False):
    """Fill DB table with data"""

    with Session(engine) as session:
        if truncate:
            table_name_str = str(table_name).rsplit(".", maxsplit=1)[-1]
            table_name_str = re.sub(r"[^\w]", " ", table_name_str)
            query = f"TRUNCATE TABLE {table_name_str} RESTART IDENTITY CASCADE"
            session.exec(query)

        if update:
            session.bulk_update_mappings(
                table_name,
                data_dict,
            )
        else:
            session.bulk_insert_mappings(
                table_name,
                data_dict,
            )

        session.commit()
