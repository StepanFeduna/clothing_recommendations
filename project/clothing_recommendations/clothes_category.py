"""
Alias extendable clothes category names with generic ones
and write them to DB table
"""

from database.database import create_db_and_tables, fill_table
from database.db_tables import ClothesCategory

def clothes_category():
    """Alias extendable clothes category names with generic ones"""

    cat_dict = {
        "shirt": {
            "short sleeve top",
            "long sleeve top",
            "vest",
            "sling",
            "shirt",
        },
        "dress": {
            "short sleeve dress",
            "long sleeve dress",
            "vest dress",
            "sling dress",
            "dress",
        },
        "outwear": {"short sleeve outwear", "long sleeve outwear", "jacket", "outwear"},
        "short": {"shorts", "trousers", "pants"},
        "skirt": {"skirt"},
    }
    for key, values in cat_dict.items():
        for value in values:
            yield {"generic_category": key, "extended_category": value}


if __name__ == "__main__":
    create_db_and_tables()
    fill_table(ClothesCategory, clothes_category(), truncate=True)
