"""
Tests for main module
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel.pool import StaticPool
from sqlmodel import SQLModel

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from main import app, get_session
from database.db_tables import UserAPI


@pytest.fixture(name="session")
async def session_fixture():
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:umimuv27@localhost:5432/clothing_db",
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: AsyncSession):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Clothing Recommendation API!"}


def test_add_image(client: TestClient):
    response = client.post(
        "/images/",
        files={
            "upload_image": (
                "filename",
                open("datasets/validation/image/000019.jpg", "rb"),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Found 1 results in your picture"}


# def test_get_image(session: AsyncSession, client: TestClient):
#     image_1 = UserAPI(
#         image_link="dsfsfdsdfsf.jpg",
#         crop_image_link="sfddsfdsf.jpg",
#         category="shirt",
#         category_id=1,
#         boundingbox=[11, 66, 460, 622],
#     )
#     image_2 = UserAPI(
#         image_link="dfsfsdfsdf.jpg",
#         crop_image_link="gfdgdgdgdg.jpg",
#         category="shirt",
#         category_id=1,
#         boundingbox=[21, 43, 432, 132],
#     )
#     session.add(image_1)
#     session.add(image_2)
#     session.commit()

#     response = client.get(
#         "/images/",
#     )

#     # json={
#     #         "image_link": "image.jpg",
#     #         "crop_image_link": "crop_image.jpg",
#     #         "category": "shirt",
#     #         "category_id": 1,
#     #         "boundingbox": [11, 66, 460, 622],
#     #     },

#     data = response.json()
#     assert response.status_code == 200
#     assert len(data) == 2
#     # assert data[0]["image_link"] == "image.jpg"
#     # assert [
#     #     data["image_link"] == "image.jpg",
#     #     data["crop_image_link"] == "crop_image.jpg",
#     #     data["category"] == "shirt",
#     #     data["category_id"] == 1,
#     #     data["boundingbox"] == [11, 66, 460, 622],
#     #     data["id"] == 1
#     # ]

#     # assert
#     # assert
#     # assert
#     # assert
#     # assert
#     # assert data["notedarray"] is None
