"""
Tests for main module
"""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client  # testing happens here


def test_root(test_app):
    response = test_app.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Clothing Recommendation API!"}


def test_add_image(test_app):
    response = test_app.post(
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
