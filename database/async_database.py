import os

from sqlmodel import SQLModel

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


# DATABASE_URL = os.environ.get(
#     "postgresql://postgres:umimuv27@localhost:5432/clothing_db"
# )
DATABASE_URL = "postgresql+asyncpg://postgres:umimuv27@localhost:5432/clothing_db"

engine = create_async_engine(DATABASE_URL, echo=True, future=True)


async def init_db():
    """Set connection with DB and create tables"""

    async with engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncSession:
    """Created a SQLAlchemy session"""

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
