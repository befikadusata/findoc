import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.models import Base, User
from app.config import settings

@pytest.fixture(scope="session")
def engine():
    """Creates a SQLAlchemy engine for the test database."""
    test_db_url = f"postgresql://{settings.db_user}:{settings.db_password.get_secret_value()}@{settings.db_host}:{settings.db_port}/test_findocai"
    return create_engine(test_db_url)

@pytest.fixture(scope="session")
def tables(engine):
    """Creates all tables in the test database."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(engine, tables, monkeypatch) -> Session:
    """
    Provides a transactional scope around a series of tests.
    Creates a new session for each test function and rolls back changes after.
    """
    # Mock the password hashing to avoid bcrypt self-test issues
    def mock_get_password_hash(password):
        return "$2b$12$LrQm32UM3e579bzX2GJ2kOQb28R/bmhVU5lXP7g2eQ6oVeA3N4n8y"  # Fixed hash

    import app.auth
    monkeypatch.setattr(app.auth, "get_password_hash", mock_get_password_hash)

    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(autocommit=False, autoflush=False, bind=connection)()

    # Seed the database with a test user using a pre-computed bcrypt hash for 'testpassword'
    # This avoids the bcrypt self-test issue which occurs when importing bcrypt during pytest collection
    test_user = User(username="testuser", email="test@example.com", hashed_password="$2b$12$LrQm32UM3e579bzX2GJ2kOQb28R/bmhVU5lXP7g2eQ6oVeA3N4n8y")  # Properly formatted bcrypt hash for 'testpassword'
    session.add(test_user)
    session.commit()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
