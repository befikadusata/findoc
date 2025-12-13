import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
import pytest
import uuid
from app.main import app, celery_app
from app.database_factory import database
from app.config import settings
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


# --- Fixtures for Test Configuration and Database ---

@pytest.fixture(scope="module", autouse=True)
def override_test_settings():
    """Overrides application settings for the test module."""
    with patch('app.config.settings', spec=True) as mock_settings:
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6380
        mock_settings.db_host = os.getenv("DB_HOST", "localhost")
        mock_settings.db_port = int(os.getenv("DB_PORT", 5434))
        mock_settings.db_name = "test_findocai_db_" + str(uuid.uuid4()).replace('-', '')[:8] # Unique test database name
        mock_settings.db_user = os.getenv("DB_USER", "findocai_user")
        mock_settings.db_password = MagicMock(get_secret_value=lambda: os.getenv("DB_PASSWORD", "findocai_password"))
        mock_settings.gemini_api_key = MagicMock(get_secret_value=lambda: "dummy_gemini_api_key")
        mock_settings.require_auth = False  # Disable authentication for tests
        mock_settings.redis_url = f"redis://{mock_settings.redis_host}:{mock_settings.redis_port}/0"
        yield mock_settings

@pytest.fixture(autouse=True)
def clean_test_database(override_test_settings):
    """
    Ensures a clean test database for each test function.
    Drops and recreates the test database before and after each test.
    """
    test_db_name = override_test_settings.db_name
    test_db_user = override_test_settings.db_user
    test_db_password = override_test_settings.db_password.get_secret_value()
    test_db_host = override_test_settings.db_host
    test_db_port = override_test_settings.db_port

    def _drop_and_create_db(action="create"):
        conn_to_postgres = None
        try:
            conn_to_postgres = psycopg2.connect(
                host=test_db_host,
                port=test_db_port,
                database='postgres', # Connect to default postgres db to manage other dbs
                user=test_db_user,
                password=test_db_password
            )
            conn_to_postgres.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor_postgres = conn_to_postgres.cursor()

            # Terminate all connections to the test database
            cursor_postgres.execute(f"SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = '{test_db_name}';")
            
            if action == "create":
                # Drop test database if it exists
                cursor_postgres.execute(f"DROP DATABASE IF EXISTS {test_db_name};")
                # Create test database
                cursor_postgres.execute(f"CREATE DATABASE {test_db_name};")
            elif action == "drop":
                cursor_postgres.execute(f"DROP DATABASE IF EXISTS {test_db_name};")

            conn_to_postgres.commit()
            cursor_postgres.close()
        except Exception as e:
            pytest.fail(f"Failed to {action} test database: {e}")
        finally:
            if conn_to_postgres:
                conn_to_postgres.close()
        
        # Now initialize tables in the newly created test database
        # Patch get_db_connection to ensure it uses the test settings from the fixture
        with patch('app.database.settings') as mock_db_settings:
            mock_db_settings.db_host = test_db_host
            mock_db_settings.db_port = test_db_port
            mock_db_settings.db_name = test_db_name
            mock_db_settings.db_user = test_db_user
            mock_db_settings.db_password = MagicMock(get_secret_value=lambda: test_db_password)
            database.init_db()

    _drop_and_create_db(action="create") # Setup before test
    yield # Run the test
    _drop_and_create_db(action="drop") # Teardown after test


# --- Tests ---, get_db_connection
from app.config import settings
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def test_upload_endpoint_integration(override_test_settings, clean_test_database):


    """Integration test for the document upload endpoint."""


    client = TestClient(app)





    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:


        tmp_file.write("This is a test document for upload testing.")


        tmp_file_path = tmp_file.name





    try:


        with patch('app.main.celery_app') as mock_celery:


            mock_task = MagicMock()


            mock_task.id = 'test-task-id'


            mock_celery.send_task.return_value = mock_task





            with open(tmp_file_path, 'rb') as f:


                response = client.post(


                    "/upload",


                    files={"file": ("test_document.txt", f, "text/plain")}


                )





            assert response.status_code == 200


            response_data = response.json()


            assert "doc_id" in response_data


            assert response_data["filename"] == "test_document.txt"


            assert response_data["task_id"] == "test-task-id"


            assert "Document uploaded successfully and processing started" in response_data["message"]





            doc_id = response_data["doc_id"]


            try:


                uuid.UUID(doc_id)


                uuid_valid = True


            except ValueError:


                uuid_valid = False


            assert uuid_valid, "doc_id should be a valid UUID"





            expected_file_path = f"./data/uploads/{doc_id}_test_document.txt"


            assert os.path.exists(expected_file_path)





            if os.path.exists(expected_file_path):


                os.remove(expected_file_path)





            mock_celery.send_task.assert_called_once()


            args, kwargs = mock_celery.send_task.call_args


            assert args[0] == 'process_document'


            task_args = kwargs.get('args', [])


            assert len(task_args) == 2


            assert task_args[0] == doc_id


            assert task_args[1] == expected_file_path





    finally:


        if os.path.exists(tmp_file_path):


            os.unlink(tmp_file_path)








def test_upload_endpoint_database_integration(override_test_settings, clean_test_database):


    """Test that document upload correctly updates the database."""


    client = TestClient(app)





    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:


        tmp_file.write("Test content for database integration.")


        tmp_file_path = tmp_file.name





    try:


        with patch('app.main.celery_app') as mock_celery:


            mock_task = MagicMock()


            mock_task.id = 'test-task-id'


            mock_celery.send_task.return_value = mock_task





            with open(tmp_file_path, 'rb') as f:


                response = client.post(


                    "/upload",


                    files={"file": ("db_test_document.txt", f, "text/plain")}


                )





            assert response.status_code == 200


            response_data = response.json()


            doc_id = response_data["doc_id"]





            document_info = database.get_document_status(doc_id)





            assert document_info is not None, "Document record should exist in the database"


            assert document_info["status"] == "queued", "Document status should be 'queued' after upload"





            status_response = client.get(f"/status/{doc_id}")


            assert status_response.status_code == 200





            status_data = status_response.json()


            assert status_data["doc_id"] == doc_id


            assert status_data["filename"] == "db_test_document.txt"


            assert status_data["status"] == "queued"





    finally:


        if os.path.exists(tmp_file_path):


            os.unlink(tmp_file_path)





        if 'doc_id' in locals():


            expected_file_path = f"./data/uploads/{doc_id}_db_test_document.txt"


            if os.path.exists(expected_file_path):


                os.remove(expected_file_path)





@patch('app.nlp.extraction.generate_summary', return_value={'summary': 'Mock summary.', 'key_points': ['Point 1']})


@patch('app.nlp.extraction.extract_entities', return_value={'invoice_number': 'INV-TEST-001', 'total_amount': 100.0})


@patch('app.rag.pipeline.index_document', return_value=True)


@patch('app.classification.model.classify_document', return_value=[{'doc_type': 'invoice', 'score': 0.99}])


@patch('app.ingestion.ocr.extract_text', return_value="This is a mock invoice text for full processing.")


def test_full_processing_pipeline(mock_extract_text, mock_classify_document, mock_index_document, mock_extract_entities, mock_generate_summary, override_test_settings, clean_test_database):


    """


    Tests the full, chained Celery pipeline from end to end with eager execution.


    """


    celery_app.conf.update(task_always_eager=True) # Force eager execution for the test


    


    client = TestClient(app)





    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:


        tmp_file.write(b"mock content")


        tmp_file_path = tmp_file.name





    try:


        # 1. Upload the document, which will trigger the eager pipeline


        with open(tmp_file_path, "rb") as f:


            response = client.post("/upload", files={"file": ("e2e_test.txt", f, "text/plain")})





        assert response.status_code == 200


        doc_id = response.json()["doc_id"]





        # 2. Verify that the mocks for the processing tasks were called


        mock_extract_text.assert_called_once_with(os.path.join("./data/uploads", f"{doc_id}_e2e_test.txt"))


        mock_classify_document.assert_called_once_with("This is a mock invoice text for full processing.", top_k=1)


        mock_index_document.assert_called_once_with(


            doc_id=doc_id,


            text="This is a mock invoice text for full processing.",


            doc_type="invoice",


            metadata={'source_file': 'e2e_test.txt'}


        )


        mock_extract_entities.assert_called_once_with("This is a mock invoice text for full processing.", "invoice")


        mock_generate_summary.assert_called_once_with("This is a mock invoice text for full processing.", "invoice")


        


        # 3. Check the final status and processed data in the database


        final_status = database.get_document_status(doc_id)


        assert final_status is not None


        assert final_status["status"] == "completed"





        summary_data = database.get_document_summary(doc_id)


        assert summary_data is not None


        assert summary_data["summary"] == "Mock summary."


        assert summary_data["key_points"] == ["Point 1"]





        entities_data = database.get_document_entities(doc_id)


        assert entities_data is not None


        assert entities_data["invoice_number"] == "INV-TEST-001"


        


    finally:


        if os.path.exists(tmp_file_path):


            os.remove(tmp_file_path)


        upload_path = f"./data/uploads/{doc_id}_e2e_test.txt"


        if os.path.exists(upload_path):


            os.remove(upload_path)


        # Reset celery config for other tests


        celery_app.conf.update(task_always_eager=False)








def test_status_endpoint_for_nonexistent_document(override_test_settings, clean_test_database):


    """Test the status endpoint with a non-existent document ID."""


    client = TestClient(app)





    nonexistent_doc_id = "nonexistent-document-id"


    response = client.get(f"/status/{nonexistent_doc_id}")





    assert response.status_code == 200


    response_data = response.json()


    assert "error" in response_data


    assert response_data["doc_id"] == nonexistent_doc_id


    assert "Document not found" in response_data["error"]

