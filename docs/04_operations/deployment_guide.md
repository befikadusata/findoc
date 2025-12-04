# Deployment Guide

This guide covers setting up the FinDocAI environment for local development and outlines the strategy for a production deployment on AWS.

## Local Development (Ubuntu)

This setup is intended for development and testing on a local machine.

```bash
# 1. Install system dependencies
# Ensure you have Python 3.10+, pip, and venv installed.
# Tesseract is required for OCR, Redis for the message queue, and PostgreSQL for metadata storage.
sudo apt update
sudo apt install -y tesseract-ocr redis-server postgresql postgresql-contrib

# 2. Clone the repository and set up the environment
git clone https://github.com/yourname/findocai.git
cd findocai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Set up PostgreSQL database
# Create a database user and database for the application
sudo -u postgres createuser --pwprompt findocai_user
sudo -u postgres createdb findocai -O findocai_user

# 4. Set environment variables
# Create a .env file or export these variables in your shell.
export GEMINI_API_KEY="your_google_ai_api_key"
export DB_HOST="localhost"
export DB_PORT="5434"  # If using the docker-compose setup
export DB_NAME="findocai"
export DB_USER="findocai_user"
export DB_PASSWORD="your_db_password"
export REDIS_HOST="localhost"
export REDIS_PORT="6380"  # Using port 6380 as per docker-compose setup

# 5. Initialize database and download ML models
python scripts/init_db.py

# 6. Start services (in separate terminal windows)
# Start the PostgreSQL and Redis services using Docker Compose
docker-compose up -d postgres redis

# Start a Celery worker to process tasks
celery -A app.worker worker --loglevel=info --concurrency=2

# Start the FastAPI web server
uvicorn app.main:app --reload --port 8000

# 7. Optional: Start the monitoring stack
# This requires Docker and Docker Compose.
docker-compose -f docker-compose.monitoring.yml up -d
```

Your local instance of FinDocAI should now be running at `http://localhost:8000`.

## Production Deployment (AWS)

This section describes a target architecture for deploying FinDocAI in a scalable, secure, and resilient manner on AWS, aligned with Kifiya's context.

```bash
# The production deployment should be managed using Infrastructure as Code.
# This example assumes Terraform is used.

cd terraform/
terraform init

# Review and apply the production configuration
terraform plan -var-file=prod.tfvars
terraform apply
```

### Production Architecture Components:
- **Compute:** An **EKS cluster** (e.g., 3 `t3.xlarge` nodes) to run the FastAPI server and Celery worker pods.
- **Database:** **Aurora PostgreSQL** for the metadata store, offering high availability and scalability.
- **Message Queue:** **ElastiCache for Redis** to manage the Celery task queue.
- **Storage:** **S3 buckets** for storing raw documents and caching embeddings.
- **AI/ML Services:**
    - **AWS Textract** for production-grade OCR.
    - **SageMaker Endpoints** for hosting the fine-tuned classifier and embedding models.
    - **Amazon Bedrock** (e.g., using Claude) or a SageMaker-hosted LLM for entity extraction and RAG.
- **Networking:** An **Application Load Balancer (ALB)** and **API Gateway** to manage ingress traffic.
- **Monitoring:** **CloudWatch** for collecting logs, metrics, and setting up alarms, with dashboards for visualization.

### Scaling Strategy:
- **Horizontal Scaling:** Celery workers can be configured with a Kubernetes Horizontal Pod Autoscaler (HPA) to automatically scale from 2 to 20 pods based on SQS queue depth.
- **Vertical Scaling:** For self-hosted LLMs, use GPU-accelerated instances (e.g., `g4dn.xlarge`) to ensure low-latency inference.
- **Database Scaling:** Aurora supports read replicas to handle high query loads from reporting or status-checking endpoints.
