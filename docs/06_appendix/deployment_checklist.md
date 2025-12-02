# Appendix: Deployment Checklist

This checklist provides a high-level overview of tasks to complete before, during, and after deploying the FinDocAI system to a production environment.

## Pre-Deployment

-   [ ] **Configuration Management:**
    -   [ ] Environment variables (`GEMINI_API_KEY`, `DATABASE_URL`, `REDIS_URL`) are managed securely (e.g., using AWS Secrets Manager or HashiCorp Vault), not hardcoded.
    -   [ ] All necessary cloud resources (S3 buckets, RDS instance, EKS cluster) are defined in Terraform/IaC scripts.
-   [ ] **Database:**
    -   [ ] Database schema and migrations are finalized and tested.
    -   [ ] A backup and recovery plan is in place for the metadata store.
-   [ ] **Models and Artifacts:**
    -   [ ] Finalized ML models (classifier, etc.) are stored in an artifact repository or S3.
    -   [ ] Embedding models are tested and ready to be deployed to a SageMaker endpoint.
-   [ ] **Security:**
    -   [ ] IAM roles and policies are configured with the principle of least privilege.
    -   [ ] Security groups and network ACLs are configured to restrict traffic.
    -   [ ] SSL/TLS certificates are generated and ready to be installed on the load balancer.

## Deployment

-   [ ] **Infrastructure Provisioning:**
    -   [ ] `terraform apply` successfully provisions all cloud resources without errors.
-   [ ] **Application Deployment:**
    -   [ ] Container images for the FastAPI application and Celery workers are built and pushed to a container registry (e.g., ECR).
    -   [ ] Kubernetes deployments/services are applied to the EKS cluster.
-   [ ] **Health Checks:**
    -   [ ] Redis connection from Celery workers and the API is successful.
    -   [ ] Database connection is successful.
    -   [ ] API health check endpoint (`/health`) is passing and reachable through the load balancer.
    -   [ ] Celery workers are successfully consuming tasks from the queue.

## Post-Deployment

-   [ ] **Monitoring and Logging:**
    -   [ ] Prometheus is successfully scraping the `/metrics` endpoint.
    -   [ ] Grafana dashboards are imported and correctly displaying data.
    -   [ ] Structured logs are flowing into CloudWatch Logs from all services.
    -   [ ] Key monitoring alerts (e.g., for high latency, error rate, queue depth) are configured and active.
-   [ ] **Testing:**
    -   [ ] A sample document is successfully processed end-to-end in the production environment.
    -   [ ] All API endpoints are tested and returning expected responses.
-   [ ] **Backup and HA:**
    -   [ ] Automated database backup schedule is confirmed.
    -   [ ] High-availability features (e.g., multi-AZ deployment) are verified.
