from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from typing import Optional

class Settings(BaseSettings):
    # BaseSettings will automatically load environment variables
    # matching the field names (case-insensitive) or from a .env file.

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6380

    # Database Configuration (PostgreSQL)
    db_host: str = "localhost"
    db_port: int = 5434
    db_name: str = "findocai"
    db_user: str = "findocai_user"
    db_password: SecretStr = Field(..., env="DB_PASSWORD") # Marked as SecretStr for sensitive info

    # Gemini API Configuration
    gemini_api_key: Optional[SecretStr] = Field(None, env="GEMINI_API_KEY") # Optional for local development/testing

    # JWT Secret Key for token signing
    jwt_secret_key: SecretStr = Field(default="a_very_secret_key_for_development", env="JWT_SECRET_KEY")

    # Authentication Configuration
    require_auth: bool = Field(default=True, env="REQUIRE_AUTH")  # Whether to require authentication

    # Celery Configuration
    # Constructed from redis_host and redis_port, or can be overridden
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# Create a settings instance that can be imported and used throughout the application
settings = Settings()
