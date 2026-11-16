from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.20
    retrieval_top_k: int = 10
    vector_store_path: str = "./data/faiss_index"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
