from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    semantic_model_path: str
    qa_dataset_path: str
    vector_database_path: str
    qa_questions_dataset_name: str
    qa_answers_dataset_name: str
    collection_name: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()