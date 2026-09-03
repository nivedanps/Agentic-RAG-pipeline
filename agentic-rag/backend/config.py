import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*urllib3.*")
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application settings loaded from .env file and environment variables.
    Pydantic automatically reads these from the environment.
    """

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL where Ollama is running"
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model to use for the AI agent"
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for embeddings"
    )

    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="Directory where ChromaDB stores vectors persistently"
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="Name of the ChromaDB collection"
    )

    data_directory: str = Field(
        default="./data",
        description="Directory where uploaded documents are stored"
    )

    chunk_size: int = Field(
        default=800,
        description="Maximum number of characters per text chunk"
    )
    chunk_overlap: int = Field(
        default=120,
        description="Number of characters overlap between adjacent chunks"
    )

    top_k: int = Field(
        default=4,
        description="Number of top similar chunks to retrieve"
    )
    max_retrieval_attempts: int = Field(
        default=3,
        description="Maximum number of retrieval attempts before giving up"
    )

    max_upload_size_mb: int = Field(
        default=50,
        description="Maximum file upload size in megabytes"
    )
    allowed_extensions: str = Field(
        default="pdf,txt,docx",
        description="Comma-separated list of allowed file extensions"
    )

    max_history_length: int = Field(
        default=10,
        description="Maximum number of conversation turns to keep in memory"
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Return allowed extensions as a list."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def data_dir(self) -> Path:
        """Return data directory as a Path object."""
        path = Path(self.data_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_dir(self) -> Path:
        """Return ChromaDB directory as a Path object."""
        path = Path(self.chroma_persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()

def setup_logging() -> logging.Logger:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger("agentic_rag")

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"agentic_rag.{name}")

app_logger = get_logger("app")
