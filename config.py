import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    COLLECTION_NAME: str = "audio_assistant"
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY")
    ASSEMBLY_API_KEY: str = os.getenv("ASSEMBLY_API_KEY")
    VECTOR_DIM: int = 1024
    DUCKDUCKGO_NUM_RESULTS: int = 5
    JINA_API_KEY: str = os.getenv("JINA_API_KEY")
