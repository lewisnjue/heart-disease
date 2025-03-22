from pydantic_settings import BaseSettings 
from functools  import lru_cache 
import os 
from pathlib import Path 

class settings(BaseSettings):
    BASE_DIR:Path = Path(__file__).resolve().parent.parent


@lru_cache
def get_settings():
    return settings()

