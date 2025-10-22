from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "BlazeVeritas AI"

    DATA_DIR: Path = Path("data")
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    XAI_DIR: Path = REPORTS_DIR / "xai"
    MET_DIR: Path = REPORTS_DIR / "metrics"

    MODEL_WEIGHTS: str = "resnet18_fire_nofire_calibrated.ckpt"
    DEVICE: str = "cuda"
    IMG_SIZE: int = 224
    TEMP_INIT: float = 1.0

    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

settings = Settings()
for p in [settings.DATA_DIR, settings.MODELS_DIR, settings.REPORTS_DIR, settings.XAI_DIR, settings.MET_DIR]:
    p.mkdir(parents=True, exist_ok=True)
