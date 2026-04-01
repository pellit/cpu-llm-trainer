import os
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "google/gemma-2-2b-it")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
BASE_MODEL_DIR = os.getenv("BASE_MODEL_PATH", "/app/models/gemma-2-2b-it")
EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_PATH", "/app/models/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def ensure_online_download_mode():
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["LOCAL_MODELS_ONLY"] = "0"


def download_repo(repo_id: str, target_dir: str):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"Descargando {repo_id} en {target_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        token=HF_TOKEN,
    )
    print(f"Listo: {repo_id}")


if __name__ == "__main__":
    ensure_online_download_mode()
    download_repo(BASE_MODEL_ID, BASE_MODEL_DIR)
    download_repo(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_DIR)
