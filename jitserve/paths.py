"""Locations of resources in an editable JITServe checkout."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QRF_MODEL = REPO_ROOT / "assets/qrf/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl"
QRF_VECTORIZER = REPO_ROOT / "assets/qrf/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl"
DEEPRESEARCH_TRAINING = REPO_ROOT / "traces/deepresearch_training.json"
