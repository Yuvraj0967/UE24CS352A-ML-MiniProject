"""Central configuration: paths, label scheme and random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GLOVE_DIR = DATA_DIR / "glove"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "docs" / "figures"

SEED = 42

# Same three classes as the Stanford CS229 paper this project follows
# (class 0 = neutral/positive, class 1 = negative attitude, class 2 = derogatory).
LABELS = ["Neutral / Positive", "Negative attitude", "Derogatory"]
SHORT_LABELS = ["Neutral/Pos", "Negative", "Derogatory"]

# CONDA (Weld et al., 2021) - Dota 2 in-game chat, human-annotated at utterance level.
CONDA_BASE_URL = "https://raw.githubusercontent.com/usydnlp/CONDA/main/data/"
CONDA_FILES = {"train": "CONDA_train.csv", "test": "CONDA_valid.csv"}

# 50-d GloVe (Wikipedia + Gigaword, 6B tokens) in word2vec text format.
GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
GLOVE_FILE = GLOVE_DIR / "glove-wiki-gigaword-50.gz"
EMBED_DIM = 50
