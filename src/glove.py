"""Download / load 50-d GloVe vectors (word2vec text format, optionally .gz)."""
import gzip
import urllib.request

import numpy as np

from .config import EMBED_DIM, GLOVE_DIR, GLOVE_FILE, GLOVE_URL


def ensure_glove():
    """Return the path of the GloVe file, downloading it (~70 MB) if needed.

    Manual alternative: put glove.6B.50d.txt from
    https://nlp.stanford.edu/projects/glove/ into data/glove/ - it is picked up
    automatically.
    """
    GLOVE_DIR.mkdir(parents=True, exist_ok=True)
    txt = GLOVE_DIR / "glove.6B.50d.txt"
    if txt.exists():
        return txt
    if not GLOVE_FILE.exists():
        print("Downloading GloVe-50d (~70 MB) ...")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE_FILE)
    return GLOVE_FILE


def build_embedding_matrix(vocab: dict, seed: int = 42, oov: str = "random"):
    """Rows for words missing from GloVe get small random vectors (slang, typos),
    or zeros when oov="zeros" (used for averaged-embedding features)."""
    path = ensure_glove()
    opener = gzip.open if str(path).endswith(".gz") else open
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab), EMBED_DIM)).astype("float32")
    if oov == "zeros":
        matrix[:] = 0.0
    if "<pad>" in vocab:
        matrix[vocab["<pad>"]] = 0.0
    found = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != EMBED_DIM + 1:      # skips the optional "400000 50" header
                continue
            idx = vocab.get(parts[0])
            if idx is not None:
                matrix[idx] = np.asarray(parts[1:], dtype="float32")
                found += 1
    return matrix, found


class GloveEmbedder:
    """Message -> average of its words' GloVe vectors (OOV slang contributes nothing)."""

    def __init__(self, texts):
        words = {w for t in texts for w in t.split()}
        self.vecs = {}
        path = ensure_glove()
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                word, _, rest = line.partition(" ")
                if word in words:
                    self.vecs[word] = np.asarray(rest.split(), dtype="float32")
        seen = sum(len(t.split()) for t in texts)
        hit = sum(1 for t in texts for w in t.split() if w in self.vecs)
        self.coverage = hit / max(seen, 1)

    def transform(self, texts):
        out = np.zeros((len(texts), EMBED_DIM), dtype="float32")
        for i, t in enumerate(texts):
            v = [self.vecs[w] for w in t.split() if w in self.vecs]
            if v:
                out[i] = np.mean(v, axis=0)
        return out
