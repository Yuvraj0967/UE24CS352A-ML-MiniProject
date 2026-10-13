"""Bi-LSTM classifier initialised with 50-d GloVe embeddings (PyTorch).

Run:  python -m src.train_lstm
Requires torch (see requirements.txt). Writes models/bilstm.pt and
models/metrics_lstm.json; the comparison table in the README/report reads them.
"""
import json
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .config import EMBED_DIM, FIG_DIR, LABELS, MODELS_DIR, SEED
from .data import load_split
from .evaluate import plot_confusion, summarise
from .glove import build_embedding_matrix

MAX_LEN, BATCH, EPOCHS, HIDDEN = 24, 128, 14, 64
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return TOKEN_RE.findall(text)


def build_vocab(texts, min_freq=2):
    counts = Counter(t for s in texts for t in tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in counts.items():
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(texts, vocab):
    out = np.zeros((len(texts), MAX_LEN), dtype="int64")
    for i, s in enumerate(texts):
        ids = [vocab.get(t, 1) for t in tokenize(s)][:MAX_LEN]
        out[i, :len(ids)] = ids
    return torch.from_numpy(out)


class BiLSTM(nn.Module):
    def __init__(self, emb_matrix, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * HIDDEN * 2, n_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.lstm(self.drop(self.emb(x)))
        h_max = h.masked_fill(~mask, -1e9).max(dim=1).values
        h_mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([h_max, h_mean], dim=1)))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 512]).argmax(1) for i in range(0, len(X), 512)]).numpy()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    train, test = load_split("train"), load_split("test")
    tr, va = train_test_split(train, test_size=0.15, stratify=train["label"], random_state=SEED)
    vocab = build_vocab(tr["text"])
    emb, found = build_embedding_matrix(vocab)
    print(f"Vocab {len(vocab)} words, {found} initialised from GloVe ({found / len(vocab):.0%})")

    l2i = {l: i for i, l in enumerate(LABELS)}
    Xtr, Xva, Xte = (encode(list(d["text"]), vocab) for d in (tr, va, test))
    ytr, yva, yte = (torch.tensor([l2i[l] for l in d["label"]]) for d in (tr, va, test))

    model = BiLSTM(emb)
    counts = np.bincount(ytr.numpy(), minlength=3)
    weights = torch.tensor(len(ytr) / (3 * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    best_f1, best_state, patience = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = loss_fn(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        f1 = f1_score(yva.numpy(), predict(model, Xva), average="macro")
        print(f"epoch {epoch:2d}  loss {loss.item():.3f}  val macro-F1 {f1:.3f}  ({time.time() - t0:.0f}s)")
        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(best_state)
    pred = predict(model, Xte)
    res = summarise(list(test["label"]), [LABELS[i] for i in pred])
    t0 = time.perf_counter()
    for i in range(200):
        predict(model, Xte[i:i + 1])
    res.update(val_macro_f1=best_f1, latency_ms=(time.perf_counter() - t0) / 200 * 1000)
    print(f"Bi-LSTM test acc {res['accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}  {res['latency_ms']:.2f} ms/msg")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state": best_state, "vocab": vocab}, MODELS_DIR / "bilstm.pt")
    json.dump(res, open(MODELS_DIR / "metrics_lstm.json", "w"), indent=2)
    plot_confusion(res["confusion_matrix"], "Bi-LSTM + GloVe - confusion matrix", FIG_DIR / "confusion_matrix_lstm.png")


if __name__ == "__main__":
    main()
