"""Shared metric helpers and plotting."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from .config import LABELS, SHORT_LABELS


def summarise(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    # "hostile" = Negative attitude or Derogatory, i.e. anything that is not Neutral / Positive
    toxic_true = np.array(y_true) != LABELS[0]
    toxic_pred = np.array(y_pred) != LABELS[0]
    tp = int((toxic_true & toxic_pred).sum())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro"),
        "per_class": {l: {k: rep[l][k] for k in ("precision", "recall", "f1-score", "support")} for l in LABELS},
        "hostile_binary": {
            "precision": tp / max(int(toxic_pred.sum()), 1),
            "recall": tp / max(int(toxic_true.sum()), 1),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def latency_ms(predict_fn, texts, repeats: int = 3) -> float:
    """Average single-message latency (ms), including vectorisation."""
    sample = list(texts[:300])
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for t in sample:
            predict_fn([t])
        best = min(best, (time.perf_counter() - t0) / len(sample))
    return best * 1000


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    short = SHORT_LABELS
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), short); ax.set_yticks(range(3), short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_comparison(results: dict, path):
    names = list(results)
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["macro_f1"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color="#4C78A8")
    b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1", color="#F58518")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                ha="center", fontsize=6.5)
    ax.set_xticks(x, names, fontsize=7, rotation=20, ha="right"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("Held-out test performance", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_class_balance(train_counts: dict, test_counts: dict, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, [train_counts[l] for l in LABELS], w, label="Train", color="#54A24B")
    ax.bar(x + w / 2, [test_counts[l] for l in LABELS], w, label="Test", color="#B279A2")
    ax.set_xticks(x, SHORT_LABELS); ax.set_ylabel("Messages")
    ax.set_title("Class distribution", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
