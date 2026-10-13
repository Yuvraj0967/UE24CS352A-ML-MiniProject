"""Train and compare six classifiers on TF-IDF and GloVe features.

TF-IDF models : K-NN, Multinomial Naive Bayes, Logistic Regression, Linear SVM (probability-calibrated),
                Random Forest
GloVe model   : feedforward neural network (MLP) on averaged 50-d GloVe vectors

Hyper-parameters are tuned on a stratified validation split of the training
data; the held-out test split is only used once, for the final numbers.
The best TF-IDF model that outputs probabilities (by validation macro-F1) is saved
as models/model.pkl together with models/vectorizer.pkl, which is what demo.py loads.
"""
import json
import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .config import FIG_DIR, LABELS, MODELS_DIR, SEED
from .data import load_split
from .evaluate import (latency_ms, plot_class_balance, plot_comparison,
                       plot_confusion, summarise)
from .features import build_vectorizer
from .glove import GloveEmbedder

TFIDF_GRIDS = {
    "K-NN": (lambda p: KNeighborsClassifier(n_neighbors=p, metric="cosine", weights="distance"),
             [15, 31, 51, 101]),
    "Naive Bayes": (lambda p: MultinomialNB(alpha=p), [0.03, 0.1, 0.3, 1.0]),
    "Logistic Regression": (lambda p: LogisticRegression(C=p, max_iter=2000, class_weight="balanced"),
                            [1, 3, 10]),
    "Linear SVM": (lambda p: CalibratedClassifierCV(LinearSVC(C=p, class_weight="balanced"), cv=3), [0.03, 0.1, 0.3, 1.0]),
    "Random Forest": (lambda p: RandomForestClassifier(n_estimators=120, criterion=p, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=SEED), ["gini", "entropy"]),
}
MLP_GRID = [(64,), (128, 64)]


def top_features(vec, model, class_name, k=12):
    """Highest-weight word features for one class from the Logistic Regression weights."""
    names = np.array(vec.get_feature_names_out())
    idx = [i for i, n in enumerate(names) if n.startswith("word__")]
    coef = model.coef_[list(model.classes_).index(class_name)][idx]
    order = np.argsort(coef)[::-1][:k]
    return [(names[idx][i].replace("word__", ""), float(coef[i])) for i in order]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    train, test = load_split("train"), load_split("test")
    tr, va = train_test_split(train, test_size=0.15, stratify=train["label"], random_state=SEED)

    vec = build_vectorizer().fit(train["text"])          # fit on all training text
    Xtr, Xva, Xte = (vec.transform(d["text"]) for d in (tr, va, test))
    Xall = vec.transform(train["text"])
    print(f"Feature dimension: {Xall.shape[1]}", flush=True)

    results, fitted, val_scores = {}, {}, {}
    for name, (make, grid) in TFIDF_GRIDS.items():
        scores = {p: f1_score(va["label"], make(p).fit(Xtr, tr["label"]).predict(Xva), average="macro")
                  for p in grid}
        best_p = max(scores, key=scores.get)
        model = make(best_p).fit(Xall, train["label"])   # refit on full training set
        res = summarise(test["label"], model.predict(Xte))
        res.update(val_macro_f1=scores[best_p], best_param=str(best_p), features="TF-IDF",
                   latency_ms=latency_ms(lambda t, m=model: m.predict(vec.transform(t)), list(test["text"])))
        results[name], fitted[name], val_scores[name] = res, model, scores[best_p]
        print(f"{name:20s} param={best_p!s:8} val-F1={scores[best_p]:.3f} test-acc={res['accuracy']:.3f} "
              f"test-F1={res['macro_f1']:.3f} {res['latency_ms']:.2f} ms/msg", flush=True)

    # ---- feedforward network on averaged GloVe vectors ----
    emb = GloveEmbedder(list(train["text"]) + list(test["text"]))
    print(f"GloVe covers {emb.coverage:.0%} of word tokens (slang is missing)", flush=True)
    Etr, Eva, Ete, Eall = (emb.transform(list(d["text"])) for d in (tr, va, test, train))
    # integer labels: sklearn's MLP early stopping cannot score string labels
    l2i = {l: i for i, l in enumerate(LABELS)}
    ytr, yva, yall = ([l2i[l] for l in d["label"]] for d in (tr, va, train))
    make_mlp = lambda h: MLPClassifier(hidden_layer_sizes=h, early_stopping=True, max_iter=300, random_state=SEED)
    scores = {h: f1_score([l2i[l] for l in va["label"]], make_mlp(h).fit(Etr, ytr).predict(Eva), average="macro")
              for h in MLP_GRID}
    best_h = max(scores, key=scores.get)
    mlp = make_mlp(best_h).fit(Eall, yall)
    to_names = lambda p: [LABELS[i] for i in p]
    res = summarise(test["label"], to_names(mlp.predict(Ete)))
    res.update(val_macro_f1=scores[best_h], best_param=str(best_h), features=f"GloVe-50 ({emb.coverage:.0%} token coverage)",
               latency_ms=latency_ms(lambda t: mlp.predict(emb.transform(t)), list(test["text"])))
    results["MLP + GloVe"], val_scores["MLP + GloVe"] = res, scores[best_h]
    print(f"{'MLP + GloVe':20s} param={best_h!s:8} val-F1={scores[best_h]:.3f} test-acc={res['accuracy']:.3f} "
          f"test-F1={res['macro_f1']:.3f} {res['latency_ms']:.2f} ms/msg", flush=True)

    # ---- pick the deployable model: TF-IDF + predict_proba ----
    deployable = {n: s for n, s in val_scores.items() if n in fitted and hasattr(fitted[n], "predict_proba")}
    best = max(deployable, key=deployable.get)
    print(f"\nDeployed model: {best}")
    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f: pickle.dump(vec, f)
    with open(MODELS_DIR / "model.pkl", "wb") as f: pickle.dump(fitted[best], f)

    pred = fitted[best].predict(Xte)
    wrong = test[pred != test["label"]].assign(pred=pred[pred != test["label"]])
    errors = wrong.sample(min(15, len(wrong)), random_state=SEED)[["text", "label", "pred"]].values.tolist()

    # Leakage check: how much of the test set was already seen verbatim in training?
    seen = test["text"].isin(set(train["text"])).values
    y = test["label"].values
    overlap = {
        "n_seen": int(seen.sum()), "share_seen": float(seen.mean()),
        "acc_seen": float((pred[seen] == y[seen]).mean()), "acc_novel": float((pred[~seen] == y[~seen]).mean()),
        "macro_f1_novel": float(f1_score(y[~seen], pred[~seen], average="macro")),
    }
    print(f"Test messages seen verbatim in training: {overlap['share_seen']:.1%}; "
          f"accuracy on novel messages {overlap['acc_novel']:.3f}")

    lr = fitted["Logistic Regression"]
    meta = {
        "overlap": overlap,
        "deployed_model": best, "n_train": len(train), "n_test": len(test), "n_features": int(Xall.shape[1]),
        "train_counts": train["label"].value_counts().to_dict(),
        "test_counts": test["label"].value_counts().to_dict(),
        "glove_coverage": emb.coverage, "results": results, "errors": errors,
        "top_features": {c: top_features(vec, lr, c) for c in LABELS[1:]},
    }
    json.dump(meta, open(MODELS_DIR / "metrics_baselines.json", "w"), indent=2)

    plot_confusion(results[best]["confusion_matrix"], f"{best} - confusion matrix", FIG_DIR / "confusion_matrix.png")
    plot_class_balance(meta["train_counts"], meta["test_counts"], FIG_DIR / "class_distribution.png")
    plot_comparison(results, FIG_DIR / "model_comparison.png")


if __name__ == "__main__":
    main()
