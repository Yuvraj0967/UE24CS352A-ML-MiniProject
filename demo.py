"""Streamlit demo - PvP chat sentiment & toxicity classifier.

Run:  streamlit run demo.py
"""
import json
import os
import pickle
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text  # noqa: E402  (same cleaning as training)

st.set_page_config(page_title="PvP Chat Classifier", page_icon="🎮", layout="centered")

STYLE = {  # class -> (emoji, headline)
    "Derogatory": ("🚫", "Hater - derogatory language"),
    "Negative attitude": ("😒", "Troll - negative attitude"),
    "Neutral / Positive": ("🤝", "Teammate - neutral or positive"),
}
EXAMPLES = [
    "gg ez trash team uninstall",
    "nice save, gg wp",
    "mid is open, push top",
    "ez mid",
    "thanks for the heal",
    "shut up and uninstall",
]


@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    with open(os.path.join(base, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    metrics_path = os.path.join(base, "metrics_baselines.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
    return model, vectorizer, metrics


try:
    model, vectorizer, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first (see README).")
    st.stop()

# ---------- header ----------
st.title("🎮 Trolls, Haters, or Teammates?")
st.caption("Classifying sentiment in PvP game chat · UE24CS352A ML Mini-Project · SRN 514 & 547")

tab_demo, tab_results, tab_about = st.tabs(["Live demo", "Model results", "About"])

# ---------- live demo ----------
with tab_demo:
    st.write("Type a chat message or pick an example, and the model labels it as "
             "**Neutral / Positive**, **Negative attitude** or **Derogatory**.")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}"):
            st.session_state["msg"] = ex
    message = st.text_input("Chat message", key="msg", value=st.session_state.get("msg", EXAMPLES[0]))

    if st.button("Analyze", type="primary"):
        if not message.strip():
            st.warning("Please enter a chat message.")
        else:
            X = vectorizer.transform([clean_text(message)])
            probs = model.predict_proba(X)[0]
            order = probs.argsort()[::-1]
            top = model.classes_[order[0]]
            emoji, headline = STYLE.get(top, ("", top))
            box = {"Derogatory": st.error, "Negative attitude": st.warning}.get(top, st.success)
            box(f"{emoji} **{headline}**  ({top}) - confidence {probs[order[0]]:.0%}")

            st.bar_chart(pd.DataFrame({"probability": probs}, index=model.classes_))
            if probs[order[0]] < 0.5:
                st.caption("Low confidence - the classes overlap for this message.")

    st.caption("Messages are lower-cased and cleaned the same way as the training data.")

# ---------- results ----------
with tab_results:
    if metrics is None:
        st.info("Run `python train_model.py` to generate evaluation metrics.")
    else:
        st.write(f"Held-out test set: **{metrics['n_test']:,}** Dota 2 chat messages "
                 f"(trained on {metrics['n_train']:,}). Deployed model: **{metrics['deployed_model']}**.")
        rows = [{"Model": n, "Accuracy": r["accuracy"], "Macro-F1": r["macro_f1"],
                 "Features": r.get("features", ""), "Latency (ms/msg)": r["latency_ms"]}
                for n, r in metrics["results"].items()]
        st.dataframe(pd.DataFrame(rows).style.format(
            {"Accuracy": "{:.3f}", "Macro-F1": "{:.3f}", "Latency (ms/msg)": "{:.2f}"}),
            hide_index=True)
        fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "figures")
        c1, c2 = st.columns(2)
        for col, name in zip((c1, c2), ("confusion_matrix.png", "model_comparison.png")):
            path = os.path.join(fig_dir, name)
            if os.path.exists(path):
                col.image(path)

# ---------- about ----------
with tab_about:
    st.markdown(
        """
**Problem.** Toxic chat hurts player retention, but banter like "gg" can be friendly *or* sarcastic.
Following the Stanford CS229 paper this project is based on, each message is classified as
**Neutral / Positive**, **Negative attitude** or **Derogatory**.

**Data.** [CONDA](https://github.com/usydnlp/CONDA) - Dota 2 in-game chat annotated by humans.
Explicit toxicity -> *Derogatory*, implicit toxicity -> *Negative attitude*, other chatter ->
*Neutral / Positive* (an approximation of the paper's hand labels, see README).

**Method.** TF-IDF (word 1-2 grams + character 2-5 grams) -> K-NN, Naive Bayes, Logistic Regression,
Linear SVM, Random Forest; plus a feedforward network on averaged GloVe vectors.

**Limitations.** Trained on Dota 2 only; sarcasm and context ("gg" after a loss) are hard;
negative-attitude messages are the hardest class; not a substitute for human moderation.
        """
    )
