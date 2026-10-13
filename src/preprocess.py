"""Text normalisation and the mapping from CONDA annotations to our 3 classes."""
import re

import pandas as pd

from .config import LABELS

NEUTRAL_POS, NEGATIVE, DEROGATORY = LABELS


def clean_text(text: str) -> str:
    """Lower-case, drop CONDA's [SEPA] markers, squeeze repeats and whitespace."""
    s = str(text).replace("[SEPA]", " ").lower()
    s = re.sub(r"https?://\S+", " url ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)  # "sooooo" -> "soo"
    return re.sub(r"\s+", " ", s).strip()


def map_label(intent: str):
    """Map a CONDA intent class to our 3 classes (None = dropped).

    * E (explicit toxicity)  -> Derogatory       (overt insults, slurs, profanity at someone)
    * I (implicit toxicity)  -> Negative attitude (subtler hostility: mockery, blame, passive aggression)
    * O (other chatter)      -> Neutral / Positive
    * A (action requests)    -> dropped (ambiguous: "report mid" can be tactical or hostile)

    This is an approximation of the paper's hand labels, not the same annotation.
    """
    return {"E": DEROGATORY, "I": NEGATIVE, "O": NEUTRAL_POS}.get(intent)


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with columns text, label, intent."""
    rows = []
    for utt, intent in zip(raw["utterance"], raw["intentClass"]):
        label, text = map_label(intent), clean_text(utt)
        if label is not None and text:
            rows.append((text, label, intent))
    df = pd.DataFrame(rows, columns=["text", "label", "intent"])
    return df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
