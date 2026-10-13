"""TF-IDF feature extraction (word 1-2 grams + character n-grams)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_vectorizer() -> FeatureUnion:
    """Word n-grams capture phrases ("gg ez"); char n-grams survive typos/leetspeak.

    token_pattern keeps single characters because chat slang such as "u" matters.
    """
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
