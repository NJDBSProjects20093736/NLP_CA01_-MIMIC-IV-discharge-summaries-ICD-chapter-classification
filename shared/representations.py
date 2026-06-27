# -*- coding: utf-8 -*-
"""
Text representation builders

"""
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from text_preprocessing import preprocess_text

# Shared vectoriser settings (match AI Studio / module benchmark approach)
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 10_000
SVD_COMPONENTS = 200
RANDOM_STATE = 42


def _vectorizer_kwargs():
    """Pass preprocessed string → token list to sklearn vectorisers."""
    return dict(
        preprocessor=preprocess_text,
        tokenizer=str.split,
        lowercase=False,  # already lowercased in preprocess_text
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
    )


def make_count_vectorizer() -> CountVectorizer:
    """Representation 1: word/phrase counts (sparse BoW matrix)."""
    return CountVectorizer(**_vectorizer_kwargs())


def make_tfidf_vectorizer() -> TfidfVectorizer:
    """Representation 2: TF-IDF weights (sparse; down-weights common terms)."""
    return TfidfVectorizer(**_vectorizer_kwargs())


def make_tfidf_svd_pipeline() -> Pipeline:
    """Representation 3: TF-IDF then TruncatedSVD (dense, reduced dimensions)."""
    return Pipeline(
        [
            ("tfidf", make_tfidf_vectorizer()),
            (
                "svd",
                TruncatedSVD(
                    n_components=SVD_COMPONENTS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


REPRESENTATIONS = {
    "Count + n-grams (BoW)": make_count_vectorizer,
    "TF-IDF + n-grams": make_tfidf_vectorizer,
    "TF-IDF + TruncatedSVD": make_tfidf_svd_pipeline,
}
