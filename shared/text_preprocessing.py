# -*- coding: utf-8 -*-
"""
CA01 Task 2 — Section 3: Clinical text preprocessing (Python / sklearn)
=======================================================================
Used inside sklearn Pipelines in Sections 4–6. AI Studio uses raw text
and performs its own feature extraction inside Auto Model (Section 4.1).
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources on first run (needs internet once)
for _resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        if _resource.startswith("punkt"):
            nltk.data.find(f"tokenizers/{_resource}")
        else:
            nltk.data.find(f"corpora/{_resource}")
    except LookupError:
        try:
            nltk.download(_resource, quiet=True)
        except Exception:
            pass

_LEMMATIZER = WordNetLemmatizer()
_STOP = set(stopwords.words("english"))
# Keep negation / direction words that matter clinically
_KEEP = {"no", "not", "nor", "against", "over", "under", "down", "up", "out"}
_STOP -= _KEEP


def normalize_whitespace(text: str) -> str:
    """Collapse line breaks and repeated spaces to a single space."""
    return re.sub(r"\s+", " ", str(text)).strip()


def preprocess_text(
    text: str,
    *,
    lemmatize: bool = True,
    remove_stopwords: bool = True,
) -> str:
    """
    Full preprocessing pipeline → returns a single space-separated string.

    Parameters
    ----------
    text : str
        Raw discharge summary (may contain line breaks and punctuation).
    lemmatize : bool
        If True, apply WordNet lemmatisation to each token.
    remove_stopwords : bool
        If True, drop high-frequency English stop words.
    """
    text = normalize_whitespace(text).lower()
    text = re.sub(r"[^\w\s\-]", " ", text)  # remove punctuation
    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP and len(t) > 1]

    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_tokens(text: str, **kwargs) -> list[str]:
    """Same as preprocess_text but returns a list of tokens."""
    return preprocess_text(text, **kwargs).split()
