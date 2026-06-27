# -*- coding: utf-8 -*-
"""
Bio_ClinicalBERT feature extraction for discharge summaries.

Used by run_bert_poc.py (Phase 1). Loads a pretrained clinical transformer from
Hugging Face and converts each note to a fixed-size embedding vector using
mean pooling over token representations (max 512 tokens per note).
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 512


class ClinicalEmbedder:
    """Load Bio_ClinicalBERT and embed batches of clinical text."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return (n_samples, 768) embedding matrix for a list of discharge notes."""
        vectors: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="BERT embed", unit="batch"):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Mask-aware mean pooling: average token vectors, ignore padding
                last_hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
            vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors)
