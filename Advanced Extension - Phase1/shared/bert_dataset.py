# -*- coding: utf-8 -*-
"""
PyTorch dataset for Bio_ClinicalBERT fine-tuning.

Used by run_bert_finetune.py. Tokenises discharge summary text and returns
input_ids, attention_mask, and integer class labels for the Hugging Face Trainer.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ClinicalTextDataset(Dataset):
    """Map discharge notes and ICD chapter labels to BERT token tensors."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
