# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from torch.utils.data import Dataset

from data import (
    QualcommInteractiveCookingDatasetNeurIPS2025,
    EgoMCBench
)

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_bert_scorer = BERTScorer(
    model_type="roberta-large", lang="en", device="cuda", rescale_with_baseline=True
)

mean = lambda lst: sum(lst) / (len(lst) + 1e-12)


def get_bert_score(pred_text: str, gt_text: str) -> float:
    """
    Calculate the BERT score between predicted text and ground truth text.

    Args:
        pred_text: The predicted text to evaluate
        gt_text: The ground truth text to compare against

    Returns:
        The BERT F1 score as a float
    """
    _, _, _bert_score = _bert_scorer.score([pred_text], [gt_text], verbose=False)
    _bert_score = float(_bert_score[0])
    return _bert_score


def get_rouge_score(pred_text: str, gt_text: str) -> float:
    """
    Calculate the ROUGE-L score between predicted text and ground truth text.

    Args:
        pred_text: The predicted text to evaluate
        gt_text: The ground truth text to compare against

    Returns:
        The ROUGE-L F-measure score as a float
    """
    _rouge_score = _rouge_scorer.score(pred_text, gt_text)
    _rouge_score = _rouge_score["rougeL"].fmeasure
    return _rouge_score


def load_dataset(dataset_version: str, plan_set: str, split: str) -> Dataset:
    """
    Load the Qualcomm Interactive Cooking Dataset.

    Args:
        dataset_root: Root directory containing the dataset
        plan_set: The plan set to use
        split: Dataset split to load (e.g., 'train', 'validation', 'test')

    Returns:
        The loaded dataset as a torch.utils.data.Dataset object
    """
    if dataset_version == "neurips2025":
        dataset = QualcommInteractiveCookingDatasetNeurIPS2025(plan_set=plan_set, split=split)
    elif dataset_version == "ego_mc_bench":
        dataset = EgoMCBench(split=split)
    else:
        raise ValueError(f"Invalid dataset_version: {dataset_version}. Expected 'neurips2025', 'ego_mc_bench'.")
    return dataset
