from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np


def edit_distance(a: Sequence, b: Sequence) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(pred: str, ref: str) -> float:
    if len(ref) == 0:
        return float(len(pred) > 0)
    return edit_distance(list(pred), list(ref)) / len(ref)


def wer(pred: str, ref: str) -> float:
    pred_words = pred.split()
    ref_words = ref.split()
    if len(ref_words) == 0:
        return float(len(pred_words) > 0)
    return edit_distance(pred_words, ref_words) / len(ref_words)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu(pred: str, ref: str, max_n: int = 4, char_level: bool = False) -> float:
    pred_tokens = list(pred) if char_level else pred.split()
    ref_tokens = list(ref) if char_level else ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = _ngrams(pred_tokens, n)
        ref_counts = _ngrams(ref_tokens, n)
        if not pred_counts:
            precisions.append(1e-9)
            continue
        overlap = sum((pred_counts & ref_counts).values())
        precisions.append(max(overlap / sum(pred_counts.values()), 1e-9))
    brevity = min(1.0, np.exp(1.0 - len(ref_tokens) / max(1, len(pred_tokens))))
    return float(brevity * np.exp(np.mean(np.log(precisions))))


def dice_score(pred_mask: np.ndarray, ref_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred_mask.astype(bool)
    ref = ref_mask.astype(bool)
    return float((2 * np.logical_and(pred, ref).sum() + eps) / (pred.sum() + ref.sum() + eps))


def iou_score(pred_mask: np.ndarray, ref_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred_mask.astype(bool)
    ref = ref_mask.astype(bool)
    return float((np.logical_and(pred, ref).sum() + eps) / (np.logical_or(pred, ref).sum() + eps))


def hd95(pred_points: np.ndarray, ref_points: np.ndarray) -> float:
    """95% Hausdorff distance for point clouds or boundary coordinates.

    Args:
        pred_points: [N, D]
        ref_points: [M, D]
    """
    from scipy.spatial.distance import cdist

    if len(pred_points) == 0 or len(ref_points) == 0:
        return float("inf")
    dists = cdist(pred_points, ref_points)
    d_pred = dists.min(axis=1)
    d_ref = dists.min(axis=0)
    return float(max(np.percentile(d_pred, 95), np.percentile(d_ref, 95)))


def mcd(mcep_pred: np.ndarray, mcep_ref: np.ndarray) -> float:
    """Mel-cepstral distortion for generated acoustic/semantic feature sequences."""
    if mcep_pred.shape != mcep_ref.shape:
        raise ValueError("mcep_pred and mcep_ref must have the same shape.")
    diff = mcep_pred - mcep_ref
    const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    return float(const * np.mean(np.sqrt(np.sum(diff**2, axis=-1))))


def accuracy(pred: Iterable[int], ref: Iterable[int]) -> float:
    pred_arr = np.asarray(list(pred))
    ref_arr = np.asarray(list(ref))
    return float((pred_arr == ref_arr).mean())


def macro_f1(pred: Iterable[int], ref: Iterable[int]) -> float:
    pred_arr = np.asarray(list(pred))
    ref_arr = np.asarray(list(ref))
    labels = np.unique(np.concatenate([pred_arr, ref_arr]))
    f1s = []
    for label in labels:
        tp = np.logical_and(pred_arr == label, ref_arr == label).sum()
        fp = np.logical_and(pred_arr == label, ref_arr != label).sum()
        fn = np.logical_and(pred_arr != label, ref_arr == label).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-8, precision + recall))
    return float(np.mean(f1s))
