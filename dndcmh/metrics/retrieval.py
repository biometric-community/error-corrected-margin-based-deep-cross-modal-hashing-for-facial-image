"""MAP / NDCG for attribute→image Hamming retrieval."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def hamming_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (Q,B) ±1, b: (N,B) ±1 → (Q,N) distances."""
    return 0.5 * (a.shape[1] - a @ b.T)


def average_precision(relevant: np.ndarray) -> float:
    """relevant: binary relevance in ranked order."""
    if relevant.sum() == 0:
        return 0.0
    hits = np.cumsum(relevant)
    ranks = np.arange(1, len(relevant) + 1)
    prec = hits / ranks
    return float((prec * relevant).sum() / relevant.sum())


def mean_average_precision(
    query_codes: np.ndarray,
    gallery_codes: np.ndarray,
    relevance: np.ndarray,
) -> float:
    """
    query_codes: (Q,B), gallery_codes: (N,B), relevance: (Q,N) binary.
    Rank gallery by Hamming distance; compute MAP.
    """
    dist = hamming_dist(query_codes.astype(np.float64), gallery_codes.astype(np.float64))
    aps: List[float] = []
    for i in range(dist.shape[0]):
        order = np.argsort(dist[i])
        rel = relevance[i, order]
        aps.append(average_precision(rel.astype(np.float64)))
    return float(np.mean(aps)) if aps else 0.0


def ndcg_at_k(
    query_codes: np.ndarray,
    gallery_codes: np.ndarray,
    relevance: np.ndarray,
    k: int = 50,
) -> float:
    dist = hamming_dist(query_codes.astype(np.float64), gallery_codes.astype(np.float64))
    scores = []
    for i in range(dist.shape[0]):
        order = np.argsort(dist[i])[:k]
        rel = relevance[i, order].astype(np.float64)
        if rel.sum() == 0:
            scores.append(0.0)
            continue
        gains = (2**rel - 1) / np.log2(np.arange(2, len(rel) + 2))
        dcg = gains.sum()
        ideal = np.sort(relevance[i])[::-1][:k].astype(np.float64)
        idcg = ((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum()
        scores.append(float(dcg / idcg) if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def build_attr_queries(
    attrs: np.ndarray,
    arity: int,
    n_queries: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample attribute conjunction queries of given arity.
    Returns query_attr_vectors (Q,40) and relevance (Q,N) where gallery row i
    is relevant if it has all selected positive attributes.
    """
    n, d = attrs.shape
    queries = []
    rels = []
    tries = 0
    while len(queries) < n_queries and tries < n_queries * 50:
        tries += 1
        idxs = rng.choice(d, size=arity, replace=False)
        # require attributes that appear as positive somewhere
        q = np.zeros(d, dtype=np.float32)
        q[idxs] = 1.0
        mask = np.ones(n, dtype=bool)
        for j in idxs:
            mask &= attrs[:, j] > 0.5
        if mask.sum() < 2:
            continue
        queries.append(q)
        rels.append(mask.astype(np.float32))
    if not queries:
        # fallback: single most frequent attrs
        freq = attrs.sum(axis=0)
        top = np.argsort(-freq)[:arity]
        q = np.zeros(d, dtype=np.float32)
        q[top] = 1.0
        mask = np.ones(n, dtype=bool)
        for j in top:
            mask &= attrs[:, j] > 0.5
        queries = [q]
        rels = [mask.astype(np.float32)]
    return np.stack(queries), np.stack(rels)
