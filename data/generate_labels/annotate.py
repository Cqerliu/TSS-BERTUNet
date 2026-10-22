# -*- coding: utf-8 -*-
"""
Promoter sequence extraction and labeling — annotation strategies
Input: sequence length L, annotation window start/end indices in the sequence (or tss_in_seq);
output: a label array of length L.

The annotation span is fixed as: 20 bp upstream of TSS + TSS + 20 bp downstream, totaling 41 bp.
"""

import numpy as np

# TSS relative position within the 41 bp window (0-based index 20)
WINDOW_LEN = 41
TSS_OFFSET = 20


def make_labels(
    seq_len: int,
    annotation_start: int,
    annotation_end: int,
    tss_in_seq: int,
    strategy: str,
    annotation_length: int = None,
) -> np.ndarray:
    """
    Build a regression label vector matching the sequence length.

    Parameters
    ----------
    seq_len : int
        Sequence length L.
    annotation_start, annotation_end : int
        Annotation window index range in the sequence [annotation_start, annotation_end);
        length must match annotation_length (defaults to 41 if unspecified).
    tss_in_seq : int
        TSS position in the sequence (1-based); if TSS is the first array element, tss_in_seq=1.
    strategy : str
        "gaussian" | "triangle" | "impulse" | "poisson"
    annotation_length : int, optional
        Annotation window length (bp); defaults to 41. TSS is at the window center.

    Returns
    -------
    np.ndarray
        Shape (seq_len,), dtype float32.
    """
    window_len = annotation_length if annotation_length is not None else WINDOW_LEN
    tss_offset = (window_len - 1) // 2 if annotation_length is not None else TSS_OFFSET
    # Distance from TSS to window edge under the triangle strategy (for normalization)
    half_span = float(tss_offset) if tss_offset > 0 else 1.0

    labels = np.zeros(seq_len, dtype=np.float32)
    if annotation_end <= annotation_start or annotation_end - annotation_start != window_len:
        return labels

    # Convert 1-based TSS position to 0-based array index
    tss_idx = int(tss_in_seq) - 1

    if strategy == "gaussian":
        # Gaussian peaked at TSS, symmetric decay within the window, max value 1
        # Set the window edge (distance x=tss_offset from TSS) near 0.01:
        #   exp(-0.5 * (x/sigma)^2) = boundary
        # => sigma = x / sqrt(-2*ln(boundary))
        boundary = 0.01
        x = float(tss_offset)
        if x <= 0:
            sigma = 1e-6
        else:
            sigma = x / np.sqrt(-2.0 * np.log(boundary))
        for i in range(window_len):
            pos = annotation_start + i
            if 0 <= pos < seq_len:
                x = i - tss_offset
                labels[pos] = float(np.exp(-0.5 * (x / sigma) ** 2))
    elif strategy == "impulse":
        if annotation_start <= tss_idx < annotation_end:
            labels[tss_idx] = 1.0
    elif strategy == "triangle":
        # Within the window, TSS at tss_offset; linear decay to 0 toward both edges
        for i in range(window_len):
            pos = annotation_start + i
            if 0 <= pos < seq_len:
                labels[pos] = 1.0 - abs(i - tss_offset) / half_span
    elif strategy == "poisson":
        # Poisson labeling: use distance d to TSS symmetrically (d=0 at TSS),
        # with regression label as the tail probability: label(d) = P(K >= d), K ~ Poisson(lambda).
        #
        # This guarantees:
        # 1) label=1 at d=0
        # 2) labels decrease monotonically with distance (near 0 far from TSS)
        # 3) lambda is chosen by binary search so label(max_dist) is near a small boundary (default 0.01)
        boundary = 0.01
        max_dist = int(tss_offset)

        if max_dist <= 0:
            if annotation_start <= tss_idx < annotation_end:
                labels[tss_idx] = 1.0
        else:
            def _tail_prob(dist: int, lam: float) -> float:
                # dist is d=distance; compute P(K >= dist)
                if dist <= 0:
                    return 1.0
                # P(K=0)
                pmf = float(np.exp(-lam))
                cum = pmf  # cum = sum_{k=0}^{dist-1} P(K=k)
                # Accumulate k=1..dist-1
                for k in range(1, dist):
                    pmf *= lam / k
                    cum += pmf
                return max(0.0, 1.0 - cum)

            # Find lambda such that tail(max_dist) ~= boundary
            low = 1e-8
            high = 100.0
            for _ in range(35):
                mid = (low + high) / 2.0
                if _tail_prob(max_dist, mid) < boundary:
                    low = mid
                else:
                    high = mid
            lam = high

            # Precompute cum[d] = sum_{k=0}^{d} P(K=k) so tail(d) = 1 - cum[d-1]
            pmf = float(np.exp(-lam))
            cum = [pmf]
            for k in range(1, max_dist + 1):
                pmf *= lam / k
                cum.append(cum[-1] + pmf)

            for i in range(window_len):
                pos = annotation_start + i
                if 0 <= pos < seq_len:
                    d = int(abs(i - tss_offset))
                    if d <= 0:
                        v = 1.0
                    else:
                        v = 1.0 - cum[d - 1]
                    labels[pos] = float(np.clip(v, 0.0, 1.0))
    else:
        raise ValueError(
            f"Unknown annotation strategy: {strategy}; supported: gaussian, triangle, impulse, poisson"
        )
    return labels


def make_labels_all_strategies(
    seq_len: int,
    annotation_start: int,
    annotation_end: int,
    tss_in_seq: int,
    annotation_length: int = None,
) -> dict:
    """Generate labels for all four strategies; return {'gaussian': array, 'triangle': array, 'impulse': array, 'poisson': array}."""
    return {
        "gaussian": make_labels(seq_len, annotation_start, annotation_end, tss_in_seq, "gaussian", annotation_length=annotation_length),
        "triangle": make_labels(seq_len, annotation_start, annotation_end, tss_in_seq, "triangle", annotation_length=annotation_length),
        "impulse": make_labels(seq_len, annotation_start, annotation_end, tss_in_seq, "impulse", annotation_length=annotation_length),
        "poisson": make_labels(seq_len, annotation_start, annotation_end, tss_in_seq, "poisson", annotation_length=annotation_length),
    }
