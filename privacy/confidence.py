"""
privacy/confidence.py
------------------------
Stage 4: Confidence Fusion.

Independent detectors (layout, OCR, regex, user overrides) each vote on
whether a region is sensitive. Instead of trusting any single detector,
overlapping votes are merged and combined probabilistically:

    fused_confidence = 1 - PRODUCT(1 - p_i * reliability_i)  for each contributing detector i

This means: a region flagged by TWO independent weak signals (e.g. layout's
"this looks like a header block" + OCR's "found a Name: label here") ends up
with higher combined confidence than either alone — which is the whole
point of not relying on OCR as the single source of truth.

Per-detector "reliability" weights reflect how trustworthy each detector's
confidence numbers tend to be in practice (regex patterns are low-false-
-positive; layout is a moderate, structural heuristic; OCR-label matching
is useful but was the exact thing that failed silently before).
"""

from __future__ import annotations

from typing import List

from privacy.types import DetectedRegion, FusedRegion

DETECTOR_RELIABILITY = {
    "layout": 0.80,
    "ocr": 0.75,
    "regex": 0.90,
    "user": 1.0,     # explicit user-specified regions (manual/custom) are ground truth, not a guess
}


def _boxes_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _union_bbox(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def _cluster_overlapping(regions: List[DetectedRegion]) -> List[List[DetectedRegion]]:
    """Simple union-find style clustering of regions whose bboxes overlap."""
    n = len(regions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(regions[i].bbox, regions[j].bbox):
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(regions[i])
    return list(clusters.values())


def fuse_detections(all_regions: List[DetectedRegion], auto_mask_threshold: float = 0.5) -> List[FusedRegion]:
    """
    Merge overlapping evidence from independent detectors into fused verdicts,
    each carrying a combined confidence and the decision on whether it clears
    the auto-mask threshold.
    """
    if not all_regions:
        return []

    clusters = _cluster_overlapping(all_regions)
    fused: List[FusedRegion] = []

    for cluster in clusters:
        # Probabilistic OR across all contributing (detector, confidence) pairs.
        prob_not_sensitive = 1.0
        for r in cluster:
            reliability = DETECTOR_RELIABILITY.get(r.detector, 0.6)
            p = max(0.0, min(1.0, r.confidence * reliability))
            prob_not_sensitive *= (1.0 - p)
        fused_confidence = 1.0 - prob_not_sensitive

        # Primary field type: whichever single contributor has the highest individual confidence.
        best = max(cluster, key=lambda r: r.confidence)
        primary_field = best.field_type

        # Union bbox across the whole cluster so the mask covers all contributing evidence.
        union_box = cluster[0].bbox
        for r in cluster[1:]:
            union_box = _union_bbox(union_box, r.bbox)

        detectors = sorted({r.detector for r in cluster})
        fused.append(
            FusedRegion(
                bbox=union_box,
                field_type=primary_field,
                confidence=round(fused_confidence, 3),
                contributing_detectors=detectors,
                auto_masked=fused_confidence >= auto_mask_threshold,
            )
        )

    return fused
