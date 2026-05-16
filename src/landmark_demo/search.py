"""Fusion ranker + name search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import AssetBundle, NameEntry, normalize_text


@dataclass
class FusionWeights:
    w_image: float
    w_text: float
    w_keyword: float

    def validate(self) -> None:
        for v in (self.w_image, self.w_text, self.w_keyword):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"가중치 범위 오류: {v}")
        total = self.w_image + self.w_text + self.w_keyword
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"가중치 합 1.0 위반: {total}")


@dataclass
class ConfidencePolicy:
    reject_threshold: float = 0.25
    weak_reject_threshold: float = 0.35
    weak_margin: float = 0.12
    match_threshold: float = 0.60
    match_floor: float = 0.50
    match_margin: float = 0.20
    text_no_keyword_reject_threshold: float = 0.35


@dataclass
class TopResult:
    rank: int
    landmark_id: str
    fusion_score: float
    image_score: float
    text_score: float
    keyword_score: float
    percentage: int


@dataclass
class SearchOutcome:
    top3: list[TopResult]
    all_scores: dict[str, dict[str, float]]   # landmark_id -> {image, text, keyword, fusion}
    below_threshold: bool
    decision: str = "matched"
    reason_codes: list[str] = None
    top1_score: float = 0.0
    top2_score: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.reason_codes is None:
            self.reason_codes = []


def _cosine_to_matrix(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """vec (D,)와 matrix (N, D) 의 cosine. 모두 L2-normalized 가정."""
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ vec


def _query_tokens(q: str) -> list[str]:
    tokens = [t for t in q.replace(",", " ").split() if t]
    compact_hints = [
        "돌담", "성벽", "성곽", "한양도성", "공원", "야경",
        "미술관", "박물관", "현대미술", "전시",
        "art", "gallery", "museum", "modern", "contemporary",
        "stone", "wall", "fortress", "city", "park", "shrine", "palace",
    ]
    for hint in compact_hints:
        if hint in q and hint not in tokens:
            tokens.append(hint)
    if not tokens and q:
        tokens = [q]
    return tokens


def _keyword_score(query: str, bundle: AssetBundle) -> dict[str, float]:
    """query를 lower-case로 정규화한 뒤 각 클래스의 alias/keyword/name과 부분 일치 횟수."""
    q = normalize_text(query)
    if not q:
        return {lid: 0.0 for lid in bundle.landmark_ids}
    tokens = _query_tokens(q)
    out: dict[str, float] = {}
    for lid in bundle.landmark_ids:
        info = bundle.info_by_id.get(lid)
        if info is None:
            out[lid] = 0.0
            continue
        keys = [normalize_text(info.name_ko), normalize_text(info.name_en)]
        keys += [normalize_text(a) for a in info.aliases]
        keys += [normalize_text(t) for t in info.tags]
        ti = bundle.text_index.get(lid)
        if ti is not None:
            keys += [normalize_text(k) for k in ti.keywords]
            keys += [normalize_text(ti.description_ko), normalize_text(ti.description_en)]
        keys = [k for k in keys if k]
        haystack = " ".join(keys)
        # full query exact substring 점수 + token 부분 일치 점수
        score = 0.0
        if q in haystack:
            score += 1.0
        hits = 0
        for tok in tokens:
            if tok in haystack:
                hits += 1
                continue
            if any(len(key) >= 2 and key in tok for key in keys):
                hits += 1
        score += hits / max(len(tokens), 1)
        out[lid] = float(score)
    # 0~1 정규화
    if out:
        max_v = max(out.values())
        if max_v > 0:
            out = {k: v / max_v for k, v in out.items()}
    return out


def _percentage(x: float) -> int:
    return int(round(max(0.0, min(1.0, x)) * 100))


def _build_outcome(
    image_score: dict[str, float],
    text_score: dict[str, float],
    keyword_score: dict[str, float],
    weights: FusionWeights,
    reject_threshold: float,
    landmark_ids: list[str],
) -> SearchOutcome:
    fusion = {
        lid: weights.w_image * image_score.get(lid, 0.0)
             + weights.w_text * text_score.get(lid, 0.0)
             + weights.w_keyword * keyword_score.get(lid, 0.0)
        for lid in landmark_ids
    }
    ranked = sorted(fusion.items(), key=lambda kv: -kv[1])[:3]
    top3: list[TopResult] = []
    for rank, (lid, score) in enumerate(ranked, start=1):
        top3.append(TopResult(
            rank=rank,
            landmark_id=lid,
            fusion_score=float(score),
            image_score=float(image_score.get(lid, 0.0)),
            text_score=float(text_score.get(lid, 0.0)),
            keyword_score=float(keyword_score.get(lid, 0.0)),
            percentage=_percentage(score),
        ))
    all_scores = {
        lid: {
            "image": float(image_score.get(lid, 0.0)),
            "text": float(text_score.get(lid, 0.0)),
            "keyword": float(keyword_score.get(lid, 0.0)),
            "fusion": float(fusion[lid]),
        }
        for lid in landmark_ids
    }
    below = top3[0].fusion_score < reject_threshold if top3 else True
    return SearchOutcome(top3=top3, all_scores=all_scores, below_threshold=below)


def apply_decision_policy(
    outcome: SearchOutcome,
    kind: str,
    policy: ConfidencePolicy,
    *,
    low_quality: bool = False,
    quality_reason_codes: list[str] | None = None,
) -> SearchOutcome:
    top1 = outcome.top3[0] if outcome.top3 else None
    top2 = outcome.top3[1] if len(outcome.top3) > 1 else None
    top1_score = float(top1.fusion_score) if top1 else 0.0
    top2_score = float(top2.fusion_score) if top2 else 0.0
    margin = top1_score - top2_score
    reason_codes: list[str] = []

    if not top1:
        decision = "out_of_scope"
        reason_codes.append("no_candidate")
    elif low_quality:
        decision = "low_quality"
        reason_codes.extend(quality_reason_codes or ["quality_low"])
    elif top1_score < policy.reject_threshold:
        decision = "out_of_scope"
        reason_codes.append("top1_below_reject")
    elif kind == "text" and top1.keyword_score <= 0.0 and top1_score < policy.text_no_keyword_reject_threshold:
        decision = "out_of_scope"
        reason_codes.extend(["keyword_miss", "top1_low"])
    elif top1_score < policy.weak_reject_threshold and margin < policy.weak_margin:
        decision = "out_of_scope"
        reason_codes.extend(["top1_weak", "margin_low"])
    elif top1_score >= policy.match_threshold:
        decision = "matched"
        reason_codes.append("top1_high")
    elif top1_score >= policy.match_floor and margin >= policy.match_margin:
        decision = "matched"
        reason_codes.extend(["top1_mid", "margin_high"])
    else:
        decision = "ambiguous"
        if top1_score < policy.match_threshold:
            reason_codes.append("top1_below_match")
        if margin < policy.match_margin:
            reason_codes.append("margin_low")
        if kind == "text" and top1.keyword_score <= 0.0:
            reason_codes.append("keyword_miss")

    outcome.decision = decision
    outcome.reason_codes = reason_codes
    outcome.top1_score = top1_score
    outcome.top2_score = top2_score
    outcome.margin = margin
    outcome.below_threshold = decision == "out_of_scope"
    return outcome


def search_by_image(
    image_embedding: np.ndarray,
    bundle: AssetBundle,
    weights: FusionWeights,
    reject_threshold: float,
    policy: ConfidencePolicy | None = None,
) -> SearchOutcome:
    sims = _cosine_to_matrix(image_embedding, bundle.proto_matrix)
    image_score = {lid: float(sims[i]) for i, lid in enumerate(bundle.landmark_ids)}
    text_score = {lid: 0.0 for lid in bundle.landmark_ids}
    keyword_score = {lid: 0.0 for lid in bundle.landmark_ids}
    outcome = _build_outcome(image_score, text_score, keyword_score, weights, reject_threshold, bundle.landmark_ids)
    return apply_decision_policy(outcome, "image", policy or ConfidencePolicy(reject_threshold=reject_threshold))


def search_by_text(
    text_embedding: Optional[np.ndarray],
    raw_query: str,
    bundle: AssetBundle,
    weights: FusionWeights,
    reject_threshold: float,
    policy: ConfidencePolicy | None = None,
) -> SearchOutcome:
    if text_embedding is not None and bundle.text_matrix is not None and bundle.text_matrix.shape[0] == len(bundle.landmark_ids):
        sims = _cosine_to_matrix(text_embedding, bundle.text_matrix)
        text_score = {lid: float(sims[i]) for i, lid in enumerate(bundle.landmark_ids)}
    else:
        text_score = {lid: 0.0 for lid in bundle.landmark_ids}
    keyword_score = _keyword_score(raw_query, bundle)
    image_score = {lid: 0.0 for lid in bundle.landmark_ids}
    outcome = _build_outcome(image_score, text_score, keyword_score, weights, reject_threshold, bundle.landmark_ids)
    return apply_decision_policy(outcome, "text", policy or ConfidencePolicy(reject_threshold=reject_threshold))


# ---- Name search ----

@dataclass
class NameSearchResult:
    matches: list[NameEntry]


def name_search(query: str, entries: list[NameEntry], limit: int = 10) -> NameSearchResult:
    q = normalize_text(query.strip())
    if not q:
        return NameSearchResult(matches=[])
    hits = [e for e in entries if q in e.key]
    # 가까운 매치(짧은 키일수록 정확) 우선
    hits.sort(key=lambda e: (-len(q) / max(len(e.key), 1), e.key))
    # 같은 landmark 중복 제거
    seen = set()
    unique: list[NameEntry] = []
    for h in hits:
        if h.landmark_id in seen:
            continue
        seen.add(h.landmark_id)
        unique.append(h)
        if len(unique) >= limit:
            break
    return NameSearchResult(matches=unique)
