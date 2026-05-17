from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from landmark_demo.data import load_asset_bundle
from landmark_demo.inference import assess_image_quality
from landmark_demo.search import (
    ConfidencePolicy,
    FusionWeights,
    SearchOutcome,
    TopResult,
    apply_decision_policy,
    search_by_text,
)


ROOT = Path(__file__).resolve().parents[1]


def test_demo_regression_text_keywords_route_to_expected_landmarks() -> None:
    result = load_asset_bundle(ROOT / "assets")
    assert result.success, result.errors
    bundle = result.bundle
    assert bundle is not None

    weights = FusionWeights(w_image=0.0, w_text=0.0, w_keyword=1.0)
    policy = ConfidencePolicy()

    cases = {
        "art gallery": "mmca_seoul",
        "modern art museum": "mmca_seoul",
        "돌담있는곳": "naksan_park",
        "미술관": "mmca_seoul",
        "large palace gate with three arches": "gwanghwamun",
        "blue roof presidential residence": "cheongwadae",
        "Royal Ancestral Shrine": "jongmyo_shrine",
        "city wall night view": "naksan_park",
    }
    for query, expected in cases.items():
        outcome = search_by_text(None, query, bundle, weights, reject_threshold=0.25, policy=policy)
        assert outcome.top3[0].landmark_id == expected
        assert outcome.decision in {"matched", "ambiguous"}


def test_image_score_policy_separates_matched_ambiguous_and_out_of_scope() -> None:
    policy = ConfidencePolicy()

    def outcome(scores: list[float]) -> SearchOutcome:
        top3 = [
            TopResult(rank=i + 1, landmark_id=f"class_{i}", fusion_score=s, image_score=s, text_score=0.0, keyword_score=0.0, percentage=int(s * 100))
            for i, s in enumerate(scores)
        ]
        return SearchOutcome(top3=top3, all_scores={}, below_threshold=False)

    assert apply_decision_policy(outcome([0.789, 0.074, 0.057]), "image", policy).decision == "matched"
    assert apply_decision_policy(outcome([0.400, 0.000, 0.000]), "image", policy).decision == "matched"
    assert apply_decision_policy(outcome([0.351, 0.257, 0.213]), "image", policy).decision == "ambiguous"
    assert apply_decision_policy(outcome([0.400, 0.300, 0.300]), "image", policy).decision == "ambiguous"
    assert apply_decision_policy(outcome([0.326, 0.251, 0.214]), "image", policy).decision == "out_of_scope"


def test_quality_gate_flags_tiny_images() -> None:
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8) + 120)
    report = assess_image_quality(image)
    assert not report.ok
    assert "too_small" in report.reason_codes
