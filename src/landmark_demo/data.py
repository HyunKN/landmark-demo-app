"""Asset 로더와 데이터 클래스."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


LANDMARK_CATALOG = [
    "bohyunsanshingak",
    "changgyeonggung",
    "cheonggyecheon",
    "cheongwadae",
    "deoksugung",
    "gwanghwamun",
    "gyeongbokgung_geunjeongmun",
    "jogyesa",
    "jongmyo_shrine",
    "mmca_seoul",
    "myeongdong_cathedral",
    "naksan_park",
    "statue_of_king_sejong",
]


@dataclass
class LandmarkInfo:
    landmark_id: str
    name_ko: str
    name_en: str
    aliases: list[str]
    description_ko: str
    latitude: float
    longitude: float
    hero_image_path: str  # 절대경로
    tags: list[str]

    @property
    def coordinates_valid(self) -> bool:
        try:
            lat = float(self.latitude)
            lon = float(self.longitude)
        except (TypeError, ValueError):
            return False
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    @property
    def map_url(self) -> str:
        return f"https://www.google.com/maps?q={self.latitude},{self.longitude}"


@dataclass
class PrototypeItem:
    landmark_id: str
    prototype: np.ndarray  # (512,)
    n_samples_used: int
    view_breakdown: dict


@dataclass
class TextIndexItem:
    landmark_id: str
    description_ko: str
    description_en: str
    keywords: list[str]
    embedding: np.ndarray  # (512,)


@dataclass
class NameEntry:
    key: str           # NFC + lower-case
    landmark_id: str
    display: str
    kind: str          # "name_ko" | "name_en" | "alias"


@dataclass
class AssetBundle:
    info_by_id: dict[str, LandmarkInfo]
    prototypes: dict[str, PrototypeItem]
    text_index: dict[str, TextIndexItem]
    name_entries: list[NameEntry]
    proto_matrix: np.ndarray             # (N, 512)
    text_matrix: Optional[np.ndarray]    # (N, 512) or None
    landmark_ids: list[str]              # proto_matrix/text_matrix 행 순서


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _build_name_entries(infos: list[LandmarkInfo]) -> list[NameEntry]:
    entries: list[NameEntry] = []
    for info in infos:
        for kind, value in (("name_ko", info.name_ko), ("name_en", info.name_en)):
            entries.append(NameEntry(
                key=normalize_text(value),
                landmark_id=info.landmark_id,
                display=value,
                kind=kind,
            ))
        for alias in info.aliases:
            entries.append(NameEntry(
                key=normalize_text(alias),
                landmark_id=info.landmark_id,
                display=alias,
                kind="alias",
            ))
    return entries


def load_landmark_info(path: Path, assets_dir: Path) -> dict[str, LandmarkInfo]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, LandmarkInfo] = {}
    for item in doc["items"]:
        hero_rel = item.get("hero_image_path", "")
        hero_abs = str((assets_dir / hero_rel).resolve()) if hero_rel else ""
        out[item["landmark_id"]] = LandmarkInfo(
            landmark_id=item["landmark_id"],
            name_ko=item["name_ko"],
            name_en=item["name_en"],
            aliases=item.get("aliases", []),
            description_ko=item.get("description_ko", ""),
            latitude=float(item.get("latitude", 0.0)),
            longitude=float(item.get("longitude", 0.0)),
            hero_image_path=hero_abs,
            tags=item.get("tags", []),
        )
    return out


def load_prototype_index(path: Path) -> tuple[dict[str, PrototypeItem], np.ndarray, list[str]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    items: dict[str, PrototypeItem] = {}
    matrix_rows: list[np.ndarray] = []
    landmark_ids: list[str] = []
    for it in doc["items"]:
        proto = np.asarray(it["prototype"], dtype=np.float32)
        items[it["landmark_id"]] = PrototypeItem(
            landmark_id=it["landmark_id"],
            prototype=proto,
            n_samples_used=int(it.get("n_samples_used", 0)),
            view_breakdown=it.get("view_breakdown", {}),
        )
        matrix_rows.append(proto)
        landmark_ids.append(it["landmark_id"])
    matrix = np.stack(matrix_rows) if matrix_rows else np.zeros((0, 512), dtype=np.float32)
    return items, matrix, landmark_ids


def load_text_index(path: Path) -> tuple[dict[str, TextIndexItem], np.ndarray, list[str]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    items: dict[str, TextIndexItem] = {}
    matrix_rows: list[np.ndarray] = []
    landmark_ids: list[str] = []
    for it in doc["items"]:
        emb = np.asarray(it["embedding"], dtype=np.float32)
        items[it["landmark_id"]] = TextIndexItem(
            landmark_id=it["landmark_id"],
            description_ko=it.get("description_ko", ""),
            description_en=it.get("description_en", ""),
            keywords=it.get("keywords", []),
            embedding=emb,
        )
        matrix_rows.append(emb)
        landmark_ids.append(it["landmark_id"])
    matrix = np.stack(matrix_rows) if matrix_rows else np.zeros((0, 512), dtype=np.float32)
    return items, matrix, landmark_ids


@dataclass
class AssetLoadResult:
    success: bool
    bundle: Optional[AssetBundle] = None
    errors: list[str] = field(default_factory=list)


def load_asset_bundle(assets_dir: Path) -> AssetLoadResult:
    """assets_dir에서 4개 파일을 로드하고 정합성 검증."""
    errors: list[str] = []
    info_path = assets_dir / "landmark_info.json"
    proto_path = assets_dir / "prototype_index.json"
    text_path = assets_dir / "landmark_text_index.json"

    if not info_path.exists():
        errors.append(f"missing: {info_path}")
    if not proto_path.exists():
        errors.append(f"missing: {proto_path}")
    if not text_path.exists():
        errors.append(f"missing (optional): {text_path}")

    if errors and not info_path.exists():
        return AssetLoadResult(success=False, errors=errors)

    try:
        info_by_id = load_landmark_info(info_path, assets_dir)
    except Exception as exc:
        errors.append(f"landmark_info parse error: {exc}")
        return AssetLoadResult(success=False, errors=errors)

    if not proto_path.exists():
        return AssetLoadResult(success=False, errors=errors)

    try:
        protos, proto_matrix, proto_ids = load_prototype_index(proto_path)
    except Exception as exc:
        errors.append(f"prototype_index parse error: {exc}")
        return AssetLoadResult(success=False, errors=errors)

    text_items: dict[str, TextIndexItem] = {}
    text_matrix = None
    if text_path.exists():
        try:
            text_items, text_matrix, _ = load_text_index(text_path)
        except Exception as exc:
            errors.append(f"text_index parse error: {exc}")
            text_matrix = None

    # 정합성 검증
    info_ids = set(info_by_id.keys())
    proto_ids_set = set(proto_ids)
    catalog = set(LANDMARK_CATALOG)
    missing_in_info = catalog - info_ids
    missing_in_proto = catalog - proto_ids_set
    if missing_in_info:
        errors.append(f"info missing landmark_ids: {sorted(missing_in_info)}")
    if missing_in_proto:
        errors.append(f"prototype missing landmark_ids: {sorted(missing_in_proto)}")

    name_entries = _build_name_entries(list(info_by_id.values()))

    bundle = AssetBundle(
        info_by_id=info_by_id,
        prototypes=protos,
        text_index=text_items,
        name_entries=name_entries,
        proto_matrix=proto_matrix,
        text_matrix=text_matrix,
        landmark_ids=proto_ids,
    )
    success = len([e for e in errors if "missing landmark_ids" in e or "missing:" in e]) == 0
    return AssetLoadResult(success=success, bundle=bundle, errors=errors)
