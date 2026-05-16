"""Add bilingual and contrast captions to Dataset/*/labels.json.

The current image classifier training path still uses image labels only, but
the dataset contract should preserve captions for text-index generation and
future image-text fine-tuning. This script updates labels in place only after
writing a timestamped backup under Dataset/_label_backups/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


CAPTION_SOURCE = "dataset_v2_text_catalog_2026_05_16"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_catalog(path: Path) -> dict[str, dict]:
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {item["landmark_id"]: item for item in doc.get("items", [])}


def first(values: list[str] | None, default: str = "") -> str:
    return values[0] if values else default


def upsert_caption(
    captions: list[dict],
    caption_type: str,
    *,
    text_en: str = "",
    text_ko: str = "",
    overwrite_ko: bool = False,
) -> bool:
    for caption in captions:
        if caption.get("caption_type") != caption_type:
            continue
        changed = False
        if text_en and not caption.get("text_en"):
            caption["text_en"] = text_en
            changed = True
        if text_ko and (overwrite_ko or not caption.get("text_ko")):
            caption["text_ko"] = text_ko
            changed = True
        if changed:
            caption["source"] = caption.get("source") or CAPTION_SOURCE
        return changed

    new_caption = {"caption_type": caption_type}
    if text_en:
        new_caption["text_en"] = text_en
    if text_ko:
        new_caption["text_ko"] = text_ko
    new_caption["source"] = CAPTION_SOURCE
    captions.append(new_caption)
    return True


def augment_record(record: dict, catalog_entry: dict, *, overwrite_ko: bool) -> bool:
    captions = record.setdefault("captions", [])
    if not isinstance(captions, list):
        record["captions"] = captions = []

    name_ko = str(record.get("landmark_name_ko") or first(catalog_entry.get("aliases_ko"), ""))
    name_en = str(record.get("landmark_name_en") or first(catalog_entry.get("aliases_en"), ""))
    changed = False
    changed |= upsert_caption(
        captions,
        "name_anchor",
        text_en=f"{name_en} photo" if name_en else "",
        text_ko=f"{name_ko} 사진" if name_ko else "",
        overwrite_ko=overwrite_ko,
    )

    visual_en = first(catalog_entry.get("visual_features_en"))
    visual_ko = first(catalog_entry.get("visual_features_ko"))
    changed |= upsert_caption(captions, "visual_feature", text_ko=visual_ko, overwrite_ko=overwrite_ko)

    if record.get("label_status") == "confirmed":
        changed |= upsert_caption(
            captions,
            "function",
            text_en=first(catalog_entry.get("user_queries_en"), visual_en),
            text_ko=first(catalog_entry.get("user_queries_ko"), visual_ko),
            overwrite_ko=overwrite_ko,
        )
        changed |= upsert_caption(
            captions,
            "class_visual_anchor",
            text_en=visual_en,
            text_ko=visual_ko,
            overwrite_ko=overwrite_ko,
        )
        changed |= upsert_caption(
            captions,
            "contrast_with",
            text_en=first(catalog_entry.get("contrast_en")),
            text_ko=first(catalog_entry.get("contrast_ko")),
            overwrite_ko=overwrite_ko,
        )

    if changed:
        record["caption_schema_version"] = "dataset-v2-bilingual-contrast"
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-ko", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    catalog = load_catalog(Path(args.catalog).resolve())
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = data_root / "_label_backups" / timestamp

    summary = {"data_root": str(data_root), "dry_run": args.dry_run, "classes": {}, "backup_root": str(backup_root)}
    for landmark_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if landmark_dir.name.startswith("_"):
            continue
        labels_path = landmark_dir / "labels.json"
        if not labels_path.exists():
            continue
        catalog_entry = catalog.get(landmark_dir.name)
        if not catalog_entry:
            summary["classes"][landmark_dir.name] = {"error": "missing_catalog_entry"}
            continue

        data = read_json(labels_path)
        if not isinstance(data, list):
            raise ValueError(f"{labels_path} must contain a list")
        changed_records = 0
        captions_added_or_updated = 0
        for record in data:
            before = json.dumps(record.get("captions", []), ensure_ascii=False, sort_keys=True)
            changed = augment_record(record, catalog_entry, overwrite_ko=args.overwrite_ko)
            after = json.dumps(record.get("captions", []), ensure_ascii=False, sort_keys=True)
            if changed:
                changed_records += 1
            if before != after:
                captions_added_or_updated += 1

        if not args.dry_run and captions_added_or_updated:
            backup_path = backup_root / landmark_dir.name / "labels.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(labels_path, backup_path)
            write_json(labels_path, data)

        summary["classes"][landmark_dir.name] = {
            "records": len(data),
            "changed_records": changed_records,
            "caption_lists_changed": captions_added_or_updated,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
