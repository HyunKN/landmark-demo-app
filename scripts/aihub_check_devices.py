"""List Qualcomm AI Hub devices that match our target SoC families.

Run after `qai-hub configure --api_token <YOUR_TOKEN>`. This script does NOT
submit any jobs; it only queries device availability so we can decide which
targets to use for compile / profile / inference.

Targets we care about for the landmark demo:
  - Snapdragon 8 Gen 2  (Galaxy S23 family)   [chipset:qualcomm-snapdragon-8gen2]
  - Snapdragon 8 Gen 3  (Galaxy S24 family)   [chipset:qualcomm-snapdragon-8gen3]
  - Snapdragon 8 Elite  (Galaxy S25 family)   [chipset:qualcomm-snapdragon-8-elite]

Usage:
    python scripts/aihub_check_devices.py
"""
from __future__ import annotations

import sys
from collections import defaultdict


TARGETS = [
    ("Snapdragon 8 Gen 2 (Galaxy S23)",  "qualcomm-snapdragon-8gen2"),
    ("Snapdragon 8 Gen 3 (Galaxy S24)",  "qualcomm-snapdragon-8gen3"),
    ("Snapdragon 8 Elite (Galaxy S25)",  "qualcomm-snapdragon-8-elite"),
]

PREFERRED_NAMES = [
    "Samsung Galaxy S25 (Family)",
    "Samsung Galaxy S24 (Family)",
    "Samsung Galaxy S23 (Family)",
    "Samsung Galaxy S25",
    "Samsung Galaxy S24",
    "Samsung Galaxy S23",
    "Samsung Galaxy S25 Ultra",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S23 Ultra",
]


def _safe_attr_join(attrs) -> str:
    if not attrs:
        return ""
    return ", ".join(sorted(str(a) for a in attrs))


def main() -> None:
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("qai-hub is not installed. pip install qai-hub")

    try:
        all_devices = hub.get_devices()
    except Exception as exc:
        sys.exit(f"qai-hub auth failed: {exc}\n"
                 "Run: qai-hub configure --api_token <YOUR_TOKEN>")

    print(f"[total] {len(all_devices)} devices visible to this account\n")

    # Group by chipset attribute
    by_chipset: dict[str, list] = defaultdict(list)
    for d in all_devices:
        chipsets = [a for a in (d.attributes or []) if a.startswith("chipset:")]
        for c in chipsets:
            by_chipset[c.split(":", 1)[1]].append(d)

    for label, key in TARGETS:
        bucket = by_chipset.get(key, [])
        print(f"=== {label}  [chipset:{key}]  ({len(bucket)} devices) ===")
        if not bucket:
            print("  (none)\n")
            continue
        # Sort: Family first, then ascending OS version
        def sort_key(d):
            name = d.name
            family_first = 0 if "(Family)" in name else 1
            os_ver = next((a.split(":", 1)[1]
                           for a in (d.attributes or []) if a.startswith("os-version:")),
                          "")
            return (family_first, name, os_ver)
        bucket.sort(key=sort_key)
        for d in bucket:
            attrs = [a for a in (d.attributes or [])
                     if a.startswith(("vendor:", "format:", "framework:"))]
            print(f"  - {d.name:36s}  os={d.os}  attrs=[{_safe_attr_join(attrs)}]")
        print()

    # Recommendation summary
    chosen = []
    for label, key in TARGETS:
        bucket = by_chipset.get(key, [])
        # Prefer the first matching name from PREFERRED_NAMES
        pick = None
        for pref in PREFERRED_NAMES:
            for d in bucket:
                if d.name == pref:
                    pick = d
                    break
            if pick:
                break
        if not pick and bucket:
            pick = bucket[0]
        if pick:
            chosen.append((label, pick))

    print("=== Recommended targets (one per generation) ===")
    if not chosen:
        print("  No matching device family found. The next step is blocked until "
              "AI Hub adds the chipset or auth is fixed.")
    else:
        for label, d in chosen:
            print(f"  {label:42s} -> name='{d.name}', os='{d.os}'")


if __name__ == "__main__":
    main()
