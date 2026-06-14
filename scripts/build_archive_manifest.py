"""Prepare a local archive manifest for A1-A7 deliverables without uploading."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit"

INCLUDE_DIRS = ("audit", "tables", "figures", "configs/formal", "scripts")
EXCLUDE_SUFFIXES = {".pyc", ".png"}  # PNG figures are listed by figure manifest.


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    AUDIT.mkdir(exist_ok=True)
    rows = []
    for dirname in INCLUDE_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            stat = path.stat()
            rows.append({
                "relative_path": path.relative_to(ROOT).as_posix(),
                "file_size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "sha256": _sha(path),
                "archive_recommendation": "include",
            })
    csv_path = AUDIT / "archive_manifest_a1_a7.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "file_size_bytes",
                "mtime_utc",
                "sha256",
                "archive_recommendation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path = AUDIT / "archive_manifest_a1_a7.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (AUDIT / "archive_manifest_a1_a7_cn.md").write_text(
        "\n".join(
            [
                "# A1-A7 archive preparation manifest",
                "",
                f"- manifest rows: {len(rows)}",
                "- 本工单只准备归档清单，不上传 Zenodo。",
                "- 建议纳入正式 raw/new result directories、source snapshot manifest、audit、tables、figures 和 formal configs。",
                "- 不建议纳入临时目录、pytest cache、`__pycache__`、旧 derived assets。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(csv_path)


if __name__ == "__main__":
    main()
