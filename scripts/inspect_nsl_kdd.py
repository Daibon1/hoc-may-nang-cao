"""Download and profile the NSL-KDD train and test files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import TextIO


FEATURES = (
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
)
EXPECTED_COLUMNS = len(FEATURES) + 2
NUMERIC_FEATURES = set(FEATURES) - {
    "protocol_type",
    "service",
    "flag",
}

ATTACK_GROUPS = {
    "DoS": {
        "apache2",
        "back",
        "land",
        "mailbomb",
        "neptune",
        "pod",
        "processtable",
        "smurf",
        "teardrop",
        "udpstorm",
        "worm",
    },
    "Probe": {
        "ipsweep",
        "mscan",
        "nmap",
        "portsweep",
        "saint",
        "satan",
    },
    "R2L": {
        "ftp_write",
        "guess_passwd",
        "httptunnel",
        "imap",
        "multihop",
        "named",
        "phf",
        "sendmail",
        "snmpgetattack",
        "snmpguess",
        "spy",
        "warezclient",
        "warezmaster",
        "xlock",
        "xsnoop",
    },
    "U2R": {
        "buffer_overflow",
        "loadmodule",
        "perl",
        "ps",
        "rootkit",
        "sqlattack",
        "xterm",
    },
}

SOURCE_REPOSITORY = "https://github.com/Jehuty4949/NSL_KDD"
SOURCE_COMMIT = "27bbbdf5c5cccf9970b8b6037a0f19dc09c5a0a9"
DATA_FILES = {
    "KDDTrain+.txt": "20002897492adc110190a10509aa694f26f7f308",
    "KDDTest+.txt": "b30e89605285a5ccd8cf8fd38dba72906c20d3dd",
}


def github_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha in DATA_FILES.items():
        destination = data_dir / filename
        if destination.exists():
            actual_sha = github_blob_sha(destination.read_bytes())
            if actual_sha == expected_sha:
                print(f"Already verified: {destination}")
                continue
            raise ValueError(
                f"{destination} exists but its Git blob SHA-1 does not match "
                "the pinned dataset; move it aside before downloading."
            )

        url = (
            f"https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/"
            f"{SOURCE_COMMIT}/{filename.replace('+', '%2B')}"
        )
        print(f"Downloading {filename} from pinned NSL-KDD mirror...")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        actual_sha = github_blob_sha(data)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Integrity check failed for {filename}: expected "
                f"{expected_sha}, got {actual_sha}."
            )

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=data_dir, prefix=f".{filename}.", delete=False
        ) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
        print(f"Downloaded and verified: {destination} ({len(data):,} bytes)")


def classify_label(label: str) -> str:
    label = label.strip().lower().removesuffix(".")
    if label == "normal":
        return "normal"
    for group, attack_types in ATTACK_GROUPS.items():
        if label in attack_types:
            return group
    return "unknown"


def profile_file(path: Path) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    category_values: dict[str, set[str]] = {
        name: set() for name in ("protocol_type", "service", "flag")
    }
    numeric_min: dict[str, float] = {}
    numeric_max: dict[str, float] = {}
    invalid_numeric: Counter[str] = Counter()
    missing_cells = 0
    malformed_rows = 0
    duplicate_rows = 0
    seen_rows: set[tuple[str, ...]] = set()
    row_count = 0
    observed_widths: Counter[int] = Counter()

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        for row in reader:
            if not row:
                continue
            row_count += 1
            observed_widths[len(row)] += 1
            if len(row) != EXPECTED_COLUMNS:
                malformed_rows += 1
                continue

            normalized = tuple(value.strip() for value in row)
            if normalized in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(normalized)

            missing_cells += sum(value == "" for value in normalized)
            label = normalized[-2].lower().removesuffix(".")
            label_counts[label] += 1
            group_counts[classify_label(label)] += 1

            for name in category_values:
                value = normalized[FEATURES.index(name)]
                if value:
                    category_values[name].add(value)

            for index, name in enumerate(FEATURES):
                if name not in NUMERIC_FEATURES:
                    continue
                value = normalized[index]
                try:
                    number = float(value)
                except ValueError:
                    invalid_numeric[name] += 1
                    continue
                if not math.isfinite(number):
                    invalid_numeric[name] += 1
                    continue
                numeric_min[name] = min(numeric_min.get(name, number), number)
                numeric_max[name] = max(numeric_max.get(name, number), number)

    return {
        "file": path.name,
        "git_blob_sha1": github_blob_sha(path.read_bytes()),
        "rows": row_count,
        "expected_columns_per_row": EXPECTED_COLUMNS,
        "observed_row_widths": {
            str(width): count for width, count in sorted(observed_widths.items())
        },
        "malformed_rows": malformed_rows,
        "missing_cells": missing_cells,
        "duplicate_rows_after_first_occurrence": duplicate_rows,
        "normal_rows": group_counts["normal"],
        "attack_rows": sum(
            count
            for group, count in group_counts.items()
            if group not in {"normal", "unknown"}
        ),
        "label_counts": dict(sorted(label_counts.items())),
        "attack_group_counts": dict(sorted(group_counts.items())),
        "unknown_labels": sorted(
            label for label in label_counts if classify_label(label) == "unknown"
        ),
        "categorical_cardinality": {
            name: len(values) for name, values in category_values.items()
        },
        "categorical_values": {
            name: sorted(values) for name, values in category_values.items()
        },
        "numeric_min": dict(sorted(numeric_min.items())),
        "numeric_max": dict(sorted(numeric_max.items())),
        "invalid_numeric_cells": dict(sorted(invalid_numeric.items())),
    }


def print_summary(profile: dict[str, object], output: TextIO) -> None:
    rows = int(profile["rows"])
    normal = int(profile["normal_rows"])
    attacks = int(profile["attack_rows"])
    print(
        f"\n{profile['file']} — {rows:,} bản ghi",
        file=output,
    )
    print(
        f"  Phân lớp: normal {normal:,} ({normal / rows:.1%}) | "
        f"attack {attacks:,} ({attacks / rows:.1%})",
        file=output,
    )
    print("  Nhóm tấn công:", file=output)
    for group in ("DoS", "Probe", "R2L", "U2R"):
        print(
            f"    {group:<5} {profile['attack_group_counts'].get(group, 0):>7,}",
            file=output,
        )
    print(
        "  Kiểm tra: "
        f"{profile['expected_columns_per_row']} cột/dòng, "
        f"{profile['malformed_rows']} dòng lỗi, "
        f"{profile['missing_cells']} ô trống, "
        f"{profile['duplicate_rows_after_first_occurrence']} dòng trùng, "
        f"{sum(profile['invalid_numeric_cells'].values())} ô số lỗi",
        file=output,
    )
    print(f"  Git blob SHA-1: {profile['git_blob_sha1']}", file=output)
    if profile["unknown_labels"]:
        print(f"  Unmapped labels: {profile['unknown_labels']}", file=output)


def print_visual_summary(profiles: list[dict[str, object]], output: TextIO) -> None:
    separator = "-" * 112
    print(f"\n{separator}\nCƠ CẤU DỮ LIỆU: BÌNH THƯỜNG VÀ TẤN CÔNG", file=output)
    print(separator, file=output)
    for profile in profiles:
        rows = int(profile["rows"])
        normal = int(profile["normal_rows"])
        attack = int(profile["attack_rows"])
        normal_bar = "#" * round(24 * normal / rows)
        attack_bar = "#" * round(24 * attack / rows)
        print(
            f"{profile['file']:<16} {rows:>9,} dòng\n"
            f"  Bình thường [{normal_bar:<24}] {normal:>8,} ({normal / rows:5.1%})\n"
            f"  Tấn công    [{attack_bar:<24}] {attack:>8,} ({attack / rows:5.1%})",
            file=output,
        )

    print(
        f"\n{separator}\nPHÂN BỐ NHÓM TẤN CÔNG",
        file=output,
    )
    print("Thanh dùng thang logarit; số lượng thực được ghi cuối dòng.", file=output)
    for profile in profiles:
        counts = profile["attack_group_counts"]
        largest = max(int(counts.get(group, 0)) for group in ATTACK_GROUPS)
        log_max = math.log10(largest + 1) if largest else 1
        print(f"\n{profile['file']}", file=output)
        for group in ATTACK_GROUPS:
            value = int(counts.get(group, 0))
            bar_length = (
                max(1, round(24 * math.log10(value + 1) / log_max)) if value else 0
            )
            print(
                f"  {group:<5} [{'#' * bar_length:<24}] {value:>8,}",
                file=output,
            )

    print(f"\n{separator}\nKẾT QUẢ KIỂM TRA CHẤT LƯỢNG", file=output)
    for profile in profiles:
        checks = {
            "43 cột/dòng": profile["malformed_rows"] == 0,
            "không ô trống": profile["missing_cells"] == 0,
            "không dòng trùng": profile["duplicate_rows_after_first_occurrence"] == 0,
            "nhãn đã ánh xạ": not profile["unknown_labels"],
            "giá trị số hợp lệ": not profile["invalid_numeric_cells"],
            "hash nguồn khớp": bool(profile.get("source_hash_verified")),
        }
        print(f"\n{profile['file']}", file=output)
        for name, passed in checks.items():
            status = "ĐẠT" if passed else "CẦN KIỂM TRA"
            print(f"  [{status:^15}] {name}", file=output)

    train = profiles[0]["categorical_cardinality"]
    test = profiles[1]["categorical_cardinality"]
    print(f"\n{separator}\nSỐ LOẠI ĐẶC TRƯNG DANH MỤC", file=output)
    print(f"{'Đặc trưng':<22}{'Train':>12}{'Test':>12}", file=output)
    print("-" * 46, file=output)
    for name in ("protocol_type", "service", "flag"):
        print(f"{name:<22}{train[name]:>12}{test[name]:>12}", file=output)
    print(separator, file=output)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_root / "data" / "raw",
        help="directory containing KDDTrain+.txt and KDDTest+.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "reports" / "nsl_kdd_data_profile.json",
        help="path for the generated JSON data profile",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the two files from the pinned public mirror if absent",
    )
    args = parser.parse_args()

    try:
        if args.download:
            download_dataset(args.data_dir)

        missing = [
            args.data_dir / filename
            for filename in DATA_FILES
            if not (args.data_dir / filename).is_file()
        ]
        if missing:
            missing_list = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Required NSL-KDD files are missing:\n"
                f"{missing_list}\nRun this script with --download or place "
                "the files in --data-dir."
            )

        profiles = [
            profile_file(args.data_dir / filename) for filename in DATA_FILES
        ]
        for profile in profiles:
            expected_sha = DATA_FILES[profile["file"]]
            if profile["git_blob_sha1"] != expected_sha:
                raise ValueError(
                    f"Integrity check failed for {profile['file']}: expected "
                    f"{expected_sha}, got {profile['git_blob_sha1']}."
                )
            profile["source_hash_verified"] = True
        train_categories = {
            name: set(values)
            for name, values in profiles[0]["categorical_values"].items()
        }
        for profile in profiles[1:]:
            profile["unseen_categories"] = {
                name: sorted(set(values) - train_categories[name])
                for name, values in profile["categorical_values"].items()
            }
        result = {
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "feature_count": len(FEATURES),
            "categorical_features": ["protocol_type", "service", "flag"],
            "label_column": "label",
            "difficulty_column": "difficulty",
            "files": profiles,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for profile in profiles:
            print_summary(profile, sys.stdout)
        print_visual_summary(profiles, sys.stdout)
        print(f"\nBáo cáo dữ liệu JSON: {args.output}")
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
