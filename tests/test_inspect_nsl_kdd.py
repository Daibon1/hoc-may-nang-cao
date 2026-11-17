import csv
import io
import tempfile
import unittest
from pathlib import Path

from scripts.inspect_nsl_kdd import (
    FEATURES,
    classify_label,
    print_visual_summary,
    profile_file,
)


class ClassifyLabelTests(unittest.TestCase):
    def test_maps_normal_and_attack_families(self) -> None:
        expected = {
            "normal": "normal",
            "neptune": "DoS",
            "nmap": "Probe",
            "guess_passwd": "R2L",
            "buffer_overflow": "U2R",
        }
        for label, group in expected.items():
            with self.subTest(label=label):
                self.assertEqual(classify_label(label), group)

    def test_keeps_unmapped_labels_explicit(self) -> None:
        self.assertEqual(classify_label("unmapped_attack"), "unknown")


class ProfileFileTests(unittest.TestCase):
    def test_counts_labels_groups_duplicates_and_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.txt"
            normal = ["0"] * len(FEATURES) + ["normal", "20"]
            normal[FEATURES.index("protocol_type")] = "tcp"
            normal[FEATURES.index("service")] = "http"
            normal[FEATURES.index("flag")] = "SF"
            dos = normal.copy()
            dos[-2] = "neptune."
            dos[FEATURES.index("src_bytes")] = "NaN"

            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(normal)
                writer.writerow(normal)
                writer.writerow(dos)
                writer.writerow(["malformed"])

            profile = profile_file(path)

        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["normal_rows"], 2)
        self.assertEqual(profile["attack_rows"], 1)
        self.assertEqual(profile["attack_group_counts"]["DoS"], 1)
        self.assertEqual(profile["duplicate_rows_after_first_occurrence"], 1)
        self.assertEqual(profile["malformed_rows"], 1)
        self.assertEqual(profile["invalid_numeric_cells"], {"src_bytes": 1})
        self.assertEqual(profile["categorical_cardinality"], {
            "protocol_type": 1,
            "service": 1,
            "flag": 1,
        })

    def test_prints_readable_text_charts_and_quality_status(self) -> None:
        def summary(file: str, rows: int, normal: int, groups: dict[str, int]) -> dict:
            return {
                "file": file,
                "rows": rows,
                "normal_rows": normal,
                "attack_rows": rows - normal,
                "attack_group_counts": groups,
                "malformed_rows": 0,
                "missing_cells": 0,
                "duplicate_rows_after_first_occurrence": 0,
                "unknown_labels": [],
                "invalid_numeric_cells": {},
                "source_hash_verified": True,
                "categorical_cardinality": {
                    "protocol_type": 3,
                    "service": 70,
                    "flag": 11,
                },
            }

        profiles = [
            summary(
                "KDDTrain+.txt",
                100,
                60,
                {"DoS": 25, "Probe": 10, "R2L": 4, "U2R": 1},
            ),
            summary(
                "KDDTest+.txt",
                80,
                30,
                {"DoS": 25, "Probe": 15, "R2L": 8, "U2R": 2},
            ),
        ]
        output = io.StringIO()

        print_visual_summary(profiles, output)

        rendered = output.getvalue()
        self.assertIn("KDDTrain+.txt", rendered)
        self.assertIn("DoS", rendered)
        self.assertIn("hash nguồn khớp", rendered)
        self.assertIn("[      ĐẠT      ]", rendered)
        self.assertIn("SỐ LOẠI ĐẶC TRƯNG DANH MỤC", rendered)
        self.assertIn("service", rendered)
        self.assertIn("70", rendered)
        self.assertIn("Thanh dùng thang logarit", rendered)
        self.assertIn("[#", rendered)


if __name__ == "__main__":
    unittest.main()
