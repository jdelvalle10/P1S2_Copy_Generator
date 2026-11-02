"""End-to-end self-test of step 2 on a small folder in step-1 layout (no network).

    python3 -m unittest discover -s tests -v
"""
import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "generate_copies.py"


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, check=True)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


class StepTwo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.sample, cls.a, cls.b = t / "sample", t / "a", t / "b"
        run(HERE / "make_sample.py", cls.sample)
        run(SCRIPT, "--in", cls.sample, "--out", cls.a, "--workers", "4", "--qa", "2")
        run(SCRIPT, "--in", cls.sample, "--out", cls.b, "--workers", "1", "--qa", "0")
        cls.copies = rows(cls.a / "manifest_copies.csv")
        cls.origs = rows(cls.a / "manifest_originals.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_cell_for_every_original(self):
        self.assertEqual(len(self.copies), len(self.origs) * 39)

    def test_same_output_with_any_worker_count(self):
        self.assertEqual((self.a / "manifest_copies.csv").read_bytes(), (self.b / "manifest_copies.csv").read_bytes())

    def test_checksums_match_files(self):
        for r in self.copies:
            if r["status"] == "ok":
                self.assertEqual(hashlib.sha256((self.a / r["file"]).read_bytes()).hexdigest(), r["sha256"])

    def test_exact_half_split(self):
        halves = [o["half"] for o in self.origs]
        self.assertEqual(halves.count("calibration"), halves.count("test"))

    def test_paste_areas_near_target(self):
        for r in self.copies:
            if r["op"] in ("splice", "copymove") and r["status"] == "ok":
                self.assertAlmostEqual(float(r["mask_share"]) / float(r["target_share"]), 1.0, delta=0.05)

    def test_removal_window_after_dilation(self):
        for r in self.copies:
            if r["op"] == "removal" and r["status"] == "ok":
                ratio = float(r["mask_share"]) / float(r["target_share"])
                self.assertTrue(0.6 <= ratio <= 1.6, ratio)

    def test_splice_donors_same_half_never_self(self):
        half = {o["image_id"]: o["half"] for o in self.origs}
        for r in self.copies:
            if r["op"] == "splice" and r["status"] == "ok":
                self.assertEqual(half[r["donor_image_id"]], r["half"])
                self.assertNotEqual(r["donor_image_id"], r["image_id"])

    def test_resume_recreates_missing_files(self):
        victim = self.origs[0]["image_id"]
        for p in (self.a / "copies" / victim).iterdir():
            p.unlink()
        run(SCRIPT, "--in", self.sample, "--out", self.a, "--workers", "2", "--qa", "0")
        self.assertEqual((self.a / "manifest_copies.csv").read_bytes(), (self.b / "manifest_copies.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
