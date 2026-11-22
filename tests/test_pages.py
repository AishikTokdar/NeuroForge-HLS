"""Public-site packaging checks, runnable without ML or third-party dependencies."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_pages import build_site


class PagesBuildTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="neuroforge-pages-")
        self.addCleanup(self.scratch.cleanup)
        self.base = Path(self.scratch.name)
        self.source = self.base / "source"
        self.source.mkdir()
        (self.source / "LICENSE").write_text("Test license", encoding="utf-8")
        self.index("<h1 id='title'>Test</h1>")
        self.output = self.base / "site"

    def index(self, body):
        (self.source / "index.html").write_text("<!doctype html><html><title>Test</title>" + body + "</html>",
                                               encoding="utf-8")

    def test_real_site_packages_every_linked_public_file(self):
        manifest = build_site(self.output, ROOT)
        for item in manifest["files"]:
            target = self.output / item["path"]
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), item["sha256"])
        self.assertEqual(len(list(self.output.rglob("*.html"))), 6)
        self.assertEqual((self.output / "index.html").read_bytes(), (ROOT / "index.html").read_bytes())
        for forbidden in [".git", ".github", ".verification", "hls", "training", "weights/params_int32"]:
            self.assertFalse((self.output / forbidden).exists(), forbidden)
        self.assertTrue((self.output / "site-manifest.json").is_file())
        for asset in ["site.css", "site.js", "theme.js"]:
            self.assertEqual((self.output / "assets" / asset).read_bytes(),
                             (ROOT / "assets" / asset).read_bytes())
        for dataset in ["classification_metrics", "training_metrics", "numeric_range_audit", "weight_validation"]:
            self.assertTrue((self.output / "docs" / (dataset + ".json")).is_file())

    def test_missing_reference_rejects_build_before_writing(self):
        self.index("<a href='visualizations/missing.html'>Missing</a>")
        with self.assertRaisesRegex(ValueError, "Missing public file"):
            build_site(self.output, self.source)
        self.assertFalse(self.output.exists())

    def test_private_source_reference_is_not_uploaded(self):
        (self.source / "training").mkdir()
        (self.source / "training/secret.txt").write_text("private", encoding="utf-8")
        self.index("<a href='training/secret.txt'>Bad</a>")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            build_site(self.output, self.source)

    def test_encoded_traversal_is_rejected(self):
        self.index("<a href='%2e%2e/outside.txt'>Bad</a>")
        with self.assertRaisesRegex(ValueError, "escapes"):
            build_site(self.output, self.source)

    def test_root_absolute_links_fail_project_subpath_check(self):
        self.index("<a href='/visualizations/example.html'>Bad</a>")
        with self.assertRaisesRegex(ValueError, "project subpath"):
            build_site(self.output, self.source)

    def test_duplicate_ids_are_rejected(self):
        self.index("<p id='same'>One</p><p id='same'>Two</p>")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_site(self.output, self.source)

    def test_stale_output_is_preserved_and_rejected(self):
        self.output.mkdir()
        marker = self.output / "old.html"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            build_site(self.output, self.source)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_output_cannot_overwrite_sources(self):
        original = (self.source / "index.html").read_bytes()
        with self.assertRaisesRegex(ValueError, "source directory"):
            build_site(self.source, self.source)
        self.assertEqual((self.source / "index.html").read_bytes(), original)

    def test_markdown_dependencies_are_packaged_transitively(self):
        (self.source / "docs").mkdir()
        (self.source / "docs/report.md").write_text(
            "[results](results.json)\n[external](https://example.org/)\n"
            "```text\n[code example](missing.txt)\n```", encoding="utf-8")
        (self.source / "docs/results.json").write_text("{}", encoding="utf-8")
        self.index("<a href='docs/report.md'>Report</a>")
        manifest = build_site(self.output, self.source)
        self.assertTrue((self.output / "docs/results.json").exists())
        self.assertIn("https://example.org/", manifest["external_dependencies"])


if __name__ == "__main__":
    unittest.main()
