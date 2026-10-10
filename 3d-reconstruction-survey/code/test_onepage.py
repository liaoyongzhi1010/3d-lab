from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONEPAGE = PROJECT_ROOT / "onepage"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []
        self.images_without_alt: list[str] = []
        self.svg_count = 0
        self.has_sidebar = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if "sidebar" in classes:
            self.has_sidebar = True
        if tag == "svg":
            self.svg_count += 1
        if tag == "a" and attributes.get("href"):
            self.paths.append(attributes["href"] or "")
        if tag == "img":
            src = attributes.get("src") or ""
            if src:
                self.paths.append(src)
            if not (attributes.get("alt") or "").strip():
                self.images_without_alt.append(src or "<missing src>")


class SurveyOnepageTests(unittest.TestCase):
    def test_document_layout_has_navigation_diagrams_and_mobile_style(self) -> None:
        index = ONEPAGE / "index.html"
        text = index.read_text(encoding="utf-8")
        parser = PageParser()
        parser.feed(text)
        self.assertTrue(parser.has_sidebar)
        self.assertGreaterEqual(parser.svg_count, 2)
        self.assertIn("@media", text)
        self.assertIn("experiment.html?doc=README.md", text)
        self.assertEqual(parser.images_without_alt, [])

    def test_local_links_resolve(self) -> None:
        index = ONEPAGE / "index.html"
        parser = PageParser()
        parser.feed(index.read_text(encoding="utf-8"))
        missing: list[str] = []
        for target in parser.paths:
            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            resolved = (index.parent / unquote(parsed.path)).resolve()
            if not resolved.exists():
                missing.append(target)
        self.assertEqual(missing, [])

    def test_experiment_reader_rejects_path_traversal_and_external_scripts(
        self,
    ) -> None:
        reader = (ONEPAGE / "experiment.html").read_text(encoding="utf-8")
        self.assertIn("safeName", reader)
        self.assertIn("experiments/", reader)
        self.assertNotIn("<script src=", reader)
        self.assertNotIn("https://cdn", reader)


if __name__ == "__main__":
    unittest.main()
