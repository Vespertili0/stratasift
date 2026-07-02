import unittest
import tempfile
from pathlib import Path
from stratasift.core.parser import parse_markdown_file


class TestParser(unittest.TestCase):
    """Unit tests for the markdown structural parser.

    Uses British spelling rules.
    """

    def test_parse_success(self) -> None:
        """Verify parsing a valid note with standard layout succeeds."""
        content = (
            "---\n"
            "title: Solid-State Synthesis of LLZO\n"
            "author: Research Group\n"
            "---\n"
            "\n"
            "## Abstract\n"
            "This study presents a solid-state synthesis route. [Link](http://google.com)\n"
            "\n"
            "## Methods\n"
            "Sintering was performed at 800°C for 6 hours.\n"
            "\n"
            "## Results and Discussion\n"
            "The final compound exhibited cubic structure. ![image](http://image.png)\n"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            lit, links, images = parse_markdown_file(temp_path)
            self.assertEqual(lit.title, "Solid-State Synthesis of LLZO")
            self.assertEqual(lit.metadata.get("author"), "Research Group")
            self.assertIn("solid-state synthesis", lit.abstract_intro)
            self.assertIn("Sintering was performed", lit.methods)
            self.assertIn("The final compound", lit.results_discussion)
            self.assertEqual(links, 1)
            self.assertEqual(images, 1)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_parse_title_fallback(self) -> None:
        """Verify parser falls back to the top H1 tag if title is missing in frontmatter."""
        content = "# Top Level Title Heading\n\n## Synopsis\nAbstract summary here.\n"

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            lit, links, images = parse_markdown_file(temp_path)
            self.assertEqual(lit.title, "Top Level Title Heading")
            self.assertIn("Abstract summary here.", lit.abstract_intro)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_parse_missing_title(self) -> None:
        """Verify parsing fails if no title is present in frontmatter or H1 header."""
        content = "## Abstract\nMissing title in this paper.\n"

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                parse_markdown_file(temp_path)
            self.assertIn("discoverable title", str(ctx.exception))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_parse_missing_sections(self) -> None:
        """Verify parsing fails if no structural layout sections match heuristics."""
        content = (
            "---\n"
            "title: Title Only\n"
            "---\n"
            "This is just some text without abstract, methods, or results headings.\n"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                parse_markdown_file(temp_path)
            self.assertIn("structural layout sections", str(ctx.exception))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_parse_missing_file(self) -> None:
        """Verify parsing raises ValueError for non-existent path."""
        with self.assertRaises(ValueError):
            parse_markdown_file(Path("non_existent_file.md"))

    def test_parse_conclusion_section(self) -> None:
        """Verify a document with a ## Conclusion heading populates lit.conclusions."""
        content = (
            "---\n"
            "title: Test Conclusion Parsing\n"
            "---\n"
            "\n"
            "## Abstract\n"
            "This study examines testing frameworks.\n"
            "\n"
            "## Methods\n"
            "We tested everything thoroughly.\n"
            "\n"
            "## Conclusion\n"
            "In conclusion, the results demonstrate significant advances.\n"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            lit, links, images = parse_markdown_file(temp_path)
            self.assertIsNotNone(lit.conclusions)
            self.assertIn("results demonstrate significant advances", lit.conclusions)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_parse_outlook_heading(self) -> None:
        """Verify ## Summary and Outlook maps to lit.conclusions."""
        content = (
            "---\n"
            "title: Test Outlook Parsing\n"
            "---\n"
            "\n"
            "## Abstract\n"
            "This study examines outlook parsing.\n"
            "\n"
            "## Summary and Outlook\n"
            "Looking forward, these methods will enable broader applications.\n"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            lit, links, images = parse_markdown_file(temp_path)
            self.assertIsNotNone(lit.conclusions)
            self.assertIn("broader applications", lit.conclusions)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_parse_conclusion_feeds_abstract_intro_separately(self) -> None:
        """Verify abstract_intro is not contaminated by conclusion content when both are present."""
        content = (
            "---\n"
            "title: Test Separation\n"
            "---\n"
            "\n"
            "## Abstract\n"
            "This is the abstract content only.\n"
            "\n"
            "## Methods\n"
            "We performed experiments.\n"
            "\n"
            "## Conclusions\n"
            "In summary, conclusions go here and should stay separate.\n"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            lit, links, images = parse_markdown_file(temp_path)
            self.assertIn("abstract content only", lit.abstract_intro)
            self.assertNotIn("conclusions go here", lit.abstract_intro)
            self.assertIsNotNone(lit.conclusions)
            self.assertIn("conclusions go here", lit.conclusions)
        finally:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    unittest.main()
