import unittest
from stratasift.core.sanitiser import sanitise_line, sanitise_text


class TestSantitiser(unittest.TestCase):
    """Unit tests for markdown line and text sanitisation logic.

    Uses British spelling rules.
    """

    def test_sanitise_line_plain_text(self) -> None:
        """Verify plain text lines are returned unaltered."""
        line = "This is a simple sentence without links or images."
        s_line, links, images = sanitise_line(line)
        self.assertEqual(s_line, line)
        self.assertEqual(links, 0)
        self.assertEqual(images, 0)

    def test_sanitise_line_with_link(self) -> None:
        """Verify academic links are stripped to their anchor text."""
        line = "Check out our paper published on [Nature](https://www.nature.com)."
        s_line, links, images = sanitise_line(line)
        self.assertEqual(s_line, "Check out our paper published on Nature.")
        self.assertEqual(links, 1)
        self.assertEqual(images, 0)

    def test_sanitise_line_with_image(self) -> None:
        """Verify markdown images are completely removed."""
        line = (
            "Here is the experimental setup: ![Setup Diagram](file:///tmp/diagram.png)"
        )
        s_line, links, images = sanitise_line(line)
        self.assertEqual(s_line, "Here is the experimental setup: ")
        self.assertEqual(links, 0)
        self.assertEqual(images, 1)

    def test_sanitise_line_table_bypass(self) -> None:
        """Verify markdown table rows are bypassed completely."""
        line = "| Yield (%) | Temperature (°C) | Precursor |"
        s_line, links, images = sanitise_line(line)
        self.assertEqual(s_line, line)
        self.assertEqual(links, 0)
        self.assertEqual(images, 0)

    def test_sanitise_text_block(self) -> None:
        """Verify text block sanitisation accumulates correct total stats."""
        text_block = (
            "## Section Title\n"
            "This is a line with an image: ![alt](https://image.png)\n"
            "| Column 1 | Column 2 |\n"
            "And a link to [ResearchGate](https://researchgate.net/paper)."
        )
        s_text, links, images = sanitise_text(text_block)

        expected_text = (
            "## Section Title\n"
            "This is a line with an image: \n"
            "| Column 1 | Column 2 |\n"
            "And a link to ResearchGate."
        )
        self.assertEqual(s_text, expected_text)
        self.assertEqual(links, 1)
        self.assertEqual(images, 1)


if __name__ == "__main__":
    unittest.main()
