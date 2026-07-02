import unittest
import tempfile
import shutil
from pathlib import Path
from pydantic import ValidationError

from stratasift.core.models import AtomicInsight
from stratasift.tools.vector_store import LanceVectorStore
from stratasift.utils.file_io import (
    sanitise_filename,
    generate_insight_id,
    format_insight,
    create_insight_file,
    append_insight_file,
)


class TestAtomicInsightModel(unittest.TestCase):
    """Unit tests for the AtomicInsight Pydantic model constraints.

    Uses British spelling rules.
    """

    def test_valid_insight(self) -> None:
        """Verify that a valid insight meets all validation constraints."""
        valid_title = "A" * 70  # Between 60 and 100
        valid_core = "B" * 250  # Between 200 and 350

        insight = AtomicInsight(
            title=valid_title,
            core_insight=valid_core,
            context_evidence="**[Source: Author (Year)]**\n* Point 1",
            related_vectors=["link1", "link2"],
        )

        self.assertEqual(insight.title, valid_title)
        self.assertEqual(insight.core_insight, valid_core)
        self.assertEqual(insight.search_block, f"{valid_title}\n{valid_core}")

    def test_invalid_title_length(self) -> None:
        """Verify validation errors when the title is too short or too long."""
        # Too short
        with self.assertRaises(ValidationError):
            AtomicInsight(
                title="Too short",
                core_insight="B" * 250,
                context_evidence="evidence",
                related_vectors=[],
            )

        # Too long
        with self.assertRaises(ValidationError):
            AtomicInsight(
                title="A" * 105,
                core_insight="B" * 250,
                context_evidence="evidence",
                related_vectors=[],
            )

    def test_invalid_core_insight_length(self) -> None:
        """Verify validation errors when core insight is outside the 200-500 limit."""
        # Too short
        with self.assertRaises(ValidationError):
            AtomicInsight(
                title="A" * 70,
                core_insight="Too short",
                context_evidence="evidence",
                related_vectors=[],
            )

        # Too long
        with self.assertRaises(ValidationError):
            AtomicInsight(
                title="A" * 70,
                core_insight="B" * 550,
                context_evidence="evidence",
                related_vectors=[],
            )


class TestFileIOUtils(unittest.TestCase):
    """Unit tests for deterministic file writer and safe markdown utilities.

    Uses British spelling rules.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_sanitise_filename(self) -> None:
        """Verify that forbidden characters are stripped from titles."""
        raw_title = "Is: electrolyte / separator | material safe?"
        clean_title = sanitise_filename(raw_title)
        self.assertEqual(clean_title, "Is electrolyte  separator  material safe")

    def test_generate_insight_id(self) -> None:
        """Verify unique ID generation formatting."""
        insight_id = generate_insight_id()
        self.assertTrue(insight_id.startswith("INSIGHT-"))
        self.assertEqual(len(insight_id.split("-")), 3)

    def test_format_insight_template(self) -> None:
        """Verify rendering using the strict markdown layout template."""
        insight = AtomicInsight(
            title="Sintering electrolyte yields optimal battery parameters and density",
            core_insight="Solid-state battery synthesis using LLZO electrolyte. Sintering protocol at 800°C for 6 hours yields a cubic phase with ionic conductivity of 1.2e-4 S/cm and an overall yield of 85%. This achieves optimal performance parameters required for high-energy density storage applications.",
            context_evidence="**[Source: Test (2026)]**\n* Extractions",
            related_vectors=["sintering_limits"],
        )
        rendered = format_insight(
            insight, "INSIGHT-2026-ABC123", "2026-06-22", ["test_tag"]
        )

        self.assertIn('id: "INSIGHT-2026-ABC123"', rendered)
        self.assertIn('status: "verified"', rendered)
        self.assertIn("tags: [test_tag]", rendered)
        self.assertIn("date_created: 2026-06-22", rendered)
        self.assertIn("# Title: Sintering electrolyte yields", rendered)
        self.assertIn("## Core Insight\n> Solid-state battery synthesis", rendered)
        self.assertIn("## Related Vectors\n* [[sintering_limits]]", rendered)
        self.assertIn("## Context & Evidence\n\n**[Source: Test (2026)]**", rendered)

    def test_create_and_append_insight(self) -> None:
        """Verify creating a new note and non-destructively appending new evidence."""
        insight = AtomicInsight(
            title="Sintering electrolyte yields optimal battery parameters and density",
            core_insight="Solid-state battery synthesis using LLZO electrolyte. Sintering protocol at 800°C for 6 hours yields a cubic phase with ionic conductivity of 1.2e-4 S/cm and an overall yield of 85%. This achieves optimal performance parameters required for high-energy density storage applications.",
            context_evidence="**[Source: Test (2026)]**\n* Extractions 1",
            related_vectors=["sintering_limits"],
        )

        # 1. Create note
        file_path = create_insight_file(self.temp_dir, insight, ["test"])
        self.assertTrue(file_path.exists())

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Extractions 1", content)

        # 2. Append note
        new_evidence = "**[Source: Second Paper (2026)]**\n* Extractions 2"
        append_insight_file(file_path, new_evidence)

        with open(file_path, "r", encoding="utf-8") as f:
            updated_content = f.read()

        self.assertIn("Extractions 1", updated_content)
        self.assertIn("Extractions 2", updated_content)
        # Ensure the immutable top section is still intact
        self.assertIn("# Title: Sintering electrolyte yields", updated_content)


class TestLanceVectorStore(unittest.TestCase):
    """Unit tests for the organic vector store insertion database logic.

    Uses British spelling rules.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_organic_insertion(self) -> None:
        """Verify inserting insights organically and preventing duplicates."""
        store = LanceVectorStore(self.temp_dir)

        # 1. Insert first record
        store.insert_insight("Note1.md", "Note 1 Title", "Search Block content 1")

        # 2. Verify table was created and has 1 record
        tbl = store.db.open_table("obsidian_vault")
        self.assertEqual(len(tbl), 1)

        # 3. Insert duplicate ID (should be ignored proactively)
        store.insert_insight(
            "Note1.md", "Note 1 Title Updated", "Search Block content updated"
        )
        tbl = store.db.open_table("obsidian_vault")
        self.assertEqual(len(tbl), 1)

        # 4. Insert second unique record
        store.insert_insight("Note2.md", "Note 2 Title", "Search Block content 2")
        tbl = store.db.open_table("obsidian_vault")
        self.assertEqual(len(tbl), 2)


if __name__ == "__main__":
    unittest.main()
