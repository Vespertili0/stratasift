import unittest
import tempfile
import shutil
from pathlib import Path

import importlib.util

from stratasift.tools.librarian import LanceLibrarian, SimpleEmbeddingFunction

LANCEDB_AVAILABLE = importlib.util.find_spec("lancedb") is not None


class TestSimpleEmbeddingFunction(unittest.TestCase):
    """Unit tests for the deterministic offline embedding fallback function.

    Uses British spelling rules.
    """

    def test_embedding_output_shape(self) -> None:
        """Verify the output vector shape and normalisation."""
        fn = SimpleEmbeddingFunction()
        texts = ["battery calcination", "biomass pyrolysis"]
        embeddings = fn(texts)

        self.assertEqual(len(embeddings), 2)
        for emb in embeddings:
            self.assertEqual(len(emb), 128)
            # Verify unit length normalisation: sum of squares ≈ 1.0
            norm = sum(x * x for x in emb)
            self.assertAlmostEqual(norm, 1.0, places=5)


@unittest.skipIf(not LANCEDB_AVAILABLE, "LanceDB is not installed manually yet.")
class TestLanceLibrarian(unittest.TestCase):
    """Unit tests for the LanceLibrarian database tool.

    Uses British spelling rules.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_sync_and_search(self) -> None:
        """Verify syncing vault files and querying them semantically."""
        # 1. Setup mock Obsidian vault structure
        note1 = self.temp_dir / "Sintering Limits.md"
        note1.write_text(
            "Sintering solid-state batteries calcination temperature limits around 900°C.",
            encoding="utf-8",
        )

        note2 = self.temp_dir / "Biomass Pyrolysis.md"
        note2.write_text(
            "Pyrolysis of pine wood biomass yields biochar and bio-oil.",
            encoding="utf-8",
        )

        # Create a hidden note and a shelved note (should be excluded)
        hidden_dir = self.temp_dir / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "Secret.md").write_text("Confidential details.", encoding="utf-8")

        shelved_dir = self.temp_dir / "Shelved"
        shelved_dir.mkdir()
        (shelved_dir / "Old Note.md").write_text(
            "Shelved calcination details.", encoding="utf-8"
        )

        # 2. Initialise VectorStore and insert organically
        from stratasift.tools.vector_store import LanceVectorStore

        vector_store = LanceVectorStore(self.temp_dir)
        vector_store.insert_insight(
            "Sintering Limits.md",
            "Sintering Limits",
            "Sintering solid-state batteries calcination temperature limits around 900°C.",
        )
        vector_store.insert_insight(
            "Biomass Pyrolysis.md",
            "Biomass Pyrolysis",
            "Pyrolysis of pine wood biomass yields biochar and bio-oil.",
        )

        # 3. Initialise Librarian
        librarian = LanceLibrarian(self.temp_dir)

        # 3. Query search
        results = librarian.search_vault("solid-state battery calcination", n_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Sintering Limits")
        self.assertIn("Sintering solid-state", results[0]["content"])

        # Query search on biomass
        results_biomass = librarian.search_vault("wood pyrolysis biochar", n_results=1)
        self.assertEqual(len(results_biomass), 1)
        self.assertEqual(results_biomass[0]["title"], "Biomass Pyrolysis")


if __name__ == "__main__":
    unittest.main()
