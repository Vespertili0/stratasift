from pathlib import Path
from typing import List, Dict, Any

import lancedb

from stratasift.tools.vector_store import SimpleEmbeddingFunction


class LanceLibrarian:
    """Librarian Tool that manages querying the local LanceDB table of vault notes.

    Uses British spelling rules.
    """

    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)
        lancedb_dir = self.vault_path / ".lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)

        # Initialise LanceDB persistent client
        self.db = lancedb.connect(str(lancedb_dir))
        self.embedding_function = SimpleEmbeddingFunction()

    def search_vault(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search the vault for semantically similar notes using cosine metric.

        Returns a list of dictionaries containing matching note details, including a calculated similarity score.
        """
        table_names = self.db.list_tables()
        table_list = (
            table_names.tables if hasattr(table_names, "tables") else list(table_names)
        )

        if "obsidian_vault" not in table_list:
            return []

        tbl = self.db.open_table("obsidian_vault")
        if len(tbl) == 0:
            return []

        query_vector = self.embedding_function([query_text])[0]

        # Execute LanceDB vector search query using cosine distance
        results = tbl.search(query_vector).metric("cosine").limit(n_results).to_list()

        matches = []
        for r in results:
            dist = r.get("_distance", 1.0)
            # cosine similarity = 1 - cosine distance
            similarity = max(0.0, min(1.0, 1.0 - dist))
            matches.append(
                {
                    "id": r["id"],
                    "title": r.get("title", Path(r["id"]).stem),
                    "content": r.get("content", ""),
                    "similarity": similarity,
                }
            )
        return matches
