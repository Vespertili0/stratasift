from pathlib import Path
from typing import List
import lancedb

from stratasift.utils.embeddings import SimpleEmbeddingFunction


class LanceVectorStore:
    """Manages the persistent LanceDB client, table initialisation, and organic insert logic.

    Uses British spelling rules.
    """

    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)
        self.lancedb_dir = self.vault_path / ".lancedb"
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)

        # Initialise LanceDB persistent client
        self.db = lancedb.connect(str(self.lancedb_dir))
        self.embedding_function = SimpleEmbeddingFunction()

    def insert_insight(self, file_id: str, title: str, search_block: str) -> None:
        """Generates and stores the vector embedding exclusively from the Search Block.

        This prevents semantic drift by excluding Context & Evidence from the vector space.
        """
        table_names = self.db.list_tables()
        table_list = (
            table_names.tables if hasattr(table_names, "tables") else list(table_names)
        )

        vector = self.embedding_function([search_block])[0]
        data = {
            "id": file_id,
            "vector": vector,
            "title": title,
            "content": search_block,
        }

        if "obsidian_vault" in table_list:
            tbl = self.db.open_table("obsidian_vault")
            # Proactively check if the ID already exists to avoid duplicate entries
            existing = tbl.search().where(f"id = '{file_id}'").to_list()
            if not existing:
                tbl.add([data])
        else:
            self.db.create_table("obsidian_vault", data=[data])
