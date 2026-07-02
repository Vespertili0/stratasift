import json
from pathlib import Path
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class EvaluationScores(BaseModel):
    extraction_faithfulness: float = Field(..., ge=0.0, le=1.0)
    synthesis_relevance: float = Field(..., ge=0.0, le=1.0)
    retrieval_recall: float = Field(..., ge=0.0, le=1.0)


class GoldenPair(BaseModel):
    source_filename: str
    reading_directive: str
    central_hypothesis: str
    context_db_snapshot: Dict[str, Any] = Field(
        ..., description="Snapshot of the tmp-ContextDB JSON state at evaluation time."
    )
    final_markdown: str = Field(
        ..., description="The final Obsidian-formatted note contents."
    )
    scores: EvaluationScores
    timestamp: str


class GoldenSet(BaseModel):
    version: str = "1.0.0"
    records: List[GoldenPair]


def bootstrap_golden_records(records: List[GoldenPair], vault_path: Path) -> None:
    """Serialises high-scoring evaluation pairs to the eval_benchmarks directory."""
    eval_dir = vault_path / ".stratasift" / "eval_benchmarks"
    # Edge Case 2 Mitigation: Explicit tree creation
    eval_dir.mkdir(parents=True, exist_ok=True)

    golden_file = eval_dir / "golden_set.json"

    existing_records = []
    if golden_file.exists():
        try:
            with open(golden_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                golden_set = GoldenSet.model_validate(data)
                existing_records = golden_set.records
        except Exception:
            pass

    existing_records.extend(records)
    new_set = GoldenSet(records=existing_records)

    with open(golden_file, "w", encoding="utf-8") as f:
        f.write(new_set.model_dump_json(indent=2))
