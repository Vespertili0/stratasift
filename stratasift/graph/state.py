from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from stratasift.core.models import SanitisedLiterature


class SpecialistWorkerState(TypedDict):
    """Sub-state for individual parallel worker nodes."""
    chunk: Dict[str, str]
    reading_directive: str
    central_hypothesis: str
    match_type: str
    feedback: Optional[Dict[str, str]]


class PaperIngestionState(TypedDict):
    """The state dictionary for the paper ingestion and synthesis workflow.

    Uses British spelling rules.
    """

    source_doc: SanitisedLiterature  # Output from Epic-02 segmenter
    reading_directive: str  # Populated by Stage 1 Supervisor

    # Annotated with operator.add for consolidated specialist node merging
    raw_extractions: Annotated[List[Dict[str, Any]], operator.add]

    retry_count: int  # Prevents infinite loops (Max 1 retry)
    insight_dossier: str  # Populated by Stage 3 Reflection
    vault_context: List[str]  # Populated by Stage 4 Librarian Tool
    final_markdown: str  # The compiled Obsidian-flavoured markdown note

    # Operational keys for coordination and routing
    relevance_score: (
        float  # Computed max(domain, methodology) for backward compatibility
    )
    feedback: Optional[
        Dict[str, str]
    ]  # Dict mapping specialist type to feedback string
    atomic_insights: Annotated[List[Any], operator.add]  # List of synthesised Pydantic AtomicInsight models
    source_filename: str  # Original name of the ingested markdown file
    routing_results: List[
        Dict[str, Any]
    ]  # List of networking results and accumulated markdown contents
    run_id: str  # Unique identifier for the current run
    context_db_path: str  # Path to the temporary JSON context DB in quarantine

    # EPIC-05: Dual-vector triage fields
    domain_relevance: float  # Domain subject-matter relevance score (0.0–1.0)
    methodology_relevance: float  # Technical methodology relevance score (0.0–1.0)
    match_type: str  # "full", "domain_only", or "methodology_only"
    central_hypothesis: str  # Formulated by triage, passed to specialist

    # EPIC-05: In-flight deduplication queue
    in_flight_notes: List[Dict[str, Any]]  # In-memory queue for note deduplication
