from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List


class SanitisedLiterature(BaseModel):
    """Pydantic model representing dynamic, heading-aware scientific literature."""

    title: str = Field(
        description="The primary title extracted from frontmatter or the top H1 tag."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted frontmatter data keys."
    )
    abstract_intro: str = Field(
        description="The extracted abstract, summary, or top-level introduction section."
    )
    toc: List[str] = Field(
        default_factory=list,
        description="Dynamic Table of Contents mapping `#` and `##` structural headings.",
    )
    section_chunks: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of logical content blocks, e.g., [{'header': 'Methods / Protocols', 'content': '...'}]",
    )


class AtomicInsight(BaseModel):
    """Pydantic model representing synthesised atomic insights."""

    title: str = Field(
        ...,
        min_length=30,
        max_length=100,
        description="A single, standalone declarative statement. No passive topics.",
    )
    core_insight: str = Field(
        ...,
        min_length=200,
        max_length=500,
        description="Exactly 2-3 sentences using explicit nouns. No vague pronouns.",
    )
    context_evidence: Any = Field(
        default="",
        description="The bulleted source data and examples extracted from the text.",
    )
    related_vectors: list[str] = Field(
        default_factory=list,
        description="List of Obsidian WikiLinks generated for vault topology.",
    )
    data_points: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted technical parameters relevant to the insight.",
    )
    source_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim source quotes supporting the insight.",
    )

    @field_validator("related_vectors", mode="before")
    @classmethod
    def flatten_related_vectors(cls, v: Any) -> list[str]:
        """Flatten any nested list structure returned by the LLM for related_vectors."""
        if isinstance(v, list):
            flat = []
            for item in v:
                if isinstance(item, list):
                    flat.extend(cls.flatten_related_vectors(item))
                elif isinstance(item, str):
                    flat.append(item)
                else:
                    flat.append(str(item))
            return flat
        return v

    @field_validator("context_evidence", mode="before")
    @classmethod
    def format_context_evidence(cls, v: Any) -> str:
        """Ensure context_evidence is always formatted as a string."""
        if isinstance(v, str):
            return v
        if isinstance(v, (dict, list)):
            import json

            return json.dumps(v, indent=2)
        return str(v)

    @property
    def search_block(self) -> str:
        """The combined ~100 token string sent to LanceDB."""
        return f"{self.title}\n{self.core_insight}"
