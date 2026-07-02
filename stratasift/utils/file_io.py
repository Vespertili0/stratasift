import datetime
import random
import string
from pathlib import Path
from typing import List, Optional
from stratasift.core.models import AtomicInsight


def sanitise_filename(title: str) -> str:
    """Sanitise the title to make it a safe filename for Obsidian vault."""
    safe_title = title
    for char in '<>:"/\\|?*':
        safe_title = safe_title.replace(char, "")
    return safe_title.strip()


def generate_insight_id() -> str:
    """Generate a zero-collision unique identifier for YAML frontmatter."""
    year = datetime.date.today().year
    # 6-character random alphanumeric string
    random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"INSIGHT-{year}-{random_suffix}"


def format_insight(
    insight: AtomicInsight, insight_id: str, date_str: str, tags: List[str]
) -> str:
    """Format the AtomicInsight into the standardised note markdown layout."""
    tags_str = ", ".join(tags)

    # Format related vectors as wiki links
    wiki_links = []
    for rv in insight.related_vectors:
        if rv.startswith("[[") and rv.endswith("]]"):
            wiki_links.append(f"* {rv}")
        else:
            wiki_links.append(f"* [[{rv}]]")
    related_vectors_str = "\n".join(wiki_links)

    return f"""---
id: "{insight_id}"
type: "insight"
status: "verified"
tags: [{tags_str}]
date_created: {date_str}
---

# Title: {insight.title}

## Core Insight
> {insight.core_insight}

## Related Vectors
{related_vectors_str}

---
## Context & Evidence

{insight.context_evidence.strip()}
"""


def create_insight_file(
    vault_path: Path, insight: AtomicInsight, tags: Optional[List[str]] = None
) -> Path:
    """Write a new .md note to the Obsidian vault adhering to the strict layout template."""
    vault_path = Path(vault_path)
    vault_path.mkdir(parents=True, exist_ok=True)

    file_name = f"{sanitise_filename(insight.title)}.md"
    file_path = vault_path / file_name

    insight_id = generate_insight_id()
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    if tags is None:
        tags = ["insight"]

    content = format_insight(insight, insight_id, date_str, tags)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def append_insight_file(file_path: Path, new_context: str) -> None:
    """Open an existing .md file, verify the context header, and cleanly append the new source block.

    This ensures that the immutable Core Insight at the top of the file remains completely untouched.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot append to non-existent note: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "## Context & Evidence" not in content:
        raise ValueError(
            f"Note at {file_path} is missing '## Context & Evidence' header."
        )

    # Append to the bottom of the file
    if not content.endswith("\n"):
        content += "\n"

    content += "\n" + new_context.strip() + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
