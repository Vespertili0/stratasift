import re
from pathlib import Path
from typing import Tuple, Set, Optional
import frontmatter

from stratasift.core.models import SanitisedLiterature
from stratasift.core.sanitiser import sanitise_line

# Heuristic mapping targets for section segmentation
ABSTRACT_REGEX = r"(?i)^##?\s+(Abstract|Summary|Synopsis|Executive Summary)"
INTRO_REGEX = r"(?i)^##?\s+(Introduction|Background|1\.\s+Introduction)"
METHODS_REGEX = (
    r"(?i)^##?\s+(Methods|Methodology|Experimental\s+Section|Materials\s+and\s+Methods)"
)
RESULTS_REGEX = r"(?i)^##?\s+(Results|Discussion|Results\s+and\s+Discussion)"
CONCLUSION_REGEX = r"(?i)^##?\s+(Conclusions?|Outlook|Summary\s+and\s+Outlook|Concluding\s+Remarks|Perspectives)"


def _segment_markdown_content(
    content: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], int, int]:
    """Segment markdown content into sections and sanitise them."""
    abstract_intro_lines = []
    methods_lines = []
    results_discussion_lines = []
    conclusions_lines = []

    current_state: Optional[str] = None
    matched_sections: Set[str] = set()

    total_links = 0
    total_images = 0

    for line in content.splitlines():
        # Match headings
        if re.match(CONCLUSION_REGEX, line):
            current_state = "conclusions"
            matched_sections.add("conclusions")
            continue
        elif re.match(ABSTRACT_REGEX, line) or re.match(INTRO_REGEX, line):
            current_state = "abstract_intro"
            matched_sections.add("abstract_intro")
            continue
        elif re.match(METHODS_REGEX, line):
            current_state = "methods"
            matched_sections.add("methods")
            continue
        elif re.match(RESULTS_REGEX, line):
            current_state = "results_discussion"
            matched_sections.add("results_discussion")
            continue

        if current_state:
            # Sanitise the line (preserves tables, strips images and reference links)
            s_line, lnk_count, img_count = sanitise_line(line)
            total_links += lnk_count
            total_images += img_count

            if current_state == "abstract_intro":
                abstract_intro_lines.append(s_line)
            elif current_state == "methods":
                methods_lines.append(s_line)
            elif current_state == "results_discussion":
                results_discussion_lines.append(s_line)
            elif current_state == "conclusions":
                conclusions_lines.append(s_line)

    # 3. Baseline validation
    if not matched_sections:
        raise ValueError("Failed to identify core document structural layout sections.")

    abstract_intro_text = "\n".join(abstract_intro_lines).strip()
    methods_text = "\n".join(methods_lines).strip() if methods_lines else None
    results_discussion_text = (
        "\n".join(results_discussion_lines).strip()
        if results_discussion_lines
        else None
    )
    conclusions_text = (
        "\n".join(conclusions_lines).strip() if conclusions_lines else None
    )

    # Check that abstract_intro is populated since it is a required field
    if not abstract_intro_text:
        raise ValueError(
            "Failed to identify core document structural layout sections (Abstract/Introduction segment is empty)."
        )

    return (
        abstract_intro_text,
        methods_text,
        results_discussion_text,
        conclusions_text,
        total_links,
        total_images,
    )


def parse_markdown_file(file_path: Path) -> Tuple[SanitisedLiterature, int, int]:
    """Parse a markdown file using regex heuristic matching and sanitise its content.

    Returns:
        A tuple of (SanitisedLiterature, total_links_sanitised, total_images_stripped).

    Raises:
        ValueError: If the file is malformed, lacks a discoverable title, or
                    fails to match any section heuristics.
    """
    if not file_path.is_file():
        raise ValueError(f"File not found: {file_path}")

    try:
        post = frontmatter.load(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse frontmatter: {str(e)}")

    metadata = post.metadata or {}
    content = post.content or ""

    # 1. Extract title from frontmatter or top H1 tag
    title = metadata.get("title")
    if not title:
        # Fall back to finding the first H1 heading
        for line in content.splitlines():
            match = re.match(r"^#\s+(.+)$", line.strip())
            if match:
                title = match.group(1).strip()
                break

    if not title:
        raise ValueError(
            "Missing a discoverable title (no title in frontmatter and no H1 header)."
        )

    # 2. Section parsing line-by-line using heuristics
    (
        abstract_intro_text,
        methods_text,
        results_discussion_text,
        conclusions_text,
        total_links,
        total_images,
    ) = _segment_markdown_content(content)

    # Construct the SanitisedLiterature Pydantic model
    try:
        sanitized_lit = SanitisedLiterature(
            title=title,
            metadata=metadata,
            abstract_intro=abstract_intro_text,
            methods=methods_text,
            results_discussion=results_discussion_text,
            conclusions=conclusions_text,
        )
    except Exception as e:
        raise ValueError(f"Data model validation failed: {str(e)}")

    return sanitized_lit, total_links, total_images
