import re
from typing import Tuple

# --- Heuristic Sanitisation Patterns ---
IMAGE_BLOCK_REGEX = re.compile(
    r"!\[.*?\]\(.*?\)"
)  # Targets and completely drops markdown images
INLINE_URL_REGEX = re.compile(
    r"\[([^\]]+)\]\((https?://[^\s)]+)\)"
)  # Extracts anchor text from reference links


def sanitise_line(line: str) -> Tuple[str, int, int]:
    """Sanitise a single line of markdown text, preserving tables untouched.

    Returns:
        A tuple of (sanitised_line, link_count, image_count).
    """
    # Check if the line is part of a markdown data table.
    # Data tables use the pipe '|' character as layout structure.
    # These must pass through completely untouched.
    if line.strip().startswith("|"):
        return line, 0, 0

    # Drop markdown images completely
    sanitised_line, image_count = IMAGE_BLOCK_REGEX.subn("", line)

    # Extract anchor text from reference links
    sanitised_line, link_count = INLINE_URL_REGEX.subn(r"\1", sanitised_line)

    return sanitised_line, link_count, image_count


def sanitise_text(text: str) -> Tuple[str, int, int]:
    """Sanitise block of markdown text line-by-line, tracking stats.

    Returns:
        A tuple of (sanitised_text, total_links, total_images).
    """
    lines = text.splitlines()
    sanitised_lines = []
    total_links = 0
    total_images = 0

    for line in lines:
        s_line, lnk_cnt, img_cnt = sanitise_line(line)
        sanitised_lines.append(s_line)
        total_links += lnk_cnt
        total_images += img_cnt

    return "\n".join(sanitised_lines), total_links, total_images
