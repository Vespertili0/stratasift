import re
from pathlib import Path
from typing import Tuple, Set, Optional, List, Dict
import frontmatter

from stratasift.core.models import SanitisedLiterature
from stratasift.core.sanitiser import sanitise_line

MIN_WORD_COUNT = 300

def _segment_markdown_content(
    content: str,
) -> Tuple[str, List[str], List[Dict[str, str]], int, int]:
    """Segment markdown content into dynamic sections and sanitise them."""
    abstract_intro_lines = []
    
    # Store tuples of (header, lines_list)
    raw_sections: List[Tuple[str, List[str]]] = []
    
    current_header: Optional[str] = None
    current_lines: List[str] = []

    total_links = 0
    total_images = 0

    for line in content.splitlines():
        # Identify heading (H1 and H2), avoid matching markdown table markers or frontmatter markers if any bleed through
        match = re.match(r"^(##?)\s+(.+)$", line)
        if match:
            # We found a new heading
            new_header = match.group(2).strip()
            
            # Save previous section if exists
            if current_header is not None:
                raw_sections.append((current_header, current_lines))
            elif current_lines and not abstract_intro_lines:
                # If we had content before any header, it's the abstract/intro
                abstract_intro_lines.extend(current_lines)
            elif current_lines:
                # If somehow abstract_intro is already set, just append
                abstract_intro_lines.extend(current_lines)
                
            current_header = new_header
            current_lines = []
        else:
            # Sanitise the line (preserves tables, strips images and reference links)
            s_line, lnk_count, img_count = sanitise_line(line)
            total_links += lnk_count
            total_images += img_count
            
            if current_header is not None:
                current_lines.append(s_line)
            else:
                abstract_intro_lines.append(s_line)
                
    # Add the last section
    if current_header is not None:
        raw_sections.append((current_header, current_lines))
    elif current_lines:
        abstract_intro_lines.extend(current_lines)
        
    abstract_intro_text = "\n".join(abstract_intro_lines).strip()
    
    if not abstract_intro_text and raw_sections:
        # Fallback: if there was no leading text before the first header, 
        # take the first section that has content as abstract/intro
        for header, lines in raw_sections:
            text = "\n".join(lines).strip()
            if text:
                abstract_intro_text = text
                break
        
    # Build TOC and chunk sections
    toc = []
    section_chunks = []
    
    if raw_sections:
        # Pre-process into header and text string
        processed_sections = []
        for header, lines in raw_sections:
            toc.append(header)
            text = "\n".join(lines).strip()
            processed_sections.append({"header": header, "content": text})
            
        # Threshold-based Section Concatenator
        for current in processed_sections:
            if not section_chunks:
                section_chunks.append(current)
                continue
            
            # Check previous chunk word count
            prev = section_chunks[-1]
            prev_word_count = len(prev["content"].split())
            
            if prev_word_count < MIN_WORD_COUNT:
                # Merge current into previous (forward merge from perspective of prev)
                merged_header = f"{prev['header']} / {current['header']}"
                merged_content = f"{prev['content']}\n\n## {current['header']}\n{current['content']}"
                section_chunks[-1] = {"header": merged_header, "content": merged_content}
            else:
                section_chunks.append(current)
                
        # Tail check for boundary safeguard
        if len(section_chunks) > 1:
            last = section_chunks[-1]
            if len(last["content"].split()) < MIN_WORD_COUNT:
                last = section_chunks.pop()
                prev = section_chunks[-1]
                merged_header = f"{prev['header']} / {last['header']}"
                merged_content = f"{prev['content']}\n\n## {last['header']}\n{last['content']}"
                section_chunks[-1] = {"header": merged_header, "content": merged_content}
                
    if not toc:
        raise ValueError("Failed to identify core document structural layout sections.")
                
    if not abstract_intro_text:
        raise ValueError(
            "Failed to identify core document structural layout sections (Abstract/Introduction segment is empty)."
        )

    return (
        abstract_intro_text,
        toc,
        section_chunks,
        total_links,
        total_images,
    )


def parse_markdown_file(file_path: Path) -> Tuple[SanitisedLiterature, int, int]:
    """Parse a markdown file using dynamic heading matching and sanitise its content.

    Returns:
        A tuple of (SanitisedLiterature, total_links_sanitised, total_images_stripped).

    Raises:
        ValueError: If the file is malformed, lacks a discoverable title, or
                    fails to match any sections.
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

    # 2. Section parsing line-by-line
    (
        abstract_intro_text,
        toc,
        section_chunks,
        total_links,
        total_images,
    ) = _segment_markdown_content(content)

    # Construct the SanitisedLiterature Pydantic model
    try:
        sanitized_lit = SanitisedLiterature(
            title=title,
            metadata=metadata,
            abstract_intro=abstract_intro_text,
            toc=toc,
            section_chunks=section_chunks,
        )
    except Exception as e:
        raise ValueError(f"Data model validation failed: {str(e)}")

    return sanitized_lit, total_links, total_images
