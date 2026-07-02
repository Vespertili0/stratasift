"""StrataSift core logic modules for models, parsing, sanitisation, and lifecycle.

All modules apply British spelling rules for terminal messages and logs.
"""

from stratasift.core.models import SanitisedLiterature, AtomicInsight
from stratasift.core.sanitiser import sanitise_line, sanitise_text
from stratasift.core.parser import parse_markdown_file
from stratasift.core.lifecycle import (
    quarantine_file,
    shelve_file,
    log_quarantine_warning,
)

__all__ = [
    "SanitisedLiterature",
    "AtomicInsight",
    "sanitise_line",
    "sanitise_text",
    "parse_markdown_file",
    "quarantine_file",
    "shelve_file",
    "log_quarantine_warning",
]
