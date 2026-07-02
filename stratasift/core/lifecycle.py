import shutil
from pathlib import Path
from datetime import datetime
import frontmatter


def log_quarantine_warning(filename: str, reason: str) -> None:
    """Append a descriptive warning to the local error trace log (stratasift.log).

    Uses British spelling rules for log messages.
    """
    log_path = Path("stratasift.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] WARNING: Failed to parse/validate clip '{filename}'. Reason: {reason}\n"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(log_message)


def quarantine_file(file_path: Path, quarantine_dir: Path, reason: str) -> Path:
    """Move the file cleanly to the quarantine directory to avoid halting processing.

    Appends a warning to the local error trace log.

    Returns:
        The target Path where the file was moved.
    """
    # Ensure the quarantine directory exists
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Define destination path
    dest_path = quarantine_dir / file_path.name

    # Handle filename collision if the file already exists in quarantine
    if dest_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = quarantine_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"

    # Perform physical move operation
    shutil.move(file_path, dest_path)

    # Log the quarantine event with warning reasons
    log_quarantine_warning(file_path.name, reason)

    return dest_path


def shelve_file(file_path: Path, vault_path: Path, relevance_score: float) -> Path:
    """Update file frontmatter with standardised metadata and move to the Shelved repository.

    Returns:
        The target Path where the file was shelved.
    """
    # 1. Update the frontmatter with standardised metadata fields without altering text payload
    try:
        post = frontmatter.load(file_path)
    except Exception:
        # Fallback for files that could not be parsed by frontmatter parser
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
        post = frontmatter.Post(content)

    post.metadata["stratasift_status"] = "shelved"
    post.metadata["relevance_score"] = relevance_score
    post.metadata["evaluated_at"] = datetime.now().strftime("%Y-%m-%d")

    with file_path.open("w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))

    # 2. Handle physical file transport to <vault_path>/Shelved/
    shelved_dir = vault_path / "Shelved"
    shelved_dir.mkdir(parents=True, exist_ok=True)

    dest_path = shelved_dir / file_path.name

    # Handle filename collision in shelving directory
    if dest_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = shelved_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"

    shutil.move(file_path, dest_path)

    return dest_path
