from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


DATABASE_DIR = Path(__file__).resolve().parent
PARSED_DATA_DIR = DATABASE_DIR / "parsed_data"
CLEANED_DATA_DIR = DATABASE_DIR / "cleaned_data"
CLEANED_DATA_REMOVE_IMAGE_DIR = DATABASE_DIR / "cleaned_data_remove_image"

IMAGE_TAG_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
SOURCE_URL_PATTERN = re.compile(r"^\s*- source_url:\s*.*$")
PERNOD_PATTERN = re.compile(r"pernod\s+ricard", re.IGNORECASE)


def extended_path(path: Path) -> str:
    path_str = str(path)
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")
    return "\\\\?\\" + path_str


def ensure_parent_dir(path: Path) -> None:
    directory = path.parent
    if directory.exists():
        return
    os.makedirs(extended_path(directory), exist_ok=True)


def read_text_file(path: Path) -> str:
    with open(extended_path(path), "r", encoding="utf-8") as handle:
        return handle.read()


def write_text_file(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    with open(extended_path(path), "w", encoding="utf-8") as handle:
        handle.write(content)


def read_bytes_file(path: Path) -> bytes:
    with open(extended_path(path), "rb") as handle:
        return handle.read()


def write_bytes_file(path: Path, content: bytes) -> None:
    ensure_parent_dir(path)
    with open(extended_path(path), "wb") as handle:
        handle.write(content)


def copy_file(source: Path, destination: Path) -> None:
    write_bytes_file(destination, read_bytes_file(source))


def image_name_from_line(line: str) -> str | None:
    match = IMAGE_TAG_PATTERN.search(line)
    if not match:
        return None
    return Path(match.group("path")).name


def find_removed_image_names(lines: list[str]) -> set[str]:
    removed: set[str] = set()
    for line in lines:
        for match in IMAGE_TAG_PATTERN.finditer(line):
            alt_text = match.group("alt")
            if PERNOD_PATTERN.search(alt_text):
                removed.add(Path(match.group("path")).name)
    return removed


def remove_pernod_tags_from_line(line: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        alt_text = match.group("alt")
        if PERNOD_PATTERN.search(alt_text):
            return ""
        return match.group(0)

    updated = IMAGE_TAG_PATTERN.sub(replacer, line)
    return re.sub(r"\s{2,}", " ", updated).strip()


def clean_markdown_text(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    removed_image_names = find_removed_image_names(lines)

    cleaned_lines: list[str] = []
    in_extracted_images = False
    skipping_image_block = False

    for line in lines:
        stripped = line.strip()

        if stripped == "## Extracted Images":
            in_extracted_images = True
            skipping_image_block = False
            cleaned_lines.append(line)
            continue

        if in_extracted_images and stripped.startswith("### "):
            image_name = stripped.removeprefix("### ").strip()
            skipping_image_block = image_name in removed_image_names
            if not skipping_image_block:
                cleaned_lines.append(line)
            continue

        if in_extracted_images and skipping_image_block:
            continue

        updated_line = remove_pernod_tags_from_line(line)
        if not updated_line.strip():
            continue

        if SOURCE_URL_PATTERN.match(updated_line):
            cleaned_lines.append("- source_url:")
            continue

        cleaned_lines.append(updated_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    if markdown_text.endswith("\n"):
        cleaned_text += "\n"
    return cleaned_text


def remove_all_image_tags_from_line(line: str) -> str:
    updated = IMAGE_TAG_PATTERN.sub("", line)
    return re.sub(r"\s{2,}", " ", updated).strip()


def clean_markdown_text_remove_images(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    cleaned_lines: list[str] = []
    in_extracted_images = False

    for line in lines:
        stripped = line.strip()

        if stripped == "## Extracted Images":
            in_extracted_images = True
            continue

        if in_extracted_images:
            continue

        updated_line = remove_all_image_tags_from_line(line)
        if not updated_line.strip():
            continue

        cleaned_lines.append(updated_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    if markdown_text.endswith("\n"):
        cleaned_text += "\n"
    return cleaned_text


def clean_markdown_file(source: Path, destination: Path) -> dict:
    original_text = read_text_file(source)
    cleaned_text = clean_markdown_text(original_text)
    write_text_file(destination, cleaned_text)

    return {
        "source": str(source),
        "destination": str(destination),
        "changed": cleaned_text != original_text,
        "removed_pernod_tags": original_text.count("Pernod Ricard") - cleaned_text.count("Pernod Ricard"),
        "cleared_source_urls": original_text.count("- source_url:") if "- source_url:" in original_text else 0,
    }


def clean_markdown_file_remove_images(source: Path, destination: Path) -> dict:
    original_text = read_text_file(source)
    cleaned_text = clean_markdown_text_remove_images(original_text)
    write_text_file(destination, cleaned_text)

    return {
        "source": str(source),
        "destination": str(destination),
        "changed": cleaned_text != original_text,
        "removed_image_tags": original_text.count("![") - cleaned_text.count("!["),
        "removed_extracted_images_section": "## Extracted Images" in original_text,
    }


def clean_all(source_root: Path = PARSED_DATA_DIR, target_root: Path = CLEANED_DATA_DIR) -> dict:
    markdown_results: list[dict] = []
    copied_files = 0

    for source_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source_root)
        destination_path = target_root / relative_path

        if source_path.suffix.lower() == ".md":
            markdown_results.append(clean_markdown_file(source_path, destination_path))
        else:
            copy_file(source_path, destination_path)
            copied_files += 1

    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "mode": "default",
        "markdown_files": len(markdown_results),
        "copied_non_markdown_files": copied_files,
        "files": markdown_results,
    }


def clean_all_remove_images(
    source_root: Path = PARSED_DATA_DIR,
    target_root: Path = CLEANED_DATA_REMOVE_IMAGE_DIR,
) -> dict:
    markdown_results: list[dict] = []
    copied_files = 0

    for source_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source_root)
        destination_path = target_root / relative_path

        if source_path.suffix.lower() == ".md":
            markdown_results.append(
                clean_markdown_file_remove_images(source_path, destination_path)
            )
        else:
            copy_file(source_path, destination_path)
            copied_files += 1

    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "mode": "remove_all_images",
        "markdown_files": len(markdown_results),
        "copied_non_markdown_files": copied_files,
        "files": markdown_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean parsed markdown into cleaned_data while preserving mirrored assets."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PARSED_DATA_DIR,
        help="Root directory containing parsed markdown and assets.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=CLEANED_DATA_DIR,
        help="Root directory where cleaned outputs will be written.",
    )
    parser.add_argument(
        "--remove-images",
        action="store_true",
        help="Remove all markdown image tags and the extracted images section.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.remove_images:
        target_root = (
            args.target_root
            if args.target_root != CLEANED_DATA_DIR
            else CLEANED_DATA_REMOVE_IMAGE_DIR
        )
        report = clean_all_remove_images(
            source_root=args.source_root,
            target_root=target_root,
        )
    else:
        report = clean_all(source_root=args.source_root, target_root=args.target_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()