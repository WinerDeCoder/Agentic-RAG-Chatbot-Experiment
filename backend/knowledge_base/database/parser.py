from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
from llama_cloud import LlamaCloud


DATABASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = DATABASE_DIR / "raw_data"
PARSED_DATA_DIR = DATABASE_DIR / "parsed_data"
DEFAULT_SKIP_EXISTING = True


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "document"


def first_words(text: str, count: int = 100) -> str:
    words = text.split()
    return " ".join(words[:count])


def extended_path(path: Path) -> str:
    path_str = str(path)
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str.lstrip("\\")
    return "\\\\?\\" + path_str


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(extended_path(path), "w", encoding="utf-8") as handle:
        handle.write(content)


def write_bytes_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(extended_path(path), "wb") as handle:
        handle.write(content)


def load_api_key() -> str:
    load_dotenv()
    load_dotenv(RAW_DATA_DIR / ".env")

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. Add it to the environment or a .env file."
        )

    os.environ["LLAMA_CLOUD_API_KEY"] = api_key
    return api_key


def find_pdf_files(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.rglob("*.pdf") if path.is_file())


def download_image(url: str, destination: Path) -> bool:
    if not url:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        write_bytes_file(destination, response.read())
    return True


def rewrite_image_paths(markdown_text: str, image_records: list[dict]) -> str:
    updated = markdown_text
    for record in image_records:
        updated = updated.replace(f"({record['filename']})", f"({record['local_path']})")
    return updated


def build_markdown(result, image_records: list[dict]) -> str:
    pages: list[str] = []
    for page_index, page in enumerate(result.markdown.pages, start=1):
        page_markdown = rewrite_image_paths(page.markdown, image_records)
        pages.append(f"<!-- page {page_index} -->\n\n{page_markdown}".strip())

    image_lines = ["## Extracted Images"]
    if image_records:
        for record in image_records:
            image_lines.append(f"### {record['filename']}")
            image_lines.append(f"- local_path: {record['local_path']}")
            image_lines.append(f"- source_url: {record['source_url']}")
            image_lines.append(f"- category: {record['category']}")
            image_lines.append(f"- content_type: {record['content_type']}")
            image_lines.append(f"- bbox: {record['bbox']}")
            image_lines.append(f"![{record['filename']}]({record['local_path']})")
            image_lines.append("")
    else:
        image_lines.append("No extracted images were returned by LlamaParse.")

    return "\n\n".join(pages + ["\n".join(image_lines)])


def output_paths_for(pdf_path: Path, source_root: Path, target_root: Path) -> dict[str, Path]:
    relative_pdf = pdf_path.relative_to(source_root)
    relative_parent = relative_pdf.parent
    parent_dir = target_root / relative_parent
    doc_name = safe_name(pdf_path.stem)
    document_dir = parent_dir / doc_name

    return {
        "parent_dir": parent_dir,
        "document_dir": document_dir,
        "markdown_path": document_dir / f"{doc_name}.md",
        "metadata_path": document_dir / f"{doc_name}.images.json",
        "images_dir": document_dir / "images",
    }


def build_skipped_result(pdf_path: Path, source_root: Path, target_root: Path) -> dict:
    paths = output_paths_for(pdf_path, source_root, target_root)
    return {
        "source_pdf": str(pdf_path.relative_to(source_root)).replace("\\", "/"),
        "status": "skipped",
        "reason": "markdown_exists",
        "markdown_path": str(paths["markdown_path"]),
        "metadata_path": str(paths["metadata_path"]),
        "images_dir": str(paths["images_dir"]),
    }


def parse_pdf(client: LlamaCloud, pdf_path: Path, source_root: Path, target_root: Path) -> dict:
    paths = output_paths_for(pdf_path, source_root, target_root)
    paths["document_dir"].mkdir(parents=True, exist_ok=True)
    paths["images_dir"].mkdir(parents=True, exist_ok=True)

    upload = client.files.create(file=pdf_path, purpose="parse")
    result = client.parsing.parse(
        file_id=upload.id,
        tier="cost_effective",
        version="latest",
        expand=["markdown", "images_content_metadata"],
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True,
                    "compact_markdown_tables": True,
                },
                "annotate_links": True,
                "inline_images": True,
            },
            "images_to_save": ["embedded"],
        },
    )

    image_records: list[dict] = []
    images_metadata = result.images_content_metadata.images if result.images_content_metadata else []
    for image in images_metadata:
        image_path = paths["images_dir"] / image.filename
        downloaded = False
        error_message = None
        try:
            downloaded = download_image(image.presigned_url, image_path)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        image_records.append(
            {
                "filename": image.filename,
                "local_path": str(image_path.relative_to(paths['document_dir'])).replace("\\", "/"),
                "source_url": image.presigned_url,
                "category": image.category,
                "content_type": image.content_type,
                "bbox": image.bbox.model_dump() if image.bbox else None,
                "downloaded": downloaded,
                "download_error": error_message,
            }
        )

    markdown_output = build_markdown(result, image_records)
    write_text_file(paths["markdown_path"], markdown_output)
    write_text_file(paths["metadata_path"], json.dumps(image_records, indent=2))

    return {
        "source_pdf": str(pdf_path.relative_to(source_root)).replace("\\", "/"),
        "status": "parsed",
        "output_name": paths["document_dir"].name,
        "markdown_path": str(paths["markdown_path"]),
        "metadata_path": str(paths["metadata_path"]),
        "images_dir": str(paths["images_dir"]),
        "image_count": len(image_records),
        "downloaded_images": sum(1 for item in image_records if item["downloaded"]),
        "markdown_chars": len(markdown_output),
        "preview": first_words(markdown_output, count=100),
    }


def parse_all_pdfs(
    source_root: Path = RAW_DATA_DIR,
    target_root: Path = PARSED_DATA_DIR,
    skip_existing: bool = DEFAULT_SKIP_EXISTING,
) -> list[dict]:
    load_api_key()
    client = LlamaCloud()

    pdf_files = find_pdf_files(source_root)
    results: list[dict] = []
    for pdf_path in pdf_files:
        relative_pdf = str(pdf_path.relative_to(source_root)).replace("\\", "/")
        paths = output_paths_for(pdf_path, source_root, target_root)

        if skip_existing and paths["markdown_path"].exists():
            print(f"Skipping {relative_pdf} (markdown exists)")
            results.append(build_skipped_result(pdf_path, source_root, target_root))
            continue

        print(f"Parsing {relative_pdf}")
        try:
            result = parse_pdf(client, pdf_path, source_root, target_root)
            results.append(result)
            print(f"Parsed {relative_pdf} -> {result['output_name']}")
            print(f"Preview (first 100 words): {result['preview']}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            results.append(
                {
                    "source_pdf": relative_pdf,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "markdown_path": str(paths["markdown_path"]),
                    "metadata_path": str(paths["metadata_path"]),
                    "images_dir": str(paths["images_dir"]),
                }
            )
            print(f"Failed {relative_pdf}: {type(exc).__name__}: {exc}")

    return results


def build_final_report(results: list[dict], skip_existing: bool) -> dict:
    summary = {
        "total": len(results),
        "parsed": sum(1 for item in results if item["status"] == "parsed"),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
    }
    return {
        "config": {
            "skip_existing": skip_existing,
        },
        "summary": summary,
        "files": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDFs from raw_data into parsed_data.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=RAW_DATA_DIR,
        help="Root directory containing raw PDFs.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=PARSED_DATA_DIR,
        help="Root directory where parsed outputs will be written.",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=DEFAULT_SKIP_EXISTING,
        help="Skip files whose markdown output already exists.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Reparse files even if markdown output already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = parse_all_pdfs(
        source_root=args.source_root,
        target_root=args.target_root,
        skip_existing=args.skip_existing,
    )
    report = build_final_report(results, skip_existing=args.skip_existing)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()