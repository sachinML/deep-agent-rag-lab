
from pathlib import Path
from hashlib import sha256
import json

from docling.document_converter import DocumentConverter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

HEADERS = [
    ("#", "heading_1"),
    ("##", "heading_2"),
    ("###", "heading_3"),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)


def load_document(path: Path, converter: DocumentConverter) -> str:
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        result = converter.convert(str(path))
        return result.document.export_to_markdown()

    raise ValueError(f"Unsupported document type: {path}")



def chunk_document(path: Path, text: str) -> list[dict]:
    document_id = sha256(
        str(path.relative_to(RAW_DIR)).encode("utf-8")
    ).hexdigest()[:16]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=False,
    )

    sections = header_splitter.split_text(text)
    chunks = []

    for section in sections:
        pieces = splitter.split_text(section.page_content)

        for content in pieces:
            content = content.strip()

            if not content:
                continue

            # Skip fragments containing only Markdown headings.
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]

            if lines and all(
                line.startswith("#") for line in lines
            ):
                continue

            chunk_number = len(chunks) + 1

            chunks.append({
                "chunk_id": f"{document_id}-{chunk_number:04d}",
                "document_id": document_id,
                "title": path.stem,
                "content": content,
                "metadata": {
                    "source": str(path),
                    "section": section.metadata,
                    "chunk_index": chunk_number,
                },
            })

    return chunks



def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
    )

    converter = DocumentConverter()
    all_chunks = []

    for path in paths:
        print(f"[INGEST] Processing {path}")

        text = load_document(path, converter)

        if not text.strip():
            print(f"[WARNING] Empty document: {path}")
            continue

        chunks = chunk_document(path, text)
        all_chunks.extend(chunks)

        print(f"[CHUNKS] {len(chunks)} created")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[DONE] {len(all_chunks)} chunks saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
