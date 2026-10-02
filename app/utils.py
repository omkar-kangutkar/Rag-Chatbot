"""Document loading and text cleaning."""

import hashlib
import re
from pathlib import Path
from typing import List

from langchain.schema import Document


def load_documents_from_dir(data_dir: str) -> List[Document]:
    """Load TXT, PDF and Markdown files from a directory."""
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        UnstructuredMarkdownLoader,
    )

    loaders = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
    }

    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist."
        )

    files_found = list(data_path.iterdir())
    if not files_found:
        raise ValueError(
            f"No files found in '{data_dir}'. Add .txt, .pdf, or .md files."
        )

    for file_path in files_found:
        ext = file_path.suffix.lower()
        if not file_path.is_file() or ext not in loaders:
            print(f"  [skip] Unsupported entry: {file_path.name}")
            continue

        print(f"  [load] {file_path.name}")
        try:
            loader = loaders[ext](str(file_path))
            docs = loader.load()
            document_id = hashlib.sha256(
                file_path.read_bytes()
            ).hexdigest()

            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["document_id"] = document_id

            documents.extend(docs)
        except Exception as exc:
            print(f"  [error] Failed to load {file_path.name}: {exc}")

    print(f"\nLoaded {len(documents)} document(s) from {data_dir}")
    return documents


def clean_text(text: str) -> str:
    """Collapse whitespace and remove non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()