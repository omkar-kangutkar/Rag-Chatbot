import os
import re
from pathlib import Path
from typing import List

from langchain.schema import Document


def load_documents_from_dir(data_dir: str) -> List[Document]:
    """
    Load all supported files from a directory.
    Supports: .txt, .pdf, .md
    Returns a list of LangChain Document objects.
    """
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
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    files_found = list(data_path.iterdir())
    if not files_found:
        raise ValueError(f"No files found in '{data_dir}'. Add .txt, .pdf, or .md files.")

    for file_path in files_found:
        ext = file_path.suffix.lower()
        if ext not in loaders:
            print(f"  [skip] Unsupported file type: {file_path.name}")
            continue

        print(f"  [load] {file_path.name}")
        try:
            loader_cls = loaders[ext]
            loader = loader_cls(str(file_path))
            docs = loader.load()

            # Tag each doc with its source filename
            for doc in docs:
                doc.metadata["source"] = file_path.name

            documents.extend(docs)
        except Exception as e:
            print(f"  [error] Failed to load {file_path.name}: {e}")

    print(f"\nLoaded {len(documents)} document(s) from {data_dir}")
    return documents


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Remove excessive whitespace / newlines
    - Strip non-printable characters
    """
    text = re.sub(r"\s+", " ", text)         # collapse whitespace
    text = re.sub(r"[^\x20-\x7E\n]", "", text)  # remove non-printable chars
    return text.strip()
