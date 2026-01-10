import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def build_or_load_chroma(
    documents: List[Document],
    embeddings,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    db.persist()
    return db


def format_docs_with_citations(docs: List[Document]) -> str:
    out = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        out.append(
            f"[{i}] ({m.get('doc_type')} | {m.get('source_file')} | page={m.get('page')})\n{d.page_content}"
        )
    return "\n\n".join(out)
