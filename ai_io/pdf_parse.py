"""
기본 PDF 파서 (PyPDFLoader 기반).

역할:
- 페이지별 텍스트를 추출하여 LangChain `Document`로 반환
- 메타데이터에 원본 파일/페이지/추출 방법을 명시해 이후 추적 용이하게 함
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_pages_parse(pdf_path: str, doc_type: str) -> List[Document]:
    """PyPDFLoader로 페이지별 문서를 로드하고 표준 메타데이터를 부여."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for d in pages:
        d.metadata = d.metadata or {}
        d.metadata.update(
            {
                "doc_type": doc_type,  # 예: "paper" | "report" | "docs"
                "source": os.path.abspath(pdf_path),  # 전체 경로
                "source_file": os.path.basename(pdf_path),  # 파일명만
                "page": d.metadata.get("page", None),  # 원본 페이지 번호
                "extract_method": "pypdf_parse",  # 추출 도구 정보
            }
        )
    return pages
