"""
Upstage Document Digitization API 파서.

역할:
- Upstage API로 PDF를 OCR/구조화하여 페이지별 텍스트로 변환
- 테이블/리스트/코드/이미지 캡션 등 구조를 최대한 보존
- 캐시를 이용해 반복 호출 비용을 줄임

주의:
- 이미지 기반 슬라이드가 많은 발표자료(paper.pdf)에 적합
- Upstage가 페이지 수를 적게 반환하는 경우 PyPDF로 폴백하도록 예외를 발생
"""
import os
import json
import hashlib
from typing import List, Dict, Any, Optional

import requests
from langchain_core.documents import Document


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_text(x: Any) -> str:
    """
    어떤 타입이 와도 안전하게 문자열로 변환.
    - Upstage 응답의 content/text/rows/cell은 dict/list일 수 있으므로 안전 변환 필요
    - ensure_ascii=False 로 JSON 직렬화 시 한글을 그대로 보존
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def upstage_document_parse(
    filename: str,
    api_key: str,
    url: str,
    ocr: str = "force",
    base64_encoding: str = "['table']",
    model: str = "document-parse",
    timeout_sec: int = 300,
) -> Dict[str, Any]:
    """Upstage Document Digitization API 호출.

    파일을 멀티파트로 업로드하여 OCR/구조화 결과(JSON)를 반환.
    ocr/base64_encoding/model 파라미터는 Upstage 문서에 따름.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"ocr": ocr, "base64_encoding": base64_encoding, "model": model}

    with open(filename, "rb") as f:
        files = {"document": f}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_sec)
        resp.raise_for_status()
        return resp.json()


def upstage_parse_to_documents(
    pdf_path: str,
    doc_type: str,
    api_key: str,
    url: str,
    ocr: str = "force",
    base64_encoding: str = "['table']",
    model: str = "document-parse",
    cache_dir: Optional[str] = None,
) -> List[Document]:
    """Upstage 결과 JSON을 받아 페이지 단위 Document 리스트로 변환.

    - paragraph/text/heading/title: 내용/레벨 기반으로 정리
    - table: rows를 Markdown 테이블로 직렬화
    - figure/image: 캡션/설명 텍스트만 추출
    - list: 각 item을 "- " 접두로 나열
    - code: 언어 힌트와 함께 fenced code block으로 래핑
    - 기타: content/text fallback
    """

    # -------------------------
    # 1) 캐시 로드/저장
    # -------------------------
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    cached: Optional[Dict[str, Any]] = None
    cache_path: Optional[str] = None

    if cache_dir:
        sha1 = _file_sha1(pdf_path)
        cache_path = os.path.join(cache_dir, f"{sha1}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

    if cached is None:
        result = upstage_document_parse(
            filename=pdf_path,
            api_key=api_key,
            url=url,
            ocr=ocr,
            base64_encoding=base64_encoding,
            model=model,
        )
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
    else:
        result = cached

    # -------------------------
    # 2) 요소(블록) 포맷터
    # -------------------------
    def format_element(elem: Any) -> str:
        if not isinstance(elem, dict):
            return _to_text(elem).strip()

        elem_type = _to_text(elem.get("type", "")).lower().strip()

        # 텍스트/문단/제목: heading level이 있으면 마크다운 제목으로 변환
        if elem_type in ("paragraph", "text", "heading", "title"):
            text = _to_text(elem.get("content") or elem.get("text") or "")
            level = elem.get("level", 0)
            if isinstance(level, int) and level > 0:
                return f"{'#' * level} {text}".strip()
            return text.strip()

        # 테이블: Markdown 직렬화 (첫 행을 헤더로 간주해 구분선 추가)
        if elem_type == "table":
            rows = elem.get("rows") or elem.get("content") or []
            if not isinstance(rows, list) or not rows:
                return ""

            lines: List[str] = []
            header_len = None

            for i, row in enumerate(rows):
                if not isinstance(row, list):
                    row = [row]
                row_cells = [_to_text(cell) for cell in row]
                row_text = " | ".join(row_cells)
                lines.append(row_text)

                if i == 0:
                    header_len = len(row_cells)
                    lines.append(" | ".join(["---"] * header_len))

            return "\n".join(lines).strip()

        # 이미지/그림: 캡션이 있으면 함께 표기, 실제 이미지 데이터는 제외
        if elem_type in ("figure", "image", "img"):
            caption = _to_text(elem.get("caption") or elem.get("description") or elem.get("text") or "")
            caption = caption.strip()
            return f"[이미지] {caption}".strip() if caption else "[이미지]"

        # 리스트: items를 문자열로 변환해 개조식으로 정리
        if elem_type == "list":
            items = elem.get("items") or elem.get("content") or []
            if not isinstance(items, list):
                items = [items]

            lines = []
            for item in items:
                if isinstance(item, dict):
                    item_text = _to_text(item.get("content") or item.get("text") or item).strip()
                else:
                    item_text = _to_text(item).strip()
                if item_text:
                    lines.append(f"- {item_text}")
            return "\n".join(lines).strip()

        # 코드: 언어 힌트와 함께 fenced code block 구성
        if elem_type == "code":
            content = _to_text(elem.get("content") or "").strip()
            language = _to_text(elem.get("language") or "").strip()
            return f"```{language}\n{content}\n```".strip()

        # 기본 fallback
        return _to_text(elem.get("content") or elem.get("text") or "").strip()

    # -------------------------
    # 3) 페이지 텍스트 구성
    # -------------------------
    def extract_page_text(page: Any) -> str:
        parts: List[str] = []

        # elements가 있으면 가장 우선: 구조 요소 순회
        if isinstance(page, dict) and isinstance(page.get("elements"), list):
            for elem in page["elements"]:
                formatted = format_element(elem)
                if formatted:
                    parts.append(formatted)

        # content/text가 dict/list여도 안전하게 문자열화하여 추가
        if isinstance(page, dict) and "content" in page:
            content = _to_text(page["content"]).strip()
            if content:
                parts.append(content)

        if isinstance(page, dict) and "text" in page:
            text = _to_text(page["text"]).strip()
            if text:
                parts.append(text)

        # 마지막 안전망: page 자체를 문자열로 변환해 추가
        if not parts:
            fallback = _to_text(page).strip()
            if fallback:
                parts.append(fallback)

        return "\n\n".join(parts).strip()

    # -------------------------
    # 4) 응답 구조에서 pages 추출 (Upstage 응답 스키마 대응)
    # -------------------------
    def _elem_page_index(elem: Any) -> Optional[int]:
        if not isinstance(elem, dict):
            return None
        for k in ("page", "pageIndex", "page_index", "page_id"):
            v = elem.get(k)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
        return None


    # --------
    # pages가 없고 elements만 있는 응답이면 elements를 page별로 묶기
    # --------
    if (not isinstance(result.get("pages"), list)) and isinstance(result.get("elements"), list):
        page_map: Dict[int, List[Any]] = {}
        for elem in result["elements"]:
            pidx = _elem_page_index(elem)
            if pidx is None:
                pidx = 0
            page_map.setdefault(pidx, []).append(elem)

        pages_text = []
        for pidx in sorted(page_map.keys()):
            fake_page = {"elements": page_map[pidx]}
            pages_text.append(extract_page_text(fake_page))
    else:
        pages_text = []
        if isinstance(result.get("pages"), list):
            for page in result["pages"]:
                pages_text.append(extract_page_text(page))
        elif isinstance(result.get("document"), dict) and isinstance(result["document"].get("pages"), list):
            for page in result["document"]["pages"]:
                pages_text.append(extract_page_text(page))
        else:
            pages_text = [extract_page_text(result)]


    # -------------------------
    # 4.5) 페이지 수 검증 및 안전한 대체: Upstage 페이지 수가 적으면 PyPDF 폴백 유도
    # -------------------------
    try:
        from pypdf import PdfReader  # lightweight page count
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
    except Exception:
        total_pages = None

    # If the original PDF has multiple pages but Upstage returned fewer,
    # signal caller to fallback to PyPDF by raising an exception.
    if total_pages and total_pages > 1 and len(pages_text) < total_pages:
        raise RuntimeError(
            f"Upstage parse returned {len(pages_text)} pages, but PDF has {total_pages}."
        )

    # -------------------------
    # 5) Document 생성: page_content와 메타데이터 설정
    # -------------------------
    docs: List[Document] = []
    for i, text in enumerate(pages_text):
        safe_text = _to_text(text).strip()
        docs.append(
            Document(
                page_content=safe_text,
                metadata={
                    "doc_type": doc_type,
                    "source": os.path.abspath(pdf_path),
                    "source_file": os.path.basename(pdf_path),
                    "page": i,
                    "extract_method": "upstage_document_parse",
                    "ocr": ocr,
                    "model": model,
                },
            )
        )

    return docs
