"""Quick parser smoke-test for docs/*.pdf.
- Loads paper.pdf/report.pdf/docs.pdf
- Tries Upstage parse (if API key present and --use-upstage set), otherwise PyPDF
- Prints page counts and first 120 chars per file
"""

import argparse
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document

from config import Config
from ai_io.pdf_parse import load_pdf_pages_parse
from ai_io.upstage_parse import upstage_parse_to_documents


def list_existing_pdfs(cfg: Config) -> List[str]:
    candidates = [
        os.path.join(cfg.docs_root, "paper.pdf"),
        os.path.join(cfg.docs_root, "report.pdf"),
        os.path.join(cfg.docs_root, "docs.pdf"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def parse_file(path: str, doc_type: str, use_upstage: bool, cfg: Config) -> List[Document]:
    if use_upstage:
        api_key = os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            print(f"[WARN] UPSTAGE_API_KEY missing, falling back to PyPDF: {path}")
            return load_pdf_pages_parse(path, doc_type=doc_type)
        try:
            return upstage_parse_to_documents(
                pdf_path=path,
                doc_type=doc_type,
                api_key=api_key,
                url=os.environ.get("UPSTAGE_API_URL", cfg.upstage_url),
                ocr=cfg.upstage_ocr,
                base64_encoding=cfg.upstage_base64_encoding,
                model=cfg.upstage_model,
                cache_dir=cfg.cache_dir,
            )
        except Exception as e:
            print(f"[WARN] Upstage parse failed, fallback to PyPDF: {e}")
            return load_pdf_pages_parse(path, doc_type=doc_type)
    else:
        return load_pdf_pages_parse(path, doc_type=doc_type)


def main() -> None:
    load_dotenv(override=True)
    cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-upstage", action="store_true", help="try Upstage document-parse first")
    args = parser.parse_args()

    pdfs = list_existing_pdfs(cfg)
    if not pdfs:
        print(f"[ERROR] No PDFs found under {cfg.docs_root} (expected paper.pdf/report.pdf/docs.pdf)")
        return

    print(f"Found {len(pdfs)} PDFs: {pdfs}")

    for path in pdfs:
        doc_type = os.path.splitext(os.path.basename(path))[0]  # paper/report/docs
        docs = parse_file(path, doc_type=doc_type, use_upstage=args.use_upstage, cfg=cfg)
        print("\n" + "=" * 60)
        print(f"{doc_type}: pages={len(docs)} | source={path}")
        for i, d in enumerate(docs, start=1):
            text = (d.page_content or "").strip().replace("\n", " ")
            preview = text[:120] + ("..." if len(text) > 120 else "")
            print(f"- page {i}: len={len(text)} | preview={preview}")
        print("=" * 60)


if __name__ == "__main__":
    main()
