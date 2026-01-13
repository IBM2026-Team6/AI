"""
발표 대본 자동 생성 메인 스크립트.

역할:
- docs 폴더의 report/docs를 벡터 DB로 구축해 RAG 참고자료로 사용
- paper.pdf를 페이지별로 파싱하여 각 페이지의 발표 대본을 생성
- IBM Watsonx 또는 Upstage(Solar) API를 CLI로 선택해 사용
- 옵션에 따라 키워드 추출 기능을 켜거나 끔

실행 예시:
    python main.py --api upstage
    python main.py --api ibm
    python main.py --api upstage --extractor y
    python main.py --api upstage --extractor n
"""
import os
import glob
import argparse
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import Config
from ai_io.pdf_parse import load_pdf_pages_parse
from ai_io.upstage_parse import upstage_parse_to_documents

from rag.splitters import split_docs
from rag.vectorstore import build_or_load_chroma
from generation.chains import build_slide_chain
from generation.postprocess import postprocess_script

# keyword_extractor 임포트
from extractor.keyword_extractor import extract_keywords_per_slide, LLMConfig as KeywordLLMConfig


def list_pdfs(folder: str) -> List[str]:
    """폴더 하위의 모든 PDF 파일 경로를 정렬하여 반환.

    참고: 현재 코드는 paper/report/docs 파일명 고정 사용으로 미사용이나,
    향후 확장 시 유틸리티로 활용 가능.
    """
    return sorted(glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True))


def main():
    """엔트리포인트: 설정 로드, API 초기화, RAG 구축, 대본 생성까지 전체 오케스트레이션."""
    load_dotenv(override=True)  # .env 값을 환경변수로 로드
    cfg = Config()  # 프로젝트 기본 설정값

    # CLI 인자 정의: 사용할 API 제공자 선택 (ibm | upstage)
    parser = argparse.ArgumentParser(description="발표 대본 자동 생성")
    parser.add_argument(
        "--api",
        choices=["ibm", "upstage"],
        default="ibm",
        help="사용할 API 선택: ibm (Watsonx) 또는 upstage (Solar)"
    )
    parser.add_argument(
        "--audience",
        choices=["expert", "general"],
        default="general",
        help="발표 대상 선택: expert(전문가) | general(비전문가)"
    )   
    parser.add_argument(
        "--nonverbal",
        choices=["y", "n"],
        default="n",
        help="비언어적 표현 포함 여부: Yes | no"
    )
    parser.add_argument(
        "--extractor",
        choices=["y", "n"],
        default="n",
        help="키워드 추출 여부: y(실행) | n(건너뛰기, 기본값)"
    )

    args = parser.parse_args()

    print("="*80)
    print("발표 대본 자동 생성 시스템")
    print(f"API: {args.api.upper()}")
    print(f"Audience: {args.audience}")
    print(f"Nonverbal: {args.nonverbal}")
    print(f"Extractor: {args.extractor}")
    print("="*80)

    # API 공통 변수 초기화 (IBM URL은 IBM 모드에서만 설정)
    api_provider = args.api
    IBM_URL = None  # IBM 사용 시에만 설정됨

    # API에 따라 Embeddings/LLM 초기화 분기
    if args.api == "upstage":
        # Upstage Solar API (OpenAI 호환): 공식 OpenAI 클라이언트를 래핑한 임베딩 사용
        UPSTAGE_API_KEY = os.environ["UPSTAGE_API_KEY"]
        
        # Upstage Embeddings: 커스텀 래퍼 사용 (OpenAI 클라이언트 직접 호출)
        from ai_io.upstage_embeddings import UpstageEmbeddings
        embeddings = UpstageEmbeddings(
            api_key=UPSTAGE_API_KEY,
            model="embedding-passage",
            base_url="https://api.upstage.ai/v1",
        )
        
        # Chat 모델 초기화 (Upstage Solar)
        llm = ChatOpenAI(
            model="solar-pro2",
            openai_api_key=UPSTAGE_API_KEY,
            openai_api_base="https://api.upstage.ai/v1",
            max_tokens=1024,
            temperature=0.3,
        )
        
        print("Upstage Solar API 초기화 완료")
        print(f"   - Embeddings: embedding-passage")
        print(f"   - LLM: solar-pro2")
    
    else:
        # IBM Watsonx API: langchain_ibm 모듈은 IBM 모드일 때만 임포트
        from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
        WATSONX_API = os.environ["API_KEY"]
        PROJECT_ID = os.environ["PROJECT_ID"]
        IBM_URL = os.environ["IBM_CLOUD_URL"]
        
        embeddings = WatsonxEmbeddings(
            model_id=cfg.embed_model,
            url=IBM_URL,
            apikey=WATSONX_API,
            project_id=PROJECT_ID,
        )
        
        # Chat 모델 초기화 (IBM Watsonx)
        llm = WatsonxLLM(
            model_id=cfg.llm_model,
            url=IBM_URL,
            apikey=WATSONX_API,
            project_id=PROJECT_ID,
            params={"decoding_method": "greedy", "max_new_tokens": 700, "temperature": 0.2},
        )
        
        print("IBM Watsonx API 초기화 완료")
        print(f"   - Embeddings: {cfg.embed_model}")
        print(f"   - LLM: {cfg.llm_model}")
    
    # Upstage Document Parse API (PDF 파싱용)
    UPSTAGE_DOC_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")
    UPSTAGE_API_URL = os.environ.get(
        "UPSTAGE_API_URL",
        "https://api.upstage.ai/v1/document-digitization"
    )

    # docs 루트 바로 아래에 paper.pdf/report.pdf/docs.pdf가 있는 구조 지원
    paper_pdfs = [os.path.join(cfg.docs_root, "paper.pdf")]
    report_pdfs = [os.path.join(cfg.docs_root, "report.pdf")]
    docs_pdfs = [os.path.join(cfg.docs_root, "docs.pdf")]

    paper_pdfs = [p for p in paper_pdfs if os.path.exists(p)]
    report_pdfs = [p for p in report_pdfs if os.path.exists(p)]
    docs_pdfs = [p for p in docs_pdfs if os.path.exists(p)]

    if not paper_pdfs:
        raise FileNotFoundError(f"paper pdf 없음: {os.path.join(cfg.docs_root, 'paper.pdf')}")

    # -------------------------
    # 1) reference: report+docs -> vectorstore (RAG 참고자료 구축)
    # -------------------------
    ref_pages: List[Document] = []
    for p in report_pdfs:
        ref_pages.extend(load_pdf_pages_parse(p, doc_type="report"))
    for p in docs_pdfs:
        ref_pages.extend(load_pdf_pages_parse(p, doc_type="docs"))

    ref_chunks = split_docs(ref_pages, cfg.chunk_size, cfg.chunk_overlap)

    # 토큰 할당량 초과 시 RAG 없이 LLM만 사용 (retriever=None)
    retriever = None
    try:
        db = build_or_load_chroma(
            documents=ref_chunks,
            embeddings=embeddings,
            persist_dir=cfg.persist_dir,
            collection_name=cfg.collection_ref,
        )
        retriever = db.as_retriever(search_kwargs={"k": cfg.top_k})
        print("[INFO] RAG vectorstore 구축 완료")
    except Exception as e:
        if "token_quota_reached" in str(e) or "403" in str(e):
            print(f"[WARN] 토큰 할당량 초과로 RAG 비활성화됨. LLM만 사용합니다.")
            print(f"       참고 자료 없이 슬라이드 내용만으로 대본을 생성합니다.")
        else:
            print(f"[WARN] Vectorstore 구축 실패 ({e}), RAG 비활성화")
        retriever = None

    # -------------------------
    # 2) paper: Upstage parse -> slide scripts (페이지별 대본 생성)
    # -------------------------
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    slide_chain = build_slide_chain(
        llm,
        retriever,
        audience=args.audience,
        nonverbal=(args.nonverbal == "y")
    )

    for paper_path in paper_pdfs:
        # paper는 Upstage가 훨씬 잘 뽑히는 경우가 많아서 우선 사용
        try:
            slides = upstage_parse_to_documents(
                pdf_path=paper_path,
                doc_type="paper",
                api_key=UPSTAGE_DOC_API_KEY,
                url=UPSTAGE_API_URL,
                ocr="force",
                base64_encoding="['table']",
                model="document-parse",
                cache_dir=cfg.cache_dir,
            )
            print(f"[INFO] Upstage parse 성공: {len(slides)}개 페이지")
        except Exception as e:
            print(f"[WARN] Upstage parse 실패 ({e}), PyPDFLoader로 대체합니다")
            slides = load_pdf_pages_parse(paper_path, doc_type="paper")

        # 모든 slide의 page_content가 str인지 확인 및 정제
        for slide in slides:
            if not isinstance(slide.page_content, str):
                slide.page_content = str(slide.page_content or "")
            slide.page_content = slide.page_content.strip()

        base = os.path.splitext(os.path.basename(paper_path))[0]
        out_path = os.path.join(cfg.out_dir, f"{base}_scripts.md")
        keywords_out_path = os.path.join(cfg.out_dir, f"{base}_keywords.txt")

        # 대본과 키워드를 저장할 컨테이너
        all_scripts = []  # 전체 대본 텍스트 (키워드 추출용)
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# 발표 대본: {base}\n\n")
            f.write(f"- extract: upstage_document_parse (ocr={cfg.upstage_ocr})\n")
            f.write(f"- RAG top_k: {cfg.top_k}\n\n")

            for idx, slide in enumerate(slides, start=1):
                page_no = slide.metadata.get("page")
                slide_text = slide.page_content or ""

                if len(slide_text) < 40:
                    f.write(f"## Slide {idx} (page={page_no})\n\n")
                    f.write("- 슬라이드 대본:\n(슬라이드 텍스트가 거의 추출되지 않았습니다.)\n\n")
                    f.write("- 핵심 메시지 3개:\n- \n- \n- \n\n")
                    f.write("- 예상 질문 2개 + 답변:\nQ1) \nA1) \n\nQ2) \nA2) \n\n---\n\n")
                    continue

                try:
                    raw = slide_chain.invoke(slide_text)
                    cooked = postprocess_script(raw)
                    # 키워드 추출용으로 대본 수집
                    all_scripts.append(f"Part {idx}: Slide {idx} ({page_no})\n{cooked}")
                except Exception as e:
                    error_msg = str(e)
                    if "token_quota_reached" in error_msg or "403" in error_msg or "quota" in error_msg.lower():
                        # API별 토큰 초과 메시지
                        print(f"\n{api_provider.upper()} API 토큰 할당량 초과")
                        print("="*80)
                        print("해결 방법:")
                        if api_provider == "ibm":
                            print("1. IBM Cloud 대시보드에서 토큰 할당량 확인/증가")
                            print("   https://cloud.ibm.com/watsonx")
                            print(f"2. 다른 지역(region) 사용 (현재: {IBM_URL})")
                            print("3. 무료 tier 리셋 대기 (시간/일 단위 제한)")
                            print("4. Upstage API로 전환: python main.py --api upstage")
                        else:  # upstage
                            print("1. Upstage 대시보드에서 토큰 할당량 확인")
                            print("   https://console.upstage.ai/")
                            print("2. API 키 재발급 또는 업그레이드")
                            print("3. IBM Watsonx로 전환: python main.py --api ibm")
                        print("="*80)
                        raise RuntimeError(f"{api_provider.upper()} API 토큰 할당량 초과. 위 해결 방법을 참고하세요.")
                    else:
                        raise

                f.write(f"## Slide {idx} (page={page_no})\n\n")
                f.write(cooked)
                f.write("\n\n---\n\n")

        print(f"[OK] {out_path}")

        # -------------------------
        # 3) 키워드 추출 및 저장 (keyword_extractor 사용)
        # -------------------------
        if args.extractor == "y":
            print(f"\n[INFO] 키워드 추출 시작")
            try:
                # 전체 대본을 하나의 스크립트로 병합
                full_script = "\n\n".join(all_scripts)
                
                # 키워드 추출 (다중 패스 5회)
                keywords_result = extract_keywords_per_slide(full_script, KeywordLLMConfig(multi_pass=5))
                
                # 키워드를 정리된 텍스트로 저장
                with open(keywords_out_path, "w", encoding="utf-8") as kf:
                    import datetime
                    kf.write(f"# 핵심 키워드: {base}\n")
                    kf.write(f"생성 날짜: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    kf.write(f"{'='*80}\n\n")
                    
                    for slide_idx, keywords in sorted(keywords_result.items()):
                        kf.write(f"## Slide {slide_idx}\n")
                        kf.write(f"키워드 ({len(keywords)}개):\n")
                        for i, kw in enumerate(keywords, 1):
                            kf.write(f"  {i}. {kw}\n")
                        kf.write("\n")
                
                print(f"[OK] 키워드 저장: {keywords_out_path}")
                for slide_idx, keywords in sorted(keywords_result.items()):
                    print(f"    Slide {slide_idx}: {', '.join(keywords)}")
            except Exception as e:
                print(f"[WARN] 키워드 추출 실패: {e}")
                print(f"       키워드 추출을 건너뜁니다.")
        else:
            print(f"\n[INFO] 키워드 추출 건너뜀 (--extractor n)")


if __name__ == "__main__":
    main()
