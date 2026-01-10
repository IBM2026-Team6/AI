from dataclasses import dataclass

@dataclass
class Config:
    docs_root: str = "./docs"
    out_dir: str = "./outputs"

    # chroma
    persist_dir: str = "./chroma_db"
    collection_ref: str = "ref_docs"
    top_k: int = 6

    # splitter
    chunk_size: int = 1024
    chunk_overlap: int = 96

    # watsonx
    embed_model: str = "ibm/granite-embedding-278m-multilingual"
    # 권장: 한국어 대응이 좋은 최신 Llama 계열
    llm_model: str = "meta-llama/llama-3-3-70b-instruct"

    # Upstage
    upstage_url: str = "https://api.upstage.ai/v1/document-digitization"
    upstage_model: str = "document-parse"
    upstage_ocr: str = "force"  # "force" 권장(슬라이드 이미지가 많아서)
    upstage_base64_encoding: str = "['table']"  # 필요시 변경

    # 하이브리드 기준(텍스트가 너무 짧으면 Upstage로)
    min_chars_for_parse: int = 80

    # 캐시 (Upstage 호출 비용 절약)
    cache_dir: str = "./.cache_upstage"
