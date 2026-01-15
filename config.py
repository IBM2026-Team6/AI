from dataclasses import dataclass

@dataclass
class Config:
    """시스템 설정"""
    docs_root: str = "./docs"
    out_dir: str = "./outputs"

    persist_dir: str = "./chroma_db"
    collection_ref: str = "ref_docs"
    top_k: int = 6

    chunk_size: int = 1024
    chunk_overlap: int = 96

    embed_model: str = "ibm/granite-embedding-278m-multilingual"
    llm_model: str = "meta-llama/llama-3-3-70b-instruct"

    upstage_url: str = "https://api.upstage.ai/v1/document-digitization"
    upstage_model: str = "document-parse"
    upstage_ocr: str = "force"
    upstage_base64_encoding: str = "['table']"

    min_chars_for_parse: int = 80
    cache_dir: str = "./.cache_upstage"

    embedding_model: str = "embedding-passage"
    lexical_threshold: float = 0.66
    semantic_threshold: float = 0.65
    semantic_skip_if_lexical_ge: float = 0.50
    stt_buffer_max_chars: int = 600
