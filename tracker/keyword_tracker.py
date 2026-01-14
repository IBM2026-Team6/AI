# keyword_tracker.py
# pip install openai python-dotenv
from __future__ import annotations

import os
import re
import math
import sys
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Mecab 형태소 분석기 (선택적)
try:
    from konlpy.tag import Mecab
    _MECAB_AVAILABLE = True
    _MECAB = Mecab()
except Exception:
    _MECAB_AVAILABLE = False
    _MECAB = None

# Sentence Transformers (선택적)
try:
    from sentence_transformers import SentenceTransformer
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# config 임포트
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# -----------------------------
# Utils: normalization & tokens
# -----------------------------

_GENERIC_STOP_TOKENS = {
    # 문법 관련
    "이번", "설명", "결과", "방식", "기반", "실험", "모델", "연구", "방법", "데이터",
    # 발표/문서 관련
    "발표", "대본", "슬라이드", "페이지", "내용", "예시", "예제", "사례", "참고",
    # 마크다운/형식 관련
    "이미지", "표", "그림", "도표", "링크", "텍스트",
    # 일반 대명사/조사
    "것", "수", "점", "쪽", "편", "경우", "관", "분", "들", "사항",
    # 동사 활용형
    "있", "되", "보", "같", "지", "넣", "놓", "두",
}

def normalize_text(text: str) -> str:
    """
    텍스트를 정규화: 특수문자 제거, 소문자 변환
    목표: "-", "_" 같은 구분자를 무시하고 일반화된 비교
    예: "t-SNE" → "tsne", "m_sequence_number" → "msequencenumber"
    """
    # 소문자 변환
    text = text.lower()
    # 하이픈, 언더스코어, 공백 제거
    text = re.sub(r'[-_\s]', '', text)
    # 특수문자 제거 (알파벳, 숫자만 유지)
    text = re.sub(r'[^a-z0-9가-힣]', '', text)
    return text

def remove_postposition_and_ending(tokens: List[str]) -> List[str]:
    """조사와 어미 제거"""
    # 자주 붙는 조사, 어미 패턴
    postpositions = ['은', '는', '이', '가', '을', '를', '에', '게', '께', '의', '와', '과', '도', '만', '서', '로', '로부터', '까지', '에서', '로서', '이라', '이다', '다', '다고', '습니다', '했습니다', '하며', '하고']
    
    result = []
    for token in tokens:
        # 조사/어미 제거 (역순으로 가장 긴 매칭부터 시도)
        for postpos in sorted(postpositions, key=len, reverse=True):
            if token.endswith(postpos) and len(token) > len(postpos):
                token = token[:-len(postpos)]
                break
        
        # 길이 2 이상 + 불용어 아님
        if len(token) >= 2 and token not in _GENERIC_STOP_TOKENS:
            result.append(token)
    
    return result

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    # keep Korean/English/digits/+/-/_ and spaces
    s = re.sub(r"[^\w가-힣\+\-\/\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text_with_steps(s: str) -> Tuple[str, Dict[str, str]]:
    """정규화 과정을 단계별로 추적"""
    steps = {}
    
    # Step 1: 원본
    steps["0_원본"] = s
    
    # Step 2: 소문자 + 양쪽 공백 제거
    s = s.strip().lower()
    steps["1_소문자"] = s
    
    # Step 3: 특수문자 제거 (Korean/English/digits/+/-/_ 만 유지)
    s = re.sub(r"[^\w가-힣\+\-\/\s]", " ", s)
    steps["2_특수문자제거"] = s
    
    # Step 4: 연속 공백 정규화
    s = re.sub(r"\s+", " ", s).strip()
    steps["3_공백정규화"] = s
    
    return s, steps

def tokenize_simple_ko_en(s: str) -> List[str]:
    """
    Tokenizer with morphological analysis support:
    - 한글: Mecab이 있으면 명사 추출, 없으면 조사/어미 제거
    - 영문/숫자: 공백 기준 분할
    """
    s = normalize_text(s)
    
    # Mecab이 가능하면 형태소 분석으로 명사 추출
    if _MECAB_AVAILABLE:
        try:
            # nouns(): 명사만 추출 (가장 정확)
            noun_tokens = _MECAB.nouns(s)
            toks = [t for t in noun_tokens if len(t) >= 2 and t not in _GENERIC_STOP_TOKENS]
            if toks:
                return toks
        except Exception:
            pass
    
    # Mecab 실패/미설치: 기본 토큰화 후 조사/어미 제거
    raw_toks = s.split(" ")
    toks = remove_postposition_and_ending(raw_toks)
    return toks


def clean_script_text(text: str) -> str:
    """발표 대본용 정제 로직을 모듈화"""
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"_(.+?)_", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)  # 제목 마크다운
    cleaned = re.sub(r"^\d{2}\s*\|", "", cleaned, flags=re.MULTILINE)  # 01 | 형태 제거
    cleaned = re.sub(r"\[.+?\]", "", cleaned)  # [텍스트] 제거
    cleaned = re.sub(r"\(.+?\)", "", cleaned)  # (텍스트) 제거
    cleaned = re.sub(r'["]', '"', cleaned)
    cleaned = re.sub(r"['']", "'", cleaned)
    cleaned = re.sub(r"[\*_\-\|]+", " ", cleaned)  # 마크다운 기호
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_scripts_by_slide(scripts_file: str) -> Dict[int, str]:
    """paper_scripts.md에서 슬라이드별 대본 파싱"""
    slide_scripts: Dict[int, str] = {}

    with open(scripts_file, "r", encoding="utf-8") as f:
        content = f.read()

    slide_pattern = r"^## Slide (\d+) \(page=\d+\)"
    matches = list(re.finditer(slide_pattern, content, re.MULTILINE))

    for idx, match in enumerate(matches):
        slide_no = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block_content = content[start:end]

        found = re.search(
            r"^- 슬라이드 대본:\s*(.*?)(?=^- (?:핵심 메시지|예상 질문)|\Z)",
            block_content,
            re.MULTILINE | re.DOTALL,
        )

        if not found:
            continue

        script_text = clean_script_text(found.group(1).strip())
        if script_text:
            slide_scripts[slide_no] = script_text

    return slide_scripts


def extract_keywords_from_script(text: str, top_n: Optional[int] = None) -> List[str]:
    """대본에서 주요 키워드 추출 (모든 토큰, top_n 지정 시 상위 N개만)"""
    cleaned = clean_script_text(text)
    tokens = tokenize_simple_ko_en(cleaned)
    if not tokens:
        return []

    long_tokens = [t for t in tokens if len(t) >= 3]
    counter = Counter(long_tokens)
    return [token for token, _ in counter.most_common(top_n)]


@dataclass
class SlideAnalysis:
    """슬라이드별 분석 결과"""
    slide_no: int
    total_keywords: int
    extracted_keywords: List[str] = field(default_factory=list)
    covered_keywords: List[str] = field(default_factory=list)
    uncovered_keywords: List[str] = field(default_factory=list)
    coverage_accuracy: float = 0.0

    def __post_init__(self):
        if self.total_keywords > 0:
            self.coverage_accuracy = len(self.covered_keywords) / self.total_keywords


def parse_keywords_from_file(keywords_file: str) -> Dict[int, List[str]]:
    """paper_keywords.txt에서 슬라이드별 키워드 파싱"""
    slide_keywords: Dict[int, List[str]] = {}

    with open(keywords_file, "r", encoding="utf-8") as f:
        content = f.read()

    slide_blocks = re.findall(
        r"## Slide (\d+)\s*\n키워드 \(\d+개\):\s*\n((?:\s+\d+\..+\n)*)",
        content,
    )

    for slide_str, keywords_str in slide_blocks:
        slide_no = int(slide_str)
        keywords = re.findall(r"\d+\.\s+(.+?)$", keywords_str, re.MULTILINE)
        slide_keywords[slide_no] = keywords

    return slide_keywords


def save_coverage_report(
    results: List["SlideAnalysis"],
    avg_accuracy: float,
    total_keywords: int,
    total_covered: int,
    output_file: Path,
) -> None:
    """커버 분석 결과를 파일에 저장"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 키워드 커버 분석 결과\n")
        f.write("=" * 80 + "\n\n")

        # 전체 요약
        f.write("## 전체 요약\n")
        f.write(f"- 총 슬라이드 수: {len(results)}개\n")
        f.write(f"- 총 키워드 수: {total_keywords}개\n")
        f.write(f"- 커버된 키워드: {total_covered}개 ({total_covered/total_keywords*100:.1f}%)\n")
        f.write(f"- 평균 커버 정확도: {avg_accuracy:.1%}\n\n")

        # 슬라이드별 상세
        f.write("## 슬라이드별 상세 분석\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            bar_length = 40
            filled = int(bar_length * result.coverage_accuracy)
            bar = "█" * filled + "░" * (bar_length - filled)

            f.write(f"### Slide {result.slide_no}\n")

            f.write(f"**정확도: {result.coverage_accuracy:.1%}** [{bar}]\n")

            f.write(f"- 커버됨: {len(result.covered_keywords)}/{result.total_keywords}\n\n")

            if result.extracted_keywords:
                f.write(f"**대본 추출 키워드:**\n")
                f.write(f"{', '.join(result.extracted_keywords)}\n\n")

            if result.covered_keywords:
                f.write("**포함된 키워드:**\n")
                for i, kw in enumerate(result.covered_keywords, 1):
                    f.write(f"  {i}. {kw}\n")
                f.write("\n")

            if result.uncovered_keywords:
                f.write("**누락된 키워드:**\n")
                for i, kw in enumerate(result.uncovered_keywords, 1):
                    f.write(f"  {i}. {kw}\n")
                f.write("\n")

            f.write("-" * 80 + "\n\n")

def token_overlap_score(keyword: str, text: str) -> float:
    """
    Overlap score in [0, 1] based on tokens + substring similarity.
    토큰 기반 매칭과 문자열 유사도를 함께 고려합니다.
    """
    # 1. 토큰 기반 매칭
    kt = tokenize_simple_ko_en(keyword)
    tt = set(tokenize_simple_ko_en(text))
    if not kt:
        return 0.0
    hit = sum(1 for t in kt if t in tt)
    token_score = hit / max(1, len(kt))
    
    # 2. 문자열 유사도 (substring matching)
    # 정규화된 텍스트에서 부분 문자열 유사도 계산
    normalized_kw = normalize_text(keyword)
    normalized_text = normalize_text(text)
    string_similarity = difflib.SequenceMatcher(None, normalized_kw, normalized_text).ratio()
    
    # 3. 두 점수를 결합 (최댓값 또는 평균)
    # 토큰 점수를 우선하되, 문자열 유사도도 고려
    combined_score = max(token_score, string_similarity * 0.9)
    
    return combined_score


# -----------------------------
# Embedding (Upstage)
# -----------------------------

def _get_upstage_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY") or os.getenv("PSTAGE_API_KEY")
    base_url = os.getenv("UPSTAGE_API_URL")

    if not api_key:
        raise RuntimeError("Missing UPSTAGE_API_KEY (or PSTAGE_API_KEY) in .env")
    if not base_url:
        raise RuntimeError("Missing UPSTAGE_API_URL in .env (e.g., https://api.upstage.ai/v1)")

    base_url = base_url.strip().rstrip("/")
    # Ensure the SDK can append /chat/... etc. correctly
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    return OpenAI(api_key=api_key, base_url=base_url)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom > 0 else 0.0


@dataclass
class EmbedderConfig:
    """
    임베딩 모델 설정
    - Upstage API: solar-embedding-1-large
    - Transformer (한국어): jhgan/ko-sroberta-multitask (442MB, 정확도 높음)
    - Transformer (경량): paraphrase-multilingual-MiniLM-L12-v2 (471MB, 빠름)
    - Transformer (초경량): paraphrase-multilingual-mpnet-base-v2 (278MB, 가장 빠름)
    """
    embedding_model: str = "solar-embedding-1-large"
    transformer_model: str = "paraphrase-multilingual-MiniLM-L12-v2"  # 경량 다국어 모델
    
    @classmethod
    def from_config(cls, cfg: Config) -> "EmbedderConfig":
        """Config 객체에서 EmbedderConfig 생성"""
        return cls(
            embedding_model=cfg.embedding_model,
            transformer_model=getattr(cfg, 'transformer_model', 'paraphrase-multilingual-MiniLM-L12-v2')
        )


class UpstageEmbedder:
    """Upstage API 기반 임베딩 (원격 API 호출)"""
    def __init__(self, cfg: Optional[EmbedderConfig] = None):
        self.cfg = cfg or EmbedderConfig()
        self.client = _get_upstage_client()
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        key = normalize_text(text)
        if key in self._cache:
            return self._cache[key]

        # OpenAI-compatible embedding endpoint
        resp = self.client.embeddings.create(
            model=self.cfg.embedding_model,
            input=text,
        )
        vec = resp.data[0].embedding
        self._cache[key] = vec
        return vec


class TransformerEmbedder:
    """Sentence Transformers 기반 임베딩 (로컬 실행, 빠름)"""
    def __init__(self, cfg: Optional[EmbedderConfig] = None):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers 라이브러리가 필요합니다. "
                "설치: pip install sentence-transformers"
            )
        self.cfg = cfg or EmbedderConfig()
        print(f"   Loading transformer model: {self.cfg.transformer_model}...")
        print(f"   (캐시: ~/.cache/huggingface/hub/)")
        
        # safetensors 파일을 자동으로 사용하도록 환경변수 설정
        import os
        os.environ['SAFETENSORS_FAST_GPU'] = '1'
        
        self.model = SentenceTransformer(
            self.cfg.transformer_model,
            trust_remote_code=False,
            device='cpu'  # CPU 사용 (MPS 사용 시 device='mps')
        )
        self._cache: Dict[str, List[float]] = {}
        print("   Transformer model loaded")

    def embed(self, text: str) -> List[float]:
        key = normalize_text(text)
        if key in self._cache:
            return self._cache[key]

        # Sentence Transformers로 임베딩 생성
        vec = self.model.encode(text, convert_to_numpy=True).tolist()
        self._cache[key] = vec
        return vec


# -----------------------------
# Keyword tracker core
# -----------------------------

@dataclass
class KeywordItem:
    text: str
    aliases: List[str] = field(default_factory=list)
    covered: bool = False
    covered_by: Optional[str] = None   # "lexical" or "semantic"
    score: float = 0.0                 # last match score


@dataclass
class TrackerConfig:
    # lexical
    lexical_threshold: float = 0.5   # 토큰 2/3 정도 맞으면 커버로 보는 기본값
    # semantic
    semantic_threshold: float = 0.78  # 임계값은 도메인/모델 따라 튜닝 필요
    # semantic triggering: lexical이 이 점수 이상이면 semantic을 굳이 안 함
    semantic_skip_if_lexical_ge: float = 0.50
    # semantic buffer
    stt_buffer_max_chars: int = 600   # 너무 짧으면 의미가 약해서 버퍼링 권장
    
    @classmethod
    def from_config(cls, cfg: Config) -> "TrackerConfig":
        """Config 객체에서 TrackerConfig 생성"""
        return cls(
            lexical_threshold=cfg.lexical_threshold,
            semantic_threshold=cfg.semantic_threshold,
            semantic_skip_if_lexical_ge=cfg.semantic_skip_if_lexical_ge,
            stt_buffer_max_chars=cfg.stt_buffer_max_chars,
        )


class SlideKeywordTracker:
    """
    슬라이드 단위로 키워드 커버 상태를 추적.
    - update_with_stt(slide_no, stt_text) 호출하면 해당 슬라이드의 키워드들이 커버되는지 갱신
    - lexical 먼저, 부족한 것만 semantic(embedding)으로 보완
    """

    def __init__(
        self,
        slide_keywords: Dict[int, List[str]],
        slide_aliases: Optional[Dict[int, Dict[str, List[str]]]] = None,
        cfg: Optional[TrackerConfig] = None,
        embedder: Optional[UpstageEmbedder] = None,
    ):
        self.cfg = cfg or TrackerConfig()
        self.embedder = embedder  # 필요할 때만 생성(비용 절약)
        self.slides: Dict[int, List[KeywordItem]] = {}
        self._buffers: Dict[int, str] = {}  # slide_no -> accumulated stt buffer

        slide_aliases = slide_aliases or {}

        for sn, kws in slide_keywords.items():
            items = []
            alias_map = slide_aliases.get(sn, {})
            for kw in kws:
                items.append(KeywordItem(text=kw, aliases=alias_map.get(kw, [])))
            self.slides[sn] = items
            self._buffers[sn] = ""

        # semantic용: 키워드 임베딩 미리 계산(옵션)
        self._kw_vecs: Dict[Tuple[int, str], List[float]] = {}

    def reset_slide(self, slide_no: int) -> None:
        for item in self.slides.get(slide_no, []):
            item.covered = False
            item.covered_by = None
            item.score = 0.0
        self._buffers[slide_no] = ""

    def get_status(self, slide_no: int) -> List[Dict[str, Any]]:
        out = []
        for it in self.slides.get(slide_no, []):
            out.append({
                "keyword": it.text,
                "covered": it.covered,
                "covered_by": it.covered_by,
                "score": it.score,
            })
        return out

    def remaining_keywords(self, slide_no: int) -> List[str]:
        return [it.text for it in self.slides.get(slide_no, []) if not it.covered]

    def _all_forms(self, item: KeywordItem) -> List[str]:
        # canonical + aliases
        forms = [item.text] + list(item.aliases or [])
        # dedup
        seen = set()
        out = []
        for f in forms:
            nf = normalize_text(f)
            if nf and nf not in seen:
                seen.add(nf)
                out.append(f)
        return out

    def update_with_stt(self, slide_no: int, stt_text: str) -> List[Dict[str, Any]]:
        """
        Returns updated status list for the slide.
        """
        if slide_no not in self.slides:
            raise KeyError(f"slide_no {slide_no} not found")

        # 1) buffer (semantic용)
        buf = (self._buffers.get(slide_no, "") + " " + stt_text).strip()
        if len(buf) > self.cfg.stt_buffer_max_chars:
            buf = buf[-self.cfg.stt_buffer_max_chars:]
        self._buffers[slide_no] = buf

        items = self.slides[slide_no]

        # 2) Lexical pass (fast)
        for it in items:
            if it.covered:
                continue

            best = 0.0
            for form in self._all_forms(it):
                s = token_overlap_score(form, stt_text)
                if s > best:
                    best = s

            if best >= self.cfg.lexical_threshold:
                it.covered = True
                it.covered_by = "lexical"
                it.score = best

        # 3) Semantic pass (only for remaining & only when needed)
        remaining = [it for it in items if not it.covered]
        if remaining:
            # semantic을 아예 안 쓰고 싶으면 embedder=None로 만들면 됨
            if self.embedder is None:
                return self.get_status(slide_no)

            # semantic은 버퍼가 어느 정도 쌓였을 때만(너무 짧으면 의미 약함)
            if len(self._buffers[slide_no]) < 40:
                return self.get_status(slide_no)

            stt_vec = self.embedder.embed(self._buffers[slide_no])

            for it in remaining:
                # lexical이 어느 정도라도 있으면 semantic 생략 (원하면 반대로 해도 됨)
                # -> 여기서는 "최근 stt_text" 기준으로 lexical 점수 다시 계산해서 판단
                best_recent = 0.0
                for form in self._all_forms(it):
                    s = token_overlap_score(form, stt_text)
                    best_recent = max(best_recent, s)

                if best_recent >= self.cfg.semantic_skip_if_lexical_ge:
                    continue

                # 키워드 임베딩 캐시
                key = (slide_no, it.text)
                if key not in self._kw_vecs:
                    self._kw_vecs[key] = self.embedder.embed(it.text)

                sim = cosine_similarity(stt_vec, self._kw_vecs[key])
                if sim >= self.cfg.semantic_threshold:
                    it.covered = True
                    it.covered_by = "semantic"
                    it.score = sim

        return self.get_status(slide_no)
