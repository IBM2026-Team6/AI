# keyword_extractor.py
# ============================================================================
# 발표 슬라이드에서 핵심 키워드를 자동 추출하는 모듈
# ============================================================================
# 
# 설치 필요 패키지:
#   pip install openai python-dotenv scikit-learn
#   (scikit-learn이 없으면 기본 n-gram 방식으로 동작)
#
# 주요 기능:
#   1. 발표 대본을 슬라이드 단위로 파싱
#   2. TF-IDF 기반 키워드 후보 추출
#   3. LLM(Upstage Solar)을 사용하여 최종 키워드 선택
#   4. 다중 패스 투표로 결과 안정화
#
# ============================================================================
# 파이프라인 플로우 (Pipeline Flow)
# ============================================================================
#
# 입력: 발표 대본 (script)
#   ↓
# [단계 1] 슬라이드 파싱 (Parsing)
#   - "Part 1: Slide 1" 또는 "페이지 1" 형식으로 분할
#   - 각 슬라이드를 독립적으로 처리
#   ↓ (각 슬라이드별로)
# [단계 2] 후보 키워드 생성 (Candidate Generation)
#   - TF-IDF 또는 n-gram 방식으로 상위 35개 후보 추출
#   - 용어스러운 특징(대문자, 숫자, 한글) 포함 여부로 스코어링
#   ↓
# [단계 3] 다중 패스 LLM 호출 (Multi-Pass Voting)
#   - 같은 슬라이드/후보로 5번 반복 호출
#   - 각 호출마다 LLM이 독립적으로 키워드 선택
#   - 결과: 5개의 키워드 세트
#   ↓
# [단계 4] 투표 및 최종 선택 (Voting & Selection)
#   - 5번 호출 결과를 집계하여 빈도 계산
#   - 많이 등장한 키워드부터 순서대로 선택
#   - 스코어링: (LLM 투표수, 후보포함여부, 용어특징, 길이)
#   - 최대 7개까지 선택
#   ↓
# [단계 5] 최소 개수 보장 (Minimum Guarantee)
#   - 만약 3개 미만이면 후보 키워드로 채우기
#   - 최종 결과: 3~7개 키워드
#   ↓
# 출력: {슬라이드_번호: [키워드1, 키워드2, ...]}
#
# ============================================================================
# 예시 흐름
# ============================================================================
#
# 입력 대본:
# "Part 1: Slide 1 (31s)
#  이번 슬라이드는 Language-Specific Neurons에 대한... LLaMA-2, BLOOM..."
#
# 단계 2 후보 생성:
#   ["language-specific", "neurons", "LLaMA", "BLOOM", ...]
#
# 단계 3 LLM 호출 5회:
#   Pass 1: ["Language-Specific Neurons", "LLaMA-2", "BLOOM"]
#   Pass 2: ["Language-Specific", "neurons", "Mistral"]
#   Pass 3: ["Language-Specific Neurons", "LLaMA", "BLOOM"]
#   Pass 4: ["neurons", "LLaMA-2", "Language Model"]
#   Pass 5: ["Language-Specific Neurons", "LLaMA", "multilingual"]
#
# 단계 4 투표 결과:
#   Language-Specific Neurons: 3회 → 가장 높은 스코어
#   LLaMA-2 / LLaMA: 3회
#   BLOOM: 2회
#   neurons: 2회
#   ...
#
# 최종 결과 (3~7개):
#   ["Language-Specific Neurons", "LLaMA-2", "BLOOM", "neurons"]
#
# ============================================================================

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Optional: TF-IDF 기반 고급 후보 생성 (scikit-learn 필요)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


# ============================================================================
# 섹션 1: 발표 대본 파싱 (script → slides)
# ============================================================================
# 발표 대본을 개별 슬라이드 단위로 분할합니다.
# 지원 형식: "Part 1: Slide 1 ..." 또는 "페이지 1 ..." 형식

def parse_slides(script: str) -> Dict[int, str]:
    """
    발표 대본을 슬라이드 단위로 파싱합니다.
    
    지원 형식:
      - "Part 1: Slide 1 ..." 블록
      - "페이지 1 ..." 블록
    
    Returns:
      {슬라이드_번호: 슬라이드_텍스트} 형태의 딕셔너리
    """
    text = script.strip()

    # 방법 1: "Part ...: Slide N" 형식 먼저 시도
    part_pattern = re.compile(
        r"(?:^|\n)Part\s*\d+\s*:\s*Slide\s*(\d+)\s*\([^)]+\)\s*\n(.*?)(?=\n-{3,}\n|(?:\nPart\s*\d+\s*:\s*Slide\s*\d+)|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    parts = part_pattern.findall(text)
    if parts:
        slides: Dict[int, str] = {}
        for sn, body in parts:
            slides[int(sn)] = body.strip()
        return slides

    # 방법 2: "페이지 N" 형식 대체
    page_pattern = re.compile(
        r"(?:^|\n)페이지\s*(\d+)\s*\n(.*?)(?=(?:\n페이지\s*\d+)|\Z)",
        re.DOTALL
    )
    pages = page_pattern.findall(text)
    if pages:
        slides = {}
        for sn, body in pages:
            slides[int(sn)] = body.strip()
        return slides

    # 방법 3: 마지막 수단 - 전체 텍스트를 슬라이드 1로 취급
    return {1: text}


# ============================================================================
# 섹션 2: 키워드 후보 생성 (하이브리드 방식)
# ============================================================================
# TF-IDF 또는 기본 n-gram 방식으로 슬라이드에서 잠재적 키워드 후보를 추출합니다.

def _basic_ngram_candidates(text: str, top_k: int = 30) -> List[str]:
    """
    외부 라이브러리 없이 기본 n-gram 방식으로 후보 추출합니다.
    
    동작:
      - 텍스트를 토큰으로 분할
      - 1~3 단어 조합(phrase) 생성
      - 대문자/숫자/한글 포함 여부로 스코어링
      - 빈도 기반으로 상위 k개 선택
    
    Args:
      text: 입력 텍스트
      top_k: 반환할 후보 개수 (기본값: 30)
    
    Returns:
      상위 k개 후보 키워드 리스트
    """
    # 정규표현식으로 텍스트 정제: 알파벳/숫자/한글/하이픈/슬래시만 보존
    cleaned = re.sub(r"[^\w가-힣\-\+\/\s]", " ", text)
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]

    # n-gram 생성 (1~3 단어 조합)
    phrases = []
    for n in range(1, 4):  # 1~3 word phrases
        for i in range(len(tokens) - n + 1):
            p = " ".join(tokens[i:i+n])
            # 너무 짧은 phrase는 제외
            if len(p) < 3:
                continue
            phrases.append(p)

    # 빈도 계산
    freq = Counter(phrases)
    
    # 스코어링: 용어스러운 특징(대문자, 특수기호, 한글) 포함 여부로 가중치 부여
    def score(p: str) -> Tuple[int, int]:
        bonus = 0
        if re.search(r"[A-Z]", p): 
            bonus += 2  # 대문자 포함
        if re.search(r"[\-\+\/\d]", p): 
            bonus += 1  # 숫자/특수기호 포함
        if re.search(r"[가-힣]", p): 
            bonus += 1  # 한글 포함
        return (bonus, freq[p])

    # 스코어 기준으로 정렬 및 중복 제거
    ranked = sorted(freq.keys(), key=lambda p: score(p), reverse=True)
    out = []
    seen = set()
    for p in ranked:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= top_k:
            break
    return out


def tfidf_ngram_candidates(text: str, top_k: int = 30) -> List[str]:
    """
    고급 후보 생성: TF-IDF 기반 n-gram 추출 (scikit-learn 사용)
    
    동작:
      - 텍스트를 1~3 단어 n-gram으로 벡터화
      - TF-IDF 점수로 중요도 계산
      - 일반적인 단어는 제외 (예: "방법", "결과", "설명")
      - 점수 기준으로 상위 k개 선택
    
    Args:
      text: 입력 텍스트
      top_k: 반환할 후보 개수 (기본값: 30)
    
    Returns:
      상위 k개 후보 키워드 리스트
    """
    # scikit-learn이 없으면 기본 방식으로 대체
    if TfidfVectorizer is None:
        return _basic_ngram_candidates(text, top_k=top_k)

    # 텍스트 정제 (알파벳/숫자/한글/하이픈/슬래시만 보존)
    cleaned = re.sub(r"[^\w가-힣\-\+\/\s]", " ", text)
    docs = [cleaned]

    # TF-IDF 벡터라이저 설정
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),      # 1~3 단어 n-gram 사용
        min_df=1,                # 최소 문서 빈도
        max_features=5000,       # 최대 피처 개수
        token_pattern=r"(?u)\b[\w가-힣\-\+\/]{2,}\b",  # 토큰 패턴
    )
    X = vectorizer.fit_transform(docs)
    feats = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]

    # TF-IDF 점수로 정렬
    pairs = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)
    
    # 너무 일반적인 단어 제외 (정지어 리스트) - 이건 주제에 맞게 설정해야 하는데 일단 예시
    stop_like = {
        "이번", "설명", "결과", "방식", "기반", "실험", "모델", 
        "연구", "방법", "첫", "두", "마지막"
    }
    out = []
    for term, _ in pairs:
        if term in stop_like:
            continue
        out.append(term)
        if len(out) >= top_k:
            break
    return out


# ============================================================================
# 섹션 3: LLM 설정 및 호출
# ============================================================================
# Upstage Solar LLM을 사용하여 최종 키워드를 선택하고 정제합니다.

@dataclass
class LLMConfig:
    """LLM 호출 설정 클래스"""
    model: str = "solar-pro2"           # Upstage Solar 모델명
    temperature: float = 0.2            # 창의성 정도 (낮을수록 일관성 높음)
    max_tokens: int = 300               # 최대 출력 토큰 수
    multi_pass: int = 5                 # 다중 패스 투표 횟수 (정확도 향상)
    keywords_min: int = 3               # 최소 키워드 개수
    keywords_max: int = 7               # 최대 키워드 개수


SYSTEM_PROMPT = """You are an expert at extracting academic presentation keywords.
Return ONLY valid JSON, no markdown, no extra text.
"""

USER_TEMPLATE = """You will be given:
1) slide_text: a single slide's speaker script
2) candidate_terms: a list of candidate terms extracted by statistics

Task:
- Select and refine 핵심 키워드 for the slide (academic, concise).
- Prefer terms grounded in the slide_text. Do NOT invent concepts not present.
- Merge duplicates/synonyms (keep the most canonical form).
- Avoid generic words (e.g., "방법", "결과", "설명", "기반", "실험").
- Output 3~7 keywords.

Output JSON schema:
{{
  "keywords": ["...","..."]
}}

slide_text:
{slide_text}

candidate_terms:
{candidate_terms}
"""


def _get_client_from_env(cfg: LLMConfig) -> ChatOpenAI:
    """
    환경변수에서 API 키를 읽어 Upstage Solar LLM 클라이언트를 초기화합니다.
    
    필요한 환경변수:
      - UPSTAGE_API_KEY: Upstage API 키
      - UPSTAGE_API_URL: Upstage API 엔드포인트 (기본값: https://api.upstage.ai/v1)
    
    Args:
      cfg: LLMConfig 설정 객체
    
    Returns:
      초기화된 ChatOpenAI 클라이언트
    
    Raises:
      RuntimeError: API 키가 없을 때 발생
    """
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    base_url = os.getenv("UPSTAGE_API_URL", "https://api.upstage.ai/v1")

    if not api_key:
        raise RuntimeError("Missing env var: UPSTAGE_API_KEY in .env")

    return ChatOpenAI(
        model=cfg.model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def _llm_once(llm: ChatOpenAI, cfg: LLMConfig, slide_text: str, candidates: List[str]) -> List[str]:
    """
    LLM에 한 번 요청하여 슬라이드 내용과 후보 키워드를 기반으로 최종 키워드를 추출합니다.
    
    동작:
      1. 슬라이드 텍스트와 후보 키워드를 프롬프트에 포함
      2. Upstage Solar LLM 호출
      3. JSON 응답 파싱하여 키워드 추출
      4. 실패 시 빈 리스트 반환 (폴백 처리)
    
    Args:
      llm: ChatOpenAI 클라이언트
      cfg: LLMConfig 설정 객체
      slide_text: 슬라이드 텍스트
      candidates: 후보 키워드 리스트
    
    Returns:
      추출된 키워드 리스트 (최대 7개, 최소 3개)
    """
    # 프롬프트 생성
    prompt = USER_TEMPLATE.format(
        slide_text=slide_text.strip(),
        candidate_terms=json.dumps(candidates, ensure_ascii=False),
    )

    # LangChain 메시지 형식으로 변환
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # LLM 호출
    try:
        resp = llm.invoke(messages)
        content = resp.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {type(e).__name__}: {e}")
        return []

    # JSON 응답 파싱
    try:
        data = json.loads(content)
        kws = data.get("keywords", [])
        if not isinstance(kws, list):
            return []
        # 키워드 정제 (중복 제거, 공백 제거)
        cleaned = []
        for k in kws:
            if not isinstance(k, str):
                continue
            kk = k.strip()
            if kk and kk not in cleaned:
                cleaned.append(kk)
        return cleaned
    except Exception:
        # JSON 파싱 실패 시, 정규표현식으로 JSON 객체 찾기 시도
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                kws = data.get("keywords", [])
                if isinstance(kws, list):
                    return [str(x).strip() for x in kws if str(x).strip()]
            except Exception:
                pass
        return []


def extract_keywords_per_slide(script: str, cfg: Optional[LLMConfig] = None) -> Dict[int, List[str]]:
    """
    메인 함수: 발표 대본 전체에서 슬라이드별 핵심 키워드를 추출합니다.
    
    전체 파이프라인:
      1. 대본을 슬라이드 단위로 파싱
      2. 각 슬라이드별로:
         a) TF-IDF 기반 후보 키워드 생성
         b) 다중 패스 투표로 LLM 호출 (안정성 향상)
         c) 후보 키워드와 LLM 결과 투표로 최종 선택
         d) 최소/최대 키워드 개수 조정
    
    Args:
      script: 전체 발표 대본 텍스트
      cfg: LLMConfig 설정 (None이면 기본값 사용)
    
    Returns:
      {슬라이드_번호: [키워드1, 키워드2, ...]} 형태의 딕셔너리
    
    예시:
      >>> script = '''Part 1: Slide 1 (31s)
      ...            대형 언어 모델의 다국어 처리 메커니즘...'''
      >>> result = extract_keywords_per_slide(script, LLMConfig(multi_pass=3))
      >>> print(result[1])
      ['Language Model', 'Multilingual', 'Neurons']
    """
    # 설정 초기화 (기본값 사용)
    cfg = cfg or LLMConfig()
    
    # LLM 클라이언트 초기화
    llm = _get_client_from_env(cfg)
    
    print(f"[INFO] LLM initialized: model={cfg.model}, temperature={cfg.temperature}, max_tokens={cfg.max_tokens}")

    # 대본을 슬라이드별로 파싱
    slides = parse_slides(script)
    results: Dict[int, List[str]] = {}

    # 각 슬라이드별 처리
    for sn in sorted(slides.keys()):
        slide_text = slides[sn]

        # 1단계: TF-IDF 기반 후보 키워드 생성 (상위 35개)
        candidates = tfidf_ngram_candidates(slide_text, top_k=35)

        # 2단계: 다중 패스 투표 (안정성 향상)
        # 같은 프롬프트로 여러 번 호출하여 결과의 일관성 확인
        runs = []
        for _ in range(max(1, cfg.multi_pass)):
            kws = _llm_once(llm, cfg, slide_text, candidates)
            runs.append(tuple(kws))

        # 3단계: 투표로 최종 키워드 선택
        # 여러 패스에서 자주 나타난 키워드를 우선적으로 선택
        flat = [k for run in runs for k in run]
        counts = Counter(flat)

        # 후보 키워드 세트 (스코어링에 사용)
        cand_set = set(candidates)

        # 키워드 스코어링 함수
        # - LLM이 뽑은 횟수 (counts)
        # - 후보 리스트 포함 여부 (bonus +2)
        # - 용어스러운 특징 (대문자, 숫자, 한글 포함)
        # - 키워드 길이
        def kw_score(k: str) -> Tuple[int, int, int]:
            bonus = 0
            if k in cand_set:
                bonus += 2  # 후보 리스트에 있으면 가산
            if re.search(r"[A-Z]", k):
                bonus += 1  # 대문자 포함
            if re.search(r"[\-\+\/\d]", k):
                bonus += 1  # 숫자/특수기호 포함
            if re.search(r"[가-힣]", k):
                bonus += 1  # 한글 포함
            return (counts[k], bonus, len(k))

        # 스코어 기준으로 정렬
        ranked = sorted(counts.keys(), key=lambda k: kw_score(k), reverse=True)

        # 4단계: 최종 키워드 결정
        # 상위 스코어부터 선택, 최대 keywords_max개까지
        final = []
        for k in ranked:
            kk = k.strip()
            if not kk:
                continue
            if kk in final:
                continue  # 중복 제거
            final.append(kk)
            if len(final) >= cfg.keywords_max:
                break

        # 5단계: 최소 개수 보장
        # LLM 결과가 부족하면 후보 키워드로 채우기
        if len(final) < cfg.keywords_min:
            for c in candidates:
                if c not in final:
                    final.append(c)
                if len(final) >= cfg.keywords_min:
                    break

        results[sn] = final

    return results


# ============================================================================
# 섹션 4: 사용 예시 및 테스트
# ============================================================================

def main_test():
    """
    모듈 테스트용 메인 함수.
    keyword_extractor 모듈을 직접 실행할 때만 사용됩니다.
    
    실행:
      python keyword_extractor.py
    """
    # 테스트용 예시 데이터
    example = """
        [LiveCoach AI Script]
        Total Time: 58s
        ==========================================

        Part 1: Slide 1 (31s)
        이번 슬라이드는 **Language-Specific Neurons**에 대한 두 편의 논문을 소개합니다. 첫 번째 논문은 대형 언어 모델(LLM)이 다국어 처리 능력을 갖추는 핵심 메커니즘을 분석한 연구로, LLaMA-2, BLOOM, Mistral 등의 모델에서 언어별 특화 뉴런(language-specific neurons)을 발견했습니다. 특히 상위/하위 레이어에 집중된 이 뉴런들을 활성화/비활성화함으로써 출력 언어를 제어할 수 있음을 실험적으로 증명했죠. 두 번째 논문은 정보 이득과 분산 기반의 특징 선택 방법을 제안하며, 텍스트 분류에서 중복성을 줄이면서도 정보 손실을 최소화하는 기법을 다룹니다. 두 연구 모두 모델의 효율성과 성능 향상에 중요한 통찰을 제공합니다.

        ------------------------------------------

        Part 2: Slide 2 (27s)
        이번 슬라이드에서는 언어별 뉴런 추출 방법과 성능 평가 결과를 설명드리겠습니다.  
        첫 번째로, 기존 논문에서는 언어별 활성화 확률을 기반으로 뉴런을 선택했습니다.  
        두 번째로, 저희는 MMR 기반 뉴런 선택 방식을 도입했는데요. 정보 획득(Relevance)과 뉴런 간 유사성(Similarity)을 동시에 고려해 최적의 뉴런을 선정합니다.  
        마지막으로, MMR 방식으로 선택된 뉴런을 비활성화했을 때 PPL(Perplexity) 점수를 측정했습니다.  
        실험 결과, MMR 기반 선택이 기존 방식보다 언어별 특성을 더 잘 보존하면서도 성능 저하를 최소화하는 것으로 나타났습니다.

        ------------------------------------------
    """
    
    # 키워드 추출 실행 (다중 패스 3회 투표)
    out = extract_keywords_per_slide(example, LLMConfig(multi_pass=3))
    
    # 결과 출력
    print("\n[결과]")
    for sn, kws in out.items():
        print(f"페이지 {sn}: {', '.join(kws)}")


if __name__ == "__main__":
    main_test()
