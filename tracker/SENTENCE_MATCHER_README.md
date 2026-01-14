# 문장 유사도 기반 키워드 매칭

## 개요

문장과 키워드 간 의미적 유사도를 계산하여 매칭하는 시스템입니다.
- **방식**: 직접 유사도 비교 (Cosine Similarity)
- **임베딩**: Transformer 또는 Upstage API
- **속도**: 140개 키워드 기준 밀리초 단위

## 왜 FAISS가 아닌 직접 비교?

### 현재 상황
- 키워드 수: 약 140개 (22개 슬라이드 × 평균 6개)
- 실시간 아님 (배치 분석)
- 정확도 우선

### 직접 비교 (선택됨)
**장점**
- 구현 간단, 정확한 결과 (Exact Search)
- 140개는 밀리초 단위로 처리
- 메모리 효율적
- 임베딩 캐싱으로 중복 계산 방지

**단점**
- 키워드 1만 개 이상이면 느림

### FAISS (미래 고려)
**장점**
- 수천~수만 개 키워드에서 빠름 (O(log n))
- GPU 가속 지원

**단점**
- 140개는 오버헤드
- 초기 인덱싱 비용
- ANN은 근사치 (정확도 손실)

**결론**: 키워드가 1000개 이상으로 늘어나면 FAISS로 전환 고려

## 파일 구조

```
tracker/
├── sentence_matcher.py      # 핵심 매칭 로직
└── keyword_tracker.py        # 기존 tracker (통합 가능)

test_sentence_matcher.py      # 테스트 스크립트
```

## 사용법

### 1. 기본 테스트
```bash
# Transformer 사용 (로컬, 빠름, 기본값)
python test_sentence_matcher.py

# Upstage API 사용
python test_sentence_matcher.py --api y
```

### 2. 대화형 모드
```bash
python test_sentence_matcher.py --interactive

# 또는 임계값 조정
python test_sentence_matcher.py -i --threshold 0.65
```

### 3. 코드에서 사용
```python
from tracker.sentence_matcher import SentenceMatcher
from tracker.keyword_tracker import TransformerEmbedder, EmbedderConfig

# 초기화
embedder = TransformerEmbedder(EmbedderConfig())
matcher = SentenceMatcher(
    embedder=embedder,
    slide_keywords={
        1: ["키워드1", "키워드2"],
        2: ["키워드3", "키워드4"]
    }
)

# 매칭
matches = matcher.find_matches(
    sentence="t-SNE로 시각화했습니다",
    threshold=0.7,
    top_k=5
)

for match in matches:
    print(f"{match.keyword}: {match.score:.3f}")

# 슬라이드 커버리지
coverage = matcher.get_coverage_for_slide(
    slide_no=13,
    script_text="대본 텍스트...",
    threshold=0.7
)
print(f"커버율: {coverage['coverage']:.1%}")
```

## 핵심 기능

### 1. find_matches()
문장과 유사한 키워드 찾기
```python
matches = matcher.find_matches(
    sentence="머신러닝 모델을 학습했습니다",
    threshold=0.7,    # 최소 유사도
    top_k=5,          # 상위 5개만
    slide_no=10       # 특정 슬라이드만 검색
)
```

**반환값**: `List[KeywordMatch]`
- `keyword`: 매칭된 키워드
- `score`: 유사도 점수 (0~1)
- `slide_no`: 슬라이드 번호 (옵션)

### 2. get_coverage_for_slide()
슬라이드별 키워드 커버리지 계산
```python
coverage = matcher.get_coverage_for_slide(
    slide_no=13,
    script_text="대본 텍스트",
    threshold=0.7
)
```

**반환값**: `Dict`
```python
{
    'total': 7,                    # 전체 키워드 수
    'covered': 5,                  # 커버된 키워드 수
    'coverage': 0.714,             # 커버율 (71.4%)
    'matched_keywords': [          # 매칭된 키워드와 점수
        ("t-SNE", 0.85),
        ("고차원 데이터", 0.78)
    ],
    'uncovered_keywords': [        # 누락된 키워드
        "2차원 축소",
        "선형 모델"
    ]
}
```

### 3. interactive_mode()
대화형 모드
```python
matcher.interactive_mode(threshold=0.7, top_k=5)
```

실행하면:
```
문장 입력> t-SNE로 시각화했습니다

✓ 3개 매칭됨:
  1. t-SNE: 0.921
  2. t-SNE 2D: 0.856
  3. 고차원 데이터: 0.743
```

## 주요 파라미터

### threshold (임계값)
- **기본값**: 0.7
- **의미**: 이 점수 이상만 매칭으로 인정
- **조정 기준**:
  - `0.8~0.9`: 엄격 (정확도 높음, 재현율 낮음)
  - `0.7`: 균형 (권장)
  - `0.5~0.6`: 느슨 (정확도 낮음, 재현율 높음)

### top_k
- **기본값**: None (전체)
- **의미**: 상위 k개만 반환
- **사용 케이스**: 
  - 대화형 모드: 5개 정도
  - 분석: None (전체)

## 성능 최적화

### 1. 임베딩 캐싱
```python
# 키워드 임베딩은 초기화 시 한 번만 계산
self._keyword_embeddings = {}  # 캐시

# 이후 검색 시 재사용
kw_emb = self._keyword_embeddings[keyword]
```

### 2. 선택적 검색
```python
# 특정 슬라이드만 검색 (전체 140개 → 7개)
matches = matcher.find_matches(
    sentence="...",
    slide_no=13  # Slide 13 키워드만 검색
)
```

### 3. 배치 처리
```python
# 여러 문장 동시 처리
results = matcher.batch_match(
    sentences=["문장1", "문장2", "문장3"],
    threshold=0.7
)
```

## 예상 결과

### Slide 13 (t-SNE 문제 해결)
**키워드**: `["t-SNE", "t-SNE 2D", "2차원 축소", ...]`
**대본**: `"sne를 이용하여 고차원 데이터를 2차원으로..."`

**기존 토큰 매칭**:
- "t-SNE" ✗ (t, sne로 분리되어 매칭 실패)

**문장 유사도 매칭**:
- "t-SNE" ✓ 0.85+ (의미적으로 유사하여 매칭 성공)

### Slide 11 (m_sequence_number 문제)
**키워드**: `["m_sequence_number", ...]`
**대본**: `"msequencenumber가 중요한..."`

**기존 토큰 매칭**:
- "m_sequence_number" ✗ (언더스코어 때문에 불일치)

**문장 유사도 매칭**:
- "m_sequence_number" ✓ 0.80+ (의미가 같아 매칭 성공)

## 미래 확장

### FAISS로 전환 (키워드 1000개 이상)
```python
# sentence_matcher.py 하단 참고용 코드 참조
import faiss
import numpy as np

class FAISSMatcher:
    def __init__(self, embedder, keywords):
        # 임베딩 계산 및 정규화
        embeddings_np = np.array([embedder.embed(kw) for kw in keywords])
        faiss.normalize_L2(embeddings_np)
        
        # 인덱스 생성
        dim = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_np)
    
    def find_matches(self, sentence, top_k=10):
        sentence_emb = np.array([embedder.embed(sentence)])
        faiss.normalize_L2(sentence_emb)
        
        scores, indices = self.index.search(sentence_emb, top_k)
        return [(keywords[idx], scores[0][i]) for i, idx in enumerate(indices[0])]
```

### GPU 가속
```python
# FAISS GPU 버전
index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(),
    0,  # GPU 0
    index
)
```

## 통합 예정

`run_tracker.py`에 통합하여 기존 토큰 매칭과 함께 사용:
1. **1차**: 토큰 매칭 (빠름, 정확한 문자열)
2. **2차**: 문장 유사도 (느림, 의미적 유사)

```python
# 하이브리드 접근
if token_score < 0.5:  # 토큰 매칭 실패
    semantic_score = sentence_matcher.find_matches(...)
    if semantic_score > 0.7:
        # 의미적으로 매칭됨
        covered = True
```
