# 발표 대본 자동 생성 및 키워드 추적 시스템

발표 슬라이드와 참고자료를 기반으로 슬라이드별 발표 대본을 자동 생성하고,
생성된 대본이 키워드를 얼마나 커버하는지 추적하는 시스템입니다.

## 주요 기능

### 1. 대본 자동 생성 (main.py)
- RAG 기반 참고문서 검색 (LangChain + ChromaDB)
- IBM Watsonx 또는 Upstage Solar LLM 선택 사용
- 슬라이드별 핵심 키워드 추출 (옵션)

### 2. 키워드 추적 (run_tracker.py)
- 생성된 대본의 키워드 커버리지 분석
- 3가지 매칭 방식:
  - `hybrid`: token + sentence 결합 (기본, 최고 성능)
  - `token`: 형태소 기반 매칭 (빠름)
  - `sentence`: 문장 유사도 기반 매칭 (정확)
- 키워드 정규화: 하이픈/언더스코어를 공백으로 치환 (기본 ON)
- 임베딩 선택:
  - Transformer (로컬, 빠름, 기본값)
  - Upstage API (정확, API 호출 필요)

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 필수 패키지
- Python 3.11.9
- langchain, langchain-openai, langchain-ibm
- chromadb
- sentence-transformers
- konlpy (한국어 형태소 분석, 옵션)

---

## 환경 변수 설정 (.env)

```env
# IBM Watsonx
API_KEY=your_ibm_api_key
PROJECT_ID=your_ibm_project_id
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com

# Upstage
UPSTAGE_API_KEY=your_upstage_api_key
UPSTAGE_API_URL=https://api.upstage.ai/v1/solar
```

---

## 사용법

### 1단계: 대본 생성

```bash
# IBM Watsonx 사용
python main.py --api ibm --extractor y

# Upstage Solar 사용
python main.py --api upstage --extractor y
```

**출력:**
- `outputs/paper_scripts.md`: 슬라이드별 대본
- `outputs/paper_keywords.txt`: 슬라이드별 키워드

### 2단계: 키워드 추적

```bash
# 기본 (hybrid + 정규화 ON, Transformer 임베딩)
python run_tracker.py

# token 모드 (정규화 OFF 예시)
python run_tracker.py -m token --normalize n

# sentence 모드 + Upstage API
python run_tracker.py -m sentence --api y
```

**출력:**
- `outputs/paper_coverage_analysis.txt`: 슬라이드별 커버리지 분석

---

## 설정 (config.py)

### 경로 설정
```python
docs_root = "./docs"           # 문서 폴더
out_dir = "./outputs"           # 결과 저장 폴더
```

### RAG 설정
```python
persist_dir = "./chroma_db"     # 벡터 DB 저장 경로
chunk_size = 400                # 청크 크기
chunk_overlap = 50              # 청크 오버랩
top_k = 3                       # 검색 결과 개수
```

### 키워드 추적 설정
```python
lexical_threshold = 0.5         # 토큰 매칭 임계값
semantic_threshold = 0.65       # 문장 유사도 임계값
```

### 임베딩 모델
```python
# Transformer (로컬)
transformer_model = "paraphrase-multilingual-MiniLM-L12-v2"

# Upstage (API)
embedding_model = "solar-embedding-1-large"
```

---

## 프로젝트 구조

```
.
├── docs/                       # 문서 폴더
│   ├── paper.pdf              # 발표 슬라이드
│   └── report.pdf             # 참고 문서
├── outputs/                    # 결과 파일
│   ├── paper_scripts.md       # 생성된 대본
│   ├── paper_keywords.txt     # 추출된 키워드
│   └── paper_coverage_analysis.txt  # 커버리지 분석
├── tracker/                    # 키워드 추적 모듈
│   ├── keyword_tracker.py     # 토큰 기반 추적
│   └── sentence_tracker.py    # 문장 유사도 추적
├── main.py                     # 대본 생성
├── run_tracker.py             # 키워드 추적
├── config.py                   # 설정
└── requirements.txt            # 의존성
```

---

## 매칭 방식 비교

| 방식 | 정확도 | 속도 | 특징 |
|------|--------|------|------|
| token | 89% | 빠름 | 형태소 기반, 정확한 단어 매칭 |
| sentence | 82% | 느림 | 의미 기반, 문맥 이해 |
| **hybrid** | **95.9%** | 중간 | token + sentence 결합, 기본 정규화(공백 치환) |

**권장:** 프레젠테이션 추적에는 `python run_tracker.py` (hybrid + 정규화 ON) 사용

---

## 주요 함수

### tracker/keyword_tracker.py
- `normalize_text()`: 텍스트 정규화
- `tokenize_simple_ko_en()`: 한영 토큰화
- `token_overlap_score()`: 토큰 오버랩 점수 계산
- `parse_keywords_from_file()`: 키워드 파일 파싱
- `parse_scripts_by_slide()`: 대본 파일 파싱

### tracker/sentence_tracker.py
- `SentenceMatcher`: 문장 유사도 매칭 클래스
- `find_matches()`: 문장에서 키워드 매칭
- `get_coverage_for_slide()`: 슬라이드 커버리지 계산

### run_tracker.py
- `main_analysis()`: 키워드 추적 메인 로직

---

## 라이센스

MIT License
