# 발표 대본 자동 생성 시스템

`docs/` 폴더에 있는 발표자료(`paper.pdf`)와 참고자료(`report.pdf`, `docs.pdf`)를 기반으로  
각 슬라이드(페이지)별 발표 대본을 자동으로 생성하는 시스템입니다.

참고자료는 RAG(LangChain) 기반으로 검색하여 근거로 활용하며,  
LLM 및 Embedding 제공자는 **IBM Watsonx** 또는 **Upstage(Solar)** 중 선택해 사용할 수 있습니다.

---

## 개요

- **RAG 기반 검색**
  - `report.pdf`, `docs.pdf`를 벡터 DB(Chroma)에 저장
  - 슬라이드 내용과 관련된 참고 문서를 검색해 대본에 반영

- **슬라이드 파싱**
  - `paper.pdf`를 페이지(슬라이드) 단위로 파싱
  - IBM API는 토큰 제한이 있어, Upstage OCR 사용을 권장

- **모델 선택**
  - 실행 시 `--api` 인자로 IBM 또는 Upstage 중 선택 가능

- **출력**
  - `outputs/` 폴더에 슬라이드별 발표 대본 파일 생성  
    (예: `outputs/paper_scripts.md`)

---

## 요구사항 및 설치

- Python 3.11.9
- 필수 패키지는 `requirements.txt`에 정의되어 있음

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt


Upstage 연동을 위해 `langchain-openai` 패키지가 필요하며,
해당 패키지는 `requirements.txt`에 이미 포함되어 있습니다.

---

## 구성 파일

### `config.py`

`config.py`에서 시스템의 기본 동작을 설정합니다.

주요 설정 항목:

* **경로**

  * `docs_root`: 문서 루트 폴더 (기본값: `./docs`)
  * `out_dir`: 결과 파일 저장 폴더 (기본값: `./outputs`)

* **RAG / 벡터 DB**

  * `persist_dir`, `collection_ref`, `top_k`
  * Chroma 벡터 DB 저장 및 검색 설정

* **텍스트 분할**

  * `chunk_size`, `chunk_overlap`
  * IBM Embeddings는 토큰 제한이 있으므로 400자 내외를 권장

* **IBM Watsonx**

  * `embed_model`, `llm_model`

* **Upstage 문서 파싱**

  * `upstage_url`, `upstage_model`
  * `upstage_ocr`: OCR 강제 여부 (`force` 권장)
  * `upstage_base64_encoding`: 테이블 인식 옵션

* **캐시**

  * `cache_dir`: Upstage 문서 파싱 결과 캐시 디렉터리
    (API 호출 비용 절감 목적)

---

## 환경 변수 설정 (`.env`)

본 프로젝트는 실행 시 `.env` 파일을 자동으로 로드합니다.
따라서 별도의 `export` 설정 없이 바로 실행할 수 있습니다.

### IBM Watsonx

```env
API_KEY=your_ibm_api_key
PROJECT_ID=your_ibm_project_id
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
```

### Upstage

```env
UPSTAGE_API_KEY=your_upstage_api_key
# 선택 사항: 문서 파싱 URL 오버라이드
UPSTAGE_API_URL=https://api.upstage.ai/v1/document-digitization
```

---

## 문서 폴더 구조

`docs/` 폴더에는 아래 파일명을 사용합니다.
파일명은 고정되어 있으며 변경하지 않는 것을 권장합니다.

```
docs/
 ├─ paper.pdf   # 발표자료(슬라이드)
 ├─ report.pdf  # 결과 보고서 / 기술 명세서
 └─ docs.pdf    # 공고, 정책, 평가 기준 등 참고자료
```

---

## 사용 방법 (`main.py`)

실행 시 `--api` 인자로 사용할 제공자를 선택합니다.
필요한 환경 변수는 `.env` 파일에 설정되어 있어야 합니다.

### Upstage(Solar) 사용

```bash
python main.py --api upstage
```

### IBM Watsonx 사용

```bash
python main.py --api ibm
```

실행이 완료되면 `outputs/` 폴더에 발표 대본 파일이 생성됩니다.

예시:

```
outputs/
 └─ paper_scripts.md
```

---

## 테스트 스크립트 (`parse_test_ibm.py`)

`parse_test_ibm.py`는 IBM Watsonx 기반 RAG 파이프라인을 단독으로 검증하기 위한 예제 스크립트입니다.

주요 내용:

* 참고 문서 로드 및 텍스트 분할
* 벡터 DB 생성
* RAG 체인 구성
* 슬라이드별 발표 대본 생성 및 저장

실행 예시:

```bash
python parse_test_ibm.py
```

---

## 테스트 스크립트 (`parse_test_upstage.py`)

`parse_test_upstage.py`는 IBM Watsonx 기반 RAG 파이프라인을 단독으로 검증하기 위한 예제 스크립트입니다.

주요 내용:

* 참고 문서 로드 및 텍스트 분할
* 벡터 DB 생성
* RAG 체인 구성
* 슬라이드별 발표 대본 생성 및 저장

실행 예시:

```bash
python parse_test_upstage.py
```
