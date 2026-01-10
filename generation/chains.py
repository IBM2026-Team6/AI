"""
발표 대본 생성 체인 구성 모듈.

역할:
- 슬라이드 텍스트만으로 대본 생성
- RAG 활성화 시, 참고 컨텍스트(report/docs)로 근거를 보강하여 대본 생성

출력 포맷은 일관된 후처리를 위해 고정되어 있으며,
`generation/postprocess.py`에서 후처리 단계를 거칠 수 있습니다.
"""

from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from rag.vectorstore import format_docs_with_citations


def _truncate_for_retriever(x: str, max_chars: int = 2000) -> str:
    """리트리버 입력을 지정 길이로 절단.

    일부 임베딩 API가 너무 긴 쿼리를 거부하는 상황을 완화하기 위한 안전장치.
    LLM에는 원본 슬라이드 텍스트가 그대로 전달되며, 리트리버 입력만 절단됨.
    """
    try:
        if not isinstance(x, str):
            x = str(x or "")
        x = x.strip()
        if len(x) > max_chars:
            return x[:max_chars]
        return x
    except Exception:
        return str(x)[:max_chars]


def build_slide_chain(llm, retriever: Optional[object] = None):
    """슬라이드 대본 생성 체인 구성.

    Args:
        llm: 대본 생성을 수행할 Chat 모델 인스턴스
        retriever: RAG 검색기 (None이면 비-RAG 모드)

    Returns:
        LCEL 파이프라인 객체. `.invoke(slide_text)`로 문자열 대본을 반환.
    """
    if retriever is None:
        # 비-RAG 모드: 슬라이드 텍스트만으로 대본 생성
        template = """
너는 발표 대본 작성자다. 다음 규칙을 반드시 지켜라.

[규칙]
1) 대본은 '현재 슬라이드 텍스트'를 최우선으로 따라가라. 슬라이드에 없는 주제는 생략한다.
2) 추측/과장 금지. 슬라이드 내용만으로 대본을 작성한다.
3) 한국어 구어체. 30~60초 분량.
4) 출력 포맷을 반드시 지켜라.

[현재 슬라이드 텍스트]
{slide_text}

[출력 포맷]
- 슬라이드 대본:
(여기에 발표 대본 작성)

- 핵심 메시지 3개:
1. 
2. 
3. 

- 예상 질문 2개 + 답변:
Q1) 
A1) 

Q2) 
A2) 
"""
        prompt = ChatPromptTemplate.from_template(template)
        return (
            RunnableParallel({"slide_text": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
    
    # RAG 활성화 모드: 슬라이드 텍스트 + 검색 컨텍스트를 함께 프롬프트에 주입
    template = """
너는 발표 대본 작성자다. 다음 규칙을 반드시 지켜라.

[규칙]
1) 대본은 '현재 슬라이드 텍스트'를 최우선으로 따라가라. 슬라이드에 없는 주제는 생략한다.
2) '참고 컨텍스트'는 report/docs 기반이므로 근거/표현/평가기준에 맞춘 보강에만 사용한다.
3) 추측/과장 금지. 컨텍스트 밖 정보 금지.
4) 한국어 구어체. 30~60초 분량.
5) 출력 포맷을 반드시 지켜라.

[현재 슬라이드 텍스트]
{slide_text}

[참고 컨텍스트]
{context}

[출력 포맷]
- 슬라이드 대본:
(여기에 발표 대본 작성)

- 핵심 메시지 3개:
1. 
2. 
3. 

- 예상 질문 2개 + 답변:
Q1) 
A1) 

Q2) 
A2) 
"""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        RunnableParallel(
            {
                "slide_text": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: _truncate_for_retriever(x))
                | retriever
                | format_docs_with_citations,
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
