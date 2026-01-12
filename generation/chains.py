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
    





def build_slide_chain(llm, retriever=None, audience="general"):
    """슬라이드 대본 생성 체인 구성.

    Args:
        llm: 대본 생성을 수행할 Chat 모델 인스턴스
        retriever: RAG 검색기 (None이면 비-RAG 모드)

    Returns:
        LCEL 파이프라인 객체. `.invoke(slide_text)`로 문자열 대본을 반환.
    """

    # 비-RAG 모드: 슬라이드 텍스트만으로 대본 생성
    if retriever is None:
        template =(
            EXPERT_PROMPT_NO_RAG
            if audience == "expert" 
            else GENERAL_PROMPT_NO_RAG
        )

        prompt = ChatPromptTemplate.from_template(template)

        return (
            RunnableParallel({"slide_text": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
    
    # RAG 활성화 모드: 슬라이드 텍스트 + 검색 컨텍스트를 함께 프롬프트에 주입
    else:
        template =(
            EXPERT_PROMPT
            if audience == "expert" 
            else GENERAL_PROMPT
        )

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



EXPERT_PROMPT = """
당신은 '전문가 대상으로 발표를 준비하는 사용자를
도와주는 발표 코치이자 발표 도우미'이다.

아래 요구사항을 기반으로,
전문가 청중을 대상으로 한 
학술적·전문적 발표 대본을 작성해라.

요구사항:
1. 슬라이드 텍스트를 최우선으로 따른다.
2. '참고 컨텍스트'는 report/docs 기반이므로 
   근거/표현/평가기준에 맞춘 보강에만 사용한다.
3. “청중 분석 → 목적 설정 → 구조화 → 전달 방식”
   이라는 논리 흐름을 분명히 유지해라.
4. 추측/과장 금지. 컨텍스트 밖 정보 금지.
5. 발표 대본 형식으로 작성하되,
   학회·세미나 발표에서 사용 가능한 수준의
   전문성과 논리를 유지해라.
6. 기본 분량은 한 페이지당 약 40초~60초 분량이지만,
   사용자가 원하는 분량이 있다면 그에 맞춰라.
7. 한국어 구어체를 사용해라.
8. 출력 포맷을 반드시 지켜라.
9. 슬라이드 내용의 흐름이 이어지도록 대본을 작성해라.

문서 내용:

[현재 슬라이드 텍스트]
{slide_text}

[참고 컨텍스트]
{context}

[출력 포맷]
- 슬라이드 대본:
(발표 대본)

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


GENERAL_PROMPT = """
당신은 ‘비전문가 대상으로 발표를 준비하는 사용자를
도와주는 발표 코치이자 발표 도우미’이다.

아래 요구사항을 기반으로,
발표 경험이 많지 않은 일반 청중(비전문가)을 대상으로 한
발표용 대본을 작성해라.

요구사항:
1. 슬라이드 텍스트를 최우선으로 따른다.
2. '참고 컨텍스트'는 report/docs 기반이므로 
   근거/표현/평가기준에 맞춘 보강에만 사용한다.
3. 전문 용어나 학술적 표현은 최대한 풀어서 설명하고,
   일상적인 언어로 쉽게 설명해라.
4. “왜 이 내용이 중요한지”를 청중의 입장에서 설명해라.
5. 추상적인 개념은 반드시 간단한 예시를 들어 설명해라.
6. 대본 형식으로 작성하며,
   실제 사람이 말하듯 자연스러운 문장으로 작성해라.
7. 기본 분량은 한 페이지당 약 40초~60초 분량이지만,
   사용자가 원하는 분량이 있다면 그에 맞춰라.
8. 한국어 구어체를 사용해라.
9. 출력 포맷을 반드시 지켜라.
10. 슬라이드 내용의 흐름이 이어지도록 대본을 작성해라.

문서 내용:

[현재 슬라이드 텍스트]
{slide_text}

[참고 컨텍스트]
{context}

[출력 포맷]
- 슬라이드 대본:
(발표 대본)

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



EXPERT_PROMPT_NO_RAG = """
당신은 '전문가 대상으로 발표를 준비하는 사용자를
도와주는 발표 코치이자 발표 도우미'이다.

아래 요구사항을 기반으로,
전문가 청중을 대상으로 한 
학술적·전문적 발표 대본을 작성해라.

요구사항:
1. 슬라이드 텍스트를 최우선으로 따른다.
2. '참고 컨텍스트'는 report/docs 기반이므로 
   근거/표현/평가기준에 맞춘 보강에만 사용한다.
3. “청중 분석 → 목적 설정 → 구조화 → 전달 방식”
   이라는 논리 흐름을 분명히 유지해라.
4. 추측/과장 금지. 컨텍스트 밖 정보 금지.
5. 발표 대본 형식으로 작성하되,
   학회·세미나 발표에서 사용 가능한 수준의
   전문성과 논리를 유지해라.
6. 기본 분량은 한 페이지당 약 40초~60초 분량이지만,
   사용자가 원하는 분량이 있다면 그에 맞춰라.
7. 한국어 구어체를 사용해라.
8. 출력 포맷을 반드시 지켜라.
9. 슬라이드 내용의 흐름이 이어지도록 대본을 작성해라.

문서 내용:

[현재 슬라이드 텍스트]
{slide_text}

[참고 컨텍스트]
X

[출력 포맷]
- 슬라이드 대본:
(발표 대본)

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


GENERAL_PROMPT_NO_RAG = """
당신은 ‘비전문가 대상으로 발표를 준비하는 사용자를
도와주는 발표 코치이자 발표 도우미’이다.

아래 요구사항을 기반으로,
발표 경험이 많지 않은 일반 청중(비전문가)을 대상으로 한
발표용 대본을 작성해라.

요구사항:
1. 슬라이드 텍스트를 최우선으로 따른다.
2. '참고 컨텍스트'는 report/docs 기반이므로 
   근거/표현/평가기준에 맞춘 보강에만 사용한다.
3. 전문 용어나 학술적 표현은 최대한 풀어서 설명하고,
   일상적인 언어로 쉽게 설명해라.
4. “왜 이 내용이 중요한지”를 청중의 입장에서 설명해라.
5. 추상적인 개념은 반드시 간단한 예시를 들어 설명해라.
6. 대본 형식으로 작성하며,
   실제 사람이 말하듯 자연스러운 문장으로 작성해라.
7. 기본 분량은 한 페이지당 약 40초~60초 분량이지만,
   사용자가 원하는 분량이 있다면 그에 맞춰라.
8. 한국어 구어체를 사용해라.
9. 출력 포맷을 반드시 지켜라.
10. 슬라이드 내용의 흐름이 이어지도록 대본을 작성해라.


문서 내용:

[현재 슬라이드 텍스트]
{slide_text}

[참고 컨텍스트]
X

[출력 포맷]
- 슬라이드 대본:
(발표 대본)

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