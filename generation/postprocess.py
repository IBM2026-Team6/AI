def postprocess_script(text: str) -> str:
    t = (text or "").strip()

    # 연속 중복 라인 제거
    lines = t.splitlines()
    new_lines = []
    prev = None
    for line in lines:
        cur = line.strip()
        if cur and cur == prev:
            continue
        new_lines.append(line)
        prev = cur
    t = "\n".join(new_lines).strip()
    
    return t if t else "(모델 출력 없음)"

    # required = [
    #     "- 슬라이드 대본:",
    #     "- 핵심 메시지 3개:", 
    #     "- 예상 질문 2개 + 답변:"
    # ]

    # if not all(k in t for k in required):
    #     t = (
    #         "- 슬라이드 대본:\n(출력 포맷이 깨졌습니다. 슬라이드 텍스트 추출/프롬프트/모델 설정 점검 필요)\n\n"
    #         "- 핵심 메시지 3개:\n- \n- \n- \n\n"
    #         "- 예상 질문 2개 + 답변:\nQ1) \nA1) \n\nQ2) \nA2) "
    #     )
    # return t
