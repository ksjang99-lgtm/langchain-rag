from typing import List, Dict, Any
from openai import OpenAI
from utils_text import robust_json_loads


def openai_answer_with_rag(
    client: OpenAI,
    model: str,
    question: str,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    contexts: [{"page_number": int, "content": str, "similarity": float}, ...]
    return: {"answer": str, "cited_pages": [int, ...]}
    """
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[page={c['page_number']}, similarity={c['similarity']:.3f}]\n{c['content']}")
    ctx_text = "\n\n---\n\n".join(ctx_lines)

    system = (
        "너는 장비 매뉴얼 PDF를 기반으로만 답하는 고객지원 챗봇이다.\n"
        "규칙:\n"
        "1) 아래 제공된 '매뉴얼 발췌'에 없는 정보는 절대 추측하지 말고, 반드시 '문서에 존재하지 않습니다.' 라고 답하라.\n"
        "2) 답변은 한국어로, 간결하되 사용자가 바로 실행할 수 있게 단계형으로 작성하라.\n"
        "3) 답변에 근거가 된 페이지 번호를 cited_pages 배열로 반드시 포함하라.\n"
        "4) 출력은 JSON 하나로만: {\"answer\": string, \"cited_pages\": number[]}\n"
    )

    user = f"질문:\n{question}\n\n매뉴얼 발췌:\n{ctx_text}\n"

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text_out = resp.output_text

    data = robust_json_loads((text_out or "").strip())
    if not data or "answer" not in data:
        return {"answer": "문서에 존재하지 않습니다.", "cited_pages": []}

    raw_pages = data.get("cited_pages", [])
    cited_pages: List[int] = []
    for p in raw_pages:
        try:
            cited_pages.append(int(p))
        except Exception:
            pass

    cited_pages = sorted(set(cited_pages))
    return {"answer": str(data.get("answer", "")).strip(), "cited_pages": cited_pages}
