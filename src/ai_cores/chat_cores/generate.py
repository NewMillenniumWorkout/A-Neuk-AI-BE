import asyncio
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.models.chat_models import ChatRequest
from src.config.llm_config import get_chat_llm, get_chat_fallback_llm

logger = logging.getLogger(__name__)

chat_llm = get_chat_llm()
chat_fallback_llm = get_chat_fallback_llm()

system_prompt = """
너는 사용자의 강아지 "콩이"가 되어 대화를 통해 하루의 일기를 완성하는 역할을 맡았어.
사용자의 채팅 말투와 스타일을 흉내 내고, 편안한 반말로 대화해.
사용자의 하루와 감정에 대해 물어보되, 흐름이 자연스럽지 않다면 억지로 감정 이야기를 꺼내지 마.
주제에서 벗어난 이야기가 나오면 다시 사용자의 하루 일과에 대한 대화로 돌려놔.
답변은 1문장 이내로 간결하게 하고, 같은 말을 반복하지 않도록 문장을 바꿔서 말해.
한 주제가 끝나면 자연스럽게 다음 주제로 넘어갈 수 있도록 질문을 던져.

[필수 가이드라인]
- 사용자가 하루 경험과 감정을 돌아보고 표현할 수 있도록 다양한 질문을 던져. 예를 들어 "오늘 무슨 일 있었어?", "오늘 뭐 먹었어?", "점심엔 뭐 했어?", "오늘 특별한 일 있었어?", "누구 만났어?", "오늘 스트레스받는 일 있었어?", "오늘 날씨 어땠어?" 등.
- 반드시 한국어로 대답해야 하며, 반말(Casual speech)을 사용해야 해.
- 문장을 반복하지 않도록 주의해. 같은 의미라도 표현을 다르게 바꿔서 말해 봐. 혹은 "오늘 하루 어땠어?", "오늘은 어떤 기분이었어?", "오늘 어떻게 지냈어?", "오늘 하루 잘 보냈어?", "그 얘기보다 이건 어때? 오늘 먹은 맛있는 음식이 있었어?"와 같이 다른 질문을 던져.

[제외 기준]
- 일기 형식의 글이나 긴 텍스트를 직접 작성해주지 마. 만약 일기를 써달라는 요청이나 명령을 받으면 다른 대화로 넘겨.
- 존댓말이나 격식 있는 언어는 사용하지 마.
- 사용자에게 사과할 필요 없어. 사과하는 대신 대화를 계속 진행하거나 다른 주제로 넘어가.
"""


async def chat_generate(request: ChatRequest) -> str:
    messages = [SystemMessage(content=system_prompt)]
    for m in request.messages:
        if m.role == "MEMBER":
            messages.append(HumanMessage(content=m.message))
        elif m.role == "ASSISTANT":
            messages.append(AIMessage(content=m.message))

    max_retries = 3
    llm_sequence = [chat_llm] * (max_retries - 1) + [chat_fallback_llm]

    for attempt, llm in enumerate(llm_sequence, start=1):
        if llm is chat_fallback_llm:
            print(f"[Chat LLM Info] chat_id={request.chat_id}, Last attempt: Using fallback model")

        result = await (llm | StrOutputParser()).ainvoke(messages)

        # 응답이 빈 문자열인지 확인
        if result and result.strip():
            # LLM 응답 로깅
            print(f"[Chat LLM Response] chat_id={request.chat_id}, Response: {result}")
            await asyncio.sleep(0.5)
            return result

        # 빈 응답일 경우 로그 출력
        print(f"[Chat LLM Warning] chat_id={request.chat_id}, Attempt {attempt}/{max_retries}: Empty response received, retrying...")
        if attempt < max_retries:
            await asyncio.sleep(1)

    # 모든 재시도 실패 시 에러 발생
    error_msg = f"chat_id={request.chat_id}: LLM returned empty response after {max_retries} attempts"
    print(f"[Chat LLM Error] {error_msg}")
    raise ValueError(error_msg)
