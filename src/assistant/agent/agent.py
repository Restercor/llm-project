"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM_PROMPT = """Ты — ассистент диспетчера сервисной службы. Отвечай на вопросы, используя инструменты.
Не выдумывай коды деталей, цены и остатки: бери их только из результатов инструментов.
Если для ответа нужен номер детали, сначала найди его через search_knowledge."""

PART_RE = re.compile(r"\b[A-Z]+-\d+[A-Z]?\b")
PRICE_RE = re.compile(r"\b\d{3,6}\b")
QTY_RE = re.compile(r"\b(\d+)\s*(?:шт|штук|штуки|единиц)", re.IGNORECASE)

def mentioned(text: str, known: set[str]) -> bool:
    return any(code in text for code in known)

def codes_in(text: str) -> set[str]:
    return set(PART_RE.findall(text))

def numbers_in(text: str) -> set[str]:
    cleaned = PART_RE.sub(" ", text)
    cleaned = re.sub(r"\bSO-\d+\b", " ", cleaned)
    return set(PRICE_RE.findall(cleaned))

def qty_in_question(question: str) -> int | None:
    match = QTY_RE.search(question)
    return int(match.group(1)) if match else None

def check_answer(answer: str, question: str, known_codes: set[str], known_numbers: set[str]) -> bool:
    for code in codes_in(answer):
        if code not in known_codes and code not in question:
            return False
    
    for number in numbers_in(answer):
        if number not in known_numbers and number not in question:
            return False
    
    return True

def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    messages = [
        {"role" : "system", "content" : SYSTEM_PROMPT},
        {"role" : "user" , "content" : question}
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    cache: dict[tuple[str, str], dict] = {}
    known_codes: set[str] = codes_in(question)
    known_numbers: set[str] = numbers_in(question)
    pending_action: dict | None = None
    
    for _ in range(max_steps):
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        
        if not response.tool_calls:
            break
        
        for call in response.tool_calls:
            name = call.name
            args = call.arguments
            step = ToolStep(tool=name, args=args)
            
            if name not in tools:
                step.error = f"неизвестный инструмент {name!r}"
                steps.append(step)
                messages.append(tool_message(call, {"error" : step.error}))
                continue
            
            if "_raw" in args:
                step.error = "аргументы инструмента не разобраны"
                steps.append(step)
                messages.append(tool_message(call, {"error" : step.error}))
                continue
            
            part_id = args.get("part_id")
            if part_id is not None and part_id not in known_codes and part_id not in question:
                step.error = f"номер детали {part_id} не встречается в вопросе или результатах"
                steps.append(step)
                messages.append(tool_message(call, {"error" : step.error}))
                continue
            
            if name == "create_service_order":
                qty = args.get("qty")
                asked = qty_in_question(question)
                
                if asked is None or qty != asked:
                    step.error = "количество в заказе не названо в вопросе"
                    steps.append(step)
                    messages.append(tool_message(call, {"error" : step.error}))
                    pending_action = None
                    continue
                
                if not confirm:
                    pending_action = {"tool": name, "args": args}
                    steps.append(step)
                    return AgentResult(
                        answer="Заказ требует подтверждения",
                        steps=steps,
                        needs_confirmation=True,
                        pending_action=pending_action
                    )
            
            cache_key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            if cache_key in cache:
                result = cache[cache_key]
            else:
                result = None
                for attempt in range(3):
                    try:
                        result = tools[name](**args)
                        break
                    except ErpUnavailable as exc:
                        if attempt == 2:
                            step.error = f"ERP недоступна: {exc}"
                            break
                        
                if result is None and step.error is None:
                    step.error = "ERP недоступна"
                if result is not None:
                    cache[cache_key] = result
            
            step.result = result
            steps.append(step)
            
            if result is not None:
                for value in result.values():
                    if isinstance(value, str):
                        known_codes |= codes_in(value)
                        known_numbers |= numbers_in(value)
                    elif isinstance(value, (int, float)):
                        known_numbers.add(str(value))
                        
                if isinstance(result.get("sources"), list):
                    sources.update(result["sources"])
                
            messages.append(tool_message(call, result if result is not None else {"error" : step.error}))
    else:
        return AgentResult(
            answer="Работа прервана: лимит шагов исчерпан.",
            steps=steps,
            sources=sorted(sources),
        )
    
    answer = response.content.strip() if response else ""
    
    if any(step.result and step.result.get("refused") for step in steps):
        return AgentResult(answer=answer, steps=steps, sources=sorted(sources), refused=True)
    
    if answer and not check_answer(answer, question, known_codes, known_numbers):
        messages.append({
            "role": "user", 
            "content": "В ответе есть коды или числа, которых нет в результатах инструментов. Исправь ответ."
            })
        response = llm.chat(messages)
        answer = response.content.strip()
        if not check_answer(answer, question, known_codes, known_numbers):
            return AgentResult(answer="в базе знаний не найдено", steps=steps, sources=sorted(sources), refused=True)
    
    if not answer:
        return AgentResult(answer="в базе знаний не найдено", steps=steps, sources=sorted(sources), refused=True)
    
    return AgentResult(answer=answer, steps=steps, sources=sorted(sources))
        
