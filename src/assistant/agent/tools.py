"""Инструменты агента. Требование FR-6."""
import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.

PART_RE = re.compile(r"^P-\d+[A-Z]?$")

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

TOOL_SCHEMAS: list[dict] = [
    {
        "type" : "function",
        "function" : {
            "name" : "search_knowledge",
            "description" : "Найти ответ на вопрос по графу знаний сервисной службы.",
            "parameters" : {
                "type" : "object",
                "properties" : {"question" : {"type" : "string", "description" : "Вопрос диспетчера"}},
                "required" : ["question"]
            }
        }
    },
    {
        "type" : "function",
        "function" : {
            "name" : "summarize_topic",
            "description" : "Обзорный отввет по сводкам сообществ графа знаний.",
            "parameters" : {
                "type" : "object",
                "properties" : {"question" : {"type" : "string"}},
                "required" : ["question"]
            }
        }  
    },
    {
        "type" : "function",
        "function" : {
            "name" : "check_stock",
            "description" : "Остаток деталей на склади и признак снятия с произвводства.",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "part_id" : {"type" : "string", "description" : "Номер детали, например P-1042B"},
                    "warehouse" : {"type" : "string", "enum" : WAREHOUSES}
                },
                "required" : ["part_id", "warehouse"]
            }
        }
    },
    {
        "type" : "function",
        "function" : {
            "name" : "get_price",
            "description" : "Цена детали в рублях и признак снятия с впроизвводства.",
            "parameters" : {
                "type" : "object",
                "properties" : {"part_id" : {"type" : "string"}},
                "required" : ["part_id"]
            }
        }
    },
    {
        "type" : "function",
        "function" : {
            "name" : "create_service_order",
            "description" : "Создать заказ детали на склад.",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "part_id" : {"type" : "string"},
                    "warehouse" : {"type" : "string", "enum" : WAREHOUSES},
                    "qty" : {"type" : "integer", "minimum" : 1},
                    "model" : {"type" : "string", "description" : "Модель оборудования, например AR-210"},
                },
                "required" : ["part_id", "warehouse", "qty", "model"]
            }
        }
    }
]

def validate_part(part_id : str) -> str | None:
    if not isinstance(part_id, str) or not PART_RE.match(part_id):
        return "номер детали должен быть вида P-1042 или P-1042B"
    return None

def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    warehouses = erp.warehouses()
    
    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}
    
    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}
    
    def check_stock(part_id : str, warehouse: str) -> dict:
        if (err := validate_part(part_id)):
            return {"error" : err}
        if warehouse not in warehouses:
            return {"error" : f"склад {warehouse!r} не найден"}
        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound as exc:
            return {"error": f"детали {part_id} нет в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}
        
    def get_price(part_id: str) -> dict:
        if (err := validate_part(part_id)):
            return {"error": err}
        try:
            return erp.get_price(part_id)
        except ErpNotFound as exc:
            return {"error": f"детали {part_id} нет в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}
        
    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if (err := validate_part(part_id)):
            return {"error" : err}
        if warehouse not in warehouses:
            return {"error" : f"склад {warehouse!r} не найден"}
        
        if not isinstance(qty, int) or qty <= 0:
            return {"error" : "количество должно быть больше 0"}
        
        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as exc:
            return {"error": f"детали {part_id} нет в ERP: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила заказ: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
