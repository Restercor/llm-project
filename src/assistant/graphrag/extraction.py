"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

SYSTEM_PROMPT = """Ты извлекаешь сущности и связи из текста документа сервисной службы.
Верни строго JSON по схеме:
{
  "entities": [{"type": "<тип>", "name": "<имя>"}],
  "relations": [{"subject_type": "...", "subject": "...", "type": "<тип связи>", "object_type": "...", "object": "..."}]
}

Типы сущностей:
- ProductLine — название линейки: Арктика, Бриз, Гольфстрим.
- Model — код модели: AR-210, AR-310, BZ-110, GF-400.
- Component — название узла строчными буквами: пусковое реле, компрессор.
- Part — номер детали: P-1042, P-1042B.
- ErrorCode — код ошибки: E-17.
- Procedure — номер регламента: SP-104.
- Supplier — название поставщика: ХолодКомплект, КомпрессорМаш.

Не извлекай как сущности:
- имена людей (инженеров, диспетчеров);
- названия городов и складов (Москва, Казань, Санкт-Петербург);
- даты (07.08.2025);
- номера отчётов и бюллетеней (R-2025-018, SB-2025-03);
- названия магазинов («Зелёная корзина»).
Если сомневаешься в типе — лучше не извлекай сущность.

Типы связей и их направление:
- BELONGS_TO: Model -> ProductLine
- HAS_PART: Model -> Part. Извлекай, если в тексте сказано, что в модели установлена деталь:
  «В AR-210 установлены узлы: … — деталь P-1042», «В состав AR-210 входит P-1042».
  Даже если деталь названа в списке через тире, это HAS_PART.
- IS_A: Part -> Component
- INDICATES: ErrorCode -> Component
- FIXED_BY: ErrorCode -> Procedure
- REPLACED_BY: Part -> Part
- SUPPLIED_BY: Part -> Supplier
- FAILED: Model -> Component

Пример 1 (руководство по модели):
Текст: "Модель AR-210 входит в линейку «Арктика». В состав входят: пусковое реле — P-1042, компрессор — P-1010. 
Поставщик P-1042 — ХолодКомплект."
Ответ: {
  "entities": [
    {"type": "Model", "name": "AR-210"},
    {"type": "ProductLine", "name": "Арктика"},
    {"type": "Part", "name": "P-1042"},
    {"type": "Part", "name": "P-1010"},
    {"type": "Component", "name": "пусковое реле"},
    {"type": "Component", "name": "компрессор"},
    {"type": "Supplier", "name": "ХолодКомплект"}
  ],
  "relations": [
    {"subject_type": "Model", "subject": "AR-210", "type": "BELONGS_TO", "object_type": "ProductLine", 
    "object": "Арктика"},
    {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", "object_type": "Part", "object": "P-1042"},
    {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", "object_type": "Part", "object": "P-1010"},
    {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", "object_type": 
    "Component", "object": "пусковое реле"},
    {"subject_type": "Part", "subject": "P-1010", "type": "IS_A", "object_type": "Component", "object": "компрессор"},
    {"subject_type": "Part", "subject": "P-1042", "type": "SUPPLIED_BY", "object_type": "Supplier", 
    "object": "ХолодКомплект"}
  ]
}

Пример 2 (бюллетень):
Текст: "Деталь P-1042 (пусковое реле) снята с производства. Вместо неё устанавливается P-1042B."
Ответ: {
  "entities": [
    {"type": "Part", "name": "P-1042"},
    {"type": "Part", "name": "P-1042B"},
    {"type": "Component", "name": "пусковое реле"}
  ],
  "relations": [
    {"subject_type": "Part", "subject": "P-1042", "type": "REPLACED_BY", "object_type": "Part", "object": "P-1042B"},
    {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", "object_type": "Component", "object": "пусковое реле"}
  ]
}

Пример 3 (руководство, раздел «Состав узлов»):
Текст: "## Состав узлов\n\nВ AR-210 установлены следующие узлы.\n\n
- компрессор – деталь P-1010, поставщик «КомпрессорМаш»\n- пусковое реле – деталь P-1042, поставщик «ХолодКомплект»"
Ответ: {
  "entities": [
    {"type": "Model", "name": "AR-210"},
    {"type": "Part", "name": "P-1010"},
    {"type": "Part", "name": "P-1042"},
    {"type": "Component", "name": "компрессор"},
    {"type": "Component", "name": "пусковое реле"},
    {"type": "Supplier", "name": "КомпрессорМаш"},
    {"type": "Supplier", "name": "ХолодКомплект"}
  ],
  "relations": [
    {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", "object_type": "Part", "object": "P-1010"},
    {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", "object_type": "Part", "object": "P-1042"},
    {"subject_type": "Part", "subject": "P-1010", "type": "IS_A", "object_type": "Component", "object": "компрессор"},
    {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", "object_type": "Component", 
    "object": "пусковое реле"},
    {"subject_type": "Part", "subject": "P-1010", "type": "SUPPLIED_BY", "object_type": "Supplier", 
    "object": "КомпрессорМаш"},
    {"subject_type": "Part", "subject": "P-1042", "type": "SUPPLIED_BY", "object_type": "Supplier", 
    "object": "ХолодКомплект"}
  ]
}
"""

CODE_TYPES = {"Part", "Model", "ErrorCode", "Procedure"}
NAME_TYPES = {"Component", "Supplier", "ProductLine"}

def resolve_entity(etype, ename, text_codes, vocab, doc_id=""):

    code = normalize_code(ename)
    if code:
        try:
            real_type = code_type(code)
        except KeyError:
            real_type = None
    else:
        real_type = None

    if real_type in CODE_TYPES:
        name = normalize_code(ename)
        if not name:
            return None
        if name not in text_codes and name not in doc_id:
            return None
        return Entity(real_type, name)
    
    if etype in NAME_TYPES:
        name = canonical_name(etype, ename, vocab)
        if name is None:
            return None
        return Entity(etype, name)

    return None

def parse_response(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    
    if not isinstance(data, dict):
        return None
    
    entities = data.get("entities")
    relations = data.get("relations")
    
    if not isinstance(entities, list) or not isinstance(relations, list):
        return None
    
    for e in entities:
        if not isinstance(e, dict) or "type" not in e or "name" not in e:
            return None
    
    for r in relations:
        required = {"subject_type", "subject", "type", "object_type", "object"}
        if not isinstance(r, dict) or not required <= r.keys():
            return None
    
    return data
    
def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    messages = [
        {"role" : "system", "content" : SYSTEM_PROMPT},
        {"role" : "user", "content" : f"Текст документа:\n\n{chunk.text}"}
    ]
    
    data = None
    
    for _ in range(2):
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
        except Exception:
            continue
        data = parse_response(response.content)
        
        if data is not None:
            break
        
    if data is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, [], [], dropped=0)
    
    # Entities filtering
    vocab = vocabulary()
    text_codes = {normalize_code(c) for c in find_codes(chunk.text)}
    for code in re.findall(r"[A-Z]+-\d+[A-Z]?", chunk.text):
        text_codes.add(normalize_code(code))
    
    kept : dict[str, Entity] = {}
    dropped = 0
    
    for e in data["entities"]:
        entity = resolve_entity(e["type"], e["name"], text_codes, vocab, chunk.doc_id)
        
        if entity is None:
            print("DROP entity:", e)
            dropped+=1
            continue
        kept[entity.node_id] = entity
    
    # Relation filtering
    relations : list[Relation] = []
    
    for r in data["relations"]:
        subj = resolve_entity(r["subject_type"], r["subject"], text_codes, vocab, chunk.doc_id)
        obj = resolve_entity(r["object_type"], r["object"], text_codes, vocab, chunk.doc_id)
        
        if subj is None or obj is None:
            print("DROP entity:", r)
            dropped+=1
            continue
        
        rel = Relation(subj, r["type"], obj)
        if not rel.matches_schema():
            reversed_rel = Relation(obj, r["type"], subj)
            
            if reversed_rel.matches_schema():
                rel = reversed_rel
            else:
                dropped += 1
                continue
            
        kept[subj.node_id] = subj
        kept[obj.node_id] = obj
        relations.append(rel)

    models = [e for e in kept.values() if e.type == "Model"]
    parts = [e for e in kept.values() if e.type == "Part"]
    text_lower = chunk.text.lower()
    composition_markers = ("установлен", "состав", "входит", "комплекту", "включает")
    if models and parts and any(m in text_lower for m in composition_markers):
        existing = {(r.subject.node_id, r.type, r.object.node_id) for r in relations}
        for model in models:
            for part in parts:
                key = (model.node_id, "HAS_PART", part.node_id)
                if key not in existing:
                    rel = Relation(model, "HAS_PART", part)
                    if rel.matches_schema():
                        relations.append(rel)
                        existing.add(key)
        
    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(kept.values()), relations, dropped=dropped)