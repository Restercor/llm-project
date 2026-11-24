"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult, EXTRACTION_SCHEMA, Entity, Relation, ENTITY_TYPES
from assistant.infra.config import vocabulary
from assistant.infra.grounding import find_codes, normalize_code, canonical_name


SYSTEM_PROMPT = """


"""

CODE_TYPES = {"Part", "Model", "ErrorCode", "Procedure"}
NAME_TYPES = {"Component", "Supplier", "ProductLine"}

def resolve_entity(etype : str, ename : str, text_codes : set[str], vocab : dict):
    if etype not in ENTITY_TYPES:
        return None
    
    if etype in CODE_TYPES:
        name = normalize_code(ename)
        if name not in text_codes:
            return None
    elif etype in NAME_TYPES:
        name = canonical_name(etype, ename, vocab)
        if name is None:
            return None
    else:
        return None
    
    return Entity(etype, name)

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
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
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
    
    kept : dict[str, Entity] = {}
    dropped = 0
    
    for e in data["entities"]:
        entity = resolve_entity(e["type"], e["name"], text_codes, vocab)
        
        if entity is None:
            dropped+=1
            continue
        kept[entity.node_id] = entity
    
    # Relation filtering
    relations : list[Relation] = []
    
    for r in data["relations"]:
        subj = resolve_entity(r["subject_type"], r["subject"], text_codes, vocab)
        obj = resolve_entity(r["object_type"], r["object"], text_codes, vocab)
        
        if subj is None or obj is None:
            dropped+=1
            continue
        
        rel = Relation(subj, r["type"], obj)
        
        if not rel.matches_schema():
            dropped+=1
            continue
        kept[subj.node_id] = subj
        kept[obj.node_id] = obj
        relations.append(rel)
        
    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(kept.values()), relations, dropped=dropped)