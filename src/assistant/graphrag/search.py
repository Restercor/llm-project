"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase, MAP_SCHEMA
from assistant.infra.grounding import find_codes, find_names, normalize_code, canonical_name

import networkx as nx
import json

LOCAL_PROMPT = """Ты отвечаешь на вопрос диспетчера сервисной службы строго по фактам из графа знаний.
Если в фактах есть цепочка замен (REPLACED_BY), называй последнюю деталь цепочки.
Не выдумывай коды, цены и остатки. Если фактов недостаточно, скажи, что в базе знаний не найдено.

Вопрос: {question}

Факты:
{facts}
"""

MAP_PROMPT = """Ниже сводка сообщества графа знаний сервисной службы.
Ответь на вопрос, если сводка содержит полезные данные. Иначе верни score 0.
Верни JSON: {{"answer": "...", "score": <0-10>}}.

Вопрос: {question}

Сводка:
{summary}
"""

REDUCE_PROMPT = """Ниже частичные ответы на вопрос, собранные по разным сообществам графа знаний.
Объедини их в один связный ответ. Не выдумывай новые факты.

Вопрос: {question}

Частичные ответы:
{partials}
"""


def node_id_for_code(graph: nx.DiGraph, code: str) -> str | None:
    norm = normalize_code(code)
    
    for node_id in graph.nodes:
        _, ename = node_id.split(":", 1)
        if ename == norm:
            return node_id
    
    return None

def node_id_for_name(graph: nx.DiGraph, etype: str, ename: str, vocab: dict) -> str | None:
    canon = canonical_name(etype, ename, vocab)
    
    if canon is None:
        return None
    
    node_id = f"{etype}:{canon}"
    
    return node_id if node_id in graph else None

def format_edge(graph: nx.DiGraph, u: str, v: str, data: dict) -> str:
    su = graph.nodes[u].get("name", u)
    sv = graph.nodes[v].get("name", v)
    return f"{su} -{data.get('type', '?')}-> {sv}"
    

def collect_facts(graph: nx.DiGraph, anchors: list[str]) -> tuple[list[str], set[str]]:
    facts: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    used_parts: set[str] = set()
    used_nodes: set[str] = set(anchors)

    def add_edge(u: str, v: str) -> None:
        data = graph.get_edge_data(u, v)
        
        if data is None:
            return
        key = (u, data.get("type", ""), v)
        
        if key in seen:
            return
        
        seen.add(key)
        facts.append(format_edge(graph, u, v, data))
        used_nodes.add(u)
        used_nodes.add(v)

        if u.startswith("Part:"):
            used_parts.add(u)
        if v.startswith("Part:"):
            used_parts.add(v)

    if len(anchors) >= 2:
        undirected = graph.to_undirected()
        for i, a in enumerate(anchors):
            for b in anchors[i + 1:]:
                try:
                    paths = list(nx.all_simple_paths(undirected, a, b, cutoff=3))
                except nx.NetworkXNoPath:
                    paths = []
                    
                for path in paths:
                    for u, v in zip(path, path[1:]):
                        if graph.has_edge(u, v):
                            add_edge(u, v)
                        if graph.has_edge(v, u):
                            add_edge(v, u)
                            
    elif len(anchors) == 1:
        anchor = anchors[0]
        
        for u, v in nx.bfs_edges(graph, anchor, depth_limit=2):
            add_edge(u, v)
        for u, v in nx.bfs_edges(graph.reverse(), anchor, depth_limit=2):
            if graph.has_edge(v, u):
                add_edge(v, u)

    used_parts.update(a for a in anchors if a.startswith("Part:"))
    queue = list(used_parts)
    
    while queue:
        part = queue.pop()
        
        for _, v, data in graph.out_edges(part, data=True):
            if data.get("type") == "REPLACED_BY" and (part, "REPLACED_BY", v) not in seen:
                add_edge(part, v)
                
                if v.startswith("Part:"):
                    queue.append(v)

    return facts, used_nodes

def parse_map(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    
    if not isinstance(data, dict):
        return None
    
    if not isinstance(data.get("answer"), str) or not isinstance(data.get("score"), int):
        return None
    
    return data
        

def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
    graph = kb.graph
    vocab = kb.vocabulary
    anchors: list[str] = []
    
    for code in find_codes(question):
        node_id = node_id_for_code(graph, code)
        
        if node_id is None:
            return Answer(text="в базе знаний не найдено", refused=True)
        
        anchors.append(node_id)
        
    for etype, ename in find_names(question, vocab):
        node_id = node_id_for_name(graph, etype, ename, vocab)
        if node_id is not None:
            anchors.append(node_id)
                
    anchors = list(dict.fromkeys(anchors)) # dedup
    
    if not anchors:
        return Answer(text="в базе знаний не найдено", refused=True)
    
    facts, used_nodes = collect_facts(graph, anchors)
    if not facts:
        return Answer(text="в базе знаний не найдено", refused=True)
    
    numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    if len(numbered) > 8000:
        numbered = numbered[:8000]
        
    prompt = LOCAL_PROMPT.format(question=question, facts=numbered)
    response = llm.chat([{"role" : "user", "content" : prompt}])
    
    sources: set[str] = set()
    for node_id in used_nodes:
        for chunk_id in graph.nodes[node_id].get("sources", []):
            sources.add(chunk_id.split("#")[0])
                                
    return Answer(text=response.content.strip(), sources=sorted(sources), facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    partials: list[str] = []
    
    for community in kb.communities:
        if not community.summary:
            continue
        
        prompt = MAP_PROMPT.format(question=question, summary=community.summary)
        response = llm.chat([{"role" : "user", "content" : prompt}], json_schema=MAP_SCHEMA)
        data = parse_map(response.content)
        
        if data is None:
            continue
        if data["score"] > 0:
            partials.append(data["answer"])
            
    if not partials:
        return Answer(text="в базе знаний не найдено", refused=True)
    
    prompt = REDUCE_PROMPT.format(question=question, partials="\n".join(f"- {p}" for p in partials))
    response = llm.chat([{"role" : "user", "content" : prompt}])
    
    return Answer(text=response.content.strip())
    
