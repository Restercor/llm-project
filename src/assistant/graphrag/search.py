"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json

import networkx as nx

from assistant.infra.grounding import canonical_name, find_codes, find_names, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

LOCAL_PROMPT = """Ты отвечаешь на вопрос диспетчера сервисной службы строго по фактам из графа знаний.
Если в фактах есть цепочка замен (REPLACED_BY), называй последнюю деталь цепочки.
Не выдумывай коды, цены и остатки. Если фактов недостаточно, скажи, что в базе знаний не найдено. 
Если в фактах есть цепочка REPLACED_BY, называй ТОЛЬКО последнюю деталь цепочки.
SP-104 — это регламент, а не деталь; детали имеют вид P-1042.

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

    def edge_type(data) -> str:
        if isinstance(data, dict) and data and all(isinstance(k, int) for k in data):
            first = next(iter(data.values()))
            
            if isinstance(first, dict):
                return first.get("type", "?")
            
        if isinstance(data, dict):
            return data.get("type", "?")
        return "?"

    def add_edge(u: str, v: str) -> None:
        if not graph.has_edge(u, v):
            return
        
        etype = edge_type(graph.get_edge_data(u, v))
        key = (u, etype, v)
        
        if key in seen:
            return
        
        seen.add(key)
        su = graph.nodes[u].get("name", u)
        sv = graph.nodes[v].get("name", v)
        
        facts.append(f"{su} -{etype}-> {sv}")
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
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    paths = []
                for path in paths:
                    for u, v in zip(path, path[1:], strict=False):
                        add_edge(u, v)
                        add_edge(v, u)

        for node in list(used_nodes):
            for nbr in list(graph.neighbors(node)):
                if not graph.has_edge(node, nbr):
                    continue
                
                et = edge_type(graph.get_edge_data(node, nbr))
                if et in ("IS_A", "HAS_PART") and nbr.startswith("Part:"):
                    if nbr in used_parts or node in anchors:
                        add_edge(node, nbr)
                        add_edge(nbr, node)
    elif len(anchors) == 1:
        anchor = anchors[0]
        for u, v in nx.bfs_edges(graph, anchor, depth_limit=2):
            add_edge(u, v)
            add_edge(v, u)
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

    for part in list(used_parts):
        for u, v, data in graph.edges(part, data=True):
            if data.get("type") == "IS_A":
                add_edge(u, v)
                
        for u, v, data in graph.in_edges(part, data=True):
            if data.get("type") == "IS_A":
                add_edge(u, v)

    for part in list(used_parts):
        for _, v, data in graph.out_edges(part, data=True):
            if data.get("type") == "SUPPLIED_BY":
                add_edge(part, v)

    for anchor in anchors:
        if anchor.startswith("ErrorCode:"):
            for _, v, data in graph.out_edges(anchor, data=True):
                if data.get("type") == "FIXED_BY":
                    add_edge(anchor, v)

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
    for u, v, data in graph.edges(data=True):
        su = graph.nodes[u].get("name", u)
        sv = graph.nodes[v].get("name", v)
        edge_str = f"{su} -{data.get('type')}-> {sv}"
        if edge_str in facts:
            for chunk_id in data.get("sources", []):
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
    used_communities = []
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
            used_communities.append(community)
            
    if not partials:
        return Answer(text="в базе знаний не найдено", refused=True)
    
    
    
    prompt = REDUCE_PROMPT.format(question=question, partials="\n".join(f"- {p}" for p in partials))
    response = llm.chat([{"role" : "user", "content" : prompt}])
    
    sources: set[str] = set()
    for community in used_communities:
        for node_id in community.nodes:
            for chunk_id in kb.graph.nodes[node_id].get("sources", []):
                sources.add(chunk_id.split("#")[0])
                
    return Answer(text=response.content.strip(), sources=sorted(sources))
    
