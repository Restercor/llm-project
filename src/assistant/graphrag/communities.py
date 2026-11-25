"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SUMMARY_PROMPT = """Ты пишешь краткую сводку сообщества графа знаний сервисной службы.
Напиши связный текст не длиннее 150 слов. Используй только приведённые ниже факты и числа.
Не выдумывай новые сущности и числа.

Узлы сообщества:
{nodes}

Суммарные веса связей по типам:
{relations}

Самые тяжёлые связи:
{edges}

Напиши сводку: какие сущности входят, как они связаны, что встречается чаще.
"""

def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    if graph.number_of_nodes() == 0:
        return []
    
    undirected = graph.to_undirected()
    
    partition = nx.community.louvain_communities(undirected, weight="weight",seed=42)
    communities = sorted(partition, key=lambda nodes: sorted(nodes))
    
    return [Community(community_id=i, nodes=sorted(nodes)) for i, nodes in enumerate(communities)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    node_lines = []
    
    for node_id in community.nodes:
        data = graph.nodes.get(node_id, {})
        etype = data.get("type", "?")
        ename = data.get("name", node_id)
        node_lines.append(f"- {etype}: {ename}")
    
    members = set(community.nodes)
    weight_by_type: Counter[str] = Counter()
    
    for u, v, data in graph.edges(data=True):
        if u in members and v in members:
            weight_by_type[data.get("type", "?")] += data.get("weight", 1)
    
    relation_lines = [f"- {rtype}: {total}" for rtype, total in sorted(weight_by_type.items())]
    
    top_edges = sorted(((u, v, data) for u, v, data in graph.edges(data=True) if u in members and v in members), 
                       key=lambda x: x[2].get("weight", 1), reverse=True)[:5]
    
    edge_lines = [f"- {graph.nodes[u].get('name', u)} -{data.get('type')}-> "
                  f"{graph.nodes[v].get('name', v)} (вес {data.get('weight', 1)})" for u, v, data in top_edges]
    
    prompt = SUMMARY_PROMPT.format(nodes="\n".join(node_lines), 
                                   relations="\n".join(relation_lines) if relation_lines else "- нет связей", 
                                   edges="\n".join(edge_lines) if edge_lines else "- нет рёбер")
    
    response = llm.chat([{"role" : "user", "content" : prompt}])
    words = response.content.split()
    
    if len(words) > 150:
        return " ".join(words[:150])
    
    return response.content.strip()
    
    