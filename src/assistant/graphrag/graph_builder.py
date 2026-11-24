"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0
    
    # Nodes
    
    for res in results:
        for entity in res.entities:
            node_id = entity.node_id
            
            if node_id not in graph:
                graph.add_node(node_id, type=entity.type, name=entity.name, sources=[])
            sources = graph.nodes[node_id]["sources"]
                
            if res.chunk_id not in sources:
                sources.append(res.chunk_id)
                    
    # Edgess
    
    edge_sources : dict[tuple[str, str, str], set[str]] = {}
    
    for res in results:
        for rel in res.relations:
            if not rel.matches_schema():
                continue
            key = (rel.subject.node_id, rel.type, rel.object.node_id)
            edge_sources.setdefault(key, set()).add(res.chunk_id)
            
    # REPLACED_BY
    
    replaced = {k : v for k, v in edge_sources.items() if k[1] == "REPLACED_BY"}
    conflicts = 0
    to_remove : set[tuple[str, str, str]] = set()
    
    for (s, _, o), sources_s in list(replaced.items()):
        reverse = (o, "REPLACED_BY", s)
        
        if reverse not in replaced:
            continue
        if (s, "REPLACED_BY", o) in to_remove or reverse in to_remove:
            continue
        
        sources_r = replaced[reverse]
        w_f, w_b = len(sources_s), len(sources_r)
        conflicts+=1
        
        if w_f > w_b:
            to_remove.add(reverse)
        elif w_b > w_f:
            to_remove.addd((s, "REPLACED_BY", o))
        else:
            to_remove.add((s, "REPLACED_BY", o))
            to_remove.add(reverse)
    
    graph.graph["conflicts"] = conflicts
    
    # Edge creation
    
    for (s, rtype, o), sources in edge_sources.items():
        if (s, rtype, o) in to_remove:
            continue
        graph.add_edge(s, o, type=rtype, weight=len(sources), sources=sorted(sources))
    
    return graph
        
