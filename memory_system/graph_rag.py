from __future__ import annotations

"""Simple graph based RAG utilities.

This module provides a naive triple extractor and a small helper class for
storing a knowledge graph built from text snippets.  The graph is backed by
``networkx`` and keeps track of which cards (short pieces of knowledge stored
by Memoria) touch particular nodes and edges.

The implementation is intentionally lightweight – it does not attempt to be a
full natural language parser.  Instead it looks for very small patterns such as
``<subject> <verb> <object>`` in the text which is often good enough for small
experiments and unit tests.
"""

import itertools
import re
from dataclasses import dataclass, field

try:  # pragma: no cover - best effort import
    import networkx as nx
except Exception:  # pragma: no cover - very small fallback graph implementation
    from collections import deque
    from types import SimpleNamespace

    class _MiniGraph:
        def __init__(self) -> None:
            self.adj: dict[str, dict[str, set[str]]] = {}

        def add_node(self, node: str) -> None:
            self.adj.setdefault(node, {})

        def add_edge(self, u: str, v: str, key: str | None = None) -> None:
            self.add_node(u)
            self.add_node(v)
            self.adj[u].setdefault(v, set()).add(key or "")

        def __getitem__(self, node: str) -> dict[str, set[str]]:
            return self.adj[node]

    def _shortest_path(graph: _MiniGraph, source: str, target: str) -> list[str]:
        if source not in graph.adj or target not in graph.adj:
            raise KeyError(target)
        q: deque[str] = deque([source])
        prev: dict[str, str | None] = {source: None}
        while q:
            u = q.popleft()
            if u == target:
                break
            for v in graph.adj.get(u, {}):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if target not in prev:
            raise ValueError(target)
        path: list[str] = [target]
        # ``prev`` maps nodes to the preceding node in the BFS search.  The
        # dictionary stores ``None`` for the source node which confuses mypy
        # when appending values to the ``path`` list.  Unroll the loop to make
        # the ``None`` case explicit and keep type checkers happy.
        while True:
            prev_node = prev[path[-1]]
            if prev_node is None:
                break
            path.append(prev_node)
        path.reverse()
        return path

    nx = SimpleNamespace(
        MultiDiGraph=_MiniGraph,
        shortest_path=_shortest_path,
        NetworkXNoPath=ValueError,
        NodeNotFound=KeyError,
    )

Triple = tuple[str, str, str]


def extract_triples(text: str) -> list[Triple]:
    """
    Extract naive (subject, predicate, object) triples from ``text``.

    The function looks for very small patterns like ``"A is B"`` or
    ``"A likes B"`` in each sentence.  It is *not* a full NLP solution but is
    sufficient for lightweight graph‑based reasoning used in tests.

    Args:
        text: Input text from which to extract triples.

    Returns:
        A list of ``(subject, predicate, object)`` tuples.

    """
    triples: list[Triple] = []
    # Very small and permissive pattern: capture words around a simple verb
    pattern = re.compile(
        r"\b([A-Za-zА-Яа-я0-9_]+)\s+(is|are|was|were|has|have|likes|like|loves|love|contains|contain|—|-)\s+([A-Za-zА-Яа-я0-9_]+)\b",
        re.IGNORECASE,
    )
    for sentence in re.split(r"[.!?\n]", text):
        for match in pattern.finditer(sentence):
            subj, pred, obj = match.groups()
            triples.append((subj.strip(), pred.strip(), obj.strip()))
    return triples


@dataclass
class GraphRAG:
    """
    Small wrapper around :class:`networkx.MultiDiGraph`.

    The class keeps a mapping from nodes/edges to the identifiers of cards that
    introduced the corresponding triple.  This allows retrieving cards when a
    path between two entities is discovered.
    """

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    node_cards: dict[str, set[str]] = field(default_factory=dict)
    edge_cards: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)
    card_content: dict[str, str] = field(default_factory=dict)

    def add_card(self, card_id: str, text: str) -> None:
        """
        Parse ``text`` into triples and store them in the graph.

        Args:
            card_id: Identifier of the card or memory snippet.
            text:    Natural language text associated with the card.

        """
        self.card_content[card_id] = text
        triples = extract_triples(text)
        for subj, pred, obj in triples:
            self.graph.add_node(subj)
            self.graph.add_node(obj)
            self.graph.add_edge(subj, obj, key=pred)
            self.node_cards.setdefault(subj, set()).add(card_id)
            self.node_cards.setdefault(obj, set()).add(card_id)
            self.edge_cards.setdefault((subj, obj, pred), set()).add(card_id)

    def find_cards_between(self, entity1: str, entity2: str) -> list[str]:
        """
        Return card ids that connect ``entity1`` and ``entity2``.

        The method searches for the shortest path between the two entities and
        collects card identifiers that are attached to nodes and edges along
        that path.  If no path exists, an empty list is returned.
        """
        try:
            path = nx.shortest_path(self.graph, entity1, entity2)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        cards: set[str] = set()
        for node in path:
            cards.update(self.node_cards.get(node, set()))
        for u, v in itertools.pairwise(path):
            for key in self.graph[u][v]:
                cards.update(self.edge_cards.get((u, v, key), set()))
        return list(cards)


# A module level instance used by ``rag_router``.  This keeps the interface very
# light‑weight while still allowing other modules to import and use the same
# graph without passing references around explicitly.
graph_store = GraphRAG()

__all__ = ["GraphRAG", "Triple", "extract_triples", "graph_store"]
