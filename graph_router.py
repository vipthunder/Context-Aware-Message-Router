from __future__ import annotations

from typing import Any, TypedDict

from media_processor import MediaProcessor
from router_agent import RouterAgent


class RouterState(TypedDict, total=False):
    message: dict[str, Any]
    context: dict[str, Any]
    result: dict[str, Any]


class LangGraphRouter:
    def __init__(self, dataset_dir="dataset"):
        self.media = MediaProcessor(dataset_dir)
        self.agent = RouterAgent()
        self.graph = self._build_graph()

    def invoke(self, message: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        state: RouterState = {"message": message, "context": context}
        if self.graph is None:
            state = self._enrich_media(state)
            state = self._route_message(state)
        else:
            state = self.graph.invoke(state)
        return state["result"]

    def _enrich_media(self, state: RouterState) -> RouterState:
        state["message"] = self.media.enrich(state["message"], state["context"])
        return state

    def _route_message(self, state: RouterState) -> RouterState:
        state["result"] = self.agent.route(state["message"], state["context"])
        return state

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None
        graph = StateGraph(RouterState)
        graph.add_node("media", self._enrich_media)
        graph.add_node("router", self._route_message)
        graph.set_entry_point("media")
        graph.add_edge("media", "router")
        graph.add_edge("router", END)
        return graph.compile()
