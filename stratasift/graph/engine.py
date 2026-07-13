from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from stratasift.graph.state import PaperIngestionState
from stratasift.graph.nodes import (
    supervisor_triage_node,
    router_proxy_node,
    specialist_worker_node,
    reflection_review_node,
    supervisor_network_node,
)


def continue_after_triage(state: PaperIngestionState):
    """Conditional router after supervisor triage.

    If the paper is irrelevant (effective relevance score < threshold),
    routes directly to Supervisor_Network. Otherwise, routes to the
    Router_Proxy for Fan-Out.
    """
    from stratasift.config import get_runtime_config

    config = get_runtime_config()
    threshold = config.blocks.supervisor_block.relevance_threshold or 0.75

    if state.get("relevance_score", 0.0) < threshold:
        return "Supervisor_Network"

    return "Router_Proxy"


def route_to_specialists(state: PaperIngestionState):
    """Dynamic Fan-Out router based on section chunks."""
    source_doc = state.get("source_doc")
    chunks = getattr(source_doc, "section_chunks", [])

    # If no chunks (e.g. monolithic fallback), send a single worker with the full text
    if not chunks:
        # Construct fallback chunk logic here
        chunks = [
            {
                "header": "Full Document",
                "content": getattr(source_doc, "abstract_intro", ""),
            }
        ]

    # Dynamically fan-out to parallel Specialist_Worker nodes
    return [
        Send(
            "Specialist_Worker",
            {
                "chunk": chunk,
                "reading_directive": state.get("reading_directive", ""),
                "central_hypothesis": state.get("central_hypothesis", ""),
                "match_type": state.get("match_type", ""),
                "feedback": state.get("feedback"),
            },
        )
        for chunk in chunks
    ]


def continue_after_reflection(state: PaperIngestionState):
    """Conditional router after reflection fact-checking.

    If feedback is present (discrepancy found) and retry count is less than max_debate_loops,
    routes back to Router_Proxy with the feedback.
    Otherwise, proceeds to the Supervisor network node.
    """
    from stratasift.config import get_runtime_config

    config = get_runtime_config()
    max_loops = (
        config.blocks.analysis_block.max_debate_loops
        if config.blocks.analysis_block.max_debate_loops is not None
        else 3
    )

    feedback = state.get("feedback")
    retry_count = state.get("retry_count", 0)

    # We restrict the retry loop to a maximum of max_loops
    if feedback and retry_count < max_loops:
        return "Router_Proxy"

    return "Supervisor_Network"


# Wire the state graph
workflow = StateGraph(PaperIngestionState)

# Add agents and tool nodes
workflow.add_node("Supervisor_Triage", supervisor_triage_node)
workflow.add_node("Router_Proxy", router_proxy_node)
workflow.add_node("Specialist_Worker", specialist_worker_node)
workflow.add_node("Reflection_Review", reflection_review_node)
workflow.add_node("Supervisor_Network", supervisor_network_node)

# Set starting transition
workflow.add_edge(START, "Supervisor_Triage")

# Triage conditional routing
workflow.add_conditional_edges(
    "Supervisor_Triage",
    continue_after_triage,
    ["Router_Proxy", "Supervisor_Network"],
)

# Dynamic Fan-Out from Proxy
workflow.add_conditional_edges(
    "Router_Proxy", route_to_specialists, ["Specialist_Worker"]
)

# Fan-in: All workers automatically resolve and proceed to Reflection
workflow.add_edge("Specialist_Worker", "Reflection_Review")

# Reflection verification routing
workflow.add_conditional_edges(
    "Reflection_Review",
    continue_after_reflection,
    ["Router_Proxy", "Supervisor_Network"],
)

# Compile final graph
workflow.add_edge("Supervisor_Network", END)
app = workflow.compile()
