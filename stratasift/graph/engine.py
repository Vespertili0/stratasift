from langgraph.graph import StateGraph, START, END
from stratasift.graph.state import PaperIngestionState
from stratasift.graph.nodes import (
    supervisor_triage_node,
    consolidated_specialist_node,
    reflection_review_node,
    supervisor_network_node,
)


def continue_to_specialist(state: PaperIngestionState):
    """Conditional router after supervisor triage.

    If the paper is irrelevant (effective relevance score < threshold),
    routes directly to Supervisor_Network. Otherwise, routes to the
    consolidated specialist node.
    """
    from stratasift.config import get_runtime_config
    config = get_runtime_config()
    threshold = config.blocks.supervisor_block.relevance_threshold or 0.75

    if state.get("relevance_score", 0.0) < threshold:
        return "Supervisor_Network"

    return "Consolidated_Specialist"


def continue_after_reflection(state: PaperIngestionState):
    """Conditional router after reflection fact-checking.

    If feedback is present (discrepancy found) and retry count is less than max_debate_loops,
    routes back to the consolidated specialist with the feedback.
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
        return "Consolidated_Specialist"

    return "Supervisor_Network"


# Wire the state graph
workflow = StateGraph(PaperIngestionState)

# Add agents and tool nodes
workflow.add_node("Supervisor_Triage", supervisor_triage_node)
workflow.add_node("Consolidated_Specialist", consolidated_specialist_node)
workflow.add_node("Reflection_Review", reflection_review_node)
workflow.add_node("Supervisor_Network", supervisor_network_node)

# Set starting transition
workflow.add_edge(START, "Supervisor_Triage")

# Triage conditional routing
workflow.add_conditional_edges(
    "Supervisor_Triage",
    continue_to_specialist,
    ["Consolidated_Specialist", "Supervisor_Network"],
)

# Consolidated specialist routes to reflection review
workflow.add_edge("Consolidated_Specialist", "Reflection_Review")

# Reflection verification routing
workflow.add_conditional_edges(
    "Reflection_Review", continue_after_reflection, ["Consolidated_Specialist", "Supervisor_Network"]
)

# Compile final graph
workflow.add_edge("Supervisor_Network", END)
app = workflow.compile()
