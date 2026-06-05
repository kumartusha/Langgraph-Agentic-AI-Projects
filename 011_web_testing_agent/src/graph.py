from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import (
    convert_user_instruction_to_actions,
    get_initial_action,
    get_website_state,
    generate_code_for_action,
    validate_generated_action,
    decide_next_path,
    handle_generation_error,
    post_process_script,
    execute_test_case,
    generate_test_report
)

def build_graph() -> StateGraph:
    """Build and compile the LangGraph for the Web Testing Agent."""
    workflow = StateGraph(GraphState)

    workflow.add_node("convert_user_instruction_to_actions", convert_user_instruction_to_actions)
    workflow.add_node("get_initial_action", get_initial_action)
    workflow.add_node("get_website_state", get_website_state)
    workflow.add_node("generate_code_for_action", generate_code_for_action)
    workflow.add_node("validate_generated_action", validate_generated_action)
    workflow.add_node("handle_generation_error", handle_generation_error)
    workflow.add_node("post_process_script", post_process_script)
    workflow.add_node("execute_test_case", execute_test_case)
    workflow.add_node("generate_test_report", generate_test_report)

    workflow.set_entry_point("convert_user_instruction_to_actions")

    workflow.add_edge("convert_user_instruction_to_actions", "get_initial_action")
    workflow.add_edge("get_initial_action", "get_website_state")
    workflow.add_edge("get_website_state", "generate_code_for_action")
    workflow.add_edge("generate_code_for_action", "validate_generated_action")

    workflow.add_conditional_edges(
        "validate_generated_action",
        decide_next_path,
        {
            "get_website_state": "get_website_state",
            "handle_generation_error": "handle_generation_error",
            "post_process_script": "post_process_script"
        }
    )

    workflow.add_edge("handle_generation_error", END)
    workflow.add_edge("post_process_script", "execute_test_case")
    workflow.add_edge("execute_test_case", "generate_test_report")
    workflow.add_edge("generate_test_report", END)

    app = workflow.compile()
    return app

# Compile the graph
app = build_graph()
