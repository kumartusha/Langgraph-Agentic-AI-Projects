import json
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import logger
from src.models import AgentState, DecisionMakingOutput, JudgeOutput
from src.prompts import (
    DECISION_MAKING_PROMPT,
    PLANNING_PROMPT,
    AGENT_PROMPT,
    JUDGE_PROMPT
)
from src.tools import tools_list, tools_dict

def format_tools_description(tools: list[BaseTool]) -> str:
    return "\n\n".join([f"- {tool.name}: {tool.description}\n Input arguments: {tool.args}" for tool in tools])

# LLMs
base_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.0)
decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
agent_llm = base_llm.bind_tools(tools_list)
judge_llm = base_llm.with_structured_output(JudgeOutput)

def decision_making_node(state: AgentState):
    """Entry point of the workflow. Based on the user query, the model can either respond directly or perform a full research, routing the workflow to the planning node"""
    logger.info("Entering decision_making_node")
    system_prompt = SystemMessage(content=DECISION_MAKING_PROMPT)
    response: DecisionMakingOutput = decision_making_llm.invoke([system_prompt] + state["messages"])
    output = {"requires_research": response.requires_research}
    if response.answer:
        output["messages"] = [AIMessage(content=response.answer)]
    return output

def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

def planning_node(state: AgentState):
    """Planning node that creates a step by step plan to answer the user query."""
    logger.info("Entering planning_node")
    system_prompt = SystemMessage(content=PLANNING_PROMPT.format(tools=format_tools_description(tools_list)))
    response = base_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    logger.info("Entering tools_node")
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        logger.info(f"Executing tool: {tool_call['name']}")
        tool_result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def agent_node(state: AgentState):
    """Agent call node that uses the LLM with tools to answer the user query."""
    logger.info("Entering agent_node")
    system_prompt = SystemMessage(content=AGENT_PROMPT)
    response = agent_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

def judge_node(state: AgentState):
    """Node to let the LLM judge the quality of its own final answer."""
    logger.info("Entering judge_node")
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        logger.warning("Reached max feedback requests. Ending judge_node.")
        return {"is_good_answer": True}

    system_prompt = SystemMessage(content=JUDGE_PROMPT)
    response: JudgeOutput = judge_llm.invoke([system_prompt] + state["messages"])
    
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": num_feedback_requests + 1
    }
    if response.feedback:
        logger.info(f"Judge feedback: {response.feedback}")
        output["messages"] = [AIMessage(content=response.feedback)]
    return output

def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"

def create_workflow() -> CompiledStateGraph:
    """Creates and compiles the LangGraph workflow."""
    logger.info("Compiling LangGraph workflow...")
    workflow = StateGraph(AgentState)

    workflow.add_node("decision_making", decision_making_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("decision_making")

    workflow.add_conditional_edges(
        "decision_making",
        router,
        {
            "planning": "planning",
            "end": END,
        }
    )
    workflow.add_edge("planning", "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": "judge",
        },
    )
    workflow.add_conditional_edges(
        "judge",
        final_answer_router,
        {
            "planning": "planning",
            "end": END,
        }
    )

    return workflow.compile()
