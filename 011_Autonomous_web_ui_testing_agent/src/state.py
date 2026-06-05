from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

class GraphState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    query: Annotated[str, "A user query containing instructions for the creation of the test case"]
    actions: Annotated[List[str], "List of actions for which to generate the code."]
    target_url: Annotated[str, "Valid URL of the website to test."]
    current_action: Annotated[int, "The index of the current action to generate the code for."]
    current_action_code: Annotated[str, "Code for the current action."]
    aggregated_raw_actions: Annotated[str, "Raw aggregation of the actions"]
    script: Annotated[str, "The generated Playwright script."]
    website_state: Annotated[str, "DOM state of the website."]
    error_message: Annotated[str, "Message that occurred during the processing of the action."]
    test_evaluation_output: Annotated[str, "Evaluation of the final test script."]
    test_name: Annotated[str, "Name of the generated test."]
    report: Annotated[str, "The final markdown report of the execution."]

class ActionList(BaseModel):
    actions: List[str] = Field(..., description="List of atomic actions for end-to-end testing")
