from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
    triage_result: str

class Router(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore', 'notify', or 'respond'."
    )
