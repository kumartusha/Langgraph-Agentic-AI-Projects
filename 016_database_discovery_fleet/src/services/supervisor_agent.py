import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from src.config.settings import settings
from src.models.domain_models import ConversationState
from src.services.discovery_agent import DiscoveryAgent
from src.services.inference_agent import InferenceAgent
from src.services.planner_agent import PlannerAgent

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self):
        self.inference_agent = InferenceAgent()
        self.planner_agent = PlannerAgent()
        self.discovery_agent = DiscoveryAgent()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=settings.OPENAI_API_KEY)

        self.db_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}

            Rules:
            1. ALWAYS include ALL results from database queries in your response
            2. Format the response clearly with each piece of information on its own line
            3. Use bullet points or numbers for multiple pieces of information
            """)
        ])

        self.chat_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AI assistant.
            Respond naturally to the user's message.
            Keep responses brief and friendly.
            Don't make up information about weather, traffic, or other external data.
            """)
        ])

    def classify_user_input(self, state: ConversationState) -> ConversationState:
        system_prompt = """You are an input classifier. Classify the user's input into one of these categories:
        - DATABASE_QUERY: Questions about data, requiring database access
        - GREETING: General greetings, how are you, etc.
        - CHITCHAT: General conversation not requiring database
        - FAREWELL: Goodbye messages

        Respond with ONLY the category name."""

        messages = [
            ("system", system_prompt),
            ("user", state['question'])
        ]

        response = self.llm.invoke(messages)
        classification = response.content.strip()
        logger.info(f"Input classified as: {classification}")

        return {
            **state,
            "input_type": classification
        }

    def discover_database(self, state: ConversationState) -> ConversationState:
        if state.get('db_graph') is None:
            logger.info("Performing one-time database schema discovery...")
            graph = self.discovery_agent.discover()
            logger.info("Database schema discovery complete.")
            return {**state, "db_graph": graph}
        return state

    def create_plan(self, state: ConversationState) -> ConversationState:
        plan = self.planner_agent.create_plan(
            question=state['question']
        )
        return {**state, "plan": plan}

    def execute_plan(self, state: ConversationState) -> ConversationState:
        results = []
        try:
            for step in state['plan']:
                if ':' not in step:
                    continue

                step_type, content = step.split(':', 1)
                content = content.strip()

                if step_type.lower().strip() == 'inference':
                    try:
                        result = self.inference_agent.query(content, state.get('db_graph'))
                        results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in inference step: {str(e)}", exc_info=True)
                        results.append(f"Step: {step}\nError: Query failed - {str(e)}")
                else:
                    results.append(f"Step: {step}\nResult: {content}")

            return {
                **state,
                "db_results": "\n\n".join(results) if results else "No results were generated."
            }

        except Exception as e:
            logger.error(f"Error in execute_plan: {str(e)}", exc_info=True)
            return {**state, "db_results": f"Error executing steps: {str(e)}"}

    def generate_response(self, state: ConversationState) -> ConversationState:
        logger.info("Generating final response")
        is_chat = state.get("input_type") in ["GREETING", "CHITCHAT", "FAREWELL"]
        prompt = self.chat_response_prompt if is_chat else self.db_response_prompt

        response = self.llm.invoke(prompt.format(
            question=state['question'],
            db_results=state.get('db_results', '')
        ))

        return {**state, "response": response.content, "plan": []}

    def compile_graph(self):
        builder = StateGraph(ConversationState)

        builder.add_node("classify_input", self.classify_user_input)
        builder.add_node("discover_database", self.discover_database)
        builder.add_node("create_plan", self.create_plan)
        builder.add_node("execute_plan", self.execute_plan)
        builder.add_node("generate_response", self.generate_response)

        builder.add_edge(START, "classify_input")

        builder.add_conditional_edges(
            "classify_input",
            lambda x: "discover_database" if x.get("input_type") == "DATABASE_QUERY" else "generate_response"
        )

        builder.add_edge("discover_database", "create_plan")

        builder.add_conditional_edges(
            "create_plan",
            lambda x: "execute_plan" if x.get("plan") is not None else "generate_response"
        )

        builder.add_edge("execute_plan", "generate_response")
        builder.add_edge("generate_response", END)

        return builder.compile()
