import logging
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config.settings import settings

logger = logging.getLogger(__name__)

class PlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=settings.OPENAI_API_KEY)
        self.planner_prompt = self.create_planner_prompt()

    def create_planner_prompt(self):
        system_template = """You are a friendly planning agent that creates specific plans to answer questions about THIS database only.

        Available actions:
        1. Inference: [query] - Use this prefix for database queries
        2. General: [response] - Use this prefix for friendly responses

        Create a SINGLE, SEQUENTIAL plan where:
        - Each step should be exactly ONE line
        - Each step must start with either 'Inference:' or 'General:'
        - Steps must be in logical order
        - DO NOT repeat steps
        - Keep the plan minimal and focused

        Example format:
        Inference: Get all artists from the database
        Inference: Count tracks per artist
        General: Provide the results in a friendly way
        """

        human_template = "Question: {question}\n\nCreate a focused plan with appropriate action steps."

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def create_plan(self, question: str) -> list:
        try:
            logger.info(f"Creating plan for question: {question}")
            response = self.llm.invoke(self.planner_prompt.format(
                question=question
            ))

            steps = [step.strip() for step in response.content.split('\n')
                     if step.strip() and not step.lower().startswith('plan:')]

            if not steps:
                return ["General: I'd love to help you explore the database! What would you like to know?"]

            return steps

        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}", exc_info=True)
            return ["General: Error occurred while creating plan"]
