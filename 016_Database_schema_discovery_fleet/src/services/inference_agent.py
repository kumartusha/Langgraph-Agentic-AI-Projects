import json
import logging
import networkx as nx
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config.settings import settings

logger = logging.getLogger(__name__)

class InferenceAgent:
    def __init__(self):
        self.db_engine = SQLDatabase.from_uri(settings.DATABASE_PATH)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=settings.OPENAI_API_KEY)
        self.toolkit = SQLDatabaseToolkit(db=self.db_engine, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        self.chat_prompt = self.create_chat_prompt()

        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15
        )

    def run_query(self, q: str) -> str:
        try:
            return self.db_engine.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def create_chat_prompt(self) -> ChatPromptTemplate:
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a database inference expert for a SQLite database.
            Your job is to answer questions by querying the database and providing clear, accurate results.

            Rules:
            1. ONLY execute queries that retrieve data
            2. DO NOT provide analysis or recommendations
            3. Format responses as:
               Query Executed: [the SQL query used]
               Results: [the query results]
               Summary: [brief factual summary of the findings]
            4. Keep responses focused on the data only
            """
        )

        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def analyze_question_with_graph(self, db_graph: nx.Graph, question: str) -> dict:
        question_lower = question.lower()
        analysis = {
            'tables': [],
            'relationships': [],
            'columns': [],
            'possible_paths': []
        }

        for node in db_graph.nodes():
            node_data = db_graph.nodes[node]
            if 'tableName' not in node_data:
                continue

            table_name = node_data['tableName'].lower()
            if not (table_name in question_lower or
                    table_name.rstrip('s') in question_lower or
                    f"{table_name}s" in question_lower):
                continue

            table_info = {'name': node_data['tableName'], 'columns': []}

            for neighbor in db_graph.neighbors(node):
                col_data = db_graph.nodes[neighbor]
                if 'columnName' in col_data and col_data['columnName'].lower() in question_lower:
                    table_info['columns'].append({
                        'name': col_data['columnName'],
                        'type': col_data['columnType'],
                        'table': node_data['tableName']
                    })

            analysis['tables'].append(table_info)

        return analysis

    def query(self, text: str, db_graph) -> str:
        try:
            if db_graph:
                logger.info(f"Analyzing query with graph: '{text}'")
                graph_analysis = self.analyze_question_with_graph(db_graph, text)

                enhanced_prompt = f"""
                Database Structure Analysis:
                - Available Tables: {[t['name'] for t in graph_analysis['tables']]}
                - Table Relationships: {graph_analysis['possible_paths']}

                User Question: {text}

                Use this structural information to form an accurate query.
                """
                return self.agent_executor.invoke({"input": enhanced_prompt})['output']

            logger.info(f"No graph available, executing standard query: '{text}'")
            return self.agent_executor.invoke({"input": text})['output']

        except Exception as e:
            logger.error(f"Error in inference query: {str(e)}")
            return f"Error processing query: {str(e)}"
