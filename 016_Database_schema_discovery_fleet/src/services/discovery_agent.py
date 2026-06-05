import json
import logging
import networkx as nx
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from src.config.settings import settings

logger = logging.getLogger(__name__)

class DiscoveryAgent:
    def __init__(self):
        self.db_engine = SQLDatabase.from_uri(settings.DATABASE_PATH)
        self.llm_gpt4 = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=settings.OPENAI_API_KEY)
        self.toolkit = SQLDatabaseToolkit(db=self.db_engine, llm=self.llm_gpt4)
        self.tools = self.toolkit.get_tools()

        self.tools.extend([
            Tool(
                name="VISUALISE_SCHEMA",
                func=self.discover,
                description="Creates a visual graph representation of the database schema showing tables, columns, and their relationships."
            )
        ])

        self.chat_prompt = self.create_chat_prompt()
        self.agent = create_tool_calling_agent(
            llm=self.llm_gpt4,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

    def create_chat_prompt(self):
        system_message = SystemMessagePromptTemplate.from_template(
            """
            You are an AI assistant for querying a SQLLite database.
            Your responses should be formatted as json only.
            Always strive for clarity, terseness and conciseness in your responses.
            Return a json array with all the tables, using the example below:

            Example output:
            ```json
            [
                {{
                    "tableName": "[NAME OF TABLE RETURNED]",
                    "columns": [
                        {{
                            "columnName": "[COLUMN 1 NAME]",
                            "columnType": "[COLUMN 1 TYPE]",
                            "isOptional": true,
                            "foreignKeyReference": {{
                                "table": "[REFERENCE TABLE NAME]",
                                "column": "[REFERENCE COLUMN NAME]"
                            }}
                        }}
                    ]
                }}
            ]
            ```

            ## mandatory
            only output json
            do not put any extra commentary
            """
        )

        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def discover(self) -> nx.Graph:
        logger.info("Performing discovery...")
        prompt = "For all tables in this database, show the table name, column name, column type, if its optional. Also show Foreign key references to other columns. Do not show examples. Output only as json."
        
        response = self.agent_executor.invoke({"input": prompt})
        graph = self.jsonToGraph(response)
        return graph

    def jsonToGraph(self, response):
        output_ = response['output']
        return self.parseJson(output_)

    def parseJson(self, output_):
        try:
            # Handle possible markdown wrapping
            if "```json" in output_:
                j = output_[output_.find('```json') + 7:output_.rfind('```')]
            else:
                j = output_
            
            data = json.loads(j.strip())
            
            G = nx.Graph()
            nodeIds = 0
            columnIds = len(data) + 1
            labeldict = {}
            canonicalColumns = {}

            for table in data:
                nodeIds += 1
                G.add_node(nodeIds)
                G.nodes[nodeIds]['tableName'] = table["tableName"]
                labeldict[nodeIds] = table["tableName"]

                for column in table["columns"]:
                    columnIds += 1
                    G.add_node(columnIds)
                    G.nodes[columnIds]['columnName'] = column["columnName"]
                    G.nodes[columnIds]['columnType'] = column["columnType"]
                    G.nodes[columnIds]['isOptional'] = column.get("isOptional", False)
                    labeldict[columnIds] = column["columnName"]
                    canonicalColumns[table["tableName"] + column["columnName"]] = columnIds
                    G.add_edge(nodeIds, columnIds)

            for table in data:
                for column in table["columns"]:
                    if column.get("foreignKeyReference"):
                        this_column = table["tableName"] + column["columnName"]
                        reference_column_ = column["foreignKeyReference"]["table"] + column["foreignKeyReference"]["column"]
                        if this_column in canonicalColumns and reference_column_ in canonicalColumns:
                            G.add_edge(canonicalColumns[this_column], canonicalColumns[reference_column_])

            return G
        except Exception as e:
            logger.error(f"Failed to parse JSON into graph: {e}")
            return nx.Graph()
