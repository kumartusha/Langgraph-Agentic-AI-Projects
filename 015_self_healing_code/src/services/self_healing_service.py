import inspect
import logging
from typing import Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.models.domain_models import HealingAgentState, CodeExecutionResponse
from src.services.memory_service import memory_service
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Initialize LLM
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=settings.LLM_MODEL, temperature=0, api_key=settings.GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Failed to load ChatGroq, falling back to OpenAI: {e}")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

class SelfHealingService:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(HealingAgentState)

        # Nodes
        builder.add_node('code_execution_node', self.code_execution_node)
        builder.add_node('code_update_node', self.code_update_node)
        builder.add_node('code_patching_node', self.code_patching_node)
        builder.add_node('bug_report_node', self.bug_report_node)
        builder.add_node('memory_search_node', self.memory_search_node)
        builder.add_node('memory_filter_node', self.memory_filter_node)
        builder.add_node('memory_modification_node', self.memory_modification_node)
        builder.add_node('memory_generation_node', self.memory_generation_node)

        # Edges
        builder.set_entry_point('code_execution_node')
        builder.add_conditional_edges('code_execution_node', self.error_router)
        builder.add_edge('bug_report_node', 'memory_search_node')
        builder.add_conditional_edges('memory_search_node', self.memory_filter_router)
        builder.add_conditional_edges('memory_filter_node', self.memory_generation_router)
        builder.add_edge('memory_generation_node', 'code_update_node')
        builder.add_conditional_edges('memory_modification_node', self.memory_update_router)

        builder.add_edge('code_update_node', 'code_patching_node')
        builder.add_edge('code_patching_node', 'code_execution_node')

        return builder.compile()

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    def code_execution_node(self, state: HealingAgentState):
        logger.info(f"Executing function {state['function_name']} with args {state['arguments']}")
        try:
            result = state['function'](*state['arguments'])
            state['error'] = False
            state['final_result'] = result
            logger.info("Function executed successfully.")
        except Exception as e:
            logger.error(f"Function raised an error: {e}")
            state['error'] = True
            state['error_description'] = str(e)
            
            # Record that we caught an error this cycle
            if not state.get('all_bug_reports'):
                state['all_bug_reports'] = []
        return state

    def code_update_node(self, state: HealingAgentState):
        logger.info("Generating code patch using LLM...")
        prompt = ChatPromptTemplate.from_template(
            'You are tasked with fixing a Python function that raised an error.\n'
            'Function: {function_string}\n'
            'Error: {error_description}\n'
            'You must provide a fix for the present error only.\n'
            'The bug fix should handle the thrown error case gracefully by returning an error message or valid fallback.\n'
            'Do not raise an error in your bug fix.\n'
            'The function must use the exact same name and parameters.\n'
            'Your response must contain only the function definition with no additional text.\n'
            'Your response must not contain any additional formatting, such as markdown code delimiters or language declarations.'
        )
        message = HumanMessage(content=prompt.format(
            function_string=state['function_string'], 
            error_description=state['error_description']
        ))
        
        # Remove markdown ticks if present
        new_func = llm.invoke([message]).content.strip()
        if new_func.startswith("```python"):
            new_func = new_func[9:]
        if new_func.startswith("```"):
            new_func = new_func[3:]
        if new_func.endswith("```"):
            new_func = new_func[:-3]
            
        state['new_function_string'] = new_func.strip()
        return state

    def code_patching_node(self, state: HealingAgentState):
        logger.info("Patching code dynamically...")
        try:
            new_code = state['new_function_string']
            namespace = {}
            # WARNING: exec() is dangerous in production!
            exec(new_code, namespace)
            
            func_name = state['function_name']
            if func_name in namespace:
                state['function'] = namespace[func_name]
                state['was_patched'] = True
                state['function_string'] = new_code
                logger.info("Patch applied successfully.")
            else:
                logger.error(f"Function {func_name} not found in patched code namespace.")
        except Exception as e:
            logger.error(f"Patching failed: {e}")
        return state

    def bug_report_node(self, state: HealingAgentState):
        logger.info("Generating bug report...")
        prompt = ChatPromptTemplate.from_template(
            'You are tasked with generating a bug report for a Python function that raised an error.\n'
            'Function: {function_string}\n'
            'Error: {error_description}\n'
            'Your response must be a comprehensive string including only crucial information on the bug report.'
        )
        message = HumanMessage(content=prompt.format(
            function_string=state['function_string'], 
            error_description=state['error_description']
        ))
        bug_report = llm.invoke([message]).content.strip()
        state['bug_report'] = bug_report
        state['all_bug_reports'].append(bug_report)
        return state

    def memory_search_node(self, state: HealingAgentState):
        logger.info("Searching for similar past bug reports...")
        prompt = ChatPromptTemplate.from_template(
            'You are tasked with archiving a bug report for a Python function that raised an error.\n'
            'Bug Report: {bug_report}.\n'
            'Your response must be a concise string including only crucial information on the bug report for future reference.\n'
            'Format: # function_name ## error_description ### error_analysis'
        )
        message = HumanMessage(content=prompt.format(bug_report=state['bug_report']))
        response = llm.invoke([message]).content.strip()

        results = memory_service.query_memories(query=response)
        state['memory_search_results'] = results
        state['memory_ids_to_update'] = []
        return state

    def memory_filter_node(self, state: HealingAgentState):
        logger.info("Filtering memory search results...")
        ids_to_update = []
        for memory in state['memory_search_results']:
            # Threshold distance < 0.3 means highly similar
            if memory['distance'] < 0.3:
                ids_to_update.append(memory['id'])
        state['memory_ids_to_update'] = ids_to_update
        return state

    def memory_generation_node(self, state: HealingAgentState):
        logger.info("Generating and saving new memory...")
        prompt = ChatPromptTemplate.from_template(
            'You are tasked with archiving a bug report for a Python function that raised an error.\n'
            'Bug Report: {bug_report}.\n'
            'Your response must be a concise string including only crucial information on the bug report for future reference.\n'
            'Format: # function_name ## error_description ### error_analysis'
        )
        message = HumanMessage(content=prompt.format(bug_report=state['bug_report']))
        response = llm.invoke([message]).content.strip()
        
        memory_service.add_memory(response)
        return state

    def memory_modification_node(self, state: HealingAgentState):
        logger.info("Updating existing memory...")
        if not state['memory_ids_to_update']:
            return state

        memory_to_update_id = state['memory_ids_to_update'].pop(0)
        
        # Remove from search results as well to match logic
        if state['memory_search_results']:
            state['memory_search_results'].pop(0)
            
        memory_to_update_content = memory_service.get_memory(memory_to_update_id)
        
        prompt = ChatPromptTemplate.from_template(
            'Update the following memories based on the new interaction:\n'
            'Current Bug Report: {bug_report}\n'
            'Prior Bug Report: {memory_to_update}\n'
            'Your response must be a concise but cumulative string including only crucial information on the current and prior bug reports for future reference.\n'
            'Format: # function_name ## error_description ### error_analysis'
        )
        message = HumanMessage(content=prompt.format(
            bug_report=state['bug_report'],
            memory_to_update=memory_to_update_content
        ))
        
        response = llm.invoke([message]).content.strip()
        memory_service.update_memory(memory_to_update_id, response)
        
        return state

    # ---------------------------------------------------------
    # Routers
    # ---------------------------------------------------------
    def error_router(self, state: HealingAgentState):
        return 'bug_report_node' if state.get('error') else END

    def memory_filter_router(self, state: HealingAgentState):
        return 'memory_filter_node' if state.get('memory_search_results') else 'memory_generation_node'

    def memory_generation_router(self, state: HealingAgentState):
        return 'memory_modification_node' if state.get('memory_ids_to_update') else 'memory_generation_node'

    def memory_update_router(self, state: HealingAgentState):
        return 'memory_modification_node' if state.get('memory_ids_to_update') else 'code_update_node'

    # ---------------------------------------------------------
    # Public Execution
    # ---------------------------------------------------------
    def run_agent(self, code_string: str, function_name: str, args: list) -> CodeExecutionResponse:
        # Initial parse and execution via namespace
        namespace = {}
        try:
            exec(code_string, namespace)
            func = namespace.get(function_name)
            if not func:
                raise ValueError(f"Function {function_name} not found in the provided code.")
        except Exception as e:
            raise ValueError(f"Failed to parse initial code: {e}")

        initial_state = {
            "function": func,
            "function_name": function_name,
            "function_string": code_string,
            "arguments": args,
            "error": False,
            "error_description": "",
            "new_function_string": "",
            "bug_report": "",
            "memory_search_results": [],
            "memory_ids_to_update": [],
            "all_bug_reports": [],
            "was_patched": False,
            "final_result": None
        }

        # Run Graph
        result_state = self.graph.invoke(initial_state)

        return CodeExecutionResponse(
            original_code=code_string,
            final_code=result_state.get('function_string', code_string),
            result=result_state.get('final_result'),
            was_patched=result_state.get('was_patched', False),
            bug_reports=result_state.get('all_bug_reports', [])
        )
