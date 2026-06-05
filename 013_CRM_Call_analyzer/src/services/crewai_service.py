from textwrap import dedent
from crewai import Task, Agent, Crew
from src.models.domain_models import CallAnalysisResult
import os

class CrewAIService:
    def __init__(self, api_key: str):
        # In CrewAI 1.14+, we use the LiteLLM format string directly, 
        # and it will automatically use the GROQ_API_KEY env variable.
        # You MUST include the "groq/" prefix so it knows not to use OpenAI!
        self.llm = "groq/llama-3.3-70b-versatile"
        os.environ["GROQ_API_KEY"] = api_key

    def _get_call_analysis_agent(self) -> Agent:
        return Agent(
            role="AI-Powered Call Analyzer",
            goal="Provide actionable insights and advanced performance analysis from sales call transcriptions, empowering agents to close deals more effectively.",
            backstory="The AI-Powered Call Analyzer evaluates sales conversations by using sentiment analysis, identifying pain points, and providing detailed recommendations for improved outcomes.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def _get_call_analysis_task(self, transcription: str, agent: Agent) -> Task:
        return Task(
            description=dedent(f"""
                Analyze a sales call transcription between a customer and an agent.
                Generate a detailed and comprehensive report following the expected output schema.
                
                If any key is not relevant or no information is found, explicitly state "No relevant information found" for that key.
                The JSON must be concise, structured, and professional.

                Here is the transcription for analysis:
                {transcription}
            """),
            expected_output="A perfectly formatted JSON object matching the CallAnalysisResult schema.",
            agent=agent,
            output_pydantic=CallAnalysisResult
        )

    def analyze_transcription(self, transcription: str) -> CallAnalysisResult:
        """
        Runs the CrewAI workflow to analyze the transcription and returns structured Pydantic object.
        """
        agent = self._get_call_analysis_agent()
        task = self._get_call_analysis_task(transcription, agent)

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
        )

        result = crew.kickoff()
        
        # CrewAI 1.14+ populates result.pydantic when output_pydantic is provided
        if hasattr(result, "pydantic") and result.pydantic:
            return result.pydantic
            
        # Fallback if it populated json_dict instead
        if hasattr(result, "json_dict") and result.json_dict:
            return CallAnalysisResult.model_validate(result.json_dict)
            
        # Final fallback: convert to dict through json if it's a raw string
        import json
        import re
        try:
            return CallAnalysisResult.model_validate_json(str(result))
        except Exception:
            match = re.search(r'```json\\s*(.*?)\\s*```', str(result), re.DOTALL)
            if match:
                return CallAnalysisResult.model_validate_json(match.group(1))
            raise ValueError(f"Could not parse result: {result}")
