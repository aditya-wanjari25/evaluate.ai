from agno.agent import Agent
from agno.models.openai import OpenAIChat
# from agno.storage.sqlite import SqliteStorage
from agno.team import Team
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response
# from agno.workflow.v2.types import WorkflowExecutionInput
# from agno.workflow.v2.workflow import Workflow
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os, json
import prompts
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.pgvector import PgVector, SearchType

from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.openai import OpenAIEmbedder

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Strucctured Outputs Classes

class AgentResponse(BaseModel):
    grade : int = Field(..., description="Student Grade")
    feedback : str = Field(... , description="Response Feedback")

class GradeResponse(BaseModel):
    grade : int = Field(..., description="Student Grade")


db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@postgres:5432/ai")

knowledge = Knowledge(
    # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
    vector_db=PgVector(
        table_name="pdf_docs",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

knowledge.add_content(
    path="Data/southwest/southwest_case_study.pdf",
    reader=PDFReader()

)

rubric_agent1 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "Industry Analysis Agent",
    role = prompts.CRITERIA1_PROMPT,
    output_schema = AgentResponse,
    )

rubric_agent2 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "Comparison Agent",
    role = prompts.CRITERIA2_PROMPT,
    output_schema = AgentResponse,
    )

rubric_agent3 = Agent(
    model = OpenAIChat(id = "gpt-5-mini", api_key = API_KEY),
    name = "RAG Agent",
    knowledge=knowledge,
    search_knowledge=True,
    role = prompts.CRITERIA3_PROMPT,
    output_schema = AgentResponse,
    )
# rubric_agent3.knowledge.load(recreate=False)

rubric_agent4 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "Presentation Agent",
    role = prompts.CRITERIA4_PROMPT,
    output_schema = AgentResponse,
    )


class StudentReport(BaseModel):
    industry_analysis_score : int
    industry_analysis_feedback : str
    comparison_score : int
    comparison_feedback : str
    rag_score : int
    rag_feedback : str
    presentation_score : int
    presentation_feedback : str

team = Team(
    name = "Evaluator Team",
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    members = [rubric_agent1,rubric_agent2,rubric_agent3,rubric_agent4],
    determine_input_for_members=False,
    instructions=[
        "You are a professor teaching Business Analytics, your job is to evaluate student assignments",
        f"Here is the question: {prompts.TEST_QUESTION}",
        "Provide the student response to all 4 agents and submit their outputs to the user"
    ],
    show_members_responses=True,
    output_schema= StudentReport

)

response = team.run(input=prompts.TEST_RESPONSE)
response = response.content


fields = StudentReport.model_fields.keys()  
values = {field: getattr(response, field) for field in fields}
report = StudentReport(**values)

data = dict(report)
# print(data)

# Save to JSON
with open("student_report.json", "w") as f:
    f.write(report.model_dump_json(indent=4))  # Pydantic v2
