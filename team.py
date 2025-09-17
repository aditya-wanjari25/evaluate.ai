from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response
from agno.workflow.v2.types import WorkflowExecutionInput
from agno.workflow.v2.workflow import Workflow
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os, json
import prompts
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.embedder.openai import OpenAIEmbedder

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Strucctured Outputs Classes

class AgentResponse(BaseModel):
    grade : int = Field(..., description="Student Grade")
    feedback : str = Field(... , description="Response Feedback")

class GradeResponse(BaseModel):
    grade : int = Field(..., description="Student Grade")

pdf_knowledge_base = PDFKnowledgeBase(
    path="Data/southwest/southwest_case_study.pdf",
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OpenAIEmbedder(id="text-embedding-3-large"),

    ),
    reader=PDFReader(chunk=True),
)

#  Defining Agents

rubric_agent1 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "Industry Analysis Agent",
    instructions = prompts.CRITERIA1_PROMPT,
    response_model = AgentResponse,
    )

rubric_agent2 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "Comparison Agent",
    instructions = prompts.CRITERIA2_PROMPT,
    response_model = AgentResponse,
    )

rubric_agent3 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "RAG Agent",
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    instructions = prompts.CRITERIA3_PROMPT,
    response_model = AgentResponse,
    )
rubric_agent3.knowledge.load(recreate=False)

rubric_agent4 = Agent(
    model = OpenAIChat(id = "gpt-4o", api_key = API_KEY),
    name = "Presentation Agent",
    instructions = prompts.CRITERIA4_PROMPT,
    response_model = AgentResponse,
    )
