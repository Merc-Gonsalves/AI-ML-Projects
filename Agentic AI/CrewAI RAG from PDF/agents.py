from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool


class CustomAgents:
    def __init__(self):
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        self.Llama3 = ChatOpenAI(
    model='ollama/llama3.2',
    base_url="http://localhost:11434"
)
    def pdf_agent(self):

        pdf_tool = PDFSearchTool("gpt-4-analysis.pdf")

        return Agent(
            role="Senior PDF Analyst",
            backstory=dedent(f"""You can find anything in a pdf.  The people need you."""),
            goal=dedent(f"""Uncover any information from pdf files exceptionally well."""),
            tools=[pdf_tool],
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def writer_agent(self):
        return Agent(
            role="Writer",
            backstory=dedent(f"""All your life you have loved writing summaries."""),
            goal=dedent(f"""Take the information from the pdf agent and summarize it nicely."""),
            verbose=True,
            llm=self.Llama3,
        )
