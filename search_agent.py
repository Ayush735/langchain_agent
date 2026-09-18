from typing import List
from pydantic import Field, BaseModel
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
import os

@tool
def search(query: str) -> str:
    """Search the internet for relevant information."""
    search = TavilySearch(max_results=5)
    result = search.invoke({"query": query})

    return "\n".join(
        f"Title: {item['title']}\nURL: {item['url']}\nContent: {item['content']}"
        for item in result["results"]
    )

class Source(BaseModel):
    """Schema for source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer & source"""
    answer: str
    # sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOllama(model="qwen3:1.7b", temperature=0)
tools = [search]
agent = create_agent(llm, tools=tools)  

def main():
    print("Hello from langchain")
    result = agent.invoke({"messages": [HumanMessage(content="Search the web and tell me one GenAI job in Pune.")]})
    final_text = result["messages"][-1].content

    structured_llm = llm.with_structured_output(AgentResponse)

    structured_response = structured_llm.invoke(
        f"""
        Convert the following answer into the required structured format.

        Answer:
        {final_text}
        """
    )

    print(structured_response)
    


if __name__ == "__main__":
    main()
