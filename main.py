from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os

load_dotenv()


def main():
    print("Hello from udemy-langchain!")
    base_template = """You are a helpful assistant. and generate some interesting facts about {topic}. Output the facts in a list format each list maximum 10 words"""

    prompt = PromptTemplate(template=base_template, input_variables=["topic"])

    llm = ChatOllama(model="qwen3:1.7b",temperature=0)

    chain = prompt | llm

    response = chain.invoke({"topic": "ayush"})

    print(response.content)

if __name__ == "__main__":
    main()
