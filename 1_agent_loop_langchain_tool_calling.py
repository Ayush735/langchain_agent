from dotenv import load_dotenv
from langchain_core import messages
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"

# Tools
@tool
def get_product_price(product_name: str) -> float:
    """Lookup the price of a product in a catalog."""
    print(f">>> Looking up price for product: {product_name}")
    prices = {
        "laptop": 999.0,
        "smartphone": 699.0,
        "headphones": 199.0,
    }
    return prices.get(product_name.lower(), "Product not found.")

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the final price.
        Available discount tiers:
            - "silver": 5% discount
            - "gold": 10% discount
            - "platinum": 15% discount"""
    print(f">>> Applying discount for tier: {discount_tier} on price: {price}") 
    discount_tiers = {
        "silver": 5,
        "gold": 10,
        "platinum": 15,
    }
    
  
    return round(price * (1 - discount_tiers.get(discount_tier, 0) / 100), 2)


#--agent loop
@traceable(name="Langchain Agent Loop")
def run_agent(question: str) -> str: 
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools }
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0, max_iterations=MAX_ITERATIONS)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 40)


    messages = [ SystemMessage(content="You are a helpful shopping assistant that can answer questions about product prices and apply discounts." \
    "you have access of product catalog tool & apply discount tool & return response in a JSON format with product & price keys"), HumanMessage(content=question) ]

    for iteration in range(1,MAX_ITERATIONS + 1):
        print(iteration, "Iteration")
        response = llm_with_tools.invoke(messages)
        tool_calls = response.tool_calls
        if not tool_calls:
            return response.content
        
        #process only first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_calls_id = tool_call.get("id")

        observation = tools_dict.get(tool_name).invoke(tool_args)
        print(f"[Tool Result] Observation : {observation}")

        messages.append(ToolMessage(content=observation, tool_name=tool_name, tool_call_id=tool_calls_id))

    print(f"Max iteration reached without a final answer. Returning the last message: {messages[-1].content}")

if __name__ == "__main__":
    print("Hello from langchain agent (.bind_tools)!")
    result = run_agent("What is the price of a laptop with a gold discount?")
    print(f"Final Answer: {result}")