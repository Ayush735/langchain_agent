import re
import inspect
from dotenv import load_dotenv
load_dotenv()
import ollama
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


@traceable(run_type='tool')
def get_product_price(product_name: str) -> float:
    """Lookup the price of a product in a catalog."""
    print(f">>> Looking up price for product: {product_name}")
    prices = {
        "laptop": 999.0,
        "smartphone": 699.0,
        "headphones": 199.0,
    }
    return prices.get(product_name.lower(), "Product not found.")


@traceable(run_type='tool')
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

tool_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_toool_descriptions(tool_dict):
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        original_function = getattr(tool_function, "_wrapped_", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}-{signature}: {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_toool_descriptions(tool_dict)
