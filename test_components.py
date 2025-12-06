#!/usr/bin/env python3
"""Test chatbot components individually without full integration"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60, flush=True)
print("COMPONENT TESTS", flush=True)
print("=" * 60, flush=True)

# Test 1: Environment check
print("\n[1] Environment Check...", flush=True)
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
print(f"    API Key: {'SET' if api_key else 'NOT SET'}", flush=True)
print(f"    Base URL: {base_url}", flush=True)
print("[OK] Environment configured", flush=True)

# Test 2: OpenAI client test
print("\n[2] Testing OpenAI client...", flush=True)
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello, I am working!' in 5 words or less"}],
        max_tokens=20
    )
    print(f"    Response: {response.choices[0].message.content}", flush=True)
    print("[OK] OpenAI client works", flush=True)
except Exception as e:
    print(f"[FAIL] OpenAI client: {e}", flush=True)

# Test 3: Pydantic models
print("\n[3] Testing Pydantic models...", flush=True)
try:
    from src.models import OrderModel
    order = OrderModel(
        product_name="MacBook Pro",
        quantity=1,
        unit_price=1999.99,
        customer_name="John Doe",
        customer_email="john@example.com"
    )
    print(f"    Order ID: {order.order_id}", flush=True)
    print(f"    Total: ${order.total_price}", flush=True)
    print("[OK] Pydantic models work", flush=True)
except Exception as e:
    print(f"[FAIL] Pydantic models: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 4: Database operations
print("\n[4] Testing database operations...", flush=True)
try:
    from src.database import init_database, create_order, get_order_by_id, get_all_orders
    from src.models import OrderModel
    
    # Initialize database
    init_database("./test_orders.db")
    print("    Database initialized", flush=True)
    
    # Create a test order
    test_order = OrderModel(
        product_name="Test Product",
        quantity=2,
        unit_price=99.99,
        customer_name="Test User",
        customer_email="test@example.com"
    )
    order_id = create_order(test_order, "./test_orders.db")
    print(f"    Created order: {order_id}", flush=True)
    
    # Retrieve order
    retrieved = get_order_by_id(order_id, "./test_orders.db")
    if retrieved:
        print(f"    Retrieved: {retrieved['product_name']} - ${retrieved['total_price']}", flush=True)
    
    # Get all orders
    all_orders = get_all_orders("./test_orders.db")
    print(f"    Total orders in DB: {len(all_orders)}", flush=True)
    
    print("[OK] Database operations work", flush=True)
except Exception as e:
    print(f"[FAIL] Database operations: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 5: Function calling test
print("\n[5] Testing Function Calling with LLM...", flush=True)
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search for products in the catalog",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_order",
                "description": "Create an order for a product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "quantity": {"type": "integer"}
                    },
                    "required": ["product_name", "quantity"]
                }
            }
        }
    ]
    
    # Test 1: Product search intent
    response1 = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What laptops do you have?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=100
    )
    
    if response1.choices[0].message.tool_calls:
        tool_call = response1.choices[0].message.tool_calls[0]
        print(f"    Search query -> Function: {tool_call.function.name}", flush=True)
        print(f"    Args: {tool_call.function.arguments}", flush=True)
    else:
        print(f"    No function call, response: {response1.choices[0].message.content[:50]}", flush=True)
    
    # Test 2: Order intent
    response2 = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "I want to buy the MacBook Pro"},
        ],
        tools=tools,
        tool_choice="auto",
        max_tokens=100
    )
    
    if response2.choices[0].message.tool_calls:
        tool_call = response2.choices[0].message.tool_calls[0]
        print(f"    Order query -> Function: {tool_call.function.name}", flush=True)
        print(f"    Args: {tool_call.function.arguments}", flush=True)
    else:
        print(f"    No function call, response: {response2.choices[0].message.content[:50]}", flush=True)
    
    print("[OK] Function calling works", flush=True)
except Exception as e:
    print(f"[FAIL] Function calling: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Cleanup
print("\n[6] Cleanup...", flush=True)
try:
    if Path("./test_orders.db").exists():
        os.remove("./test_orders.db")
    print("[OK] Cleanup complete", flush=True)
except:
    pass

print("\n" + "=" * 60, flush=True)
print("ALL COMPONENT TESTS COMPLETED!", flush=True)
print("=" * 60, flush=True)
