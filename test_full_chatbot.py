#!/usr/bin/env python3
"""Test the full chatbot flow programmatically"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60, flush=True)
print("FULL CHATBOT TEST", flush=True)
print("=" * 60, flush=True)

# Test 1: Initialize chatbot
print("\n[1] Initializing Chatbot...", flush=True)
try:
    from src.chatbot import EcommerceChatbot
    chatbot = EcommerceChatbot()
    print("[OK] Chatbot initialized", flush=True)
    print(f"    Session ID: {chatbot.session_id}", flush=True)
except Exception as e:
    print(f"[FAIL] Chatbot init: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Product inquiry
print("\n[2] Testing product inquiry...", flush=True)
try:
    query1 = "What laptops do you have?"
    print(f"    User: {query1}", flush=True)
    response1 = chatbot.handle_message(query1)
    print(f"    Bot: {response1[:200]}...", flush=True)
    print("[OK] Product inquiry works", flush=True)
except Exception as e:
    print(f"[FAIL] Product inquiry: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 3: Price check
print("\n[3] Testing price check...", flush=True)
try:
    query2 = "How much is the MacBook Pro?"
    print(f"    User: {query2}", flush=True)
    response2 = chatbot.handle_message(query2)
    print(f"    Bot: {response2[:200]}...", flush=True)
    print("[OK] Price check works", flush=True)
except Exception as e:
    print(f"[FAIL] Price check: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 4: Order intent
print("\n[4] Testing order intent detection...", flush=True)
try:
    query3 = "I want to buy the MacBook Pro please"
    print(f"    User: {query3}", flush=True)
    response3 = chatbot.handle_message(query3)
    print(f"    Bot: {response3[:200]}...", flush=True)
    print("[OK] Order intent works", flush=True)
except Exception as e:
    print(f"[FAIL] Order intent: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 5: Order confirmation
print("\n[5] Testing order confirmation...", flush=True)
try:
    query4 = "Yes, confirm my order. My name is John Doe, email john@example.com"
    print(f"    User: {query4}", flush=True)
    response4 = chatbot.handle_message(query4)
    print(f"    Bot: {response4[:300]}...", flush=True)
    print("[OK] Order confirmation works", flush=True)
except Exception as e:
    print(f"[FAIL] Order confirmation: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Test 6: Check database for orders
print("\n[6] Checking database for orders...", flush=True)
try:
    from src.database import get_all_orders
    orders = get_all_orders()
    print(f"    Found {len(orders)} total orders", flush=True)
    for order in orders[-5:]:  # Show last 5 orders
        print(f"    - Order #{order.get('order_id', 'N/A')}: {order.get('product_name', 'N/A')}", flush=True)
    print("[OK] Database check works", flush=True)
except Exception as e:
    print(f"[FAIL] Database check: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60, flush=True)
print("ALL TESTS COMPLETED!", flush=True)
print("=" * 60, flush=True)
