#!/usr/bin/env python3
"""Quick test for keyword search"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from src.agents.rag_agent import RAGAgent

agent = RAGAgent()

print("Testing laptop search...")
results = agent.search_products('laptops', k=3)
print(f"Found {len(results)} products:")
for p in results:
    print(f"  - {p['name']}: ${p['price']}")

print("\nTesting phone search...")
results2 = agent.search_products('phone', k=3)
print(f"Found {len(results2)} products:")
for p in results2:
    print(f"  - {p['name']}: ${p['price']}")

print("\nDone!")
