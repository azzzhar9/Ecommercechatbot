#!/usr/bin/env python3
"""Test RAG Agent functionality"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.rag_agent import RAGAgent
from dotenv import load_dotenv

load_dotenv()

def test_rag_agent():
    """Test RAG agent search functionality"""
    print("\n" + "="*60)
    print("TEST 2: RAG AGENT - PRODUCT SEARCH")
    print("="*60)
    
    # Initialize RAG agent
    print("\nInitializing RAG Agent...")
    agent = RAGAgent()
    print("[OK] RAG Agent initialized")
    
    # Test 1: Search for laptop
    print("\n--- Test 2.1: Search for laptop ---")
    query = "I am looking for a laptop for programming"
    print(f"Query: {query}")
    results = agent.search_products(query, k=3)
    print(f"Found {len(results)} products:")
    for i, result in enumerate(results, 1):
        name = result.get("name", "Unknown")
        price = result.get("price", 0)
        desc = result.get("description", "")[:80]
        print(f"  {i}. {name} - ${price}")
        print(f"     {desc}...")
    
    # Test 2: Search for headphones
    print("\n--- Test 2.2: Search for wireless headphones ---")
    query2 = "I need wireless headphones under 150"
    print(f"Query: {query2}")
    results2 = agent.search_products(query2, k=2)
    print(f"Found {len(results2)} products:")
    for i, result in enumerate(results2, 1):
        name = result.get("name", "Unknown")
        price = result.get("price", 0)
        print(f"  {i}. {name} - ${price}")
    
    # Test 3: Search for monitor
    print("\n--- Test 2.3: Search for gaming monitor ---")
    query3 = "gaming monitor with 144hz refresh rate"
    print(f"Query: {query3}")
    results3 = agent.search_products(query3, k=2)
    print(f"Found {len(results3)} products:")
    for i, result in enumerate(results3, 1):
        name = result.get("name", "Unknown")
        price = result.get("price", 0)
        print(f"  {i}. {name} - ${price}")
    
    print("\n[OK] RAG Agent Tests Completed Successfully!")

if __name__ == "__main__":
    test_rag_agent()
