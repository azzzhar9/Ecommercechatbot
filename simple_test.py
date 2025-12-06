#!/usr/bin/env python3
"""Simple test for RAG Agent"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("SIMPLE RAG AGENT TEST")
print("=" * 60)

# Test 1: Check environment
print("\n[1] Checking environment...")
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
print(f"API Key: {'SET' if api_key else 'NOT SET'}")
print(f"Base URL: {base_url}")

# Test 2: Check vector store
print("\n[2] Checking vector store...")
import chromadb
from chromadb.config import Settings

try:
    client = chromadb.PersistentClient(
        path="./vector_store",
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection("products")
    count = collection.count()
    print(f"[OK] Vector store connected - {count} products")
except Exception as e:
    print(f"[ERROR] Vector store: {e}")
    sys.exit(1)

# Test 3: Initialize RAG Agent
print("\n[3] Initializing RAG Agent...")
try:
    from src.agents.rag_agent import RAGAgent
    agent = RAGAgent()
    print("[OK] RAG Agent initialized")
except Exception as e:
    print(f"[ERROR] RAG Agent init: {e}")
    sys.exit(1)

# Test 4: Search products
print("\n[4] Testing product search...")
try:
    query = "laptop for programming"
    print(f"Query: '{query}'")
    results = agent.search_products(query, k=3)
    print(f"[OK] Found {len(results)} products:")
    for i, p in enumerate(results, 1):
        print(f"  {i}. {p.get('name', 'N/A')} - ${p.get('price', 0)}")
except Exception as e:
    print(f"[ERROR] Search failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
