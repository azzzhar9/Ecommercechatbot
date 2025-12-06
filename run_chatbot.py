"""Simple script to run the chatbot."""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("E-commerce Chatbot")
print("=" * 60)
print()

# Check if vector store exists
vector_store_path = "./vector_store"
if not Path(vector_store_path).exists():
    print("Vector store not found. Initializing...")
    try:
        from src.initialize_vector_store import initialize_vector_store
        initialize_vector_store()
        print("Vector store initialized successfully!")
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("Vector store found.")

print()

# Run chatbot
try:
    from src.chatbot import EcommerceChatbot
    
    chatbot = EcommerceChatbot()
    chatbot.run()
except KeyboardInterrupt:
    print("\n\nGoodbye!")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

