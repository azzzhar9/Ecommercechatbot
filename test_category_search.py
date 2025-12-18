#!/usr/bin/env python3
"""Test category search functionality."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.chatbot import EcommerceChatbot

def test_category_queries():
    """Test various category search queries."""
    print("=" * 70)
    print("Testing Category Search Functionality")
    print("=" * 70)
    
    chatbot = EcommerceChatbot()
    
    test_queries = [
        # Critical queries that were failing
        ("show me books,Garden and sports stock", ["Books", "Home & Garden", "Sports"]),
        ("show me the Home,Sports and clothing", ["Home & Garden", "Sports", "Clothing"]),
        ("show me books and garden", ["Books", "Home & Garden"]),
        # Single category queries
        ("show me clothing", ["Clothing"]),
        ("show me sports", ["Sports"]),
        ("show me garden", ["Home & Garden"]),
        ("show me electronics", ["Electronics"]),
        ("show me phones", ["Phones"]),
        ("show me laptops", ["Laptops"]),
        # Additional multi-category and edge cases
        ("show me clothing, sports and garden", ["Clothing", "Sports", "Home & Garden"]),
        ("show me phones, laptops and books", ["Phones", "Laptops", "Books"]),
        ("show me electronics, books, and clothing", ["Electronics", "Books", "Clothing"]),
        ("show me home and garden, sports", ["Home & Garden", "Sports"]),
        ("show me books, books, books", ["Books"]),
        ("show me laptops, laptops and laptops", ["Laptops"]),
        ("show me phones and sports", ["Phones", "Sports"]),
        ("show me garden, clothing", ["Home & Garden", "Clothing"]),
        ("show me electronics and electronics", ["Electronics"]),
        ("show me phones, laptops, books, sports, clothing, garden", ["Phones", "Laptops", "Books", "Sports", "Clothing", "Home & Garden"]),
        ("show me books, garden, sports, clothing, electronics", ["Books", "Home & Garden", "Sports", "Clothing", "Electronics"]),
        ("show me laptops and garden", ["Laptops", "Home & Garden"]),
        # Case variations
        ("show me GARDEN", ["Home & Garden"]),
        ("show me Garden", ["Home & Garden"]),
        ("show me SPORTS", ["Sports"]),
        ("show me Sports", ["Sports"]),
        ("show me CLOTHING", ["Clothing"]),
        ("show me Clothing", ["Clothing"]),
    ]
    
    for query, expected_categories in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"Expected categories: {expected_categories}")
        print("-" * 70)
        
        try:
            response = chatbot.handle_message(query)
            
            # Check if response contains expected categories
            response_lower = response.lower()
            found_categories = []
            for cat in expected_categories:
                # Check for category indicators
                if "home & garden" in response_lower or "home and garden" in response_lower or "🏠" in response:
                    if "Home & Garden" not in found_categories:
                        found_categories.append("Home & Garden")
                elif cat.lower() in response_lower or ("📚" in response and "Books" not in found_categories and cat == "Books"):
                    if cat not in found_categories:
                        found_categories.append(cat)
                elif "⚽" in response and "Sports" not in found_categories and cat == "Sports":
                    if cat not in found_categories:
                        found_categories.append(cat)
                elif "👕" in response and "Clothing" not in found_categories and cat == "Clothing":
                    if cat not in found_categories:
                        found_categories.append(cat)
            
            print(f"Response preview (first 500 chars):")
            try:
                print(response[:500].encode('utf-8', errors='replace').decode('utf-8'))
            except:
                print(response[:500])
            print(f"\nFound categories in response: {found_categories}")
            
            # Simple validation
            if len(found_categories) > 0:
                print("[OK] Response contains category information")
            else:
                print("[WARNING] Could not detect expected categories in response")
                
        except Exception as e:
            print(f"[ERROR] Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Testing complete!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_category_queries()
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
