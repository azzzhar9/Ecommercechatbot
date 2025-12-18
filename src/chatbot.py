"""Main chatbot application with function calling and session management."""

import os
import uuid
import time
import json
from typing import Dict, List, Optional

from openai import OpenAI

from src.agents.order_agent import OrderAgent
from src.agents.rag_agent import RAGAgent
from src.database import init_database
from src.logger import get_logger, setup_logger
from src.tracing import get_tracer, traced
from src.cart import ShoppingCart, CartManager
from src.cache import get_stock_cache, get_product_cache

logger = get_logger()


class EcommerceChatbot:
    """Main chatbot application with RAG and Order agents."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "./orders.db",
        vector_store_path: str = "./vector_store"
    ):
        """
        Initialize chatbot.
        
        Args:
            api_key: OpenAI API key
            db_path: Path to database file
            vector_store_path: Path to vector store
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not provided")
        
        # Configure client for OpenRouter if base_url is provided
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            logger.info(f"Using OpenRouter API at {base_url}")
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("Using OpenAI API")
        self.db_path = db_path
        self.vector_store_path = vector_store_path
        
        # Initialize database
        init_database(db_path)
        
        # Initialize agents
        self.rag_agent = RAGAgent(
            vector_store_path=vector_store_path,
            api_key=self.api_key
        )
        self.order_agent = OrderAgent(
            api_key=self.api_key,
            db_path=db_path,
            vector_store_path=vector_store_path
        )
        
        # Session state
        self.session_id = str(uuid.uuid4())
        self.chat_history: List[Dict] = []
        self.last_product: Optional[str] = None
        self.pending_order: Optional[Dict] = None
        self.order_count = 0
        
        # Initialize shopping cart
        self.cart = CartManager.get_cart(self.session_id)
        
        # Session memory for browsed products
        self.browsed_products: List[Dict] = []
        
        # Initialize caches
        self.stock_cache = get_stock_cache()
        self.product_cache = get_product_cache()
        
        # Initialize Langfuse tracer
        self.tracer = get_tracer()
        
        # Setup logger with session ID
        setup_logger(session_id=self.session_id)
        
        logger.info(f"Chatbot initialized with session ID: {self.session_id}")
    
    def _flush_tracer_async(self):
        """Flush tracer asynchronously to avoid blocking response."""
        import threading
        def flush_in_background():
            try:
                self.tracer.flush()
            except Exception as e:
                logger.debug(f"Background flush error (non-critical): {e}")
        
        # Start flush in background thread
        thread = threading.Thread(target=flush_in_background, daemon=True)
        thread.start()
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input.
        
        Args:
            text: User input text
        
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        # Remove potentially dangerous characters
        import re
        sanitized = re.sub(r'[<>"\';\\]', '', text)
        return sanitized.strip()
    
    def _resolve_product_name(self, product_name: str) -> Optional[Dict]:
        """
        Resolve partial/plural product names to exact product names.
        
        Args:
            product_name: Partial or plural product name (e.g., "MacBooks", "iPhone")
        
        Returns:
            Product dict with exact name and price, or None if not found
        """
        # Check cache first
        cache_key = f"product_resolve:{product_name.lower().strip()}"
        cached_result = self.product_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for product resolution: {product_name}")
            return cached_result
        
        import json
        from pathlib import Path
        
        try:
            products_file = Path("./data/products.json")
            if not products_file.exists():
                return None
            
            with open(products_file, 'r', encoding='utf-8') as f:
                all_products = json.load(f)
            
            product_name_lower = product_name.lower().strip()
            
            # Normalize: remove trailing 's' for plurals
            if product_name_lower.endswith('s') and len(product_name_lower) > 3:
                product_name_normalized = product_name_lower[:-1]
            else:
                product_name_normalized = product_name_lower
            
            # Try exact match first
            for product in all_products:
                if product_name_lower in product.get('name', '').lower():
                    result = product
                    # Cache the result
                    self.product_cache.set(cache_key, result)
                    return result
            
            # Try normalized match (plural handling)
            for product in all_products:
                product_name_db = product.get('name', '').lower()
                if product_name_normalized in product_name_db or product_name_db.startswith(product_name_normalized):
                    result = product
                    # Cache the result
                    self.product_cache.set(cache_key, result)
                    return result
            
            # Try fuzzy match - check if any word matches or if query is substring
            # This handles cases like "iPhone" -> "iPhone 15 Pro"
            query_words = set(product_name_normalized.split())
            best_match = None
            best_score = 0
            
            for product in all_products:
                product_name_db = product.get('name', '').lower()
                
                # Check if normalized query is a prefix or substring of product name
                # This handles "iPhone" matching "iPhone 15 Pro"
                if product_name_normalized in product_name_db:
                    # Prefer matches where query is at the start (better match)
                    if product_name_db.startswith(product_name_normalized):
                        result = product  # Best match - return immediately
                        # Cache the result
                        self.product_cache.set(cache_key, result)
                        return result
                    # Otherwise, track as potential match
                    if best_match is None or len(product_name_db) < len(best_match.get('name', '').lower()):
                        best_match = product
                        best_score = 1.0
                        continue
                
                product_words = set(product_name_db.split())
                
                # Count matching words
                matches = len(query_words.intersection(product_words))
                if matches > best_score:
                    best_score = matches
                    best_match = product
            
            result = best_match if best_score > 0 else None
            # Cache the result (even if None to avoid repeated lookups)
            if result:
                self.product_cache.set(cache_key, result)
            else:
                self.product_cache.set(cache_key, None, ttl=60)  # Shorter TTL for misses
            return result
            
        except Exception as e:
            logger.error(f"Error resolving product name: {str(e)}")
            return None
    
    def get_function_tools(self) -> List[Dict]:
        """
        Define function tools for OpenAI Function Calling.
        
        Returns:
            List of function tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products by name, category, or description. Use this when the user asks about product information, prices, or availability. Supports filters like 'laptops under $1000' or 'cheap phones'. For multi-category queries (e.g., 'books, garden and sports' or 'Home,Sports and clothing'), pass the ENTIRE query as a single function call - do NOT split into multiple calls. The search engine will automatically detect and group results by category.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for products (e.g., 'iPhone', 'laptops under $1000', 'cheap phones', 'books, garden and sports'). For multi-category queries, include the full query with all categories."
                            },
                            "sort_by": {
                                "type": "string",
                                "enum": ["relevance", "price_low", "price_high"],
                                "description": "Sort order for results"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_to_cart",
                    "description": "Add a product to the shopping cart. Use when user says 'add to cart', 'I want this', 'add X to my cart'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Name of the product to add"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Quantity to add (default 1)",
                                "minimum": 1,
                                "default": 1
                            },
                            "unit_price": {
                                "type": "number",
                                "description": "Unit price of the product"
                            }
                        },
                        "required": ["product_name", "unit_price"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "view_cart",
                    "description": "View the current shopping cart contents. Use when user asks 'show my cart', 'what's in my cart', 'view cart'.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_from_cart",
                    "description": "Remove a product from the shopping cart.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Name of the product to remove"
                            }
                        },
                        "required": ["product_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_coupon",
                    "description": "Apply a coupon code to the cart for a discount.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coupon_code": {
                                "type": "string",
                                "description": "The coupon code to apply (e.g., SAVE10, SAVE20, WELCOME)"
                            }
                        },
                        "required": ["coupon_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "checkout",
                    "description": "Checkout and place an order for all items in the cart. Use when user says 'checkout', 'place order', 'buy now', 'complete order'. Customer name and email are optional - if not provided, defaults will be used. User can provide them in the query like 'checkout with name John and email john@example.com'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_name": {
                                "type": "string",
                                "description": "Customer name (optional - will use default if not provided)"
                            },
                            "customer_email": {
                                "type": "string",
                                "description": "Customer email address (optional - will use default if not provided)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recommendations",
                    "description": "Get product recommendations based on a product. Use when user asks for similar products or recommendations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Name of the product to get recommendations for"
                            }
                        },
                        "required": ["product_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_order",
                    "description": "Create a single order directly (bypasses cart). Use for quick single-item purchases.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {
                                "type": "string",
                                "description": "Name of the product to order"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Quantity to order (must be at least 1)",
                                "minimum": 1
                            },
                            "unit_price": {
                                "type": "number",
                                "description": "Unit price of the product (must be greater than 0)",
                                "minimum": 0.01
                            },
                            "customer_name": {
                                "type": "string",
                                "description": "Customer name (optional)"
                            },
                            "customer_email": {
                                "type": "string",
                                "description": "Customer email address (optional)"
                            }
                        },
                        "required": ["product_name", "quantity", "unit_price"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_info",
                    "description": "Get stock information for all products. Use when user asks for stock details, stock info, inventory, or wants to see all products with their stock status.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_categories",
                    "description": "List all available product categories. Use when user asks for categories, wants to see all categories, or asks what categories are available.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
    
    def execute_function(self, function_name: str, arguments: Dict) -> Dict:
        """
        Execute function based on function name.
        
        Args:
            function_name: Name of the function to execute
            arguments: Function arguments
        
        Returns:
            Function result dictionary
        """
        try:
            if function_name == "search_products":
                query = arguments.get("query", "")
                sort_by = arguments.get("sort_by", "relevance")
                products = self.rag_agent.search_products(query, k=5, sort_by=sort_by)
                
                # Check if results are grouped by category (dict) or flat list
                if isinstance(products, dict):
                    # Multi-category results - format with category headers
                    result_text = "I found products in multiple categories:\n\n"
                    all_products = []
                    # Category display names
                    category_names = {
                        'computers': '💻 Laptops & Computers',
                        'phones': '📱 Phones',
                        'books': '📚 Books',
                        'audio': '🎧 Audio & Headphones',
                        'gaming': '🎮 Gaming',
                        'wearables': '⌚ Wearables',
                        'electronics': '⚡ Electronics',
                        'home_garden': '🏠 Home & Garden',
                        'sports': '⚽ Sports',
                        'clothing': '👕 Clothing'
                    }
                    for category, category_products in products.items():
                        if category_products:
                            # Normalize category key to lowercase to match category_names mapping
                            # Handle variations like 'Clothing', 'CLOTHING', 'home_garden', 'Home_garden', etc.
                            category_normalized = category.lower().strip()
                            # Also handle special cases like 'Home & Garden' -> 'home_garden'
                            if 'home' in category_normalized and 'garden' in category_normalized:
                                category_normalized = 'home_garden'
                            elif category_normalized == 'clothing':
                                category_normalized = 'clothing'
                            elif category_normalized == 'sports':
                                category_normalized = 'sports'
                            
                            display_name = category_names.get(category_normalized, category.title())
                            # #region agent log
                            try:
                                import json
                                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"id":f"log_category_display","timestamp":int(__import__('time').time()*1000),"location":"chatbot.py:456","message":"Displaying category","data":{"category":category,"category_normalized":category_normalized,"display_name":display_name,"product_count":len(category_products)},"sessionId":"debug-session","runId":"run2","hypothesisId":"A"}) + '\n')
                            except: pass
                            # #endregion
                            result_text += f"**{display_name}:**\n"
                            for i, p in enumerate(category_products, 1):
                                stock_msg = "in stock" if p['stock_status'] == "in_stock" else f"{p['stock_status']}"
                                result_text += f"  {i}. {p['name']} - ${p['price']:.2f} ({stock_msg})\n"
                            result_text += "\n"
                            all_products.extend(category_products)
                            if not self.last_product and category_products:
                                self.last_product = category_products[0]['name']
                            self.browsed_products.extend(category_products[:2])
                    if not all_products:
                        return {
                            "success": False,
                            "result": "No products found matching your search. Please try different keywords or filters.",
                            "products": [],
                            "query": query
                        }
                    return {
                        "success": True,
                        "result": result_text,
                        "products": all_products,
                        "grouped_by_category": True,
                        "query": query
                    }
                # Single category or no category - existing logic
                if not products:
                    return {
                        "success": False,
                        "result": "No products found matching your search. Please try different keywords or filters.",
                        "products": [],
                        "query": query
                    }
                # Only use filtered products, no fallback
                self.last_product = products[0]['name']
                self.browsed_products.extend(products[:3])
                if len(products) == 1:
                    p = products[0]
                    stock_msg = "in stock" if p['stock_status'] == "in_stock" else f"{p['stock_status']}"
                    result_text = (
                        f"The {p['name']} is priced at ${p['price']:.2f} and is currently {stock_msg}.\n"
                        f"{p['description'][:200]}..."
                    )
                else:
                    result_text = f"I found {len(products)} product(s) matching your search:\n\n"
                    for i, p in enumerate(products, 1):
                        stock_msg = "in stock" if p['stock_status'] == "in_stock" else f"{p['stock_status']}"
                        result_text += f"{i}. **{p['name']}** - ${p['price']:.2f} ({stock_msg})\n"
                        result_text += f"   {p['description'][:100]}...\n\n"
                return {
                    "success": True,
                    "result": result_text,
                    "products": products,
                    "query": query
                }
            
            elif function_name == "add_to_cart":
                product_name = arguments.get("product_name", "")
                quantity = arguments.get("quantity", 1)
                unit_price = arguments.get("unit_price", 0.0)
                
                # Resolve product name if unit_price is 0 or product_name is partial/plural
                # PRIORITY: Try to resolve provided product_name first, only use last_product as fallback
                if unit_price == 0.0 or not product_name:
                    # First, try to resolve the provided product_name
                    if product_name:
                        resolved = self._resolve_product_name(product_name)
                        if resolved:
                            product_name = resolved['name']
                            unit_price = resolved['price']
                    
                    # Only use last_product if product_name resolution failed
                    if (unit_price == 0.0 or not product_name) and self.last_product:
                        resolved = self._resolve_product_name(self.last_product)
                        if resolved:
                            product_name = resolved['name']
                            unit_price = resolved['price']
                
                # If still no price, try to find product using search
                if unit_price == 0.0 and product_name:
                    # First try direct resolution
                    resolved = self._resolve_product_name(product_name)
                    if resolved:
                        product_name = resolved['name']
                        unit_price = resolved['price']
                    else:
                        # Fallback: search for products matching the name
                        search_results = self.rag_agent.search_products(product_name, k=1)
                        if search_results:
                            # Handle both list and dict results
                            if isinstance(search_results, dict):
                                # Get first product from first category
                                for category_products in search_results.values():
                                    if category_products:
                                        product_name = category_products[0]['name']
                                        unit_price = category_products[0]['price']
                                        break
                            elif isinstance(search_results, list) and len(search_results) > 0:
                                product_name = search_results[0]['name']
                                unit_price = search_results[0]['price']
                
                if not product_name or unit_price == 0.0:
                    original_name = arguments.get("product_name", product_name or "that product")
                    return {
                        "success": False,
                        "result": f"I couldn't find the product '{original_name}'. Please try searching for it first, or specify the exact product name."
                    }
                
                self.cart.add_item(
                    product_id=product_name.lower().replace(" ", "_"),
                    product_name=product_name,
                    unit_price=unit_price,
                    quantity=quantity
                )
                
                return {
                    "success": True,
                    "result": f"Added {quantity}x {product_name} to your cart! 🛒\n\nCart now has {self.cart.item_count} item(s). Total: ${self.cart.total:.2f}",
                    "cart": self.cart.to_dict()
                }
            
            elif function_name == "view_cart":
                if self.cart.is_empty:
                    return {
                        "success": True,
                        "result": "Your cart is empty. Browse our products to add items!"
                    }
                return {
                    "success": True,
                    "result": self.cart.get_summary(),
                    "cart": self.cart.to_dict()
                }
            
            elif function_name == "remove_from_cart":
                product_name = arguments.get("product_name", "")
                if self.cart.remove_item(product_name):
                    return {
                        "success": True,
                        "result": f"Removed {product_name} from your cart."
                    }
                return {
                    "success": False,
                    "result": f"'{product_name}' not found in your cart."
                }
            
            elif function_name == "apply_coupon":
                coupon_code = arguments.get("coupon_code", "")
                success, message = self.cart.apply_coupon(coupon_code)
                return {
                    "success": success,
                    "result": message + (f"\nNew total: ${self.cart.total:.2f}" if success else "")
                }
            
            elif function_name == "checkout":
                if self.cart.is_empty:
                    return {
                        "success": False,
                        "result": "Your cart is empty. Add products before checking out!"
                    }
                
                customer_name = arguments.get("customer_name")
                customer_email = arguments.get("customer_email")
                
                # Use defaults if not provided (for testing/demo purposes)
                if not customer_name:
                    customer_name = "Guest Customer"
                if not customer_email:
                    customer_email = "guest@example.com"
                
                # Set cart customer info
                self.cart.customer_name = customer_name
                self.cart.customer_email = customer_email
                
                # Create orders for each item
                order_ids = []
                errors = []
                for item in self.cart.items:
                    order_data = {
                        "product_name": item.product_name,
                        "quantity": item.quantity,
                        "unit_price": item.unit_price,  # Include unit_price from cart
                        "customer_name": self.cart.customer_name,
                        "customer_email": self.cart.customer_email
                    }
                    # Use process_order_without_confirmation for checkout (skip interactive confirmation)
                    success, message, order_id = self.order_agent.process_order_without_confirmation(order_data)
                    if success:
                        order_ids.append(order_id)
                        self.order_count += 1
                    else:
                        errors.append(f"{item.product_name}: {message}")
                
                # Generate receipt
                if order_ids:
                    receipt = f"""
✅ **Order Confirmed!**

{self.cart.get_summary()}

Order ID(s): {', '.join(order_ids)}

Thank you for your purchase! 🎉
"""
                    # Clear cart after successful checkout
                    self.cart.clear()
                    
                    return {
                        "success": True,
                        "result": receipt,
                        "order_ids": order_ids
                    }
                else:
                    # Some or all orders failed
                    error_msg = "Failed to process checkout. "
                    if errors:
                        error_msg += "Errors: " + "; ".join(errors)
                    else:
                        error_msg += "No items could be processed."
                    
                    return {
                        "success": False,
                        "result": error_msg
                    }
            
            elif function_name == "get_recommendations":
                product_name = arguments.get("product_name", "")
                
                # Use last product if not provided
                if not product_name:
                    if self.last_product:
                        product_name = self.last_product
                    elif self.browsed_products:
                        # Use most recently browsed product
                        product_name = self.browsed_products[-1].get('name', '')
                    else:
                        return {
                            "success": False,
                            "result": "I don't have a product to base recommendations on. Please search for a product first, or specify which product you'd like recommendations for."
                        }
                
                # Resolve product name if partial
                resolved = self._resolve_product_name(product_name)
                if resolved:
                    product_name = resolved['name']
                
                recommendations = self.rag_agent.get_recommendations(product_name, k=3)
                
                if recommendations:
                    result_text = f"Based on {product_name}, you might also like:\n\n"
                    for p in recommendations:
                        result_text += f"• **{p['name']}** - ${p['price']:.2f} ({p.get('category', 'N/A')})\n"
                    return {
                        "success": True,
                        "result": result_text,
                        "products": recommendations
                    }
                return {
                    "success": False,
                    "result": f"No recommendations available for {product_name}."
                }
            
            elif function_name == "get_stock_info":
                # Get stock information for all products (with caching)
                cache_key = "stock_info:all"
                
                # Check cache first
                cached_result = self.stock_cache.get(cache_key)
                if cached_result is not None:
                    logger.info("Cache hit for stock information")
                    return cached_result
                
                # Get all products with stock information
                import json
                from pathlib import Path
                
                try:
                    products_file = Path("./data/products.json")
                    if not products_file.exists():
                        return {
                            "success": False,
                            "result": "Product database not found."
                        }
                    
                    with open(products_file, 'r', encoding='utf-8') as f:
                        all_products = json.load(f)
                    
                    # Group by category
                    categories = {}
                    for product in all_products:
                        category = product.get('category', 'Uncategorized')
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(product)
                    
                    # Format stock information with professional layout
                    result_text = "## 📦 Stock Information for All Products\n\n"
                    result_text += "---\n\n"
                    
                    # Category icons mapping
                    category_icons = {
                        'Electronics': '⚡',
                        'Clothing': '👕',
                        'Books': '📚',
                        'Home & Kitchen': '🏠',
                        'Home & Garden': '🏠',
                        'Sports & Outdoors': '⚽',
                        'Sports': '⚽',
                        'Toys & Games': '🎮',
                        'Beauty & Personal Care': '💄',
                        'Health & Household': '💊'
                    }
                    
                    # Calculate statistics
                    total_in_stock = 0
                    total_low_stock = 0
                    total_out_of_stock = 0
                    
                    for category, products in sorted(categories.items()):
                        # Get category icon
                        icon = category_icons.get(category, '📦')
                        result_text += f"### {icon} {category}\n\n"
                        
                        # Sort products by name for better readability
                        sorted_products = sorted(products, key=lambda x: x.get('name', ''))
                        
                        for product in sorted_products:
                            name = product.get('name', 'N/A')
                            price = product.get('price', 0)
                            stock_status = product.get('stock_status', 'unknown')
                            
                            # Format stock status with consistent spacing
                            if stock_status == 'in_stock':
                                stock_display = "✅ In Stock"
                                total_in_stock += 1
                            elif stock_status == 'low_stock':
                                stock_display = "⚠️ Low Stock"
                                total_low_stock += 1
                            elif stock_status == 'out_of_stock':
                                stock_display = "❌ Out of Stock"
                                total_out_of_stock += 1
                            else:
                                stock_display = f"❓ {stock_status}"
                            
                            # Format with consistent alignment (using fixed-width spacing)
                            # Product name, price aligned, stock status
                            result_text += f"  • **{name}** - ${price:.2f} - {stock_display}\n"
                        
                        result_text += "\n"
                    
                    # Professional summary section
                    result_text += "---\n\n"
                    result_text += "### 📊 Summary Statistics\n\n"
                    result_text += f"| Metric | Count |\n"
                    result_text += f"|--------|-------|\n"
                    result_text += f"| **Total Products** | {len(all_products)} |\n"
                    result_text += f"| **Total Categories** | {len(categories)} |\n"
                    result_text += f"| **✅ In Stock** | {total_in_stock} |\n"
                    result_text += f"| **⚠️ Low Stock** | {total_low_stock} |\n"
                    result_text += f"| **❌ Out of Stock** | {total_out_of_stock} |\n"
                    
                    result = {
                        "success": True,
                        "result": result_text,
                        "products": all_products,
                        "categories": list(categories.keys())
                    }
                    
                    # Cache the result
                    self.stock_cache.set(cache_key, result)
                    return result
                    
                except Exception as e:
                    logger.error(f"Error getting stock info: {str(e)}", exc_info=True)
                    return {
                        "success": False,
                        "result": f"Error retrieving stock information: {str(e)}"
                    }
            
            elif function_name == "list_categories":
                # List all available categories
                import json
                from pathlib import Path
                
                try:
                    products_file = Path("./data/products.json")
                    if not products_file.exists():
                        return {
                            "success": False,
                            "result": "Product database not found."
                        }
                    
                    with open(products_file, 'r', encoding='utf-8') as f:
                        all_products = json.load(f)
                    
                    # Get unique categories
                    categories = set()
                    category_counts = {}
                    for product in all_products:
                        category = product.get('category', 'Uncategorized')
                        categories.add(category)
                        category_counts[category] = category_counts.get(category, 0) + 1
                    
                    # Format category list
                    result_text = "**Available Product Categories:**\n\n"
                    
                    # Category display mapping
                    category_display = {
                        'Electronics': '⚡ Electronics',
                        'Clothing': '👕 Clothing',
                        'Books': '📚 Books',
                        'Home & Kitchen': '🏠 Home & Kitchen',
                        'Sports & Outdoors': '⚽ Sports & Outdoors',
                        'Toys & Games': '🎮 Toys & Games',
                        'Beauty & Personal Care': '💄 Beauty & Personal Care',
                        'Health & Household': '💊 Health & Household'
                    }
                    
                    for category in sorted(categories):
                        display_name = category_display.get(category, category)
                        count = category_counts[category]
                        result_text += f"  • {display_name} ({count} product{'s' if count != 1 else ''})\n"
                    
                    result_text += f"\n**Total Categories:** {len(categories)}"
                    result_text += f"\n**Total Products:** {len(all_products)}"
                    
                    # Also provide category keywords for search
                    result_text += "\n\n**You can search by category keywords like:**"
                    result_text += "\n  • Laptops, Phones, Headphones, Gaming, Books, etc."
                    result_text += "\n  • Or ask me to 'show me [category]' to browse products"
                    
                    return {
                        "success": True,
                        "result": result_text,
                        "categories": sorted(list(categories)),
                        "category_counts": category_counts
                    }
                    
                except Exception as e:
                    logger.error(f"Error listing categories: {str(e)}", exc_info=True)
                    return {
                        "success": False,
                        "result": f"Error retrieving categories: {str(e)}"
                    }
            
            elif function_name == "create_order":
                # Extract order details
                product_name = arguments.get("product_name", "")
                quantity = arguments.get("quantity", 1)
                unit_price = arguments.get("unit_price", 0.0)
                customer_name = arguments.get("customer_name")
                customer_email = arguments.get("customer_email")
                
                # Process order through Order Agent
                order_data = {
                    "product_name": product_name,
                    "quantity": quantity,
                    "customer_name": customer_name,
                    "customer_email": customer_email
                }
                
                success, message, order_id = self.order_agent.process_order(order_data)
                
                if success:
                    self.order_count += 1
                    logger.info(f"Order created: {order_id}")
                
                return {
                    "success": success,
                    "result": message,
                    "order_id": order_id
                }
            
            else:
                return {
                    "success": False,
                    "result": f"Unknown function: {function_name}"
                }
                
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "result": f"An error occurred while executing {function_name}. Please try again."
            }
    
    def resolve_ambiguous_product(self, chat_history: List[Dict]) -> str:
        """
        Resolve ambiguous product references.
        
        Args:
            chat_history: Conversation history
        
        Returns:
            Clarification question or resolved product name
        """
        if self.last_product:
            return f"Which product do you mean? {self.last_product} or would you like to search for something else?"
        else:
            return "I'm not sure which product you're referring to. Could you please specify the product name?"
    
    def handle_message(self, user_input: str) -> str:
        """
        Handle user message and return response.
        
        Args:
            user_input: User's message
        
        Returns:
            Bot's response
        """
        # #region agent log
        handle_start = time.time()
        try:
            with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:954","message":"handle_message START","data":{"user_input":user_input[:50],"sessionId":self.session_id},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
        except: pass
        # #endregion
        
        logger.debug(f"[DEBUG] handle_message called with user_input: {user_input}")
        # Create trace for this conversation turn
        trace = self.tracer.trace(
            name="handle_message",
            session_id=self.session_id,
            user_id="demo_user",
            metadata={"input_length": len(user_input)},
            tags=["chatbot", "conversation"]
        )
        
        try:
            # Sanitize input
            sanitized_input = self.sanitize_input(user_input)
            if not sanitized_input:
                return "I didn't catch that. Could you please repeat?"
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": sanitized_input})
            
            # Build messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful e-commerce chatbot assistant.
You help customers find products and place orders.
Use the search_products function to find product information.
Use the create_order function ONLY when the user explicitly wants to place an order.
Be conversational, friendly, and helpful.
Always use exact prices from search results."""
                }
            ]
            
            # Add chat history - use reduced context for better latency
            # For order processing, minimal context is sufficient since details come from function arguments
            # For general queries, 5 messages provide sufficient context
            context_size = 5  # Default: last 5 messages
            messages.extend(self.chat_history[-context_size:])
            
            # Get response with function calling
            tools = self.get_function_tools()
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Use model from environment or default
                    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
                    logger.info(f"Making API call to {model}...")
                    
                    # #region agent log
                    llm_call_start = time.time()
                    try:
                        with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1023","message":"LLM call START","data":{"model":model,"messages_count":len(messages),"retry_count":retry_count},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + '\n')
                    except: pass
                    # #endregion
                    
                    # Create LLM generation span for Langfuse
                    llm_gen = self.tracer.generation(
                        trace=trace,
                        name="chat_completion",
                        model=model,
                        input={"messages": messages[-5:]},  # Last 5 messages for context
                        metadata={"retry_count": retry_count}
                    )
                    
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    # #region agent log
                    llm_call_end = time.time()
                    llm_call_duration = llm_call_end - llm_call_start
                    try:
                        with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1044","message":"LLM call END","data":{"duration_ms":llm_call_duration*1000,"has_tool_calls":bool(response.choices[0].message.tool_calls)},"sessionId":"debug-session","runId":"run1","hypothesisId":"D"}) + '\n')
                    except: pass
                    # #endregion
                    
                    # End generation with output and usage
                    usage_info = {}
                    if hasattr(response, 'usage') and response.usage:
                        usage_info = {
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                            "total": response.usage.total_tokens
                        }
                    llm_gen.end(
                        output={"content": response.choices[0].message.content},
                        usage=usage_info
                    )
                    logger.info("API call completed successfully")
                    
                    message = response.choices[0].message
                    bot_response = None  # Initialize early to prevent UnboundLocalError
                    
                    # Check for function calls
                    if message.tool_calls:
                        # Execute functions
                        function_results = []
                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                            import json
                            arguments = json.loads(tool_call.function.arguments)
                            
                            logger.info(f"Function called: {function_name} with args: {arguments}")
                            
                            # #region agent log
                            func_start = time.time()
                            try:
                                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1068","message":"Function execution START","data":{"function_name":function_name},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + '\n')
                            except: pass
                            # #endregion
                            
                            # Create span for function execution
                            func_span = self.tracer.span(
                                trace=trace,
                                name=f"function_{function_name}",
                                input={"function": function_name, "arguments": arguments}
                            )
                            
                            # Execute function
                            function_result = self.execute_function(function_name, arguments)
                            
                            # #region agent log
                            func_end = time.time()
                            func_duration = func_end - func_start
                            try:
                                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1072","message":"Function execution END","data":{"function_name":function_name,"duration_ms":func_duration*1000,"success":function_result.get("success",False)},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + '\n')
                            except: pass
                            # #endregion
                            function_results.append(function_result)
                            
                            # End function span
                            func_span.end(output={"result": str(function_result)[:500]})
                            
                            # Add function result to messages
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(function_result)
                            })
                        
                        # Prepare formatted response from function results (immediate response)
                        bot_response = None
                        if function_results:
                            result = function_results[0]
                            logger.info(f"Processing function result: success={result.get('success')}, keys={list(result.keys())}")
                            # Priority: Use formatted result if available (handles multi-category, cart, etc.)
                            if result.get("success"):
                                # Check for grouped results first
                                if result.get("grouped_by_category") and result.get("result"):
                                    bot_response = result.get("result")
                                # Then check for regular result (cart operations, order operations, etc.)
                                elif result.get("result"):
                                    bot_response = result.get("result")
                                # Check for cart operations (they return result field)
                                elif result.get("cart"):
                                    # Cart operations return result in "result" field
                                    bot_response = result.get("result", "Cart operation completed.")
                                # Check for order operations
                                elif result.get("order_id"):
                                    bot_response = result.get("result", "Your order has been processed.")
                                # Fallback to products if no formatted result
                                elif result.get("products"):
                                    products = result["products"]
                                    if products:
                                        # If the user query is a stock/inventory request, show all products, not just 5
                                        user_query = self.chat_history[-1]["content"].lower() if self.chat_history else ""
                                        show_all_stock = any(
                                            kw in user_query for kw in [
                                                "show me all stock", "show all stock", "all stock", "show me the stock", "stock information", "inventory", "all products", "list all products"
                                            ]
                                        )
                                        if len(products) == 1:
                                            product = products[0]
                                            stock_msg = "in stock" if product['stock_status'] == "in_stock" else f"{product['stock_status']}"
                                            bot_response = (
                                                f"The {product['name']} is priced at ${product['price']:.2f} "
                                                f"and is currently {stock_msg}."
                                            )
                                        else:
                                            response_parts = [f"I found {len(products)} product(s):\n\n"]
                                            # Show all products for stock queries, else show only 5
                                            display_products = products if show_all_stock else products[:5]
                                            for i, product in enumerate(display_products, 1):
                                                stock_msg = "in stock" if product['stock_status'] == "in_stock" else f"{product['stock_status']}"
                                                response_parts.append(
                                                    f"{i}. {product['name']} - ${product['price']:.2f} ({stock_msg})\n"
                                                )
                                            if not show_all_stock and len(products) > 5:
                                                response_parts.append(f"...and {len(products) - 5} more. Ask for 'all stock' to see everything.\n")
                                            bot_response = "".join(response_parts)
                                    else:
                                        # Check if this was a filtered search (category + price)
                                        # Get query from result metadata if available
                                        query_text = result.get("query", "")
                                        
                                        # Check for category and price filter in query
                                        has_category = False
                                        has_price_filter = False
                                        category_name = "products"
                                        price_part = ""
                                        
                                        if query_text:
                                            query_lower = query_text.lower()
                                            # Check for categories
                                            if 'laptop' in query_lower:
                                                category_name = "laptops"
                                                has_category = True
                                            elif 'phone' in query_lower or 'iphone' in query_lower or 'samsung' in query_lower:
                                                category_name = "phones"
                                                has_category = True
                                            elif 'book' in query_lower and 'notebook' not in query_lower and 'macbook' not in query_lower:
                                                category_name = "books"
                                                has_category = True
                                            elif 'headphone' in query_lower or 'earbud' in query_lower:
                                                category_name = "headphones"
                                                has_category = True
                                            elif 'gaming' in query_lower or 'console' in query_lower:
                                                category_name = "gaming products"
                                                has_category = True
                                            
                                            # Check for price filter
                                            import re
                                            price_match = re.search(r'under\s*\$?\s*(\d+(?:\.\d+)?)', query_lower)
                                            if price_match:
                                                has_price_filter = True
                                                price_part = f" under ${price_match.group(1)}"
                                            else:
                                                price_match = re.search(r'below\s*\$?\s*(\d+(?:\.\d+)?)', query_lower)
                                                if price_match:
                                                    has_price_filter = True
                                                    price_part = f" below ${price_match.group(1)}"
                                        
                                        # Provide specific error message if category and price filter detected
                                        if has_category and has_price_filter:
                                            bot_response = f"No {category_name} found{price_part}. Please try different filters or browse all {category_name}."
                                        elif has_category:
                                            bot_response = f"No {category_name} found. Please try different keywords or browse all {category_name}."
                                        else:
                                            bot_response = "I couldn't find that product. Could you try rephrasing your search?"
                                else:
                                    # Final fallback - check if there's any result field
                                    bot_response = result.get("result", "I processed your request.")
                            else:
                                # Function returned failure - use error message
                                bot_response = result.get("result", "I couldn't process that request. Please try again.")
                        
                        # If we have a response from function results, use it (skip slow LLM call for faster response)
                        # BUT: If multiple function calls were made (multi-category query), aggregate all results
                        # Check if result has grouped_by_category - if so, use the pre-formatted result directly
                        has_grouped_category = any(r.get("grouped_by_category") for r in function_results if r.get("success") and r.get("result"))
                        if has_grouped_category and len(function_results) == 1:
                            # Single grouped result - use it directly without re-categorization
                            bot_response = function_results[0].get("result")
                            logger.info("Using pre-formatted grouped category result directly")
                        elif bot_response and len(function_results) == 1:
                            # Single result - use formatted response directly
                            logger.info(f"Using formatted response from function results: {bot_response[:100]}...")
                        elif function_results:
                            # Multiple results or no response - aggregate all function results
                            if bot_response and len(function_results) > 1:
                                # Reset bot_response to trigger aggregation for multi-category queries
                                bot_response = None
                                logger.info(f"Multiple function calls detected ({len(function_results)}), aggregating all results...")
                            # Fallback: Aggregate all function results for multi-intent queries
                            # Deduplicate category headers and product listings before formatting
                            from collections import defaultdict, OrderedDict
                            category_to_products = OrderedDict()
                            product_ids_seen = set()
                            for idx, result in enumerate(function_results):
                                if not result.get("success"):
                                    continue
                                # If grouped by category, preserve the original category grouping from search
                                if result.get("grouped_by_category") and result.get("products"):
                                    # When results are already grouped by category, use the product's actual category from data
                                    # Map with case-insensitive matching
                                    category_display_map_grouped = {
                                        "Home & Garden": "🏠 Home & Garden",
                                        "home & garden": "🏠 Home & Garden",
                                        "Home and Garden": "🏠 Home & Garden",
                                        "home and garden": "🏠 Home & Garden",
                                        "Sports": "⚽ Sports",
                                        "sports": "⚽ Sports",
                                        "sport": "⚽ Sports",
                                        "Clothing": "👕 Clothing",
                                        "clothing": "👕 Clothing",
                                        "clothes": "👕 Clothing",
                                        "Books": "📚 Books",
                                        "books": "📚 Books",
                                        "book": "📚 Books",
                                        "Electronics": "⚡ Electronics",
                                        "electronics": "⚡ Electronics"
                                    }
                                    for p in result["products"]:
                                        pid = p.get("product_id")
                                        if pid in product_ids_seen:
                                            continue
                                        # Use the product's actual category from the data
                                        original_category = p.get("category")
                                        # Try exact match first, then case-insensitive
                                        label = category_display_map_grouped.get(original_category) or category_display_map_grouped.get(original_category.lower() if original_category else "") or "⚡ Electronics"
                                        if label not in category_to_products:
                                            category_to_products[label] = []
                                        category_to_products[label].append(p)
                                        product_ids_seen.add(pid)
                                    continue
                                # If single/multi product result, group by product's actual category from data
                                elif result.get("products") is not None:
                                    products = result["products"]
                                    if isinstance(products, list) and len(products) > 0:
                                        # Use product's actual category from data instead of re-categorizing
                                        # Map with case-insensitive matching and handle variations
                                        category_display_map = {
                                            "Home & Garden": "🏠 Home & Garden",
                                            "home & garden": "🏠 Home & Garden",
                                            "Home and Garden": "🏠 Home & Garden",
                                            "home and garden": "🏠 Home & Garden",
                                            "Sports": "⚽ Sports",
                                            "sports": "⚽ Sports",
                                            "sport": "⚽ Sports",
                                            "Clothing": "👕 Clothing",
                                            "clothing": "👕 Clothing",
                                            "clothes": "👕 Clothing",
                                            "Books": "📚 Books",
                                            "books": "📚 Books",
                                            "book": "📚 Books",
                                            "Electronics": "⚡ Electronics",
                                            "electronics": "⚡ Electronics",
                                            "Computers": "💻 Laptops & Computers",
                                            "computers": "💻 Laptops & Computers",
                                            "Phones": "📱 Phones",
                                            "phones": "📱 Phones",
                                            "Audio": "🎧 Audio & Headphones",
                                            "audio": "🎧 Audio & Headphones",
                                            "Gaming": "🎮 Gaming",
                                            "gaming": "🎮 Gaming",
                                            "Wearables": "⌚ Wearables",
                                            "wearables": "⌚ Wearables"
                                        }
                                        
                                        # Fallback category inference from product name/description
                                        def infer_category_from_product(p):
                                            """Infer category from product name/description if category field is missing."""
                                            pname = p.get("name", "").lower()
                                            pdesc = p.get("description", "").lower()
                                            
                                            # Home & Garden keywords
                                            if any(kw in pname or kw in pdesc for kw in ["vacuum", "roomba", "dyson", "instant pot", "nespresso", "philips hue", "appliance", "kitchen"]):
                                                return "🏠 Home & Garden"
                                            # Sports keywords
                                            if any(kw in pname or kw in pdesc for kw in ["peloton", "yoga", "mat", "dumbbell", "water bottle", "fitness", "gym", "running", "bike"]):
                                                return "⚽ Sports"
                                            # Clothing keywords
                                            if any(kw in pname or kw in pdesc for kw in ["nike", "adidas", "levi", "patagonia", "north face", "sneakers", "jeans", "jacket", "sweater", "shoes"]):
                                                return "👕 Clothing"
                                            # Books keywords (exclude MacBook, notebook)
                                            if any(kw in pname or kw in pdesc for kw in ["book", "novel", "reading"]) and not any(ex in pname for ex in ["macbook", "notebook", "kindle"]):
                                                return "📚 Books"
                                            # Default to Electronics
                                            return "⚡ Electronics"
                                        
                                        for p in products:
                                            pid = p.get("product_id")
                                            if pid in product_ids_seen:
                                                continue
                                            # Use the product's actual category from the data
                                            original_category = p.get("category")
                                            
                                            # Debug logging to see what category values we're getting
                                            if not original_category:
                                                logger.warning(f"Product {p.get('name', 'Unknown')} missing category field, inferring from name/description")
                                            
                                            # Try exact match first
                                            label = category_display_map.get(original_category)
                                            
                                            # If not found, try case-insensitive match
                                            if not label and original_category:
                                                label = category_display_map.get(original_category.lower())
                                            
                                            # If still not found, infer from product details
                                            if not label:
                                                label = infer_category_from_product(p)
                                                logger.debug(f"Inferred category {label} for product {p.get('name', 'Unknown')} (original_category: {original_category})")
                                            
                                            if label not in category_to_products:
                                                category_to_products[label] = []
                                            category_to_products[label].append(p)
                                            product_ids_seen.add(pid)
                                else:
                                    # Check for non-product results (cart, orders, etc.)
                                    if result.get("result") and not result.get("products"):
                                        # This is a cart/order operation - store it for later
                                        if not hasattr(self, '_non_product_results'):
                                            self._non_product_results = []
                                        self._non_product_results.append(result.get("result"))
                                    continue
                            
                            # Now format the deduplicated response
                            bot_response_parts = []
                            
                            # Add non-product results first (cart operations, orders, etc.)
                            if hasattr(self, '_non_product_results') and self._non_product_results:
                                bot_response_parts.extend(self._non_product_results)
                                delattr(self, '_non_product_results')
                            
                            # Add product results
                            # Only include categories that were requested in the query
                            # Map query to extracted categories (normalized)
                            requested_categories = []
                            if hasattr(self, 'last_query') and self.last_query:
                                from src.search import HybridSearch
                                requested_categories = HybridSearch()._extract_categories(self.last_query)
                            # Normalize category keys
                            def normalize(cat):
                                return cat.strip().replace(' ', '_').lower()
                            # Map to display labels
                            label_map = {
                                'clothing': '👕 Clothing',
                                'sports': '⚽ Sports',
                                'home_garden': '🏠 Home & Garden',
                                'books': '📚 Books',
                                'phones': '📱 Phones',
                                'computers': '💻 Laptops',
                                'audio': '🎧 Audio & Headphones',
                                'gaming': '🎮 Gaming',
                                'wearables': '⌚ Wearables',
                                'electronics': '⚡ Electronics'
                            }
                            # Only show requested categories, in order, with fallback for missing
                            any_found = False
                            for cat in requested_categories:
                                norm_cat = normalize(cat)
                                label = label_map.get(norm_cat, cat.capitalize())
                                group_products = category_to_products.get(label, [])
                                if not group_products:
                                    # Try fallback: check for label with different case or spaces
                                    alt_label = label_map.get(cat.lower(), cat.capitalize())
                                    group_products = category_to_products.get(alt_label, [])
                                if not group_products:
                                    bot_response_parts.append(f"**{label}:**\nNo products found for this category.\n")
                                    continue
                                any_found = True
                                if len(group_products) == 1:
                                    product = group_products[0]
                                    stock_msg = "in stock" if product['stock_status'] == "in_stock" else f"{product['stock_status']}"
                                    header = f"**{label}:**\n"
                                    bot_response_parts.append(
                                        f"{header}The {product['name']} is priced at ${product['price']:.2f} and is currently {stock_msg}."
                                    )
                                else:
                                    header = f"**{label}:**\n"
                                    response_parts = [header + f"I found {len(group_products)} product(s):\n\n"]
                                    for i, product in enumerate(group_products, 1):
                                        stock_msg = "in stock" if product['stock_status'] == "in_stock" else f"{product['stock_status']}"
                                        response_parts.append(
                                            f"{i}. {product['name']} - ${product['price']:.2f} ({stock_msg})\n"
                                        )
                                    bot_response_parts.append("".join(response_parts))
                            if not any_found:
                                bot_response_parts.append("No products found for any of the requested categories. Please try different keywords.")
                            
                            bot_response = "\n\n".join(bot_response_parts) if bot_response_parts else None
                            
                            if not bot_response:
                                logger.warning("No response from function results aggregation")
                        
                        # If we still don't have a response, try LLM call
                        if not bot_response:
                            # #region agent log
                            fallback_start = time.time()
                            try:
                                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1294","message":"FALLBACK LLM call triggered","data":{"chat_history_length":len(self.chat_history)},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
                            except: pass
                            # #endregion
                            try:
                                logger.info("Making API call to generate response...")
                                # Fix: Use self.client instead of client, and use reduced context
                                response = self.client.chat.completions.create(
                                    model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
                                    messages=messages[-3:],  # Use minimal context for fallback
                                    temperature=0.7,
                                    timeout=30.0
                                )
                                message = response.choices[0].message
                                bot_response = message.content
                                logger.info("LLM response generated successfully")
                                
                                # #region agent log
                                fallback_end = time.time()
                                fallback_duration = fallback_end - fallback_start
                                try:
                                    with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1305","message":"FALLBACK LLM call END","data":{"duration_ms":fallback_duration*1000},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
                                except: pass
                                # #endregion
                            except Exception as e:
                                # #region agent log
                                try:
                                    with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1307","message":"FALLBACK LLM call ERROR","data":{"error":str(e)[:100]},"sessionId":"debug-session","runId":"run1","hypothesisId":"E"}) + '\n')
                                except: pass
                                # #endregion
                                logger.error(f"Failed to generate LLM response: {str(e)}")
                                bot_response = "I processed your request, but couldn't generate a response. Please try again."
                        
                        # Final fallback
                        if not bot_response:
                            bot_response = "I'm sorry, I couldn't process that request. Please try again."
                            logger.warning("bot_response was None, using final fallback")
                    else:
                        # No tool calls - use message content directly
                        if message.content:
                            bot_response = message.content
                        else:
                            bot_response = "I received your message but couldn't generate a response."
                    
                    # Ensure bot_response is always set before using it
                    if not bot_response:
                        bot_response = "I'm sorry, I couldn't process that request. Please try again."
                    
                    self.chat_history.append({"role": "assistant", "content": bot_response})
                    trace.end(output={"response": bot_response[:200], "success": True})
                    # Flush asynchronously to avoid blocking response
                    self._flush_tracer_async()
                    
                    # #region agent log
                    handle_end = time.time()
                    total_duration = handle_end - handle_start
                    try:
                        with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"chatbot.py:1320","message":"handle_message END","data":{"total_duration_ms":total_duration*1000,"response_length":len(bot_response)},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
                    except: pass
                    # #endregion
                    
                    logger.info(f"Returning response to user: {bot_response[:100] if len(bot_response) > 100 else bot_response}")
                    return bot_response
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(f"API call failed, retrying in {wait_time}s... ({retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to get response after {max_retries} attempts: {str(e)}", exc_info=True)
                        trace.end(output={"error": str(e), "success": False})
                        # Flush asynchronously to avoid blocking response
                        self._flush_tracer_async()
                        return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            trace.end(output={"error": str(e), "success": False})
            # Flush asynchronously to avoid blocking response
            self._flush_tracer_async()
            return "I encountered an error. Please try again."
    
    def run(self):
        """Run interactive chatbot."""
        print("=" * 60)
        print("Welcome to the E-commerce Chatbot!")
        print("Ask me about products, prices, or place an order.")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                response = self.handle_message(user_input)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}", exc_info=True)
                print("I encountered an error. Please try again.\n")
        
        # Exit summary
        if self.order_count > 0:
            print(f"\n{'='*60}")
            print(f"You placed {self.order_count} order(s). Thank you!")
            print(f"{'='*60}\n")
        else:
            print("\nThank you for using the E-commerce Chatbot!\n")


if __name__ == "__main__":
    """Run chatbot."""
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="E-commerce Chatbot")
    parser.add_argument(
        "--db-path",
        type=str,
        default="./orders.db",
        help="Path to database file"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="./vector_store",
        help="Path to vector store directory"
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = EcommerceChatbot(
            db_path=args.db_path,
            vector_store_path=args.vector_store
        )
        chatbot.run()
    except Exception as e:
        print(f"Error starting chatbot: {str(e)}")
        logger.error(f"Error starting chatbot: {str(e)}", exc_info=True)

