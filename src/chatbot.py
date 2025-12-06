"""Main chatbot application with function calling and session management."""

import os
import uuid
from typing import Dict, List, Optional

from openai import OpenAI

from src.agents.order_agent import OrderAgent
from src.agents.rag_agent import RAGAgent
from src.database import init_database
from src.logger import get_logger, setup_logger
from src.tracing import get_tracer, traced
from src.cart import ShoppingCart, CartManager

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
        
        # Initialize Langfuse tracer
        self.tracer = get_tracer()
        
        # Setup logger with session ID
        setup_logger(session_id=self.session_id)
        
        logger.info(f"Chatbot initialized with session ID: {self.session_id}")
    
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
                    "description": "Search for products by name, category, or description. Use this when the user asks about product information, prices, or availability. Supports filters like 'laptops under $1000' or 'cheap phones'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for products (e.g., 'iPhone', 'laptops under $1000', 'cheap phones')"
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
                    "description": "Checkout and place an order for all items in the cart. Use when user says 'checkout', 'place order', 'buy now', 'complete order'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_name": {
                                "type": "string",
                                "description": "Customer name (optional)"
                            },
                            "customer_email": {
                                "type": "string",
                                "description": "Customer email address (optional)"
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
                
                if products:
                    # Update last product and browsed products
                    self.last_product = products[0]['name']
                    self.browsed_products.extend(products[:3])  # Remember top 3
                
                # Format results
                result_text = "\n\n".join([
                    f"Product: {p['name']}\n"
                    f"Price: ${p['price']:.2f}\n"
                    f"Description: {p['description'][:200]}...\n"
                    f"Stock: {p['stock_status']}\n"
                    f"Category: {p['category']}"
                    for p in products
                ])
                
                return {
                    "success": True,
                    "result": f"Found {len(products)} product(s):\n\n{result_text}" if products else "No products found.",
                    "products": products
                }
            
            elif function_name == "add_to_cart":
                product_name = arguments.get("product_name", "")
                quantity = arguments.get("quantity", 1)
                unit_price = arguments.get("unit_price", 0.0)
                
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
                
                if customer_name:
                    self.cart.customer_name = customer_name
                if customer_email:
                    self.cart.customer_email = customer_email
                
                # Create orders for each item
                order_ids = []
                for item in self.cart.items:
                    order_data = {
                        "product_name": item.product_name,
                        "quantity": item.quantity,
                        "customer_name": self.cart.customer_name,
                        "customer_email": self.cart.customer_email
                    }
                    success, message, order_id = self.order_agent.process_order(order_data)
                    if success:
                        order_ids.append(order_id)
                        self.order_count += 1
                
                # Generate receipt
                receipt = f"""
✅ **Order Confirmed!**

{self.cart.get_summary()}

Order ID(s): {', '.join(order_ids)}

Thank you for your purchase! 🎉
"""
                
                # Clear cart after checkout
                self.cart.clear()
                
                return {
                    "success": True,
                    "result": receipt,
                    "order_ids": order_ids
                }
            
            elif function_name == "get_recommendations":
                product_name = arguments.get("product_name", "")
                if not product_name and self.last_product:
                    product_name = self.last_product
                
                recommendations = self.rag_agent.get_recommendations(product_name, k=3)
                
                if recommendations:
                    result_text = "Based on your interest, you might also like:\n\n"
                    for p in recommendations:
                        result_text += f"• **{p['name']}** - ${p['price']:.2f}\n"
                    return {
                        "success": True,
                        "result": result_text,
                        "products": recommendations
                    }
                return {
                    "success": False,
                    "result": "No recommendations available."
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
            
            # Add chat history
            messages.extend(self.chat_history[-10:])  # Last 10 messages for context
            
            # Get response with function calling
            tools = self.get_function_tools()
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Use model from environment or default
                    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
                    logger.info(f"Making API call to {model}...")
                    
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
                    
                    # Check for function calls
                    if message.tool_calls:
                        # Execute functions
                        function_results = []
                        for tool_call in message.tool_calls:
                            function_name = tool_call.function.name
                            import json
                            arguments = json.loads(tool_call.function.arguments)
                            
                            logger.info(f"Function called: {function_name} with args: {arguments}")
                            
                            # Create span for function execution
                            func_span = self.tracer.span(
                                trace=trace,
                                name=f"function_{function_name}",
                                input={"function": function_name, "arguments": arguments}
                            )
                            
                            # Execute function
                            function_result = self.execute_function(function_name, arguments)
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
                            if result.get("success") and result.get("products"):
                                products = result["products"]
                                if products:
                                    product = products[0]
                                    # Create a natural, conversational response
                                    stock_msg = "in stock" if product['stock_status'] == "in_stock" else f"has {product['stock_status']}"
                                    bot_response = (
                                        f"The {product['name']} is priced at ${product['price']:.2f} "
                                        f"and is currently {stock_msg}. "
                                    )
                                    # Add description if available
                                    if product.get('description'):
                                        desc = product['description'][:150].replace('\n', ' ')
                                        bot_response += f"{desc}..."
                                else:
                                    bot_response = "I couldn't find that product. Could you try rephrasing your search?"
                            elif result.get("success") and result.get("order_id"):
                                # Order was created
                                bot_response = result.get("result", "Your order has been processed.")
                            else:
                                bot_response = result.get("result", "I processed your request.")
                        
                        # If we have a response, use it (skip slow LLM call for faster response)
                        if not bot_response:
                            # Fallback if somehow bot_response is None
                            bot_response = "I processed your request, but couldn't generate a response. Please try again."
                            logger.warning("bot_response was None, using fallback")
                    else:
                        bot_response = message.content
                    
                    # Ensure we have a response
                    if not bot_response:
                        bot_response = "I'm sorry, I couldn't process that request. Please try again."
                    
                    # Add to chat history
                    self.chat_history.append({"role": "assistant", "content": bot_response})
                    
                    # End trace with success
                    trace.end(output={"response": bot_response[:200], "success": True})
                    self.tracer.flush()
                    
                    logger.info(f"Returning response to user: {bot_response[:100] if len(bot_response) > 100 else bot_response}")
                    return bot_response
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        import time
                        wait_time = 2 ** retry_count
                        logger.warning(f"API call failed, retrying in {wait_time}s... ({retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to get response after {max_retries} attempts: {str(e)}", exc_info=True)
                        trace.end(output={"error": str(e), "success": False})
                        self.tracer.flush()
                        return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            trace.end(output={"error": str(e), "success": False})
            self.tracer.flush()
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

