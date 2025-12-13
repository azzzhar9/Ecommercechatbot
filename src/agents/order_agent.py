"""Order Agent for processing orders with stock verification and confirmation."""

import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from src.database import check_stock, create_order
from src.logger import get_logger
from src.models import OrderModel
from src.utils import sanitize_input, validate_email

logger = get_logger()


class OrderAgent:
    """Order Agent for processing orders with validation and stock checking."""
    
    def __init__(self, api_key: Optional[str] = None, db_path: str = "./orders.db", vector_store_path: str = "./vector_store"):
        """
        Initialize Order Agent.
        
        Args:
            api_key: OpenAI API key
            db_path: Path to database file
            vector_store_path: Path to vector store for stock checking
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not provided")
        
        # Configure client for OpenRouter if base_url is provided
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.db_path = db_path
        self.vector_store_path = vector_store_path
        logger.info("Order Agent initialized")
    
    def detect_order_intent(self, chat_history: List[Dict], max_retries: int = 3) -> bool:
        """
        Detect if user wants to place an order.
        
        Args:
            chat_history: Conversation history
            max_retries: Maximum retry attempts
        
        Returns:
            True if order intent detected, False otherwise
        """
        try:
            # Check for explicit order phrases
            order_phrases = [
                "i'll take it", "i'll take", "place order", "buy", "purchase",
                "confirm", "yes, please", "yes please", "order it", "i want to buy",
                "add to cart", "checkout", "proceed with order"
            ]
            
            last_user_message = ""
            for msg in reversed(chat_history):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "").lower()
                    break
            
            if any(phrase in last_user_message for phrase in order_phrases):
                return True
            
            # Use LLM for more nuanced detection
            system_prompt = """You are an order intent detection system.
Analyze the conversation and determine if the user wants to place an order.
Respond with only "YES" or "NO"."""
            
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                for msg in chat_history[-5:]  # Last 5 messages
            ])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nDoes the user want to place an order? (YES/NO)"}
            ]
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Use model from environment or default
                    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=10
                    )
                    answer = response.choices[0].message.content.strip().upper()
                    return "YES" in answer
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        import time
                        time.sleep(2 ** retry_count)
                    else:
                        logger.error(f"Error detecting order intent: {str(e)}", exc_info=True)
                        return False
            
        except Exception as e:
            logger.error(f"Error in order intent detection: {str(e)}", exc_info=True)
            return False
    
    def extract_order_details(self, chat_history: List[Dict], max_retries: int = 3) -> Optional[Dict]:
        """
        Extract order details from chat history.
        
        Args:
            chat_history: Conversation history
            max_retries: Maximum retry attempts
        
        Returns:
            Dictionary with order details or None
        """
        try:
            system_prompt = """You are an order extraction system.
Extract the following information from the conversation:
- product_name: The name of the product the user wants to order
- quantity: The quantity (default to 1 if not specified)
- customer_name: Customer name if mentioned
- customer_email: Customer email if mentioned

Return a JSON object with these fields. If information is missing, use null."""
            
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                for msg in chat_history
            ])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation:\n{conversation_text}\n\nExtract order details as JSON:"}
            ]
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Use model from environment or default
                    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=200,
                        response_format={"type": "json_object"}
                    )
                    import json
                    order_data = json.loads(response.choices[0].message.content)
                    
                    # Validate and sanitize
                    product_name = sanitize_input(order_data.get("product_name", ""), max_length=200)
                    quantity = int(order_data.get("quantity", 1))
                    customer_name = sanitize_input(order_data.get("customer_name", ""), max_length=100) if order_data.get("customer_name") else None
                    customer_email = order_data.get("customer_email") if validate_email(order_data.get("customer_email", "")) else None
                    
                    if not product_name:
                        logger.warning("Could not extract product name from conversation")
                        return None
                    
                    return {
                        "product_name": product_name,
                        "quantity": max(1, quantity),  # Ensure at least 1
                        "customer_name": customer_name,
                        "customer_email": customer_email
                    }
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        import time
                        time.sleep(2 ** retry_count)
                    else:
                        logger.error(f"Error extracting order details: {str(e)}", exc_info=True)
                        return None
            
        except Exception as e:
            logger.error(f"Error in order extraction: {str(e)}", exc_info=True)
            return None
    
    def verify_stock(self, product_name: str) -> Tuple[bool, str]:
        """
        Verify stock status for a product.
        
        Args:
            product_name: Name of the product
        
        Returns:
            Tuple of (can_proceed, message)
        """
        try:
            stock_status = check_stock(product_name, self.vector_store_path)
            
            if stock_status == "out_of_stock":
                return False, f"Sorry, {product_name} is currently out of stock. Please check back later or consider a similar product."
            elif stock_status == "low_stock":
                return True, f"Warning: {product_name} has low stock. Would you like to proceed with your order?"
            else:
                return True, f"{product_name} is in stock and ready to order."
                
        except Exception as e:
            logger.error(f"Error verifying stock: {str(e)}", exc_info=True)
            # Default to allowing order on error (could be changed to False for safety)
            return True, "Unable to verify stock status. Proceeding with order."
    
    def get_product_price(self, product_name: str) -> Optional[float]:
        """
        Get product price from vector store metadata.
        
        Args:
            product_name: Name of the product
        
        Returns:
            Price as float or None
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_collection("products")
            
            results = collection.get(
                where={"name": product_name},
                limit=1
            )
            
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                price = results['metadatas'][0].get('price')
                if price is not None:
                    return float(price)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting product price: {str(e)}", exc_info=True)
            return None
    
    def request_confirmation(self, order_summary: Dict) -> bool:
        """
        Request final confirmation from user.
        
        Args:
            order_summary: Dictionary with order details
        
        Returns:
            True if confirmed, False otherwise
        """
        try:
            summary = (
                f"Product: {order_summary['product_name']}\n"
                f"Quantity: {order_summary['quantity']}\n"
                f"Unit Price: ${order_summary['unit_price']:.2f}\n"
                f"Total: ${order_summary['total_price']:.2f}"
            )
            
            print(f"\n{'='*50}")
            print("ORDER CONFIRMATION")
            print(f"{'='*50}")
            print(summary)
            print(f"{'='*50}")
            
            response = input("\nConfirm this order? (yes/no): ").strip().lower()
            return response in ['yes', 'y']
            
        except Exception as e:
            logger.error(f"Error requesting confirmation: {str(e)}", exc_info=True)
            return False
    
    def process_order(self, order_data: Dict) -> Tuple[bool, str, Optional[str]]:
        """
        Process and save order to database.
        
        Args:
            order_data: Dictionary with order details
        
        Returns:
            Tuple of (success, message, order_id)
        """
        try:
            # Verify stock
            can_proceed, stock_message = self.verify_stock(order_data['product_name'])
            if not can_proceed:
                return False, stock_message, None
            
            # Get price from metadata
            unit_price = self.get_product_price(order_data['product_name'])
            if unit_price is None:
                return False, f"Could not retrieve price for {order_data['product_name']}. Please try again.", None
            
            # Calculate total
            quantity = order_data['quantity']
            total_price = quantity * unit_price
            
            # Create OrderModel
            order = OrderModel(
                product_name=order_data['product_name'],
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                customer_name=order_data.get('customer_name'),
                customer_email=order_data.get('customer_email')
            )
            
            # Request confirmation
            order_summary = {
                'product_name': order.product_name,
                'quantity': order.quantity,
                'unit_price': order.unit_price,
                'total_price': order.total_price
            }
            
            if not self.request_confirmation(order_summary):
                return False, "Order cancelled by user.", None
            
            # Save to database
            order_id = create_order(order, self.db_path)
            
            confirmation_message = (
                f"Your order has been confirmed!\n"
                f"Order ID: {order_id}\n"
                f"Product: {order.product_name} x{order.quantity}\n"
                f"Total: ${order.total_price:.2f}\n"
                f"Thank you for your purchase!"
            )
            
            return True, confirmation_message, order_id
            
        except Exception as e:
            logger.error(f"Error processing order: {str(e)}", exc_info=True)
            return False, "An error occurred while processing your order. Please try again.", None
    
    def process_order_without_confirmation(self, order_data: Dict) -> Tuple[bool, str, Optional[str]]:
        """
        Process and save order to database WITHOUT interactive confirmation (for checkout).
        
        Args:
            order_data: Dictionary with order details
        
        Returns:
            Tuple of (success, message, order_id)
        """
        try:
            # Verify stock
            can_proceed, stock_message = self.verify_stock(order_data['product_name'])
            if not can_proceed:
                return False, stock_message, None
            
            # Get price from metadata (or use unit_price from cart if available)
            unit_price = self.get_product_price(order_data['product_name'])
            if unit_price is None:
                # Try to get price from order_data if it's a cart item
                unit_price = order_data.get('unit_price')
                if unit_price is None:
                    return False, f"Could not retrieve price for {order_data['product_name']}. Please try again.", None
            
            # Calculate total
            quantity = order_data['quantity']
            total_price = quantity * unit_price
            
            # Create OrderModel
            order = OrderModel(
                product_name=order_data['product_name'],
                quantity=quantity,
                unit_price=unit_price,
                total_price=total_price,
                customer_name=order_data.get('customer_name'),
                customer_email=order_data.get('customer_email')
            )
            
            # Save to database (skip confirmation for checkout)
            order_id = create_order(order, self.db_path)
            
            confirmation_message = (
                f"Order confirmed for {order.product_name} x{order.quantity} - ${order.total_price:.2f}"
            )
            
            return True, confirmation_message, order_id
            
        except Exception as e:
            logger.error(f"Error processing order without confirmation: {str(e)}", exc_info=True)
            return False, f"An error occurred while processing your order: {str(e)}", None

