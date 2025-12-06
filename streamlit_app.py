"""Streamlit UI for E-commerce Chatbot - Impressive Demo Interface"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="E-Commerce AI Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChat message {
        padding: 10px;
        border-radius: 10px;
    }
    .product-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .cart-item {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #ffc107;
    }
    .order-success {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #28a745;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .coupon-badge {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    try:
        from src.chatbot import EcommerceChatbot
        st.session_state.chatbot = EcommerceChatbot()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.init_error = str(e)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'order_count' not in st.session_state:
    st.session_state.order_count = 0

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=100)
    st.title("🛒 E-Commerce Chatbot")
    st.markdown("---")
    
    # Shopping Cart Display
    if st.session_state.initialized:
        cart = st.session_state.chatbot.cart
        
        st.markdown("### 🛒 Shopping Cart")
        if cart.is_empty:
            st.info("Your cart is empty")
        else:
            for item in cart.items:
                st.markdown(f"""
                <div class="cart-item">
                    <strong>{item.product_name}</strong><br>
                    {item.quantity}x @ ${item.unit_price:.2f} = ${item.subtotal:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Subtotal:** ${cart.subtotal:.2f}")
            if cart.discount_percent > 0:
                st.markdown(f"**Discount ({cart.coupon_code}):** -${cart.discount_amount:.2f}")
            st.markdown(f"**Tax (8%):** ${cart.tax_amount:.2f}")
            if cart.shipping_cost > 0:
                st.markdown(f"**Shipping:** ${cart.shipping_cost:.2f}")
            else:
                st.markdown("**Shipping:** FREE ✓")
            st.markdown(f"### Total: ${cart.total:.2f}")
            
            # Coupon code input
            with st.expander("🎟️ Have a coupon?"):
                coupon = st.text_input("Enter code:", key="coupon_input")
                if st.button("Apply"):
                    success, msg = cart.apply_coupon(coupon)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                st.caption("Try: SAVE10, SAVE20, DEMO")
        
        st.markdown("---")
    
    st.markdown("### 💡 Try These Commands:")
    st.markdown("""
    - "Show me laptops under $1500"
    - "Add iPhone to my cart"
    - "What's in my cart?"
    - "Apply coupon DEMO"
    - "Checkout"
    - "Show similar products"
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    
    if st.session_state.initialized:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Orders", st.session_state.order_count)
        
        st.markdown(f"**Session ID:**")
        st.code(st.session_state.chatbot.session_id[:8] + "...")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🛒 Clear Cart", use_container_width=True):
        if st.session_state.initialized:
            st.session_state.chatbot.cart.clear()
            st.rerun()
    
    if st.button("📦 View Orders", use_container_width=True):
        st.session_state.show_orders = True
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **LLM**: GPT-4o-mini (OpenRouter)
    - **Search**: Hybrid BM25 + Semantic
    - **Database**: SQLite
    - **Cart**: Multi-item support
    - **Validation**: Pydantic
    - **Tracing**: Langfuse
    """)

# Main content
st.title("🤖 AI Shopping Assistant")
st.markdown("Welcome! I can help you find products and place orders. Just ask me anything!")

# Check initialization
if not st.session_state.initialized:
    st.error(f"Failed to initialize chatbot: {st.session_state.get('init_error', 'Unknown error')}")
    st.stop()

# Show orders modal
if st.session_state.get('show_orders', False):
    with st.expander("📦 Recent Orders", expanded=True):
        try:
            from src.database import get_all_orders
            orders = get_all_orders()
            if orders:
                for order in orders[-5:]:  # Show last 5 orders
                    st.markdown(f"""
                    <div class="product-card">
                        <strong>Order #{order['order_id']}</strong><br>
                        Product: {order['product_name']}<br>
                        Quantity: {order['quantity']} | Total: ${order['total_price']:.2f}<br>
                        <small>{order.get('timestamp', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No orders yet. Start shopping!")
        except Exception as e:
            st.error(f"Error loading orders: {e}")
        
        if st.button("Close"):
            st.session_state.show_orders = False
            st.rerun()

# Chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about products or place an order..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chatbot.handle_message(prompt)
                st.markdown(response)
                
                # Check if order was created
                if "order" in response.lower() and ("confirmed" in response.lower() or "created" in response.lower() or "ORD-" in response):
                    st.session_state.order_count += 1
                    st.balloons()
                
            except Exception as e:
                response = f"Sorry, I encountered an error: {str(e)}"
                st.error(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔒 Secure Checkout**")
with col2:
    st.markdown("**📞 24/7 Support**")
with col3:
    st.markdown("**🚚 Fast Delivery**")
