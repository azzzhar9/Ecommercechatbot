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
    .product-card-modern {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid #e0e0e0;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .product-card-modern:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .product-image {
        width: 100%;
        height: 200px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
        margin-bottom: 15px;
    }
    .product-name {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .product-price {
        font-size: 24px;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 10px;
    }
    .stock-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .stock-in-stock {
        background-color: #d1fae5;
        color: #065f46;
    }
    .stock-low-stock {
        background-color: #fef3c7;
        color: #92400e;
    }
    .stock-out-of-stock {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .category-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        background-color: #f3f4f6;
        color: #4b5563;
        margin-bottom: 12px;
    }
    .product-description {
        font-size: 13px;
        color: #6b7280;
        line-height: 1.5;
        margin-bottom: 15px;
        flex-grow: 1;
    }
    .add-to-cart-btn {
        width: 100%;
        padding: 10px;
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .add-to-cart-btn:hover {
        background-color: #1d4ed8;
    }
    .category-filter-btn {
        padding: 8px 16px;
        border-radius: 20px;
        border: 2px solid #e5e7eb;
        background: white;
        cursor: pointer;
        transition: all 0.2s;
        margin: 5px;
    }
    .category-filter-btn:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }
    .category-filter-btn.active {
        background: #2563eb;
        color: white;
        border-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with caching
@st.cache_resource
def get_chatbot():
    """Get or create chatbot instance (cached across reruns to avoid cold starts)."""
    from src.chatbot import EcommerceChatbot
    return EcommerceChatbot()

if 'chatbot' not in st.session_state:
    try:
        st.session_state.chatbot = get_chatbot()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.init_error = str(e)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'order_count' not in st.session_state:
    st.session_state.order_count = 0

if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "All"

if 'product_search' not in st.session_state:
    st.session_state.product_search = ""

# Helper functions
@st.cache_data
def load_products():
    """Load products from JSON file."""
    import json
    from pathlib import Path
    
    products_file = Path("./data/products.json")
    if not products_file.exists():
        return []
    
    try:
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        return products
    except Exception as e:
        st.error(f"Error loading products: {e}")
        return []

def get_category_icon(category):
    """Get emoji icon for category."""
    icons = {
        "Electronics": "⚡",
        "Clothing": "👕",
        "Books": "📚",
        "Home & Kitchen": "🏠",
        "Sports & Outdoors": "⚽",
        "Toys & Games": "🎮",
        "Beauty & Personal Care": "💄",
        "Health & Household": "💊"
    }
    return icons.get(category, "📦")

def get_stock_badge_html(stock_status):
    """Get HTML for stock status badge."""
    if stock_status == "in_stock":
        return '<span class="stock-badge stock-in-stock">✅ In Stock</span>'
    elif stock_status == "low_stock":
        return '<span class="stock-badge stock-low-stock">⚠️ Low Stock</span>'
    elif stock_status == "out_of_stock":
        return '<span class="stock-badge stock-out-of-stock">❌ Out of Stock</span>'
    else:
        return '<span class="stock-badge">❓ Unknown</span>'

def render_product_card(product, chatbot):
    """Render a professional product card."""
    icon = get_category_icon(product.get('category', 'Electronics'))
    stock_badge = get_stock_badge_html(product.get('stock_status', 'unknown'))
    
    card_html = f"""
    <div class="product-card-modern">
        <div class="product-image">{icon}</div>
        <div class="product-name">{product.get('name', 'Unknown Product')}</div>
        <div class="product-price">${product.get('price', 0):.2f}</div>
        {stock_badge}
        <div class="category-badge">{product.get('category', 'Uncategorized')}</div>
        <div class="product-description">{product.get('description', '')[:100]}...</div>
    </div>
    """
    return card_html

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
    - **RAG**: Keyword Search
    - **Database**: SQLite
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

# Create tabs for Browse Products and Chat
tab1, tab2 = st.tabs(["🛍️ Browse Products", "💬 Chat"])

# Browse Products Tab
with tab1:
    st.header("Browse Our Products")
    
    # Load products
    all_products = load_products()
    
    if all_products:
        # Filters section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 Search products...",
                value=st.session_state.product_search,
                placeholder="Search by name, category, or description",
                key="product_search_input"
            )
            st.session_state.product_search = search_query
        
        with col2:
            sort_option = st.selectbox(
                "Sort by",
                ["Price: Low to High", "Price: High to Low", "Name: A-Z", "Name: Z-A"],
                key="sort_option"
            )
        
        # Category filter
        categories = ["All"] + sorted(list(set(p.get('category', 'Uncategorized') for p in all_products)))
        
        st.markdown("### Categories")
        category_cols = st.columns(min(len(categories), 8))
        for idx, category in enumerate(categories[:8]):
            with category_cols[idx]:
                if st.button(
                    category,
                    key=f"cat_{category}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_category == category else "secondary"
                ):
                    st.session_state.selected_category = category
                    st.rerun()
        
        # Filter products
        filtered_products = all_products.copy()
        
        # Apply category filter
        if st.session_state.selected_category != "All":
            filtered_products = [p for p in filtered_products if p.get('category') == st.session_state.selected_category]
        
        # Apply search filter
        if search_query:
            query_lower = search_query.lower()
            filtered_products = [
                p for p in filtered_products
                if query_lower in p.get('name', '').lower()
                or query_lower in p.get('description', '').lower()
                or query_lower in p.get('category', '').lower()
            ]
        
        # Sort products
        if sort_option == "Price: Low to High":
            filtered_products.sort(key=lambda x: x.get('price', 0))
        elif sort_option == "Price: High to Low":
            filtered_products.sort(key=lambda x: x.get('price', 0), reverse=True)
        elif sort_option == "Name: A-Z":
            filtered_products.sort(key=lambda x: x.get('name', ''))
        elif sort_option == "Name: Z-A":
            filtered_products.sort(key=lambda x: x.get('name', ''), reverse=True)
        
        # Display products
        st.markdown(f"### Found {len(filtered_products)} product(s)")
        st.markdown("---")
        
        if filtered_products:
            # Display products in grid (3 columns)
            for i in range(0, len(filtered_products), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(filtered_products):
                        product = filtered_products[i + j]
                        with col:
                            # Product card
                            icon = get_category_icon(product.get('category', 'Electronics'))
                            stock_status = product.get('stock_status', 'unknown')
                            
                            # Display product info
                            st.markdown(f"### {icon} {product.get('name', 'Unknown')}")
                            st.markdown(f"**${product.get('price', 0):.2f}**")
                            
                            # Stock badge
                            if stock_status == "in_stock":
                                st.success("✅ In Stock")
                            elif stock_status == "low_stock":
                                st.warning("⚠️ Low Stock")
                            elif stock_status == "out_of_stock":
                                st.error("❌ Out of Stock")
                            
                            # Category
                            st.caption(f"📁 {product.get('category', 'Uncategorized')}")
                            
                            # Description
                            with st.expander("View Details"):
                                st.write(product.get('description', 'No description available'))
                            
                            # Add to cart button
                            if stock_status != "out_of_stock":
                                if st.button(
                                    "🛒 Add to Cart",
                                    key=f"add_{product.get('product_id', i+j)}_{i+j}",
                                    use_container_width=True
                                ):
                                    try:
                                        # Add product to cart via chatbot
                                        result = st.session_state.chatbot.execute_function(
                                            "add_to_cart",
                                            {
                                                "product_name": product.get('name'),
                                                "quantity": 1,
                                                "unit_price": product.get('price', 0)
                                            }
                                        )
                                        if result.get('success'):
                                            st.success(f"✅ Added {product.get('name')} to cart!")
                                            st.rerun()
                                        else:
                                            st.error(result.get('result', 'Failed to add to cart'))
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                            else:
                                st.info("Out of stock")
                            
                            st.markdown("---")
        else:
            st.info("No products found matching your filters. Try adjusting your search or category selection.")
    else:
        st.warning("No products available. Please check the product database.")

# Chat Tab
with tab2:

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
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        try:
            response = st.session_state.chatbot.handle_message(prompt)
            
            # Check if order was created
            if "order" in response.lower() and ("confirmed" in response.lower() or "created" in response.lower() or "ORD-" in response):
                st.session_state.order_count += 1
            
        except Exception as e:
            response = f"Sorry, I encountered an error: {str(e)}"
        
        # Add assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to display updated messages
        st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔒 Secure Checkout**")
with col2:
    st.markdown("**📞 24/7 Support**")
with col3:
    st.markdown("**🚚 Fast Delivery**")
