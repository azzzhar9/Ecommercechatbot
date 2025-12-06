# E-commerce Chatbot with RAG and Autonomous Order Processing

## Project Description

This project implements an intelligent e-commerce chatbot that combines Retrieval-Augmented Generation (RAG) with autonomous multi-agent orchestration to provide a seamless shopping experience. The system features two specialized agents: a RAG Agent that retrieves product information from a vector store, and an Order Agent that processes orders autonomously using OpenAI Function Calling. The chatbot can answer product questions, provide pricing information, check stock availability, and process complete orders through natural conversation without requiring forms or manual intervention.

The system uses ChromaDB for vector storage, SQLite for order persistence, and OpenAI's GPT models (via OpenRouter API) with function calling for intelligent agent orchestration. All prices are retrieved from vector store metadata to ensure accuracy, and orders are validated using Pydantic models before database persistence. The implementation includes comprehensive error handling, input sanitization, stock verification, and order confirmation workflows.

### Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid BM25 Search** | Advanced search combining BM25 ranking with semantic matching |
| **Shopping Cart** | Multi-product cart with tax, shipping, and coupon support |
| **Product Recommendations** | Similarity-based recommendations during conversation |
| **Streamlit Web UI** | Beautiful chat interface for demos and live presentations |
| **Langfuse Tracing** | Full observability with trace logging for every conversation |
| **OpenRouter Support** | Use GPT-4o-mini via OpenRouter API for cost-effective inference |
| **Session Memory** | Remembers browsed products within session |
| **Coupon System** | Apply discount codes (SAVE10, SAVE20, DEMO) |

## Architecture

The system follows a two-agent architecture with clear separation of concerns:

```
User Input (CLI or Streamlit UI)
    ↓
┌─────────────────────────────────────────┐
│         Chatbot Orchestrator            │
│  (Session Memory, Cart, Function Call)  │
└─────────────────────────────────────────┘
    ↓
┌───────────┬───────────────┬─────────────┐
│           │               │             │
│ RAG Agent │ Cart Manager  │ Order Agent │
│           │               │             │
└───────────┴───────────────┴─────────────┘
    ↓             ↓               ↓
┌─────────┐ ┌───────────┐ ┌─────────────┐
│ Hybrid  │ │  Shopping │ │  Database   │
│ Search  │ │   Cart    │ │  (SQLite)   │
│ (BM25)  │ │ (Coupons) │ │             │
└─────────┴─┴───────────┴─┴─────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Langfuse Tracing (Observability)       │
└─────────────────────────────────────────┘
```

**Agent Workflow:**
1. **RAG Agent**: Handles product queries by searching the vector store, retrieving product information with prices from metadata, and providing conversational responses.
2. **Order Agent**: Activates when purchase intent is detected, extracts order details from conversation history, verifies stock availability, requests confirmation, and persists orders to the database.
3. **Function Calling**: Enables autonomous tool selection - the LLM decides when to call `search_products` or `create_order` based purely on conversation context.

**Handoff Mechanism:**
- The chatbot analyzes conversation context to determine which agent should handle the request
- Order intent is detected through phrase matching and LLM analysis
- Smooth transition: "Perfect! Let me process your order for [Product]..."
- Order Agent extracts all details from chat history without re-asking the user

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- OpenRouter API key (get from https://openrouter.ai/keys) OR OpenAI API key
- Langfuse account (optional, for tracing - https://cloud.langfuse.com)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AIFinalProject
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   ```
   
   ```env
   # OpenRouter Configuration (recommended)
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   
   # OR OpenAI Configuration
   # OPENAI_API_KEY=sk-your-key-here
   
   # Langfuse Configuration (optional - for tracing)
   LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
   LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

5. **Initialize vector store:**
   ```bash
   python src/initialize_vector_store.py
   ```
   This will:
   - Load products from `data/products.json`
   - Generate embeddings using OpenAI
   - Store in ChromaDB vector store

6. **Initialize database:**
   The database is automatically created on first run, but you can verify:
   ```bash
   python -c "from src.database import init_database; init_database()"
   ```

## Usage

### Running the Chatbot (Command Line)

```bash
python src/chatbot.py
```

### Running the Streamlit Web UI (Recommended for Demos)

```bash
# Set PYTHONPATH first
$env:PYTHONPATH="E:\AIFinalProject"  # Windows PowerShell
# OR
export PYTHONPATH=$(pwd)  # Linux/Mac

# Run Streamlit
streamlit run streamlit_app.py --server.headless true
```

Then open **http://localhost:8501** in your browser.

**Streamlit Features:**
- 💬 Interactive chat interface
- 📦 Real-time order history sidebar
- 🎨 Dark theme styling
- 🔄 Reset conversation button
- 📊 Langfuse tracing integration
- 🛒 Live shopping cart in sidebar
- 🎟️ Coupon code support

### Example Queries

#### Product Search
| Query | Description |
|-------|-------------|
| "Show me laptops" | Basic product search |
| "What phones do you have?" | Category browsing |
| "Show me laptops under $1500" | Price-filtered search |
| "cheap phones" | Budget-friendly search |
| "expensive headphones" | Premium product search |
| "gaming accessories" | Category search |

#### Shopping Cart
| Query | Description |
|-------|-------------|
| "Add iPhone 15 Pro to my cart" | Add single item |
| "Add 2 MacBooks to cart" | Add multiple quantity |
| "What's in my cart?" | View cart contents |
| "Remove iPhone from cart" | Remove item |
| "Clear my cart" | Empty the cart |

#### Coupons & Checkout
| Query | Description |
|-------|-------------|
| "Apply coupon DEMO" | 25% discount |
| "Apply coupon SAVE10" | 10% discount |
| "Apply coupon SAVE20" | 20% discount |
| "Apply coupon WELCOME" | 15% discount |
| "Checkout" | Process order for all cart items |

#### Recommendations
| Query | Description |
|-------|-------------|
| "Show me similar products" | Based on last viewed |
| "What else do you recommend?" | Get recommendations |
| "Products like MacBook Pro" | Specific recommendations |

### Example Conversation (Cart Flow)

```
You: Show me phones under $1000
Bot: The Google Pixel 8 Pro is priced at $899.99 and is currently in stock...

You: Add Google Pixel 8 Pro to my cart
Bot: Added 1x Google Pixel 8 Pro to your cart! 🛒
     Cart now has 1 item(s). Total: $971.99

You: Add AirPods Pro too
Bot: Added 1x AirPods Pro to your cart! 🛒
     Cart now has 2 item(s). Total: $1241.98

You: Apply coupon DEMO
Bot: Coupon applied! 25% discount
     New total: $1007.48

You: Checkout
Bot: ✅ Order Confirmed!
     
     ** Shopping Cart **
     - Google Pixel 8 Pro x1 @ $899.99 = $899.99
     - AirPods Pro x1 @ $249.99 = $249.99
     
     Subtotal: $1149.98
     Discount (DEMO - 25%): -$287.50
     Tax (8%): $68.99
     Shipping: FREE
     
     **Total: $931.48**
     
     Order ID(s): ORD-ABC12345, ORD-DEF67890
     Thank you for your purchase! 🎉
```

### Example Conversation (Direct Order)

```
You: What's the price of iPhone 15 Pro?
Bot: The iPhone 15 Pro is priced at $999.99 and is currently in stock. 
     It features the latest A17 Pro chip, 256GB storage, and a Pro camera system.

You: I'll take 2 of them
Bot: Perfect! Let me process your order for the iPhone 15 Pro...

     ==================================================
     ORDER CONFIRMATION
     ==================================================
     Product: iPhone 15 Pro
     Quantity: 2
     Unit Price: $999.99
     Total: $1999.98
     ==================================================

Bot: Your order has been confirmed!
     Order ID: #ORD-ABC12345
     Product: iPhone 15 Pro x2
     Total: $1999.98
     Thank you for your purchase!
```

### Command Line Options

```bash
python src/chatbot.py --db-path ./orders.db --vector-store ./vector_store
```

## Product Data Management

### Adding/Updating Products

Edit `data/products.json` to add or modify products. Each product must have:
- `product_id`: Unique identifier (e.g., "PROD001")
- `name`: Product name
- `description`: Detailed description
- `price`: Price as float (must be > 0)
- `category`: Product category
- `stock_status`: "in_stock", "low_stock", or "out_of_stock"

**Example Product Entry:**
```json
{
  "product_id": "PROD001",
  "name": "iPhone 15 Pro",
  "description": "Latest iPhone with A17 Pro chip...",
  "price": 999.99,
  "category": "Electronics",
  "stock_status": "in_stock"
}
```

After updating products, reinitialize the vector store:
```bash
python src/initialize_vector_store.py
```

## Database Schema

The orders table structure:

| Field | Type | Description |
|-------|------|-------------|
| order_id | TEXT (PK) | Unique order identifier (format: ORD-XXXXXXXX) |
| product_name | TEXT | Name of ordered product |
| quantity | INTEGER | Order quantity (must be > 0) |
| unit_price | REAL | Price per unit |
| total_price | REAL | Total price (quantity × unit_price) |
| customer_name | TEXT | Customer name (optional) |
| customer_email | TEXT | Customer email (optional) |
| timestamp | DATETIME | Order timestamp |

### Viewing Saved Orders

```python
from src.database import get_all_orders

orders = get_all_orders()
for order in orders:
    print(f"Order {order['order_id']}: {order['product_name']} x{order['quantity']} = ${order['total_price']}")
```

## Technical Decisions

### Why Hybrid BM25 Search?

The hybrid search engine combines multiple ranking signals for superior product retrieval:

| Component | Purpose |
|-----------|---------|
| **BM25 Scoring** | Term frequency with document length normalization |
| **Synonym Expansion** | "phone" → ["phone", "iphone", "samsung", "galaxy", "smartphone"] |
| **Price Filtering** | Automatic extraction of "under $X", "cheap", "premium" |
| **Name Match Bonus** | Higher weight for matches in product name |
| **Stock Bonus** | In-stock products ranked higher |

This approach provides:
- Fast keyword matching without embedding API calls
- Intelligent query understanding (price filters, synonyms)
- Relevance ranking that prioritizes exact matches
- OpenRouter compatibility (no embedding API needed)

### Why Shopping Cart?

The cart system enables realistic e-commerce flows:

| Feature | Benefit |
|---------|---------|
| Multi-product orders | "Add laptop and mouse to cart" |
| Tax calculation | 8% automatic tax |
| Shipping logic | Free shipping over $100 |
| Coupon system | SAVE10, SAVE20, DEMO, WELCOME |
| Session persistence | Cart survives page refresh |

### Function Tools Available

The chatbot uses 7 function tools for autonomous operation:

| Function | Trigger Examples | Purpose |
|----------|-----------------|---------|
| `search_products` | "Show me phones", "laptops under $1000" | Hybrid product search |
| `add_to_cart` | "Add iPhone to cart", "I want this" | Add items to cart |
| `view_cart` | "What's in my cart?", "Show cart" | Display cart contents |
| `remove_from_cart` | "Remove laptop from cart" | Remove cart items |
| `apply_coupon` | "Apply coupon DEMO" | Apply discount codes |
| `checkout` | "Checkout", "Place order", "Buy now" | Process cart order |
| `get_recommendations` | "Similar products", "Recommend" | Product suggestions |
| `create_order` | "I'll take 2 of them" | Direct single-item order |

### Why RAG for Products?

RAG (Retrieval-Augmented Generation) is ideal for product catalogs because it ensures accuracy and prevents hallucination. By storing product information in a vector store with embeddings, we can:
- Retrieve exact product details based on semantic similarity
- Maintain a single source of truth (products.json → vector store)
- Scale to thousands of products efficiently
- Update product information without retraining models
- Ensure prices and stock status are always accurate

The vector store allows natural language queries ("laptops under $1000") to find relevant products, while metadata ensures prices and stock status are retrieved exactly as stored, not generated by the LLM.

### Why Function Calling?

OpenAI Function Calling enables autonomous agent orchestration without manual keyword routing. The LLM analyzes conversation context and decides:
- When to search for products (user asks about products)
- When to create orders (user confirms purchase intent)

This creates a natural, conversational flow where the system adapts to user intent rather than requiring specific commands. Function calling also provides structured outputs that can be validated and processed reliably.

### Why Two Agents Instead of One?

Specialization improves accuracy and maintainability:
- **RAG Agent**: Optimized for information retrieval, maintains product context, handles search queries
- **Order Agent**: Specialized for order processing, stock verification, confirmation workflows, database operations

This separation allows each agent to have focused system prompts and logic, making the system easier to debug, test, and improve. The handoff is seamless because both agents share the same conversation history.

### How Does Handoff Work?

Handoff is autonomous and context-driven:
1. Chatbot analyzes conversation to detect intent (product query vs. order)
2. If order intent detected (phrases like "I'll take it", "place order", "buy"), Order Agent activates
3. Order Agent extracts product, quantity, and price from chat history
4. Stock is verified, confirmation requested, order processed
5. Smooth transition message: "Perfect! Let me process your order..."

The system never requires manual switching - it's all based on conversation analysis.

### Database Choice Rationale

SQLite was chosen for:
- **Simplicity**: No separate database server required
- **Reliability**: ACID compliance, data persistence
- **Portability**: Single file, easy to backup/transfer
- **Performance**: Sufficient for this use case
- **Security**: Parameterized queries prevent SQL injection

For production at scale, PostgreSQL would be recommended, but SQLite is perfect for this prototype.

## Stock Verification

The system includes comprehensive stock verification:
- **Out of Stock**: Order is blocked, user informed
- **Low Stock**: Warning displayed, user can still proceed
- **In Stock**: Order proceeds normally

Stock status is checked from vector store metadata before order creation, ensuring real-time accuracy.

## Memory and Conversation Recovery

The chatbot maintains conversation context:
- **Last Product Tracking**: Remembers last discussed product
- **Ambiguous Resolution**: If user says "I'll take it" without context, asks for clarification
- **Multi-turn Context**: Maintains full conversation history for order extraction

## Price Source of Truth

**Critical**: Prices are ALWAYS retrieved from vector store metadata, never from LLM text output. This ensures:
- No hallucinated prices
- Exact price matching
- Consistency across queries
- Easy price updates (change products.json, reinitialize vector store)

## Order Confirmation Workflow

Every order requires explicit confirmation:
1. Order details extracted from conversation
2. Stock verified
3. Order summary displayed
4. User must type "yes" to confirm
5. Order saved to database only after confirmation

This prevents accidental orders and ensures user intent.

## Enhanced Logging

The system includes comprehensive logging:
- **Timestamps**: ISO format for all log entries
- **Session IDs**: Track conversations across sessions
- **Structured Format**: `[timestamp] [session_id] [level] [message]`
- **Log Events**: Order creation, API calls, errors, vector retrieval, stock checks

Logs are written to `logs/chatbot.log` and console.

## Error Handling

The system handles various error scenarios:
- **Network/API Failures**: Retry with exponential backoff
- **Vector Store Errors**: File missing/corrupted handling, fallback mechanisms
- **Database Errors**: Locked DB retry, disk full detection, user-friendly messages
- **Input Validation**: Sanitization, email validation, quantity/price validation

All errors are logged with session IDs for debugging.

## Security

Security measures implemented:
- **Input Sanitization**: Removes dangerous characters, prevents SQL injection
- **Parameterized Queries**: All database operations use parameterized queries
- **Email Validation**: Regex validation before storage
- **Error Message Sanitization**: Never expose raw SQL errors to users

## Scalability Features

- **Batching**: Embeddings generated in batches (50-100 products)
- **Caching**: Optional cache for frequent queries (can be added)
- **Async Support**: Architecture supports async operations (can be enhanced)

## Testing

Run tests with:
```bash
python -m pytest tests/test_chatbot.py -v
```

Or:
```bash
python -m unittest tests.test_chatbot
```

See `examples/test_conversations.md` for manual test scenarios.

## Optional Features

### Category Browsing
Query: "Show me electronics under $500"
- Searches by category and price filter
- Returns matching products

### Product Recommendations
Query: "Do you have similar products?"
- Uses vector similarity to find related products
- Based on product embeddings

### Exit Summary
On quit, displays: "You placed [N] orders. Thank you!"

## Troubleshooting

### Vector Store Not Found
```bash
python src/initialize_vector_store.py
```

### Database Errors
Check disk space and file permissions. Database is created automatically.

### API Key Issues
Verify `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is set in `.env` file.

### Import Errors
Ensure you're in the project root directory and virtual environment is activated.
Set PYTHONPATH:
```bash
$env:PYTHONPATH="E:\AIFinalProject"  # Windows
export PYTHONPATH=$(pwd)              # Linux/Mac
```

### Langfuse Not Connecting
1. Verify Langfuse keys are set in `.env`
2. Check internet connection
3. Tracing works without Langfuse (graceful fallback)

### Streamlit Won't Start
```bash
pip install streamlit
streamlit run streamlit_app.py --server.headless true
```

## Known Limitations

- Single-user sessions (multi-user support can be added)
- No payment processing (mock can be added)
- No email confirmations (can be simulated)
- Vector store must be reinitialized after product updates

## Future Enhancements

### ✅ Completed Enhancements
| Feature | Status | Description |
|---------|--------|-------------|
| Web Interface | ✅ DONE | Streamlit UI with cart sidebar |
| Langfuse Tracing | ✅ DONE | Full observability for every conversation |
| Hybrid BM25 Search | ✅ DONE | Fast keyword search with synonym expansion |
| Shopping Cart | ✅ DONE | Multi-product cart with tax, shipping, coupons |
| Product Recommendations | ✅ DONE | Similarity-based suggestions |
| Price Filtering | ✅ DONE | "under $X", "cheap", "premium" queries |

### 🚀 Suggestions for Extra Credit

| Enhancement | Description | Difficulty |
|-------------|-------------|------------|
| **Unit Test Coverage Expansion** | Increase coverage around cart manager and hybrid search logic | Medium |
| **Streaming Chat Responses** | Partial streaming improves UX for longer answers | Medium |
| **Multi-user Session Isolation** | Per-session cart and memory using session keys in DB or Redis | Hard |
| **Payment Simulation + Order Status** | Simple stub or state transition (Pending → Paid → Fulfilled) | Medium |
| **Categorization LLM Fallback** | If metadata missing, classify product by description | Easy |

### Implementation Notes for Future Work

**Streaming Responses:**
```python
# Use OpenAI streaming API
for chunk in client.chat.completions.create(..., stream=True):
    yield chunk.choices[0].delta.content
```

**Multi-user Sessions:**
```python
# Store cart per session ID in Redis
cart_key = f"cart:{session_id}"
redis.hset(cart_key, product_id, quantity)
```

**Order Status Transitions:**
```python
class OrderStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    FULFILLED = "fulfilled"
```

## Langfuse Tracing

The project includes comprehensive Langfuse tracing for observability:

### What's Traced
- **Conversation Turns**: Every user message → bot response
- **LLM Generations**: Token usage, model, latency
- **Function Calls**: search_products, create_order with arguments
- **Session Tracking**: Group traces by conversation session

### Viewing Traces
1. Go to https://cloud.langfuse.com
2. Sign in with your account
3. View traces, spans, and generations in real-time

### Trace Structure
```
handle_message (trace)
├── chat_completion (generation) - LLM call with token usage
├── function_search_products (span) - Product search
└── function_create_order (span) - Order creation
```

## OpenRouter Integration

The project uses OpenRouter API for cost-effective inference:

### Why OpenRouter?
- **Lower costs**: Pay-per-token pricing
- **Multiple models**: Access to GPT-4, Claude, Llama, etc.
- **No rate limits**: Higher throughput than direct OpenAI
- **Same API**: Drop-in replacement for OpenAI SDK

### Configuration
```env
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### Note on Embeddings
OpenRouter doesn't support the embeddings API, so the RAG Agent uses **keyword-based search** with synonym expansion instead of vector similarity. This provides fast, accurate product matching.

## Project Structure

```
AIFinalProject/
├── .env                          # Environment variables (API keys)
├── .env.example                  # Example environment file
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── streamlit_app.py              # Streamlit web UI
├── orders.db                     # SQLite database (auto-created)
├── data/
│   └── products.json             # Product catalog (35 products)
├── src/
│   ├── __init__.py
│   ├── chatbot.py                # Main chatbot orchestrator (7 function tools)
│   ├── database.py               # SQLite operations
│   ├── models.py                 # Pydantic validation models
│   ├── logger.py                 # Logging configuration
│   ├── tracing.py                # Langfuse tracing utilities (v3.x API)
│   ├── search.py                 # Hybrid BM25 search engine (NEW)
│   ├── cart.py                   # Shopping cart with coupons (NEW)
│   ├── initialize_vector_store.py # Vector store setup
│   └── agents/
│       ├── __init__.py
│       ├── rag_agent.py          # Product search agent
│       └── order_agent.py        # Order processing agent
├── tests/
│   ├── test_chatbot.py
│   ├── test_components.py
│   └── test_search.py
└── logs/
    └── chatbot.log               # Application logs
```

## License

This project is for educational purposes.

## Contact

For questions or issues, please open an issue in the repository.

