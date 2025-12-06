# E-commerce Chatbot with RAG and Autonomous Order Processing

## Project Description

This project implements an intelligent e-commerce chatbot that combines Retrieval-Augmented Generation (RAG) with autonomous multi-agent orchestration to provide a seamless shopping experience. The system features two specialized agents: a RAG Agent that retrieves product information from a vector store, and an Order Agent that processes orders autonomously using OpenAI Function Calling. The chatbot can answer product questions, provide pricing information, check stock availability, and process complete orders through natural conversation without requiring forms or manual intervention.

The system uses ChromaDB for vector storage, SQLite for order persistence, and OpenAI's GPT models (via OpenRouter API) with function calling for intelligent agent orchestration. All prices are retrieved from vector store metadata to ensure accuracy, and orders are validated using Pydantic models before database persistence. The implementation includes comprehensive error handling, input sanitization, stock verification, and order confirmation workflows.

### New Features
- **Streamlit Web UI**: Beautiful chat interface for demos and live presentations
- **Langfuse Tracing**: Full observability with trace logging for every conversation
- **OpenRouter Support**: Use GPT-4o-mini via OpenRouter API for cost-effective inference

## Architecture

The system follows a two-agent architecture with clear separation of concerns:

```
User Input (CLI or Streamlit UI)
    ↓
Chatbot (Main Orchestrator)
    ↓
┌─────────────────┬─────────────────┐
│                 │                 │
RAG Agent    Function Calling   Order Agent
│                 │                 │
↓                 ↓                 ↓
Vector Store   OpenRouter API   Database
(ChromaDB)     (GPT-4o-mini)    (SQLite)
    ↓
Langfuse Tracing (Observability)
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

### Example Conversation

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

     Confirm this order? (yes/no): yes

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

- ~~Web interface (Streamlit/Gradio)~~ ✅ **DONE** - Streamlit UI implemented
- ~~Langfuse tracing~~ ✅ **DONE** - Full observability added
- Multi-user session handling
- Email confirmation simulation
- Payment API mock integration
- Async/await for concurrent operations
- Caching for popular products

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
│   ├── chatbot.py                # Main chatbot orchestrator
│   ├── database.py               # SQLite operations
│   ├── models.py                 # Pydantic validation models
│   ├── logger.py                 # Logging configuration
│   ├── tracing.py                # Langfuse tracing utilities
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

