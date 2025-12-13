# Demo Day Presentation – E-commerce Chatbot

## :stopwatch: 1 Minute – Elevator Pitch

- **Assignment Chosen:** E-commerce Chatbot with Retrieval-Augmented Generation (RAG) and Autonomous Order Processing.
- **Problem Solved:** Online shopping can be confusing and time-consuming. Users want quick, accurate answers about products, prices, and orders—without filling out forms or navigating complex menus.
- **Solution Overview:** My project is an intelligent chatbot that lets users search for products, ask questions, and place orders—all through natural conversation. It combines advanced search with autonomous order handling for a seamless shopping experience.
- **Agents Involved:**
  - **RAG Agent:** Finds and explains product information using a hybrid search engine.
  - **Order Agent:** Detects purchase intent, verifies stock, and processes orders automatically.

---

## :stopwatch: 3 Minutes – Project Demo

- **Input:** The user types a message, such as “Show me laptops under $1500” or “I’ll take 2 of them.”
- **What Happens:**
  1. The chatbot analyzes the message and decides which agent to use.
  2. For product questions, the RAG Agent searches the catalog and returns relevant products with prices and stock status.
  3. If the user wants to buy, the Order Agent extracts order details, checks stock, and asks for confirmation.
  4. Once confirmed, the order is saved and a summary is shown.
- **Final Output:** The chatbot responds with clear, friendly messages:
  - Product lists grouped by category (if needed)
  - Shopping cart updates and order summaries
  - Order confirmation with details and order ID
- **Demo Flow Example:**
  1. User: “Show me phones under $1000”
     - Bot: Lists matching phones with prices and stock.
  2. User: “Add iPhone 15 Pro to my cart”
     - Bot: Confirms item added to cart.
  3. User: “Apply coupon DEMO”
     - Bot: Applies discount and updates total.
  4. User: “Checkout”
     - Bot: Confirms order, shows summary, and thanks the user.

---

## :stopwatch: 2 Minutes – Question

**Sample Question:**  
*What was the biggest challenge you faced, and how did you solve it?*

**Answer:**  
The biggest challenge was making sure the chatbot could accurately understand when a user wanted to place an order, especially in natural, multi-turn conversations. To solve this, I combined phrase detection with LLM-based intent analysis, so the system can reliably detect order intent even if the user doesn’t use exact keywords. This makes the experience much more natural and user-friendly.

---

## :stopwatch: 1 Minute – Feedback

*(Reserved for evaluator feedback on communication, clarity, and structure.)*

---

## Project Block Flow Diagram

```mermaid
graph TD
    A[User Input (CLI or Streamlit UI)] --> B[Chatbot Orchestrator\n(Session Memory, Cart, Function Call)]
    B --> C1[RAG Agent]
    B --> C2[Cart Manager]
    B --> C3[Order Agent]
    C1 --> D1[Hybrid Search (BM25)]
    C2 --> D2[Shopping Cart (Coupons)]
    C3 --> D3[Database (SQLite)]
    B --> E[Langfuse Tracing (Observability)]
```

---
