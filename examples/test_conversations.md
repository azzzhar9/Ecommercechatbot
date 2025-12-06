# Test Conversation Scenarios

This document describes test conversation flows with expected outcomes.

## Test 1: Product Price Query

**User:** "What's the price of iPhone 15 Pro?"

**Expected Bot Response:**
- Bot should call `search_products("iPhone 15 Pro")`
- Bot should return price from vector store metadata: $999.99
- Response should be conversational: "The iPhone 15 Pro is priced at $999.99 and is currently in stock."

**Verification:**
- Price comes from metadata, not LLM hallucination
- Stock status included
- Natural, conversational tone

---

## Test 2: Multi-turn Product Discussion

**Conversation:**
1. **User:** "Tell me about laptops"
2. **Bot:** Lists available laptops with prices
3. **User:** "What about the MacBook Pro?"
4. **Bot:** Provides detailed information about MacBook Pro
5. **User:** "What's the price?"
6. **Bot:** "The MacBook Pro 14-inch is priced at $1999.99 and has low stock."

**Expected Behavior:**
- Bot maintains context across multiple turns
- Last product (MacBook Pro) is remembered
- Price retrieved from metadata
- Stock status accurately reported

**Verification:**
- Context maintained in chat_history
- last_product updated correctly
- No need to re-ask for product name

---

## Test 3: Order Confirmation with Extraction

**Conversation:**
1. **User:** "I want to buy the iPhone 15 Pro"
2. **Bot:** Provides product details and asks for quantity
3. **User:** "I'll take 2 of them"
4. **Bot:** Detects order intent
5. **Bot:** Extracts order details (product: iPhone 15 Pro, quantity: 2)
6. **Bot:** Checks stock (in_stock)
7. **Bot:** Requests confirmation: "Confirm order: iPhone 15 Pro x2 = $1999.98? (yes/no)"
8. **User:** "yes"
9. **Bot:** "Your order has been confirmed! Order ID: #ORD-XXXXX. Total: $1999.98. Thank you!"

**Expected Behavior:**
- Order intent detected correctly
- Product, quantity, and price extracted from conversation
- Stock verified before order creation
- Confirmation step included
- Order saved to database with unique order_id
- User receives order confirmation with order ID

**Verification:**
- Order appears in database
- Order ID is unique
- All fields populated correctly
- Total price = quantity × unit_price

---

## Test 4: Ambiguous Query Handling

**Conversation:**
1. **User:** "Tell me about laptops"
2. **Bot:** Lists multiple laptops (MacBook Pro, Dell XPS 15)
3. **User:** "I'll take it"
4. **Bot:** "Which product do you mean? MacBook Pro 14-inch or Dell XPS 15? Please specify."

**Expected Behavior:**
- Bot recognizes ambiguous reference
- Bot asks for clarification
- Bot uses last_product context if available
- Bot provides options from recent conversation

**Verification:**
- Ambiguity detected
- Clarification requested
- Context from chat_history used

---

## Test 5: Invalid Order Rejection

**Scenario A: Out of Stock**
1. **User:** "I want to buy PlayStation 5"
2. **Bot:** Checks stock
3. **Bot:** "Sorry, PlayStation 5 is currently out of stock. Please check back later or consider a similar product."

**Expected Behavior:**
- Stock status checked: out_of_stock
- Order blocked
- User informed clearly
- No order created

**Scenario B: Invalid Quantity**
1. **User:** "I want to order iPhone 15 Pro"
2. **Bot:** "How many?"
3. **User:** "0"
4. **Bot:** Order validation fails (quantity must be > 0)
5. **Bot:** "Quantity must be at least 1. Please specify a valid quantity."

**Expected Behavior:**
- Pydantic validation catches invalid quantity
- User-friendly error message
- No order created

**Scenario C: Negative Price**
1. Order with negative price should fail validation
2. Error message: "Price must be greater than 0"

---

## Test 6: Low Stock Warning

**Conversation:**
1. **User:** "I want to buy MacBook Pro"
2. **Bot:** Checks stock (low_stock)
3. **Bot:** "Warning: MacBook Pro 14-inch has low stock. Would you like to proceed with your order?"
4. **User:** "yes"
5. **Bot:** Proceeds with order confirmation

**Expected Behavior:**
- Low stock detected
- Warning message displayed
- User can still proceed
- Order created if confirmed

---

## Test 7: Price Accuracy Verification

**Test:** Verify that prices always come from vector store metadata, never from LLM.

**Method:**
1. Query product price
2. Check that price matches exactly with products.json
3. Verify no hallucinated prices
4. Test with multiple products

**Expected:**
- All prices match metadata exactly
- No rounding errors beyond 2 decimal places
- No made-up prices

---

## Test 8: Function Calling - Search Products

**User:** "Show me electronics under $500"

**Expected:**
- Function `search_products` called with query
- Results filtered by category and price
- Products listed with prices from metadata

---

## Test 9: Function Calling - Create Order

**User:** "I'll take 2 iPhone 15 Pro"

**Expected:**
- Order intent detected
- Function `create_order` called
- Stock verified
- Confirmation requested
- Order created after confirmation

---

## Test 10: Error Handling - Network Failure

**Scenario:** Simulate API failure

**Expected:**
- Retry logic activated
- User-friendly error message
- No crash
- Logs error with session_id

---

## Test 11: Error Handling - Database Locked

**Scenario:** Simulate database lock

**Expected:**
- Retry with exponential backoff
- User-friendly message after max retries
- Error logged
- No data corruption

---

## Test 12: Input Sanitization

**User Input:** "Product'; DROP TABLE orders; --"

**Expected:**
- Dangerous characters removed
- SQL injection prevented
- Safe input processed
- No database damage

---

## Test 13: Email Validation

**Test Cases:**
- Valid: "user@example.com" → Accepted
- Invalid: "notanemail" → Rejected
- Invalid: "@example.com" → Rejected
- Invalid: "user@" → Rejected

---

## Test 14: Exit Summary

**Conversation:**
1. User places 3 orders
2. User types "quit"
3. Bot: "You placed 3 orders. Thank you!"

**Expected:**
- Order count tracked correctly
- Summary displayed on exit
- Session ends gracefully

---

## Test 15: Multi-Product Search

**User:** "Show me all phones"

**Expected:**
- Multiple products returned
- All prices from metadata
- Stock status for each
- Natural formatting

