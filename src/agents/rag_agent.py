"""RAG Agent for product information retrieval."""

import os
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from src.logger import get_logger
from src.search import get_search_engine, HybridSearch
from src.cache import get_search_cache

logger = get_logger()


class RAGAgent:
    """RAG Agent for retrieving product information from vector store."""
    
    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        embedding_model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG Agent.
        
        Args:
            vector_store_path: Path to vector store directory
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to environment variable)
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not provided")
        
        # Configure client for OpenRouter if base_url is provided
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.last_product: Optional[str] = None
        
        # Initialize vector store connection
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_collection("products")
            logger.info(f"RAG Agent initialized with vector store at {vector_store_path}")
        except Exception as e:
            logger.error(f"Error initializing vector store connection: {str(e)}", exc_info=True)
            raise
        
        # Initialize hybrid search engine
        self.hybrid_search = get_search_engine()
        
        # Initialize cache
        self.search_cache = get_search_cache()
    
    def search_products(
        self,
        query: str,
        k: int = 5,
        max_retries: int = 3,
        sort_by: str = 'relevance'
    ) -> List[Dict]:
        """
        Search for products using hybrid BM25 + semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            max_retries: Maximum retry attempts (unused, kept for compatibility)
            sort_by: Sort order ('relevance', 'price_low', 'price_high')
        
        Returns:
            List of product dictionaries with metadata
        """
        # #region agent log
        import time
        import json
        search_start = time.time()
        try:
            with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"rag_agent.py:90","message":"search_products START","data":{"query":query,"k":k},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + '\n')
        except: pass
        # #endregion
        
        # Create cache key from query parameters
        cache_key = f"search:{query.lower().strip()}:k{k}:sort{sort_by}"
        
        # Check cache first
        cached_result = self.search_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for query: {query}")
            # #region agent log
            try:
                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"rag_agent.py:98","message":"search_products CACHE HIT","data":{"query":query},"sessionId":"debug-session","runId":"run1","hypothesisId":"F"}) + '\n')
            except: pass
            # #endregion
            return cached_result
        
        logger.info(f"Searching products with query: {query}")
        
        # Extract filters before search (for fallback if needed)
        price_filter = self.hybrid_search._extract_price_filter(query)
        categories = self.hybrid_search._extract_categories(query)
        # #region agent log
        try:
            with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_categories_extracted","timestamp":int(time.time()*1000),"location":"rag_agent.py:115","message":"Categories extracted from query","data":{"query":query,"categories":categories},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
        except: pass
        # #endregion
        category_filter = categories[0] if categories else None
        
        # Use hybrid search engine (BM25 + semantic matching)
        products = self.hybrid_search.search(query, k=k, sort_by=sort_by)
        
        # #region agent log
        search_end = time.time()
        search_duration = search_end - search_start
        # #endregion
        
        # Check if results are grouped by category (dict) or flat list
        if isinstance(products, dict):
            # Multi-category results - extract first product from first category
            all_products = []
            for category_products in products.values():
                all_products.extend(category_products)
            if all_products:
                self.last_product = all_products[0].get('name')
                logger.info(f"Found products in {len(products)} categories for query: {query}")
            # Cache the result
            self.search_cache.set(cache_key, products)
            # #region agent log
            try:
                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"rag_agent.py:114","message":"search_products END","data":{"query":query,"duration_ms":search_duration*1000,"result_type":"dict","categories":len(products)},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + '\n')
            except: pass
            # #endregion
            return products
        elif products:
            self.last_product = products[0].get('name')
            logger.info(f"Found {len(products)} products for query: {query}")
            # Cache the result
            self.search_cache.set(cache_key, products)
            # #region agent log
            try:
                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"rag_agent.py:120","message":"search_products END","data":{"query":query,"duration_ms":search_duration*1000,"result_type":"list","count":len(products)},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + '\n')
            except: pass
            # #endregion
        else:
            # Fallback to basic keyword search WITH filters applied
            products = self._keyword_search(query, k, price_filter=price_filter, category_filter=category_filter)
            if products:
                self.last_product = products[0].get('name')
                logger.info(f"Fallback found {len(products)} products for query: {query} (with filters applied)")
                # Cache the result
                self.search_cache.set(cache_key, products)
            else:
                logger.warning(f"No products found for query: {query}")
                # Cache empty result to avoid repeated searches
                self.search_cache.set(cache_key, [], ttl=60)  # Shorter TTL for empty results
            # #region agent log
            try:
                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"id":f"log_{int(time.time()*1000)}","timestamp":int(time.time()*1000),"location":"rag_agent.py:128","message":"search_products END (fallback)","data":{"query":query,"duration_ms":search_duration*1000,"count":len(products) if products else 0},"sessionId":"debug-session","runId":"run1","hypothesisId":"C"}) + '\n')
            except: pass
            # #endregion
        
        return products
    
    def get_recommendations(self, product_name: str, k: int = 3) -> List[Dict]:
        """
        Get product recommendations based on a product.
        
        Args:
            product_name: Name of the reference product
            k: Number of recommendations
        
        Returns:
            List of recommended products
        """
        return self.hybrid_search.get_recommendations(product_name, k=k)

    def _keyword_search(
        self, 
        query: str, 
        k: int = 3,
        price_filter: Optional[Tuple[float, float]] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """Keyword-based search using products.json directly with price and category filters."""
        import json
        from pathlib import Path
        
        try:
            products_file = Path("./data/products.json")
            if not products_file.exists():
                return []
            
            with open(products_file, 'r') as f:
                all_products = json.load(f)
            
            # Enhanced keyword matching with synonyms
            query_lower = query.lower()
            
            # Remove common suffixes for better matching
            def normalize_word(word):
                """Remove common suffixes."""
                if word.endswith('s') and len(word) > 3:
                    return word[:-1]  # laptops -> laptop
                return word
            
            # Common synonyms/related terms
            synonyms = {
                'laptop': ['laptop', 'macbook', 'dell', 'notebook', 'xps', 'portable'],
                'phone': ['phone', 'iphone', 'samsung', 'galaxy', 'smartphone', 'mobile'],
                'headphones': ['headphones', 'headphone', 'earbuds', 'airpods', 'earphones', 'headset', 'wh-1000'],
                'watch': ['watch', 'smartwatch', 'apple watch', 'fitness'],
                'tablet': ['tablet', 'ipad', 'surface'],
                'computer': ['computer', 'laptop', 'macbook', 'desktop', 'pc'],
                'monitor': ['monitor', 'display', 'screen', 'lg'],
                'keyboard': ['keyboard', 'logitech', 'mechanical'],
                'mouse': ['mouse', 'logitech'],
            }
            
            # Expand keywords with synonyms
            keywords = [normalize_word(k) for k in query_lower.split()]
            expanded_keywords = set(keywords)
            for keyword in keywords:
                if keyword in synonyms:
                    expanded_keywords.update(synonyms[keyword])
            
            scored_products = []
            for product in all_products:
                # Apply price filter if present
                if price_filter:
                    price = float(product.get('price', 0))
                    min_price, max_price = price_filter
                    if price < min_price or price > max_price:
                        continue
                
                # Apply category filter if present
                if category_filter:
                    product_category = product.get('category', '').lower()
                    product_name = product.get('name', '').lower()
                    product_desc = product.get('description', '').lower()
                    
                    category_match = False
                    if category_filter == 'phones':
                        phone_keywords = ['iphone', 'samsung', 'galaxy', 'smartphone', 'mobile phone', 'cell phone']
                        exclude_keywords = ['headphone', 'earbud', 'airpod', 'speaker']
                        is_excluded = any(kw in product_name or kw in product_desc for kw in exclude_keywords)
                        if not is_excluded:
                            category_match = any(kw in product_name or kw in product_desc for kw in phone_keywords) or 'phone' in product_name
                    elif category_filter == 'computers':
                        computer_keywords = ['laptop', 'macbook', 'computer', 'pc', 'desktop', 'xps', 'notebook']
                        category_match = any(kw in product_name or kw in product_desc for kw in computer_keywords)
                    elif category_filter == 'audio':
                        audio_keywords = ['headphone', 'earbud', 'speaker', 'airpod', 'audio', 'sound']
                        category_match = any(kw in product_name or kw in product_desc for kw in audio_keywords)
                    elif category_filter == 'gaming':
                        gaming_keywords = ['playstation', 'xbox', 'nintendo', 'console', 'controller', 'switch', 'ps5']
                        category_match = any(kw in product_name or kw in product_desc for kw in gaming_keywords)
                        if any(exclude in product_name or exclude in product_desc for exclude in ['laptop', 'macbook', 'computer', 'xps', 'dell']):
                            category_match = False
                    elif category_filter == 'books':
                        if 'book' in product_category.lower():
                            category_match = True
                        else:
                            book_keywords = ['book', 'novel', 'reading']
                            exclude_keywords = ['macbook', 'notebook', 'laptop']
                            has_book_keyword = any(kw in product_name or kw in product_desc for kw in book_keywords)
                            is_excluded = any(kw in product_name for kw in exclude_keywords)
                            category_match = has_book_keyword and not is_excluded
                    elif category_filter == 'wearables':
                        wearable_keywords = ['watch', 'smartwatch', 'fitness', 'tracker', 'wearable']
                        category_match = any(kw in product_name or kw in product_desc for kw in wearable_keywords)
                    elif category_filter == 'electronics':
                        category_match = 'electronics' in product_category.lower()
                    
                    if not category_match:
                        continue
                
                # Score products by keyword matching
                score = 0
                searchable = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}".lower()
                for keyword in expanded_keywords:
                    if keyword in searchable:
                        score += 1
                        # Bonus for exact name match
                        if keyword in product.get('name', '').lower():
                            score += 2
                
                if score > 0:
                    scored_products.append((score, product))
            
            # Sort by score descending
            scored_products.sort(key=lambda x: x[0], reverse=True)
            
            # Return top k
            return [p[1] for p in scored_products[:k]]
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []

    def _vector_search(
        self,
        query: str,
        k: int = 3,
        max_retries: int = 3
    ) -> List[Dict]:
        """Vector similarity search using embeddings."""
        # Generate query embedding
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=[query]
                )
                query_embedding = response.data[0].embedding
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    import time
                    wait_time = 2 ** retry_count
                    logger.warning(f"Embedding generation failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate query embedding: {str(e)}", exc_info=True)
                    raise
        
        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        products = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for i, metadata in enumerate(results['metadatas'][0]):
                product = {
                    "product_id": metadata.get("product_id"),
                    "name": metadata.get("name"),
                    "description": results['documents'][0][i] if results.get('documents') else "",
                    "price": float(metadata.get("price", 0)),  # Get price from metadata
                    "category": metadata.get("category"),
                    "stock_status": metadata.get("stock_status")
                }
                products.append(product)
        
        if not products:
            logger.warning(f"No products found for query: {query}")
        
        return products
    
    def get_price_from_metadata(self, product_name: str) -> Optional[float]:
        """
        Get product price from vector store metadata (NOT from LLM text).
        
        Args:
            product_name: Name of the product
        
        Returns:
            Price as float or None if not found
        """
        try:
            results = self.collection.get(
                where={"name": product_name},
                limit=1
            )
            
            if results and results['metadatas'] and len(results['metadatas']) > 0:
                price = results['metadatas'][0].get('price')
                if price is not None:
                    return float(price)
            
            logger.warning(f"Price not found in metadata for product: {product_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting price from metadata: {str(e)}", exc_info=True)
            return None
    
    def answer_query(
        self,
        user_query: str,
        chat_history: List[Dict],
        max_retries: int = 3
    ) -> str:
        """
        Answer user query using RAG.
        
        Args:
            user_query: User's question
            chat_history: Previous conversation history
            max_retries: Maximum retry attempts for API calls
        
        Returns:
            Agent's response
        """
        try:
            # Search for relevant products
            products = self.search_products(user_query, k=3)
            
            if not products:
                return "I couldn't find any products matching your query. Could you try rephrasing it?"
            
            # Build context from retrieved products
            context = "\n\n".join([
                f"Product: {p['name']}\n"
                f"Price: ${p['price']:.2f}\n"
                f"Description: {p['description']}\n"
                f"Stock Status: {p['stock_status']}\n"
                f"Category: {p['category']}"
                for p in products
            ])
            
            # Update last product reference
            if products:
                self.last_product = products[0]['name']
            
            # Build system prompt
            system_prompt = """You are a helpful product information assistant for an e-commerce store.
Your role is to answer customer questions about products using the information provided.
Always use the exact prices from the product information (never make up prices).
Be conversational, friendly, and helpful.
If asked about a product, provide the price, description, and stock status.
If multiple products match, list them all."""
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Product Information:\n{context}\n\nUser Question: {user_query}"}
            ]
            
            # Add recent chat history for context
            if chat_history:
                recent_history = chat_history[-3:]  # Last 3 exchanges
                for msg in recent_history:
                    if msg.get("role") in ["user", "assistant"]:
                        messages.insert(-1, {"role": msg["role"], "content": msg["content"]})
            
            # Call OpenAI API with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Use model from environment or default
                    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500
                    )
                    answer = response.choices[0].message.content
                    return answer
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        import time
                        wait_time = 2 ** retry_count
                        logger.warning(f"API call failed, retrying in {wait_time}s... ({retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to get response after {max_retries} attempts: {str(e)}", exc_info=True)
                        return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}", exc_info=True)
            return "I encountered an error while processing your query. Please try again."

