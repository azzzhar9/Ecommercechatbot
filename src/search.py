"""Advanced hybrid search with BM25 + semantic matching."""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from src.logger import get_logger

logger = get_logger()


class HybridSearch:
    """
    Hybrid search engine combining BM25 keyword matching with semantic similarity.
    
    Features:
    - BM25 algorithm for term frequency scoring
    - Synonym expansion for better recall
    - Price and category filtering
    - Attribute extraction (size, color, capacity)
    - Product ranking by relevance, price, popularity
    """
    
    # BM25 parameters
    K1 = 1.5  # Term frequency saturation
    B = 0.75  # Length normalization
    
    # Synonyms for query expansion
    SYNONYMS = {
        'laptop': ['laptop', 'macbook', 'macbooks', 'dell', 'notebook', 'xps', 'portable', 'ultrabook'],
        'phone': ['phone', 'phones', 'iphone', 'samsung', 'galaxy', 'smartphone', 'mobile', 'cellphone'],
        'headphones': ['headphones', 'headphone', 'earbuds', 'airpods', 'earphones', 'headset', 'wh-1000', 'audio'],
        'watch': ['watch', 'smartwatch', 'apple watch', 'fitness', 'wearable'],
        'tablet': ['tablet', 'ipad', 'surface', 'tab'],
        'computer': ['computer', 'laptop', 'macbook', 'desktop', 'pc', 'workstation'],
        'monitor': ['monitor', 'display', 'screen', 'lg', 'ultragear'],
        'keyboard': ['keyboard', 'logitech', 'mechanical', 'wireless keyboard'],
        'mouse': ['mouse', 'logitech', 'gaming mouse', 'wireless mouse'],
        'camera': ['camera', 'canon', 'sony', 'dslr', 'mirrorless', 'photography'],
        'speaker': ['speaker', 'bluetooth speaker', 'sonos', 'bose', 'audio'],
        'tv': ['tv', 'television', 'smart tv', 'oled', '4k', 'display'],
        'gaming': ['gaming', 'ps5', 'playstation', 'xbox', 'nintendo', 'console', 'controller', 'accessories', 'accessory'],
        'cheap': ['cheap', 'budget', 'affordable', 'inexpensive', 'low cost'],
        'expensive': ['expensive', 'premium', 'high-end', 'flagship', 'pro', 'ultra'],
    }
    
    # Price modifiers
    PRICE_KEYWORDS = {
        'cheap': (0, 200),
        'budget': (0, 300),
        'affordable': (0, 500),
        'mid-range': (300, 800),
        'expensive': (800, float('inf')),
        'premium': (1000, float('inf')),
        'flagship': (1000, float('inf')),
    }
    
    def __init__(self, products_path: str = "./data/products.json"):
        """Initialize the hybrid search engine."""
        self.products_path = products_path
        self.products: List[Dict] = []
        self.doc_count = 0
        self.avg_doc_length = 0
        self.idf_cache: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.term_frequencies: List[Dict[str, int]] = []
        
        self._load_products()
        self._build_index()
    
    def _load_products(self):
        """Load products from JSON file."""
        try:
            products_file = Path(self.products_path)
            if products_file.exists():
                with open(products_file, 'r', encoding='utf-8') as f:
                    self.products = json.load(f)
                logger.info(f"Loaded {len(self.products)} products for hybrid search")
            else:
                logger.warning(f"Products file not found: {self.products_path}")
                self.products = []
        except Exception as e:
            logger.error(f"Error loading products: {e}")
            self.products = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into normalized terms."""
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Stem simple suffixes
        stemmed = []
        for token in tokens:
            if token.endswith('s') and len(token) > 3:
                token = token[:-1]
            if token.endswith('ing') and len(token) > 5:
                token = token[:-3]
            stemmed.append(token)
        
        return stemmed
    
    def _build_index(self):
        """Build BM25 index for all products."""
        if not self.products:
            return
        
        self.doc_count = len(self.products)
        total_length = 0
        
        # Calculate term frequencies and document lengths
        for product in self.products:
            doc_text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}"
            tokens = self._tokenize(doc_text)
            
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            
            tf = Counter(tokens)
            self.term_frequencies.append(tf)
        
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
        
        # Pre-compute IDF for all terms
        all_terms = set()
        for tf in self.term_frequencies:
            all_terms.update(tf.keys())
        
        for term in all_terms:
            doc_freq = sum(1 for tf in self.term_frequencies if term in tf)
            self.idf_cache[term] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        tokens = self._tokenize(query)
        expanded = set(tokens)
        
        for token in tokens:
            if token in self.SYNONYMS:
                expanded.update(self.SYNONYMS[token])
        
        return list(expanded)
    
    def _extract_price_filter(self, query: str) -> Optional[Tuple[float, float]]:
        """Extract price range from query."""
        query_lower = query.lower()
        
        # Check for explicit price mentions
        price_match = re.search(r'under\s*\$?\s*(\d+(?:\.\d+)?)', query_lower)
        if price_match:
            max_price = float(price_match.group(1))
            return (0.0, max_price)
        
        price_match = re.search(r'below\s*\$?(\d+)', query_lower)
        if price_match:
            return (0, float(price_match.group(1)))
        
        price_match = re.search(r'above\s*\$?(\d+)', query_lower)
        if price_match:
            return (float(price_match.group(1)), float('inf'))
        
        price_match = re.search(r'over\s*\$?(\d+)', query_lower)
        if price_match:
            return (float(price_match.group(1)), float('inf'))
        
        # Check for price keywords
        for keyword, price_range in self.PRICE_KEYWORDS.items():
            if keyword in query_lower:
                return price_range
        
        return None
    
    def _extract_categories(self, query: str) -> List[str]:
        """Extract multiple categories from query with support for comma/and-separated lists. STRICT for price queries."""
        # #region agent log
        try:
            import json
            with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_extract_cat_start","timestamp":int(__import__('time').time()*1000),"location":"search.py:177","message":"_extract_categories START","data":{"query":query},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
        except: pass
        # #endregion
        query_lower = query.lower()
        categories = {
            'phones': ['phone', 'phones', 'mobile', 'mobiles', 'smartphone', 'cellphone', 'iphone', 'iphones', 'samsung', 'galaxy'],
            'computers': ['computer', 'laptop', 'laptops', 'pc', 'desktop', 'notebook', 'macbook', 'macbooks'],
            'audio': ['audio', 'sound', 'music', 'headphone', 'speaker', 'earbuds', 'earbud'],
            'gaming': ['gaming', 'game', 'console', 'playstation', 'xbox', 'nintendo', 'accessories', 'accessory'],
            'wearables': ['wearable', 'watch', 'fitness', 'tracker'],
            'books': ['book', 'books', 'novel', 'novels', 'reading'],
            'electronics': ['electronics', 'electronic', 'tech', 'gadget'],
            'home_garden': ['garden', 'gardening', 'home', 'home & garden', 'home and garden', 'household', 'appliance', 'vacuum', 'kitchen', 'home garden'],
            'clothing': ['clothing', 'clothes', 'apparel', 'wear', 'fashion', 'shoes', 'sneakers', 'jeans', 'jacket'],
            'sports': ['sports', 'sport', 'fitness', 'exercise', 'workout', 'gym', 'yoga', 'running'],
        }
        # Remove common words and clean up query
        query_clean = query_lower.strip()
        for remove_word in ['show me', 'i want', 'show', 'give me', 'find', 'search for', 'looking for', 'stock']:
            query_clean = query_clean.replace(remove_word, '').strip()
        # Remove extra whitespace
        query_clean = ' '.join(query_clean.split())
        segments = []
        has_comma = ',' in query_clean
        has_and = ' and ' in query_clean
        if has_comma or has_and:
            comma_parts = [part.strip() for part in query_clean.split(',')]
            for part in comma_parts:
                and_parts = [p.strip() for p in part.split(' and ')]
                segments.extend(and_parts)
        else:
            segments = [query_clean]
        segments = [s for s in segments if s and len(s) > 0]
        found_categories = []
        def is_explicit_book_segment(segment):
            book_keywords = ['book', 'books', 'novel', 'novels', 'reading']
            return any(bk in segment for bk in book_keywords)
        # Check if price filter is present (for fallback logic)
        price_filter_present = bool(re.search(r'(under|below|over|above|less than|more than|cheaper than|costing less than|costing more than|\$\d+)', query_clean))
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            segment_normalized = segment.rstrip('s') if segment.endswith('s') and len(segment) > 3 else segment
            words = segment.split()
            segment_matched = False
            
            # First, try exact segment match (e.g., "home" should match "home_garden")
            for category, keywords in categories.items():
                if category == 'books' and not is_explicit_book_segment(segment):
                    continue
                # Check if segment exactly matches any keyword
                if segment in keywords or segment_normalized in keywords:
                    if category not in found_categories:
                        found_categories.append(category)
                    segment_matched = True
                    break
                # Check if any keyword exactly matches the segment
                for keyword in keywords:
                    if segment == keyword or segment_normalized == keyword:
                        if category not in found_categories:
                            found_categories.append(category)
                        segment_matched = True
                        break
                # Explicit handling for common single-word category matches
                if segment == 'home' and category == 'home_garden':
                    if category not in found_categories:
                        found_categories.append(category)
                    segment_matched = True
                    break
                if segment == 'sport' and category == 'sports':
                    if category not in found_categories:
                        found_categories.append(category)
                    segment_matched = True
                    break
                if segment in ['cloth', 'clothes', 'clothing'] and category == 'clothing':
                    if category not in found_categories:
                        found_categories.append(category)
                    segment_matched = True
                    break
                if segment_matched:
                    break
            
            # If segment didn't match, try word-by-word matching
            if not segment_matched:
                # #region agent log
                try:
                    import json
                    with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"id":f"log_segment_not_matched","timestamp":int(__import__('time').time()*1000),"location":"search.py:242","message":"Segment not matched, checking words","data":{"segment":segment,"words":words},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + '\n')
                except: pass
                # #endregion
                for word in words:
                    word = word.strip().lower()  # Ensure lowercase for consistent matching
                    if not word:
                        continue
                    word_normalized = word.rstrip('s') if word.endswith('s') and len(word) > 3 else word
                    for category, keywords in categories.items():
                        if category == 'books' and not is_explicit_book_segment(word):
                            continue
                        # Exact match (case-insensitive)
                        if word in keywords or word_normalized in keywords:
                            # #region agent log
                            try:
                                import json
                                with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                                    f.write(json.dumps({"id":f"log_word_matched","timestamp":int(__import__('time').time()*1000),"location":"search.py:252","message":"Word matched category","data":{"word":word,"category":category},"sessionId":"debug-session","runId":"run1","hypothesisId":"B"}) + '\n')
                            except: pass
                            # #endregion
                            if category not in found_categories:
                                found_categories.append(category)
                            break
                        # Check if any keyword contains the word (handles "accessories" matching "gaming accessories")
                        # Also check if word matches keyword (handles "garden" matching "garden" in home_garden keywords)
                        for keyword in keywords:
                            keyword_lower = keyword.lower()
                            if word == keyword_lower or word_normalized == keyword_lower:
                                if category == 'books' and not is_explicit_book_segment(word):
                                    continue
                                if category not in found_categories:
                                    found_categories.append(category)
                                break
                            # Use word boundary matching to prevent category leakage (e.g., "book" matching "notebook")
                            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                            if re.search(pattern, word) or re.search(pattern, word_normalized):
                                if category == 'books' and not is_explicit_book_segment(word):
                                    continue
                                if category not in found_categories:
                                    found_categories.append(category)
                                break
                        # Prefix match for longer words
                        if len(word) >= 4:
                            for keyword in keywords:
                                keyword_lower = keyword.lower()
                                if keyword_lower.startswith(word) or word.startswith(keyword_lower):
                                    if category == 'books' and not is_explicit_book_segment(word):
                                        continue
                                    if category not in found_categories:
                                        found_categories.append(category)
                                    break
        # #region agent log
        try:
            import json
            with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_extract_cat_segments","timestamp":int(__import__('time').time()*1000),"location":"search.py:280","message":"_extract_categories segments","data":{"segments":segments,"found_categories":found_categories},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
        except: pass
        # #endregion
        # Fallback: if no categories found in segments, check whole query
        # For price queries, still try to find categories but be more careful
        if not found_categories:
            for category, keywords in categories.items():
                if category == 'books':
                    # Only add books if explicit book keywords found
                    if any(bk in query_lower for bk in ['book', 'books', 'novel', 'novels', 'reading']):
                        if 'books' not in found_categories:
                            found_categories.append('books')
                    continue
                for keyword in keywords:
                    # Check if keyword appears as a standalone word (not part of another word)
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, query_lower):
                        if category not in found_categories:
                            found_categories.append(category)
                        break
                    # Also check substring match for compound words like "gaming accessories"
                    if keyword in query_lower:
                        if category not in found_categories:
                            found_categories.append(category)
                        break
        
        # #region agent log
        try:
            import json
            with open(r'e:\AIFinalProject\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id":f"log_extract_cat_end","timestamp":int(__import__('time').time()*1000),"location":"search.py:324","message":"_extract_categories END","data":{"found_categories":found_categories},"sessionId":"debug-session","runId":"run1","hypothesisId":"A"}) + '\n')
        except: pass
        # #endregion
        return found_categories if found_categories else []
    
    def _extract_category(self, query: str) -> Optional[str]:
        """Extract single category (backward compatibility)."""
        categories = self._extract_categories(query)
        return categories[0] if categories else None
    
    def decompose_query(self, query: str) -> Dict:
        """
        Decompose query into structured intent without performing search.
        
        Returns structured intent dict with:
        - products: List of product names/terms mentioned
        - categories: List of detected categories
        - filters: Dict with price, stock, and other filters
        
        Args:
            query: User search query
            
        Returns:
            Dict with structured intent information
        """
        # Extract categories
        categories = self._extract_categories(query)
        
        # Extract price filter
        price_filter = self._extract_price_filter(query)
        
        # Extract product terms (simplified - just tokenize and remove stop words)
        query_terms = self._tokenize(query)
        stop_words = {'show', 'me', 'i', 'want', 'find', 'search', 'for', 'looking', 'stock', 'the', 'a', 'an', 'and', 'or', 'but'}
        product_terms = [term for term in query_terms if term not in stop_words and len(term) > 2]
        
        # Build structured intent
        intent = {
            "products": product_terms,
            "categories": categories,
            "filters": {
                "price": price_filter,
                "stock": None  # Can be extended later
            },
            "original_query": query
        }
        
        return intent
    
    def _bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        if doc_idx >= len(self.term_frequencies):
            return 0.0
        
        tf = self.term_frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            
            term_freq = tf[term]
            idf = self.idf_cache.get(term, 0)
            
            # BM25 formula
            numerator = term_freq * (self.K1 + 1)
            denominator = term_freq + self.K1 * (1 - self.B + self.B * doc_length / self.avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _name_match_bonus(self, query_terms: List[str], product: Dict) -> float:
        """Give bonus points for matches in product name."""
        name_tokens = set(self._tokenize(product.get('name', '')))
        matches = len(name_tokens.intersection(set(query_terms)))
        return matches * 2.0  # 2x bonus for name matches
    
    def _stock_bonus(self, product: Dict) -> float:
        """Give bonus for in-stock products."""
        stock = product.get('stock_status', 'out_of_stock')
        if stock == 'in_stock':
            return 0.5
        elif stock == 'low_stock':
            return 0.2
        return 0
    
    def _matches_category(self, product: Dict, category_filter: str, query: str = "") -> bool:
        """
        Check if a product matches a given category filter.
        
        Args:
            product: Product dictionary
            category_filter: Category to check against
            query: Original query (for context-dependent matching)
            
        Returns:
            True if product matches category
        """
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
        elif category_filter == 'electronics':
            category_match = 'electronics' in product_category
        elif category_filter == 'audio':
            audio_keywords = ['headphone', 'earbud', 'speaker', 'airpod', 'audio', 'sound']
            category_match = any(kw in product_name or kw in product_desc for kw in audio_keywords)
        elif category_filter == 'computers':
            computer_keywords = ['laptop', 'macbook', 'computer', 'pc', 'desktop', 'xps', 'notebook']
            category_match = any(kw in product_name or kw in product_desc for kw in computer_keywords)
        elif category_filter == 'gaming':
            gaming_keywords = ['playstation', 'xbox', 'nintendo', 'console', 'controller', 'switch', 'ps5', 'gaming']
            category_match = any(kw in product_name or kw in product_desc for kw in gaming_keywords)
            if 'accessories' in query.lower() or 'accessory' in query.lower():
                if 'accessories' in product_name.lower() or 'accessory' in product_name.lower():
                    if any(gk in product_name or gk in product_desc for gk in ['console', 'controller', 'playstation', 'xbox', 'nintendo']):
                        category_match = True
            if any(exclude in product_name or exclude in product_desc for exclude in ['laptop', 'macbook', 'computer', 'xps', 'dell']):
                category_match = False
        elif category_filter == 'wearables':
            wearable_keywords = ['watch', 'smartwatch', 'fitness', 'tracker', 'wearable']
            category_match = any(kw in product_name or kw in product_desc for kw in wearable_keywords)
        elif category_filter == 'books':
            if 'book' in product_category.lower():
                category_match = True
            else:
                book_keywords = ['book', 'novel', 'reading']
                exclude_keywords = ['macbook', 'notebook', 'laptop']
                has_book_keyword = any(kw in product_name or kw in product_desc for kw in book_keywords)
                is_excluded = any(kw in product_name for kw in exclude_keywords)
                category_match = has_book_keyword and not is_excluded
        elif category_filter == 'home_garden':
            category_match = (
                ('home' in product_category and 'garden' in product_category) or 
                product_category == 'home & garden' or
                product_category == 'home and garden' or
                product_category == 'home garden'
            )
            if not category_match:
                home_keywords = ['vacuum', 'appliance', 'coffee', 'kitchen', 'roomba', 'dyson', 'instant pot', 'nespresso', 'philips hue']
                category_match = any(kw in product_name or kw in product_desc for kw in home_keywords)
        elif category_filter == 'clothing':
            category_match = 'clothing' in product_category.lower()
            if not category_match:
                clothing_keywords = ['shoes', 'sneakers', 'jeans', 'jacket', 'sweater', 'nike', 'adidas', 'levi', 'patagonia', 'north face']
                category_match = any(kw in product_name.lower() or kw in product_desc.lower() for kw in clothing_keywords)
        elif category_filter == 'sports':
            category_match = 'sport' in product_category.lower()
            if not category_match:
                sports_keywords = ['yoga', 'mat', 'fitness', 'gym', 'running', 'bike', 'peloton', 'water bottle', 'dumbbell']
                category_match = any(kw in product_name.lower() or kw in product_desc.lower() for kw in sports_keywords)
        
        return category_match
    
    def search(
        self,
        query: str,
        k: int = 10,
        sort_by: str = 'relevance',  # 'relevance', 'price_low', 'price_high'
        in_stock_only: bool = False
    ):
        """
        Search products using unified BM25 search (single pass for all categories).
        
        Args:
            query: Search query
            k: Number of results to return per category (for multi-category) or total (for single)
            sort_by: Sort order ('relevance', 'price_low', 'price_high')
            in_stock_only: Filter to only in-stock products
        
        Returns:
            List[Dict] for single category, Dict[str, List[Dict]] for multiple categories
        """
        if not self.products:
            return []
        
        # Expand query with synonyms
        query_terms = self._expand_query(query)
        
        # Extract filters
        price_filter = self._extract_price_filter(query)
        categories = self._extract_categories(query)
        
        # Unified retrieval: Single pass through all products
        # Score all products once, then group by category if needed
        scored_products = []
        
        for idx, product in enumerate(self.products):
            # Apply stock filter
            if in_stock_only and product.get('stock_status') == 'out_of_stock':
                continue
            
            # Apply price filter if present
            if price_filter:
                price = float(product.get('price', 0))
                min_price, max_price = price_filter
                if price < min_price or price > max_price:
                    continue
            
            # Category matching: For multi-category, check if product matches ANY category
            # For single category, check if it matches that category
            # For no category, include all products
            category_match = True
            matched_categories = []
            
            if len(categories) > 1:
                # Multi-category: Check if product matches any of the requested categories
                for category in categories:
                    if self._matches_category(product, category, query):
                        category_match = True
                        matched_categories.append(category)
                        break
                else:
                    category_match = False
            elif len(categories) == 1:
                # Single category: Check if product matches the category
                category_match = self._matches_category(product, categories[0], query)
                if category_match:
                    matched_categories = [categories[0]]
            # else: no category filter, include all products (category_match = True)
            
            if not category_match:
                continue
            
            # Calculate relevance scores
            bm25_score = self._bm25_score(query_terms, idx)
            name_bonus = self._name_match_bonus(query_terms, product)
            stock_bonus = self._stock_bonus(product)
            total_score = bm25_score + name_bonus + stock_bonus
            
            if total_score > 0:
                scored_products.append({
                    'product': product,
                    'score': total_score,
                    'bm25_score': bm25_score,
                    'name_bonus': name_bonus,
                    'matched_categories': matched_categories if matched_categories else (categories if categories else [])
                })
        
        # Sort results
        if sort_by == 'price_low':
            scored_products.sort(key=lambda x: x['product'].get('price', float('inf')))
        elif sort_by == 'price_high':
            scored_products.sort(key=lambda x: x['product'].get('price', 0), reverse=True)
        else:
            scored_products.sort(key=lambda x: x['score'], reverse=True)
        
        # Group by category for multi-category queries
        if len(categories) > 1:
            grouped_results = {}
            # Initialize empty lists for each requested category
            for cat in categories:
                grouped_results[cat] = []
            
            for item in scored_products:
                product = item['product']
                # Only use matched_categories - products must match at least one requested category
                product_categories = item.get('matched_categories', [])
                
                # Filter: only include categories that are in the requested categories list
                product_categories = [cat for cat in product_categories if cat in categories]
                
                # If no match found, skip this product (strict filtering)
                if not product_categories:
                    continue
                
                # Add product to each matching category group
                product_id = product.get('product_id') or product.get('name', '')
                for cat in product_categories:
                    # Deduplicate: check if product already in this category
                    if not any((p.get('product_id') or p.get('name', '')) == product_id for p in grouped_results[cat]):
                        grouped_results[cat].append(product)
                        # Limit results per category
                        if len(grouped_results[cat]) >= k:
                            break
            
            # Limit results per category
            for cat in grouped_results:
                grouped_results[cat] = grouped_results[cat][:k]
            
            # Return dict only if we have results in multiple categories
            if len(grouped_results) > 1:
                return grouped_results
            elif len(grouped_results) == 1:
                return list(grouped_results.values())[0]
            else:
                return []
        
        # Single category or no category - return flat list
        return [item['product'] for item in scored_products[:k]]
    
    def _search_by_category(
        self,
        query: str,
        category_filter: str,
        price_filter: Optional[Tuple[float, float]],
        query_terms: List[str],
        k: int,
        sort_by: str,
        in_stock_only: bool
    ) -> List[Dict]:
        """Search products for a specific category."""
        scored_products = []
        
        for idx, product in enumerate(self.products):
            # Apply filters
            if in_stock_only and product.get('stock_status') == 'out_of_stock':
                continue
            
            # Category filter
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
            elif category_filter == 'electronics':
                category_match = 'electronics' in product_category
            elif category_filter == 'audio':
                audio_keywords = ['headphone', 'earbud', 'speaker', 'airpod', 'audio', 'sound']
                category_match = any(kw in product_name or kw in product_desc for kw in audio_keywords)
            elif category_filter == 'computers':
                computer_keywords = ['laptop', 'macbook', 'computer', 'pc', 'desktop', 'xps', 'notebook']
                category_match = any(kw in product_name or kw in product_desc for kw in computer_keywords)
            elif category_filter == 'gaming':
                gaming_keywords = ['playstation', 'xbox', 'nintendo', 'console', 'controller', 'switch', 'ps5', 'gaming']
                # Also check for "accessories" if query mentions gaming
                category_match = any(kw in product_name or kw in product_desc for kw in gaming_keywords)
                # Check for gaming accessories (accessories keyword with gaming context)
                if 'accessories' in query.lower() or 'accessory' in query.lower():
                    if 'accessories' in product_name.lower() or 'accessory' in product_name.lower():
                        # Only match if it's actually gaming-related (console, controller, etc.)
                        if any(gk in product_name or gk in product_desc for gk in ['console', 'controller', 'playstation', 'xbox', 'nintendo']):
                            category_match = True
                # Exclude laptops/computers that just mention gaming
                if any(exclude in product_name or exclude in product_desc for exclude in ['laptop', 'macbook', 'computer', 'xps', 'dell']):
                    category_match = False
            elif category_filter == 'wearables':
                wearable_keywords = ['watch', 'smartwatch', 'fitness', 'tracker', 'wearable']
                category_match = any(kw in product_name or kw in product_desc for kw in wearable_keywords)
            elif category_filter == 'books':
                # Check category first
                if 'book' in product_category.lower():
                    category_match = True
                else:
                    # Check for book-related keywords but exclude "MacBook", "notebook" (computer)
                    book_keywords = ['book', 'novel', 'reading']
                    exclude_keywords = ['macbook', 'notebook', 'laptop']  # Exclude computer-related
                    has_book_keyword = any(kw in product_name or kw in product_desc for kw in book_keywords)
                    is_excluded = any(kw in product_name for kw in exclude_keywords)
                    category_match = has_book_keyword and not is_excluded
            elif category_filter == 'home_garden':
                # Match products with "Home & Garden" category (exact match or contains both words)
                # Handle variations: "home & garden", "home and garden", "home garden"
                category_match = (
                    ('home' in product_category and 'garden' in product_category) or 
                    product_category == 'home & garden' or
                    product_category == 'home and garden' or
                    product_category == 'home garden'
                )
                # Also match common home/garden keywords
                if not category_match:
                    home_keywords = ['vacuum', 'appliance', 'coffee', 'kitchen', 'roomba', 'dyson', 'instant pot', 'nespresso', 'philips hue']
                    category_match = any(kw in product_name or kw in product_desc for kw in home_keywords)
            elif category_filter == 'clothing':
                # Match products with "Clothing" category (case-insensitive)
                category_match = 'clothing' in product_category.lower()
                # Also match common clothing keywords
                if not category_match:
                    clothing_keywords = ['shoes', 'sneakers', 'jeans', 'jacket', 'sweater', 'nike', 'adidas', 'levi', 'patagonia', 'north face']
                    category_match = any(kw in product_name.lower() or kw in product_desc.lower() for kw in clothing_keywords)
            elif category_filter == 'sports':
                # Match products with "Sports" category (case-insensitive)
                category_match = 'sport' in product_category.lower()
                # Also match common sports keywords (exclude 'garmin', 'fitbit' to avoid wearables being shown as separate)
                if not category_match:
                    sports_keywords = ['yoga', 'mat', 'fitness', 'gym', 'running', 'bike', 'peloton', 'water bottle', 'dumbbell']
                    category_match = any(kw in product_name.lower() or kw in product_desc.lower() for kw in sports_keywords)
            
            if not category_match:
                continue
            
            # Price filter - apply STRICTLY (exclude products outside range)
            if price_filter:
                price = float(product.get('price', 0))
                min_price, max_price = price_filter
                # Exclude if price is below minimum OR above maximum
                if price < min_price or price > max_price:
                    continue
            
            # Calculate scores
            bm25_score = self._bm25_score(query_terms, idx)
            name_bonus = self._name_match_bonus(query_terms, product)
            stock_bonus = self._stock_bonus(product)
            
            total_score = bm25_score + name_bonus + stock_bonus
            
            if total_score > 0:
                scored_products.append({
                    'product': product,
                    'score': total_score,
                    'bm25_score': bm25_score,
                    'name_bonus': name_bonus
                })
        
        # Sort results
        if sort_by == 'price_low':
            scored_products.sort(key=lambda x: x['product'].get('price', float('inf')))
        elif sort_by == 'price_high':
            scored_products.sort(key=lambda x: x['product'].get('price', 0), reverse=True)
        else:
            scored_products.sort(key=lambda x: x['score'], reverse=True)
        
        return [item['product'] for item in scored_products[:k]]
    
    def get_recommendations(
        self,
        product_name: str,
        k: int = 3,
        exclude_same: bool = True
    ) -> List[Dict]:
        """
        Get product recommendations based on a product.
        
        Args:
            product_name: Name of the reference product
            k: Number of recommendations
            exclude_same: Exclude the reference product
        
        Returns:
            List of recommended products
        """
        # Find the reference product
        ref_product = None
        for product in self.products:
            if product_name.lower() in product.get('name', '').lower():
                ref_product = product
                break
        
        if not ref_product:
            return []
        
        ref_category = ref_product.get('category', '')
        ref_price = ref_product.get('price', 0)
        ref_name = ref_product.get('name', '')
        
        # Score products by similarity
        scored = []
        for product in self.products:
            if exclude_same and product.get('name') == ref_name:
                continue
            
            score = 0
            
            # Same category bonus
            if product.get('category') == ref_category:
                score += 3
            
            # Similar price range (within 30%)
            price = product.get('price', 0)
            if ref_price > 0 and abs(price - ref_price) / ref_price < 0.3:
                score += 2
            
            # In stock bonus
            if product.get('stock_status') == 'in_stock':
                score += 1
            
            if score > 0:
                scored.append((score, product))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:k]]
    
    def get_products_by_category(self, category: str, k: int = 10) -> List[Dict]:
        """Get products by category."""
        matching = []
        for product in self.products:
            if category.lower() in product.get('category', '').lower():
                matching.append(product)
        return matching[:k]
    
    def get_price_range(self, min_price: float, max_price: float) -> List[Dict]:
        """Get products within a price range."""
        return [
            p for p in self.products
            if min_price <= p.get('price', 0) <= max_price
        ]
    
    def merge_results(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        bm25_weight: float = 0.6,
        vector_weight: float = 0.4,
        k: int = 10
    ) -> List[Dict]:
        """
        Merge BM25 and vector search results with score fusion.
        
        Args:
            bm25_results: List of products from BM25 search (with scores)
            vector_results: List of products from vector search
            bm25_weight: Weight for BM25 scores (default 0.6)
            vector_weight: Weight for vector scores (default 0.4)
            k: Number of final results to return
            
        Returns:
            Merged and deduplicated list of products sorted by combined score
        """
        # Create a map of product_id -> merged result
        merged = {}
        
        # Normalize BM25 scores (0-1 range)
        bm25_scores = {}
        if bm25_results:
            max_bm25 = max(item.get('score', 0) for item in bm25_results if isinstance(item, dict) and 'score' in item)
            if max_bm25 > 0:
                for item in bm25_results:
                    if isinstance(item, dict):
                        product = item.get('product', item)  # Handle both formats
                        product_id = product.get('product_id') or product.get('name', '')
                        score = item.get('score', 0) / max_bm25
                        bm25_scores[product_id] = score
                        if product_id not in merged:
                            merged[product_id] = product.copy()
                            merged[product_id]['_bm25_score'] = score
                            merged[product_id]['_vector_score'] = 0.0
                    else:
                        # Direct product dict
                        product_id = item.get('product_id') or item.get('name', '')
                        if product_id not in merged:
                            merged[product_id] = item.copy()
                            merged[product_id]['_bm25_score'] = 0.5  # Default score
                            merged[product_id]['_vector_score'] = 0.0
        
        # Normalize vector scores (assume they come with similarity scores 0-1)
        vector_scores = {}
        if vector_results:
            # Vector results should have similarity scores, but if not, assign based on position
            for idx, product in enumerate(vector_results):
                product_id = product.get('product_id') or product.get('name', '')
                # If vector results have scores, use them; otherwise use position-based score
                vector_score = product.get('similarity', product.get('score', 1.0 - (idx / len(vector_results))))
                vector_scores[product_id] = vector_score
                if product_id not in merged:
                    merged[product_id] = product.copy()
                    merged[product_id]['_bm25_score'] = 0.0
                    merged[product_id]['_vector_score'] = vector_score
                else:
                    merged[product_id]['_vector_score'] = vector_score
        
        # Calculate combined scores and sort
        scored_merged = []
        for product_id, product in merged.items():
            bm25_score = product.get('_bm25_score', 0.0)
            vector_score = product.get('_vector_score', 0.0)
            combined_score = (bm25_score * bm25_weight) + (vector_score * vector_weight)
            
            # Remove internal score fields
            clean_product = {k: v for k, v in product.items() if not k.startswith('_')}
            scored_merged.append({
                'product': clean_product,
                'combined_score': combined_score,
                'bm25_score': bm25_score,
                'vector_score': vector_score
            })
        
        # Sort by combined score descending
        scored_merged.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top k products
        return [item['product'] for item in scored_merged[:k]]


# Singleton instance
_search_engine: Optional[HybridSearch] = None


def get_search_engine() -> HybridSearch:
    """Get or create the hybrid search engine singleton."""
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearch()
    return _search_engine
