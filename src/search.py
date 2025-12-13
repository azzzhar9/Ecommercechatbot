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
        query_lower = query.lower()
        categories = {
            'phones': ['phone', 'phones', 'mobile', 'mobiles', 'smartphone', 'cellphone', 'iphone', 'iphones', 'samsung', 'galaxy'],
            'computers': ['computer', 'laptop', 'laptops', 'pc', 'desktop', 'notebook', 'macbook', 'macbooks'],
            'audio': ['audio', 'sound', 'music', 'headphone', 'speaker', 'earbuds', 'earbud'],
            'gaming': ['gaming', 'game', 'console', 'playstation', 'xbox', 'nintendo', 'accessories', 'accessory'],
            'wearables': ['wearable', 'watch', 'fitness', 'tracker'],
            'books': ['book', 'books', 'novel', 'novels', 'reading'],
            'electronics': ['electronics', 'electronic', 'tech', 'gadget'],
        }
        # Remove common words
        query_clean = query_lower.strip()
        for remove_word in ['show me', 'i want', 'show', 'give me', 'find', 'search for', 'looking for']:
            query_clean = query_clean.replace(remove_word, '').strip()
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
            for category, keywords in categories.items():
                if category == 'books' and not is_explicit_book_segment(segment):
                    continue
                for keyword in keywords:
                    if segment == keyword or segment_normalized == keyword:
                        if category not in found_categories:
                            found_categories.append(category)
                        segment_matched = True
                        break
                    if len(segment) >= 4 and keyword.startswith(segment):
                        if category not in found_categories:
                            found_categories.append(category)
                        segment_matched = True
                        break
                    if segment == keyword[:len(segment)] and len(segment) >= 3:
                        if category == 'books' and not is_explicit_book_segment(segment):
                            continue
                        if category not in found_categories:
                            found_categories.append(category)
                        segment_matched = True
                        break
                if segment_matched:
                    break
            if not segment_matched:
                for word in words:
                    word = word.strip()
                    if not word:
                        continue
                    word_normalized = word.rstrip('s') if word.endswith('s') and len(word) > 3 else word
                    for category, keywords in categories.items():
                        if category == 'books' and not is_explicit_book_segment(word):
                            continue
                        # Exact match
                        if word in keywords or word_normalized in keywords:
                            if category not in found_categories:
                                found_categories.append(category)
                            break
                        # Check if any keyword contains the word (handles "accessories" matching "gaming accessories")
                        for keyword in keywords:
                            if word in keyword or keyword in word:
                                if category == 'books' and not is_explicit_book_segment(word):
                                    continue
                                if category not in found_categories:
                                    found_categories.append(category)
                                break
                        # Prefix match for longer words
                        if len(word) >= 4:
                            for keyword in keywords:
                                if keyword.startswith(word):
                                    if category == 'books' and not is_explicit_book_segment(word):
                                        continue
                                    if category not in found_categories:
                                        found_categories.append(category)
                                    break
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
        
        return found_categories if found_categories else []
    
    def _extract_category(self, query: str) -> Optional[str]:
        """Extract single category (backward compatibility)."""
        categories = self._extract_categories(query)
        return categories[0] if categories else None
    
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
    
    def search(
        self,
        query: str,
        k: int = 5,
        sort_by: str = 'relevance',  # 'relevance', 'price_low', 'price_high'
        in_stock_only: bool = False
    ):
        """
        Search products using hybrid BM25 + semantic matching.
        
        Args:
            query: Search query
            k: Number of results to return
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
        
        # Decision point: Multiple categories vs Single category vs No category
        # Multiple categories (explicit separators like comma, "and") -> return grouped dict
        # Single category (no separators) -> return flat list
        # No category -> search all products (with price filter if present)
        if len(categories) > 1:
            # Multi-category query: Search each category separately and return grouped results
            grouped_results = {}
            for category in categories:
                category_results = self._search_by_category(query, category, price_filter, query_terms, k, sort_by, in_stock_only)
                if category_results:
                    grouped_results[category] = category_results
            # Return dict only if we have results in multiple categories
            if len(grouped_results) > 1:
                return grouped_results
            # If only one category has results, return as list for backward compatibility
            elif len(grouped_results) == 1:
                return list(grouped_results.values())[0]
            else:
                return []
        
        # Single category - filter by category and price
        category_filter = categories[0] if categories else None
        if category_filter:
            return self._search_by_category(query, category_filter, price_filter, query_terms, k, sort_by, in_stock_only)
        
        # No category specified - search ALL products (with price filter if present)
        # Score all products
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


# Singleton instance
_search_engine: Optional[HybridSearch] = None


def get_search_engine() -> HybridSearch:
    """Get or create the hybrid search engine singleton."""
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearch()
    return _search_engine
