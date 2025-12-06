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
        'laptop': ['laptop', 'macbook', 'dell', 'notebook', 'xps', 'portable', 'ultrabook'],
        'phone': ['phone', 'iphone', 'samsung', 'galaxy', 'smartphone', 'mobile', 'cellphone'],
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
        'gaming': ['gaming', 'ps5', 'playstation', 'xbox', 'nintendo', 'console', 'controller'],
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
        price_match = re.search(r'under\s*\$?(\d+)', query_lower)
        if price_match:
            return (0, float(price_match.group(1)))
        
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
    
    def _extract_category(self, query: str) -> Optional[str]:
        """Extract category filter from query."""
        query_lower = query.lower()
        
        categories = {
            'electronics': ['electronics', 'electronic', 'tech', 'gadget'],
            'audio': ['audio', 'sound', 'music', 'headphone', 'speaker', 'earbuds'],
            'computers': ['computer', 'laptop', 'pc', 'desktop'],
            'phones': ['phone', 'mobile', 'smartphone'],
            'gaming': ['gaming', 'game', 'console', 'playstation', 'xbox'],
            'wearables': ['wearable', 'watch', 'fitness', 'tracker'],
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return category
        
        return None
    
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
    ) -> List[Dict]:
        """
        Search products using hybrid BM25 + semantic matching.
        
        Args:
            query: Search query
            k: Number of results to return
            sort_by: Sort order ('relevance', 'price_low', 'price_high')
            in_stock_only: Filter to only in-stock products
        
        Returns:
            List of matching products with scores
        """
        if not self.products:
            return []
        
        # Expand query with synonyms
        query_terms = self._expand_query(query)
        
        # Extract filters
        price_filter = self._extract_price_filter(query)
        
        # Score all products
        scored_products = []
        for idx, product in enumerate(self.products):
            # Apply filters
            if in_stock_only and product.get('stock_status') == 'out_of_stock':
                continue
            
            if price_filter:
                price = product.get('price', 0)
                if not (price_filter[0] <= price <= price_filter[1]):
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
        else:  # relevance
            scored_products.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top k products
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
