"""Simple in-memory cache with TTL (Time-To-Live) for performance optimization."""

import time
from typing import Any, Optional, Dict, Tuple
from threading import Lock
from src.logger import get_logger

logger = get_logger()


class TTLCache:
    """
    Thread-safe cache with Time-To-Live (TTL) expiration.
    
    Features:
    - Automatic expiration of cached items
    - Thread-safe operations
    - Configurable TTL per cache
    - Size limits to prevent memory issues
    """
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """
        Initialize TTL cache.
        
        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
            max_size: Maximum number of items in cache (default: 1000)
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiration_time)
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiration = self._cache[key]
            
            # Check if expired
            if time.time() > expiration:
                del self._cache[key]
                logger.debug(f"Cache expired for key: {key}")
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            # Evict oldest if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            ttl = ttl or self.default_ttl
            expiration = time.time() + ttl
            self._cache[key] = (value, expiration)
            logger.debug(f"Cached key: {key} (TTL: {ttl}s)")
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (first) entry from cache."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared cache ({count} items)")
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expiration) in self._cache.items()
                if now > expiration
            ]
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            return len(expired_keys)


# Global cache instances for different use cases
_search_cache = TTLCache(default_ttl=600, max_size=500)  # 10 minutes for searches
_stock_cache = TTLCache(default_ttl=300, max_size=100)   # 5 minutes for stock info
_product_cache = TTLCache(default_ttl=1800, max_size=200)  # 30 minutes for product data


def get_search_cache() -> TTLCache:
    """Get the search results cache."""
    return _search_cache


def get_stock_cache() -> TTLCache:
    """Get the stock information cache."""
    return _stock_cache


def get_product_cache() -> TTLCache:
    """Get the product data cache."""
    return _product_cache


def clear_all_caches() -> None:
    """Clear all caches."""
    _search_cache.clear()
    _stock_cache.clear()
    _product_cache.clear()
    logger.info("All caches cleared")

