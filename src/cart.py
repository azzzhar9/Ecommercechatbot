"""Shopping cart model for multi-product orders."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from src.logger import get_logger

logger = get_logger()


@dataclass
class CartItem:
    """Single item in the shopping cart."""
    product_id: str
    product_name: str
    unit_price: float
    quantity: int
    category: Optional[str] = None
    added_at: datetime = field(default_factory=datetime.now)
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal for this item."""
        return self.unit_price * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "unit_price": self.unit_price,
            "quantity": self.quantity,
            "category": self.category,
            "subtotal": self.subtotal,
            "added_at": self.added_at.isoformat()
        }


@dataclass
class ShoppingCart:
    """
    Shopping cart supporting multiple products.
    
    Features:
    - Add/remove/update items
    - Calculate totals with tax and shipping
    - Apply coupon codes
    - Session persistence
    """
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    items: List[CartItem] = field(default_factory=list)
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    coupon_code: Optional[str] = None
    discount_percent: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Tax and shipping rates
    TAX_RATE = 0.08  # 8% tax
    FREE_SHIPPING_THRESHOLD = 100.0
    SHIPPING_COST = 9.99
    
    # Available coupon codes
    COUPON_CODES = {
        "SAVE10": 10.0,
        "SAVE20": 20.0,
        "WELCOME": 15.0,
        "DEMO": 25.0,
    }
    
    def add_item(
        self,
        product_id: str,
        product_name: str,
        unit_price: float,
        quantity: int = 1,
        category: Optional[str] = None
    ) -> bool:
        """
        Add an item to the cart.
        
        If the product already exists, increase quantity.
        
        Returns:
            True if successful
        """
        # Check if product already in cart
        for item in self.items:
            if item.product_id == product_id or item.product_name.lower() == product_name.lower():
                item.quantity += quantity
                self.updated_at = datetime.now()
                logger.info(f"Updated cart: {product_name} x{item.quantity}")
                return True
        
        # Add new item
        item = CartItem(
            product_id=product_id,
            product_name=product_name,
            unit_price=unit_price,
            quantity=quantity,
            category=category
        )
        self.items.append(item)
        self.updated_at = datetime.now()
        logger.info(f"Added to cart: {product_name} x{quantity} @ ${unit_price}")
        return True
    
    def remove_item(self, product_name: str) -> bool:
        """Remove an item from the cart."""
        for i, item in enumerate(self.items):
            if item.product_name.lower() == product_name.lower():
                removed = self.items.pop(i)
                self.updated_at = datetime.now()
                logger.info(f"Removed from cart: {removed.product_name}")
                return True
        return False
    
    def update_quantity(self, product_name: str, quantity: int) -> bool:
        """Update quantity of an item."""
        if quantity <= 0:
            return self.remove_item(product_name)
        
        for item in self.items:
            if item.product_name.lower() == product_name.lower():
                item.quantity = quantity
                self.updated_at = datetime.now()
                logger.info(f"Updated quantity: {product_name} x{quantity}")
                return True
        return False
    
    def clear(self):
        """Clear all items from cart."""
        self.items = []
        self.coupon_code = None
        self.discount_percent = 0.0
        self.updated_at = datetime.now()
        logger.info("Cart cleared")
    
    def apply_coupon(self, code: str) -> tuple[bool, str]:
        """
        Apply a coupon code.
        
        Returns:
            (success, message)
        """
        code_upper = code.upper().strip()
        if code_upper in self.COUPON_CODES:
            self.coupon_code = code_upper
            self.discount_percent = self.COUPON_CODES[code_upper]
            self.updated_at = datetime.now()
            return True, f"Coupon applied! {self.discount_percent}% discount"
        return False, "Invalid coupon code"
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal before discounts."""
        return sum(item.subtotal for item in self.items)
    
    @property
    def discount_amount(self) -> float:
        """Calculate discount amount."""
        return self.subtotal * (self.discount_percent / 100)
    
    @property
    def subtotal_after_discount(self) -> float:
        """Subtotal after applying discount."""
        return self.subtotal - self.discount_amount
    
    @property
    def tax_amount(self) -> float:
        """Calculate tax amount."""
        return self.subtotal_after_discount * self.TAX_RATE
    
    @property
    def shipping_cost(self) -> float:
        """Calculate shipping cost."""
        if self.subtotal_after_discount >= self.FREE_SHIPPING_THRESHOLD:
            return 0.0
        return self.SHIPPING_COST
    
    @property
    def total(self) -> float:
        """Calculate total including tax and shipping."""
        return self.subtotal_after_discount + self.tax_amount + self.shipping_cost
    
    @property
    def item_count(self) -> int:
        """Total number of items."""
        return sum(item.quantity for item in self.items)
    
    @property
    def is_empty(self) -> bool:
        """Check if cart is empty."""
        return len(self.items) == 0
    
    def get_summary(self) -> str:
        """Get a formatted cart summary."""
        if self.is_empty:
            return "Your cart is empty."
        
        lines = ["** Shopping Cart **", ""]
        
        for item in self.items:
            lines.append(f"- {item.product_name} x{item.quantity} @ ${item.unit_price:.2f} = ${item.subtotal:.2f}")
        
        lines.append("")
        lines.append(f"Subtotal: ${self.subtotal:.2f}")
        
        if self.discount_percent > 0:
            lines.append(f"Discount ({self.coupon_code} - {self.discount_percent}%): -${self.discount_amount:.2f}")
        
        lines.append(f"Tax (8%): ${self.tax_amount:.2f}")
        
        if self.shipping_cost > 0:
            lines.append(f"Shipping: ${self.shipping_cost:.2f}")
        else:
            lines.append("Shipping: FREE")
        
        lines.append(f"**Total: ${self.total:.2f}**")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert cart to dictionary."""
        return {
            "session_id": self.session_id,
            "items": [item.to_dict() for item in self.items],
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "coupon_code": self.coupon_code,
            "discount_percent": self.discount_percent,
            "subtotal": self.subtotal,
            "discount_amount": self.discount_amount,
            "tax_amount": self.tax_amount,
            "shipping_cost": self.shipping_cost,
            "total": self.total,
            "item_count": self.item_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class CartManager:
    """Manage shopping carts for multiple sessions."""
    
    _carts: Dict[str, ShoppingCart] = {}
    
    @classmethod
    def get_cart(cls, session_id: str) -> ShoppingCart:
        """Get or create a cart for a session."""
        if session_id not in cls._carts:
            cls._carts[session_id] = ShoppingCart(session_id=session_id)
        return cls._carts[session_id]
    
    @classmethod
    def clear_cart(cls, session_id: str):
        """Clear a session's cart."""
        if session_id in cls._carts:
            cls._carts[session_id].clear()
    
    @classmethod
    def remove_cart(cls, session_id: str):
        """Remove a cart completely."""
        if session_id in cls._carts:
            del cls._carts[session_id]
