"""Pydantic models for product and order validation."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class StockStatus(str, Enum):
    """Stock status enumeration."""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"


class Product(BaseModel):
    """Product model with validation."""
    product_id: str = Field(..., min_length=1, description="Unique product identifier")
    name: str = Field(..., min_length=1, description="Product name")
    description: str = Field(..., description="Product description")
    price: float = Field(..., gt=0, description="Product price (must be greater than 0)")
    category: str = Field(..., min_length=1, description="Product category")
    stock_status: StockStatus = Field(..., description="Current stock status")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class OrderModel(BaseModel):
    """Order model with validation and custom validators."""
    order_id: str = Field(default_factory=lambda: f"ORD-{uuid4().hex[:8].upper()}", description="Unique order identifier")
    product_name: str = Field(..., min_length=1, description="Product name")
    quantity: int = Field(..., gt=0, description="Order quantity (must be greater than 0)")
    unit_price: float = Field(..., gt=0, description="Unit price (must be greater than 0)")
    total_price: Optional[float] = Field(default=None, description="Total price (quantity × unit_price)")
    customer_name: Optional[str] = Field(None, min_length=1, description="Customer name")
    customer_email: Optional[str] = Field(None, description="Customer email")
    timestamp: datetime = Field(default_factory=datetime.now, description="Order timestamp")

    def __init__(self, **data):
        """Initialize and compute total_price if not provided."""
        if 'total_price' not in data or data['total_price'] is None:
            if 'quantity' in data and 'unit_price' in data:
                data['total_price'] = data['quantity'] * data['unit_price']
        super().__init__(**data)

    @field_validator('customer_email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format if provided."""
        if v is not None:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, v):
                raise ValueError("Invalid email format")
        return v

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

