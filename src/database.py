"""Database operations with error handling and security."""

import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, text
from sqlalchemy.exc import OperationalError, DatabaseError
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.logger import get_logger
from src.models import OrderModel
from src.utils import sanitize_input, validate_email

logger = get_logger()

Base = declarative_base()


class Order(Base):
    """SQLAlchemy model for orders table."""
    __tablename__ = "orders"
    
    order_id = Column(String, primary_key=True)
    product_name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    customer_name = Column(String, nullable=True)
    customer_email = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False)




@contextmanager
def get_db_session(db_path: str):
    """
    Context manager for database sessions with error handling.
    
    Args:
        db_path: Path to database file
    
    Yields:
        Database session
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}", exc_info=True)
        raise
    finally:
        session.close()


def init_database(db_path: str = "./orders.db") -> None:
    """
    Initialize database and create tables if they don't exist.
    
    Args:
        db_path: Path to database file
    """
    try:
        # Create directory if it doesn't exist
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        logger.info(f"Database initialized at {db_path}")
    except OperationalError as e:
        logger.error(f"Database initialization error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {str(e)}", exc_info=True)
        raise


def create_order(order: OrderModel, db_path: str = "./orders.db") -> str:
    """
    Create a new order in the database with error handling.
    
    Args:
        order: OrderModel instance
        db_path: Path to database file
    
    Returns:
        Order ID
    
    Raises:
        DatabaseError: If order creation fails
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Sanitize inputs
            sanitized_product_name = sanitize_input(order.product_name, max_length=200)
            sanitized_customer_name = sanitize_input(order.customer_name, max_length=100) if order.customer_name else None
            sanitized_customer_email = order.customer_email if (order.customer_email and validate_email(order.customer_email)) else None
            
            with get_db_session(db_path) as session:
                db_order = Order(
                    order_id=order.order_id,
                    product_name=sanitized_product_name,
                    quantity=order.quantity,
                    unit_price=order.unit_price,
                    total_price=order.total_price,
                    customer_name=sanitized_customer_name,
                    customer_email=sanitized_customer_email,
                    timestamp=order.timestamp
                )
                session.add(db_order)
                session.commit()
                
            logger.info(f"Order created successfully: {order.order_id}")
            return order.order_id
            
        except OperationalError as e:
            error_msg = str(e).lower()
            if "locked" in error_msg or "database is locked" in error_msg:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Database locked, retrying ({retry_count}/{max_retries})...")
                    import time
                    time.sleep(0.5 * retry_count)  # Exponential backoff
                    continue
                else:
                    logger.error("Database locked after multiple retries")
                    raise DatabaseError("Database is currently locked. Please try again later.")
            elif "disk" in error_msg or "full" in error_msg:
                logger.error("Disk full error")
                raise DatabaseError("Insufficient disk space. Please contact support.")
            else:
                logger.error(f"Database operational error: {str(e)}", exc_info=True)
                raise DatabaseError("An error occurred while creating the order. Please try again.")
        except DatabaseError as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating order: {str(e)}", exc_info=True)
            raise DatabaseError("An unexpected error occurred. Please try again.")
    
    raise DatabaseError("Failed to create order after multiple attempts.")


def get_order_by_id(order_id: str, db_path: str = "./orders.db") -> Optional[Dict]:
    """
    Retrieve order by order ID.
    
    Args:
        order_id: Order ID to retrieve
        db_path: Path to database file
    
    Returns:
        Order dictionary or None if not found
    """
    try:
        with get_db_session(db_path) as session:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order:
                return {
                    "order_id": order.order_id,
                    "product_name": order.product_name,
                    "quantity": order.quantity,
                    "unit_price": order.unit_price,
                    "total_price": order.total_price,
                    "customer_name": order.customer_name,
                    "customer_email": order.customer_email,
                    "timestamp": order.timestamp.isoformat() if order.timestamp else None
                }
            return None
    except Exception as e:
        logger.error(f"Error retrieving order {order_id}: {str(e)}", exc_info=True)
        return None


def get_all_orders(db_path: str = "./orders.db") -> List[Dict]:
    """
    Retrieve all orders from database.
    
    Args:
        db_path: Path to database file
    
    Returns:
        List of order dictionaries
    """
    try:
        with get_db_session(db_path) as session:
            orders = session.query(Order).all()
            return [
                {
                    "order_id": order.order_id,
                    "product_name": order.product_name,
                    "quantity": order.quantity,
                    "unit_price": order.unit_price,
                    "total_price": order.total_price,
                    "customer_name": order.customer_name,
                    "customer_email": order.customer_email,
                    "timestamp": order.timestamp.isoformat() if order.timestamp else None
                }
                for order in orders
            ]
    except Exception as e:
        logger.error(f"Error retrieving all orders: {str(e)}", exc_info=True)
        return []


def check_stock(product_name: str, vector_store_path: str = "./vector_store") -> str:
    """
    Check stock status for a product from vector store metadata.
    
    Args:
        product_name: Name of the product
        vector_store_path: Path to vector store
    
    Returns:
        Stock status: "in_stock", "low_stock", or "out_of_stock"
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Load vector store
        client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection("products")
        
        # Search for product
        results = collection.get(
            where={"name": product_name},
            limit=1
        )
        
        if results and results['metadatas'] and len(results['metadatas']) > 0:
            stock_status = results['metadatas'][0].get('stock_status', 'in_stock')
            return stock_status
        else:
            logger.warning(f"Product not found in vector store: {product_name}")
            return "out_of_stock"
            
    except Exception as e:
        logger.error(f"Error checking stock for {product_name}: {str(e)}", exc_info=True)
        # Default to out_of_stock on error for safety
        return "out_of_stock"

