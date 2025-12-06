"""Initialize vector store with product embeddings and metadata."""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

from src.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger with error handling
try:
    logger = get_logger()
except Exception as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ecommerce_chatbot")
    logger.warning(f"Could not initialize custom logger: {e}. Using basic logger.")


def load_products(json_path: str) -> List[Dict]:
    """
    Load products from JSON file.
    
    Args:
        json_path: Path to products JSON file
    
    Returns:
        List of product dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        logger.info(f"Loaded {len(products)} products from {json_path}")
        return products
    except FileNotFoundError:
        logger.error(f"Products file not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in products file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading products: {str(e)}", exc_info=True)
        raise


def generate_embeddings_batch(
    texts: List[str],
    client: OpenAI,
    model: Optional[str] = None,
    batch_size: int = 100,
    max_retries: int = 3
) -> List[List[float]]:
    """
    Generate embeddings in batches with retry logic.
    
    Args:
        texts: List of texts to embed
        client: OpenAI client
        model: Embedding model name
        batch_size: Number of texts per batch
        max_retries: Maximum retry attempts
    
    Returns:
        List of embedding vectors
    """
    # Use model from environment or default
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} items)")
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"Embedding generation failed, retrying in {wait_time}s... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embeddings after {max_retries} attempts: {str(e)}", exc_info=True)
                    raise
    
    return all_embeddings


def initialize_vector_store(
    products_path: str = "./data/products.json",
    vector_store_path: str = "./vector_store",
    embedding_model: Optional[str] = None,
    batch_size: int = 50
) -> None:
    """
    Initialize vector store with product embeddings and metadata.
    
    Args:
        products_path: Path to products JSON file
        vector_store_path: Path to vector store directory
        embedding_model: OpenAI embedding model name
        batch_size: Batch size for embedding generation
    """
    try:
        # Load products
        products = load_products(products_path)
        
        if not products:
            raise ValueError("No products found in JSON file")
        
        # Initialize OpenAI client (supports OpenRouter)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            logger.error("OPENAI_API_KEY or OPENROUTER_API_KEY environment variable not set. Please check your .env file.")
            raise ValueError("API key not set. Please ensure your .env file contains: OPENAI_API_KEY=your-key-here")
        
        # Configure client for OpenRouter if base_url is provided
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Using OpenRouter API at {base_url}")
        else:
            client = OpenAI(api_key=api_key)
            logger.info("Using OpenAI API")
        
        # Create vector store directory
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            chroma_client = chromadb.PersistentClient(
                path=vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            logger.warning(f"Error loading existing vector store: {str(e)}. Creating new one...")
            # Backup corrupted store if it exists
            backup_path = f"{vector_store_path}_backup_{int(time.time())}"
            if Path(vector_store_path).exists():
                import shutil
                shutil.move(vector_store_path, backup_path)
                logger.info(f"Backed up corrupted vector store to {backup_path}")
            # Create new client
            chroma_client = chromadb.PersistentClient(
                path=vector_store_path,
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection("products")
            logger.info("Using existing 'products' collection")
        except:
            collection = chroma_client.create_collection(
                name="products",
                metadata={"description": "Product catalog with embeddings"}
            )
            logger.info("Created new 'products' collection")
        
        # Prepare texts and metadata
        texts = []
        metadatas = []
        ids = []
        
        for product in products:
            # Create text for embedding (name + description)
            text_content = f"{product['name']} {product['description']}"
            texts.append(text_content)
            
            # Store metadata with price (critical for price source of truth)
            metadata = {
                "product_id": product["product_id"],
                "name": product["name"],
                "price": float(product["price"]),  # Store as float in metadata
                "category": product["category"],
                "stock_status": product["stock_status"]
            }
            metadatas.append(metadata)
            ids.append(product["product_id"])
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} products in batches of {batch_size}...")
        embeddings = generate_embeddings_batch(
            texts=texts,
            client=client,
            model=embedding_model,
            batch_size=batch_size
        )
        
        # Add to collection
        logger.info("Adding products to vector store...")
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully initialized vector store with {len(products)} products at {vector_store_path}")
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    """Run vector store initialization."""
    import argparse
    
    print("Initializing vector store...")
    
    parser = argparse.ArgumentParser(description="Initialize vector store with product embeddings")
    parser.add_argument(
        "--products",
        type=str,
        default="./data/products.json",
        help="Path to products JSON file"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="./vector_store",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI embedding model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding generation"
    )
    
    args = parser.parse_args()
    
    try:
        initialize_vector_store(
            products_path=args.products,
            vector_store_path=args.vector_store,
            embedding_model=args.model,
            batch_size=args.batch_size
        )
        print("Vector store initialized successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

