import sys
import chromadb
print("Loading ChromaDB...", flush=True)
c = chromadb.PersistentClient(path='./vector_store')
print("Getting collection...", flush=True)
coll = c.get_collection('products')
print("Got collection, getting sample...", flush=True)
sample = coll.peek(limit=2)
print(f"Sample IDs: {sample['ids']}", flush=True)
print("SUCCESS!", flush=True)
sys.stdout.flush()
