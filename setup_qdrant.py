"""
One-time Qdrant setup script for JKF.

Run once after creating your Qdrant instance:
    python setup_qdrant.py

Creates the 'jkf_kb' collection with hybrid dense+sparse vectors
and the required payload indexes.
"""
from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams, PayloadSchemaType,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "jkf_kb"
VECTOR_SIZE = 3072  # text-embedding-3-large


def create_collection(qdrant: QdrantClient):
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
    )
    print(f"Created collection '{COLLECTION_NAME}'.")

    # Payload indexes for fast filtered search
    qdrant.create_payload_index(COLLECTION_NAME, "type",        PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "source_type", PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(COLLECTION_NAME, "file_id",     PayloadSchemaType.KEYWORD)
    print("Created payload indexes (type, source_type, file_id).")


if __name__ == "__main__":
    print(f"=== JKF Qdrant Setup ===")
    print(f"Connecting to {QDRANT_URL} …")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, prefer_grpc=False)
    create_collection(qdrant)
    print("\nDone! You can now run the scraper or sync Q&A entries to populate the knowledge base.")
