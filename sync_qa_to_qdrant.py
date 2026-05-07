"""
Sync all active Q&A entries from the JKF database to Qdrant.

Run this after adding new entries to the knowledge base if you prefer
manual syncing over the in-app button:
    python sync_qa_to_qdrant.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
import re
import uuid
import hashlib
from collections import Counter

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from app import app, db, KnowledgeBase

COLLECTION_NAME = "jkf_kb"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def compute_sparse_vector(text: str):
    words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    if not words:
        return [], []
    counts = Counter(words)
    total = len(words)
    seen: dict = {}
    indices, values = [], []
    for word, count in counts.items():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 50000
        if idx in seen:
            values[seen[idx]] += count / total
        else:
            seen[idx] = len(indices)
            indices.append(idx)
            values.append(count / total)
    return indices, values


def main():
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, prefer_grpc=False)

    with app.app_context():
        entries = KnowledgeBase.query.filter_by(is_active=True, qdrant_synced=False).all()
        print(f"Found {len(entries)} unsynced Q&A entries.")

        synced = 0
        for entry in entries:
            try:
                text = f"Spørgsmål: {entry.question}\nSvar: {entry.answer}"
                emb = openai_client.embeddings.create(model="text-embedding-3-large", input=text)
                dense = emb.data[0].embedding
                sp_idx, sp_val = compute_sparse_vector(text)

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"jkf-kb-{entry.id}"))
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id=point_id,
                        vector={"dense": dense, "sparse": SparseVector(indices=sp_idx, values=sp_val)},
                        payload={
                            "text": text,
                            "type": "qa",
                            "question": entry.question,
                            "answer": entry.answer,
                            "source_type": "qa",
                            "created_at": entry.created_at.isoformat() if entry.created_at else None,
                        }
                    )]
                )
                entry.qdrant_synced = True
                synced += 1
                print(f"  ✓ [{entry.id}] {entry.question[:60]}…")
            except Exception as e:
                print(f"  ✗ [{entry.id}] Error: {e}")

        db.session.commit()
        print(f"\nSynced {synced}/{len(entries)} entries.")


if __name__ == "__main__":
    main()
