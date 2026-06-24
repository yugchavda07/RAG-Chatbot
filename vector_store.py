"""
Vector Store Manager
Handles interaction with ChromaDB to persist document text chunks, metadata, and embeddings,
and performs similarity search queries.
"""

import re
import os
from typing import List, Dict, Any
import chromadb

class VectorStoreManager:
    """Manages the persistent vector store using ChromaDB."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        # Ensure persistence directory exists
        os.makedirs(persist_dir, exist_ok=True)
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def sanitize_collection_name(self, filename: str) -> str:
        """
        Sanitizes filenames to meet ChromaDB collection name requirements:
        - Must be 3-63 characters long.
        - Must start and end with an alphanumeric character.
        - Can contain only lowercase letters, numbers, underscores, and hyphens.
        - Cannot have consecutive hyphens or underscores.
        """
        # Convert to lowercase
        name = filename.lower()
        # Replace non-alphanumeric chars (excluding hyphens/underscores) with hyphens
        name = re.sub(r"[^a-z0-9_-]", "-", name)
        # Replace consecutive hyphens/underscores with a single hyphen
        name = re.sub(r"[-_]{2,}", "-", name)
        # Strip leading/trailing non-alphanumeric chars
        name = re.sub(r"^[^a-z0-9]+", "", name)
        name = re.sub(r"[^a-z0-9]+$", "", name)
        
        # Ensure proper length (between 3 and 63 chars)
        if len(name) < 3:
            name = f"collection-{name}"
        name = name[:63]
        
        # Final validation check
        if not re.match(r"^[a-z0-9]([a-z0-9_-]{1,61}[a-z0-9])?$", name):
            # Safe fallback if naming regex still fails
            import uuid
            name = f"pdf-db-{uuid.uuid4().hex[:12]}"
            
        return name

    def add_documents(self, 
                      collection_name: str, 
                      chunks: List[Dict[str, Any]], 
                      embeddings: List[List[float]]) -> None:
        """
        Ingests text chunks, metadata, and pre-calculated embeddings into ChromaDB.
        Args:
            collection_name (str): Cleaned collection name.
            chunks (List[Dict[str, Any]]): Text chunks and page metadata from PDFProcessor.
            embeddings (List[List[float]]): Computed vectors for the chunks.
        """
        # Create or fetch the collection
        collection = self.client.get_or_create_collection(name=collection_name)
        
        # Prepare inputs
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add to collection in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def collection_exists_and_has_data(self, collection_name: str) -> bool:
        """
        Checks if a collection exists and is populated.
        """
        try:
            # Check list of existing collections
            collections = [col.name for col in self.client.list_collections()]
            if collection_name in collections:
                col = self.client.get_collection(name=collection_name)
                return col.count() > 0
        except Exception:
            pass
        return False

    def query_similarity(self, 
                         collection_name: str, 
                         query_embedding: List[float], 
                         top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for semantically similar document chunks.
        Args:
            collection_name (str): Name of target document collection.
            query_embedding (List[float]): Generated query vector.
            top_k (int): Number of top results to return.
        Returns:
            List[Dict[str, Any]]: Chunks with text, page number, and distance score.
        """
        collection = self.client.get_collection(name=collection_name)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results for consumption
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc,
                    "page_number": meta.get("page_number", "Unknown"),
                    "distance": dist
                })
                
        return formatted_results

    def delete_collection(self, collection_name: str) -> None:
        """Deletes a collection from the database."""
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
