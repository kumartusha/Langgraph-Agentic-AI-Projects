import chromadb
import uuid
from typing import List, Dict
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self):
        # We use a PersistentClient so bug reports are saved across server restarts.
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(name=settings.CHROMA_COLLECTION_NAME)

    def query_memories(self, query: str, n_results: int = 1) -> List[Dict]:
        """
        Query the vector database for similar bug reports.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            parsed_results = []
            if results and results.get('ids') and results['ids'][0]:
                for index, doc_id in enumerate(results['ids'][0]):
                    parsed_results.append({
                        'id': doc_id,
                        'memory': results['documents'][0][index],
                        'distance': results['distances'][0][index] if 'distances' in results and results['distances'] else 0.0
                    })
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return []

    def add_memory(self, content: str) -> str:
        """
        Save a new bug report into memory.
        """
        try:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                ids=[doc_id],
                documents=[content]
            )
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return ""

    def update_memory(self, doc_id: str, content: str) -> bool:
        """
        Update an existing memory with cumulative insights.
        """
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[content]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {doc_id}: {e}")
            return False

    def get_memory(self, doc_id: str) -> str:
        """
        Retrieve a specific memory document by ID.
        """
        try:
            results = self.collection.get(ids=[doc_id])
            if results and results.get('documents') and results['documents']:
                return results['documents'][0]
            return ""
        except Exception as e:
            logger.error(f"Failed to get memory {doc_id}: {e}")
            return ""

memory_service = MemoryService()
