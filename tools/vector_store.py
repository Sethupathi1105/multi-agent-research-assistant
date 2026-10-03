import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self, persist_dir="data/chroma_db", collection_name="research_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, chunks: list[str], metadatas: list[dict] = None, ids: list[str] = None):
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        return results


if __name__ == "__main__":
    # quick standalone test
    store = VectorStore()
    store.add_documents(
        chunks=["The sky is blue.", "CrewAI is a multi-agent framework.", "Docker packages apps into containers."],
        ids=["1", "2", "3"]
    )
    results = store.query("What is CrewAI?")
    print(results)