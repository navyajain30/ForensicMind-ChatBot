import chromadb
from chromadb.config import Settings
import os

from langchain_community.embeddings import OllamaEmbeddings

# Initialize local embedding model via Ollama
# We use nomic-embed-text for high performance local embeddings
embeddings_processor = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# Custom embedding wrapper for ChromaDB using Langchain Ollama
class ChromaOllamaEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Chroma passes a list of texts
        return embeddings_processor.embed_documents(input)

embedding_function = ChromaOllamaEmbeddingFunction()

# Initialize ChromaDB client
CHROMA_DB_DIR = "chroma_db_storage"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection for multi-modal case texts
case_collection = client.get_or_create_collection(
    name="crime_cases",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

def add_documents(docs: list[str], metadatas: list[dict], ids: list[str]):
    """Add chunked documents to the vector store."""
    if not docs:
        return
    case_collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )

def basic_search(query: str, n_results: int = 5, where: dict = None):
    """
    Standard Semantic search using vector cosine similarity.
    """
    # The embedding function auto-embeds the query string here
    results = case_collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where
    )
    return results
