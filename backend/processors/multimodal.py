import fitz  # PyMuPDF
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.vector_store import add_documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

def process_pdf(file_path: str, case_id: str, metadata: dict) -> list[str]:
    """
    Extracts text from PDF, chunks it, and adds to ChromaDB.
    Returns the loaded chunk IDs.
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF {file_path}: {e}")
        return []

    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
        
    chunks = text_splitter.split_text(full_text)
    
    ids = [f"{case_id}-{uuid.uuid4()}" for _ in chunks]
    metadatas = [
        {
            **metadata,
            "chunk_index": i
        } for i in range(len(chunks))
    ]
    
    add_documents(chunks, metadatas, ids)
    return ids

def process_image(file_path: str, case_id: str, metadata: dict) -> list[str]:
    """
    Simulated Vision Processing (e.g. LLaVA via Ollama or Groq Vision).
    In a fully functional backend, this would use a vision endpoint to 
    describe the CCTV frame/map, extract timestamps, and return text.
    """
    # Simulate vision extraction
    vision_description = (
        f"Image Analysis of {file_path}:\n"
        "Visible timestamp: 2024-05-12 23:15:00 UTC\n"
        "Scene notes: Two individuals seen near the entrance. "
        "One appears to be holding a bag. Lighting is poor."
    )
    
    chunk_id = f"{case_id}-IMG-{uuid.uuid4()}"
    metadatas = [{ **metadata, "content_type": "image_description" }]
    
    add_documents([vision_description], metadatas, [chunk_id])
    return [chunk_id]

def process_txt(file_path: str, case_id: str, metadata: dict) -> list[str]:
    """
    Extracts text from a standard .txt file, chunks it, and adds to ChromaDB.
    Returns the loaded chunk IDs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        print(f"Error opening TXT {file_path}: {e}")
        return []

    chunks = text_splitter.split_text(full_text)
    if not chunks:
        return []
    
    ids = [f"{case_id}-{uuid.uuid4()}" for _ in chunks]
    metadatas = [
        {
            **metadata,
            "chunk_index": i
        } for i in range(len(chunks))
    ]
    
    add_documents(chunks, metadatas, ids)
    return ids
