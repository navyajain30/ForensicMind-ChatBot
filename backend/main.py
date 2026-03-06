from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import uuid
import tempfile
from typing import List, Optional

from processors.multimodal import process_pdf, process_image, process_txt
from rag.basic    import generate_basic_rag_response
from rag.advanced import generate_advanced_rag_response
from evaluation.evaluate_rag import run_evaluation

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Crime Intelligence & IPC Legal Mapping API",
    description="Backend API for Advanced Multimodal RAG Pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js",  StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")),  name="js")

@app.get("/app", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# === Models ===
class ChatQuery(BaseModel):
    query: str
    case_id: str
    advanced_rag: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    timeline: Optional[List[dict]] = None
    crime_type: Optional[str] = None
    confidence: Optional[float] = None
    ipc_sections: Optional[List[dict]] = None


# === Endpoints ===

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Crime Intel system operational."}


@app.post("/upload")
async def upload_evidence(
    files: List[UploadFile] = File(...),
    location: str = Form(...),
    year: int = Form(...),
    crime_type: str = Form(...)
):
    """
    Upload FIRs, witness statements, or scene images.
    Extracts text/visuals and ingests into Vector Store.
    """

    case_id = f"CASE-{year}-{location[:3].upper()}-{str(uuid.uuid4())[:8].upper()}"

    upload_dir = os.path.join(tempfile.gettempdir(), "crime_intel_uploads", case_id)
    os.makedirs(upload_dir, exist_ok=True)

    metadata_base = {
        "case_id": case_id,
        "location": location,
        "year": year,
        "type": crime_type,
    }

    total_chunks = 0
    saved_files = []

    for file in files:
        # Sanitize filename to prevent path traversal attacks
        safe_name = os.path.basename(file.filename)
        file_path = os.path.join(upload_dir, safe_name)

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        saved_files.append(safe_name)

        metadata = {**metadata_base, "source": safe_name}
        filename_lower = safe_name.lower()

        try:
            if filename_lower.endswith(".pdf"):
                chunks_added = process_pdf(file_path, case_id, metadata)
                total_chunks += len(chunks_added)
            elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
                chunks_added = process_image(file_path, case_id, metadata)
                total_chunks += len(chunks_added)
            elif filename_lower.endswith(".txt"):
                chunks_added = process_txt(file_path, case_id, metadata)
                total_chunks += len(chunks_added)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {safe_name}. Allowed: .pdf, .txt, .png, .jpg, .jpeg"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {safe_name}: {str(e)}"
            )

    if total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No extractable content found. Ensure files contain readable text or clear images."
        )

    return {
        "status": "success",
        "message": f"Ingested {len(saved_files)} file(s) → {total_chunks} chunks into knowledge base.",
        "case_id": case_id,
        "files": saved_files,
        "chunks": total_chunks,
        "metadata": metadata_base
    }


@app.get("/evaluate")
async def evaluate_rag():
    """Run Precision / Recall / MRR evaluation over golden test dataset."""
    try:
        metrics = run_evaluation()
        return {
            "status":            "success",
            "precision":         metrics["precision"],
            "recall":            metrics["recall"],
            "mrr":               metrics["mrr"],
            "queries_evaluated": 8
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatQuery):
    """
    Chat endpoint.
    Routes to Advanced RAG (HyDE + Multi-query + Re-ranking) when
    advanced_rag=true, otherwise uses the Basic RAG pipeline.
    """
    if not request.case_id:
        raise HTTPException(status_code=400, detail="case_id is required.")

    if request.advanced_rag:
        try:
            rag_res = generate_advanced_rag_response(
                query=request.query, case_id=request.case_id
            )
        except Exception as e:
            # Graceful fallback to basic RAG if advanced pipeline errors out
            rag_res = generate_basic_rag_response(
                query=request.query, case_id=request.case_id
            )
            rag_res["answer"] = f"[Advanced RAG failed, using Basic RAG]\n\n{rag_res['answer']}"
    else:
        rag_res = generate_basic_rag_response(
            query=request.query, case_id=request.case_id
        )

    return ChatResponse(
        answer       = rag_res["answer"],
        sources      = rag_res.get("sources", []),
        timeline     = rag_res.get("timeline", []),
        crime_type   = rag_res.get("crime_type"),
        confidence   = rag_res.get("confidence"),
        ipc_sections = rag_res.get("ipc_sections", []),
    )



if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[".", "rag", "processors", "utils"],
        reload_excludes=[
            "temp_uploads",
            "chroma_db_store",
            "chroma_db_storage",
            "__pycache__",
            "*.pyc"
        ]
    )