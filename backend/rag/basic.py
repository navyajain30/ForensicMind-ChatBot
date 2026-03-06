import json
import os
from rag.vector_store import basic_search
from utils.llm import llm_service

# ── Load IPC knowledge base once at module import ─────────────────────────────
_IPC_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ipc_sections.json")
with open(_IPC_DB_PATH, "r", encoding="utf-8") as f:
    IPC_DATABASE = json.load(f)

CATEGORY_LABELS = {
    "theft":    "Theft",
    "robbery":  "Robbery",
    "assault":  "Assault",
    "fraud":    "Fraud / Cyber",
    "homicide": "Homicide",
    "other":    "Other",
}

CATEGORY_KEYWORDS = {
    "homicide": ["murder", "killed", "dead body", "death", "poisoned", "strangulated", "shot dead"],
    "robbery":  ["snatched", "robbed", "robbery", "motorcycle", "fled", "forcibly took", "dacoity"],
    "theft":    ["stolen", "theft", "burglary", "broke into", "missing", "pickpocket"],
    "assault":  ["attacked", "beaten", "assault", "hit", "stabbed", "hurt", "injury", "molested", "rape"],
    "fraud":    ["cheated", "fraud", "scam", "fake", "forged", "phishing", "otp", "cyber", "online"],
}

# ── Legal-intent keywords ──────────────────────────────────────────────────────
# These signal the user is specifically asking for IPC / legal analysis.
_LEGAL_KEYWORDS = [
    "ipc", "section", "legal", "law", "penal", "punish", "offence", "offense",
    "charge", "arrest", "bail", "court", "warrant", "fir", "cognizable",
    "classify", "crime type", "what crime", "which crime", "applicable",
    "liable", "convict", "sentence",
]


def _is_legal_query(query: str) -> bool:
    """
    Returns True only when the user is explicitly asking for legal /
    IPC-section analysis.  All other queries get a plain evidence answer.
    """
    q = query.lower()
    return any(kw in q for kw in _LEGAL_KEYWORDS)


def _classify_crime_from_context(context: str):
    """Keyword-based crime category detection. Returns (category, confidence)."""
    context_lower = context.lower()
    scores = {cat: sum(1 for kw in kws if kw in context_lower)
              for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    if scores[best] == 0:
        return "other", 0.50
    return best, round(min(scores[best] / total, 0.99), 2)


def _get_ipc_sections_for_category(category: str, context: str) -> list:
    """Return top-3 IPC sections for the detected crime category."""
    context_lower = context.lower()
    matching = [{"section": sec, **data}
                for sec, data in IPC_DATABASE.items()
                if data["category"] == category]
    if not matching:
        matching = [{"section": sec, **data}
                    for sec, data in IPC_DATABASE.items()
                    if data["category"] == "other"]

    def relevance(item):
        words = (item["title"] + " " + item["description"]).lower().split()
        return sum(1 for w in words if len(w) > 4 and w in context_lower)

    return sorted(matching, key=relevance, reverse=True)[:3]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_basic_rag_response(query: str, case_id: str = None) -> dict:
    """
    RAG pipeline:
      - All queries → retrieve evidence → LLM answers from evidence only
      - Legal queries only → also run crime classification + IPC mapping
    """

    # ── Step 1: Retrieve evidence from ChromaDB ────────────────────────────────
    where_clause = {"case_id": case_id} if case_id else None
    search_results = basic_search(query=query, n_results=5, where=where_clause)

    if not search_results or not search_results.get("documents") or not search_results["documents"][0]:
        return {
            "answer": "No relevant evidence found for this case. Please ensure you have uploaded FIR and witness statement files.",
            "sources": [],
            "crime_type": None,
            "confidence": None,
            "ipc_sections": [],
            "timeline": [],
        }

    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]
    distances = search_results.get("distances", [[0.5] * len(documents)])[0]
    context   = "\n\n---\n\n".join(documents)

    # ── Step 2: Decide if IPC analysis is needed ──────────────────────────────
    legal_query = _is_legal_query(query)

    crime_type   = None
    confidence   = None
    ipc_sections = []

    if legal_query:
        # Only run IPC pipeline for legal queries
        crime_category, confidence_score = _classify_crime_from_context(context)
        crime_type   = CATEGORY_LABELS.get(crime_category, "Other")
        confidence   = round(confidence_score * 100, 1)
        ipc_sections = _get_ipc_sections_for_category(crime_category, context)

        ipc_ref = "\n".join(
            f"  • IPC {s['section']} — {s['title']}: {s['description'][:100]}..."
            for s in ipc_sections
        )

        system_prompt = (
            "You are an expert criminal investigator and Indian Penal Code (IPC) legal analyst.\n"
            "The user is asking a legal question. Use the evidence below to:\n"
            "1. Identify which IPC sections apply and explain WHY, with specific references to the evidence.\n"
            "2. Be concise and factual. Only cite what is clearly in the evidence.\n"
            "3. Do not invent facts not present in the documents."
        )
        user_prompt = (
            f"CASE EVIDENCE:\n{context}\n\n"
            f"RELEVANT IPC SECTIONS TO CONSIDER:\n{ipc_ref}\n\n"
            f"LEGAL QUERY: {query}"
        )

    else:
        # Plain factual query — answer directly from evidence only
        system_prompt = (
            "You are a factual case analysis assistant.\n"
            "Answer the investigator's question using ONLY the provided case evidence below.\n"
            "Be direct and specific. Quote names, times, locations, and actions from the documents.\n"
            "Do NOT mention IPC sections or legal analysis unless asked.\n"
            "If the evidence does not contain the answer, say so clearly."
        )
        user_prompt = (
            f"CASE EVIDENCE:\n{context}\n\n"
            f"QUESTION: {query}"
        )

    # ── Step 3: Generate answer ────────────────────────────────────────────────
    answer, is_error = llm_service.generate(system_prompt, user_prompt)

    if is_error:
        # Fallback: show raw retrieved chunks directly
        chunks_summary = "\n".join(
            f"[{i+1}] {meta.get('source', '?')}: {doc[:200]}..."
            for i, (doc, meta) in enumerate(zip(documents, metadatas))
        )
        answer = f"⚠️ LLM unavailable — showing retrieved evidence directly.\n{answer}\n\n**Retrieved Evidence:**\n{chunks_summary}"

    # ── Step 4: Build sources list ─────────────────────────────────────────────
    sources = [
        {
            "chunk_text": doc[:150] + "...",
            "score":      round(1.0 - dist, 2),
            "source":     meta.get("source", "Unknown"),
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    return {
        "answer":      answer,
        "sources":     sources,
        "crime_type":  crime_type,    # None for non-legal queries
        "confidence":  confidence,    # None for non-legal queries
        "ipc_sections": ipc_sections, # [] for non-legal queries
        "timeline":    [],
    }
