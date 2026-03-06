"""
rag/advanced.py — Advanced RAG Pipeline
=========================================
Implements three enhancements over basic RAG:

  1. HyDE  (Hypothetical Document Embedding)
     — LLM generates a plausible hypothetical answer first,
       then that answer is used as the ChromaDB search query,
       retrieving semantically richer chunks than the raw question would.

  2. Multi-Query Retrieval
     — LLM rewrites the original query into 3 semantically distinct variants.
       All variants are run through ChromaDB independently.
       Results are merged and deduplicated by document content.

  3. Re-ranking
     — Every retrieved chunk is scored by keyword overlap with both the
       original query and the hypothetical answer.
       Only the top-N highest-scoring chunks reach the LLM.

The final answer + sources format is identical to basic.py so the
/chat endpoint and all frontend code need zero changes.
"""

import json
import os
from rag.vector_store import basic_search
from utils.llm import llm_service
from rag.basic import (          # reuse IPC logic from basic.py — no duplication
    _is_legal_query,
    _classify_crime_from_context,
    _get_ipc_sections_for_category,
    CATEGORY_LABELS,
    IPC_DATABASE,
)

# ── Constants ──────────────────────────────────────────────────────────────────
N_PER_QUERY = 5    # chunks to fetch per query variant
N_FINAL     = 5    # top chunks to keep after re-ranking
N_VARIANTS  = 3    # number of multi-query variants to generate


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — HyDE: Generate a hypothetical answer
# ─────────────────────────────────────────────────────────────────────────────

def _generate_hypothetical_answer(query: str) -> str | None:
    """
    Ask the LLM to write a short hypothetical document that would perfectly
    answer the query.  This hypothetical text is then used as the search
    vector instead of the bald question, pulling back denser semantic matches.
    """
    system = (
        "You are a legal document writer. The user gives you a question about a crime case. "
        "Write a short, realistic document excerpt (2–4 sentences) that would directly answer "
        "that question. Use realistic investigative language. Do not say 'I don't know'."
    )
    user = f"Question: {query}\n\nWrite a hypothetical document excerpt that answers this:"

    hypothesis, is_error = llm_service.generate(system, user)
    if is_error:
        return None       # fall back to raw query if LLM unavailable
    return hypothesis.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Multi-Query: Generate query variants
# ─────────────────────────────────────────────────────────────────────────────

def _generate_query_variants(query: str) -> list[str]:
    """
    Ask the LLM for N_VARIANTS rephrased versions of the query.
    Each variant is a semantically distinct rewrite, covering different
    angles so ChromaDB retrieves a broader set of relevant chunks.
    """
    system = (
        f"You are a search query optimizer for a legal case retrieval system. "
        f"Rewrite the user's question into exactly {N_VARIANTS} different phrasings. "
        f"Output ONLY the {N_VARIANTS} queries, each on its own numbered line (1. ... 2. ... 3. ...). "
        f"Make each query semantically different from the others."
    )
    user = f"Original query: {query}"

    result, is_error = llm_service.generate(system, user)
    if is_error:
        return [query]    # safe fallback: just use the original

    # Parse numbered lines — accept "1." / "1)" / "- " etc.
    variants = []
    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering / bullets
        for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "-", "*"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if len(line) > 8:          # ignore very short/empty lines
            variants.append(line)

    # Always include the original query and cap at N_VARIANTS
    all_variants = list(dict.fromkeys([query] + variants))   # dedup, preserve order
    return all_variants[:N_VARIANTS + 1]                     # +1 for the original


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Multi-query retrieval + deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _multi_query_retrieve(queries: list[str], case_id: str | None) -> list[dict]:
    """
    Run ChromaDB search for each query variant.
    Deduplicate by document text and return a flat list of chunk dicts.
    """
    where_clause = {"case_id": case_id} if case_id else None
    seen_texts   = set()
    all_chunks   = []

    for q in queries:
        try:
            results = basic_search(query=q, n_results=N_PER_QUERY, where=where_clause)
            if not results or not results.get("documents") or not results["documents"][0]:
                continue
            docs  = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results.get("distances", [[0.5] * len(docs)])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                # Deduplicate on first 120 chars of text
                key = doc[:120].strip()
                if key not in seen_texts:
                    seen_texts.add(key)
                    all_chunks.append({
                        "text":   doc,
                        "meta":   meta,
                        "dist":   dist,
                        "score":  0.0,   # filled in by re-ranker
                    })
        except Exception:
            continue   # skip a failed variant silently

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(chunks: list[dict], query: str, hypothesis: str | None) -> list[dict]:
    """
    Score each chunk by:
      a) Keyword overlap with original query
      b) Keyword overlap with hypothetical answer (if available)
      c) ChromaDB cosine distance (lower = better)
    Return the top N_FINAL chunks sorted by combined score (descending).
    """
    # Build token sets for scoring
    def tokenise(text: str) -> set:
        return {w.lower() for w in (text or "").split() if len(w) > 3}

    query_tokens      = tokenise(query)
    hypo_tokens       = tokenise(hypothesis or "")   # empty set if no hypothesis

    for chunk in chunks:
        doc_tokens = tokenise(chunk["text"])

        overlap_query = len(query_tokens & doc_tokens)  / max(len(query_tokens), 1)
        overlap_hypo  = len(hypo_tokens  & doc_tokens)  / max(len(hypo_tokens), 1)
        cosine_score  = 1.0 - chunk["dist"]             # higher = more similar

        # Weighted combination
        chunk["score"] = (
            0.35 * overlap_query +
            0.35 * overlap_hypo  +
            0.30 * cosine_score
        )

    ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
    return ranked[:N_FINAL]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_advanced_rag_response(query: str, case_id: str = None) -> dict:
    """
    Full Advanced RAG pipeline:
      HyDE → Multi-query retrieval → Re-ranking → LLM answer generation.

    Returns the same dict shape as basic.py:
      { answer, sources, crime_type, confidence, ipc_sections, timeline }
    """

    # ── 1. HyDE ───────────────────────────────────────────────────────────────
    hypothesis = None
    try:
        hypothesis = _generate_hypothetical_answer(query)
    except Exception:
        pass   # non-fatal; re-ranker works without it

    # ── 2. Build query set (original + variants + hypothesis) ─────────────────
    try:
        variants = _generate_query_variants(query)
    except Exception:
        variants = [query]

    search_queries = list(dict.fromkeys(variants + ([hypothesis] if hypothesis else [])))

    # ── 3. Multi-query retrieval ───────────────────────────────────────────────
    chunks = _multi_query_retrieve(search_queries, case_id)

    if not chunks:
        # Nothing in the DB — fall back to a single basic_search
        try:
            results = basic_search(query=query, n_results=N_FINAL,
                                   where={"case_id": case_id} if case_id else None)
            if results and results.get("documents") and results["documents"][0]:
                docs  = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results.get("distances", [[0.5] * len(docs)])[0]
                chunks = [{"text": d, "meta": m, "dist": x, "score": 1.0 - x}
                          for d, m, x in zip(docs, metas, dists)]
        except Exception:
            pass

    if not chunks:
        return {
            "answer": "No relevant evidence found for this case. Please upload FIR and witness statement files.",
            "sources": [], "crime_type": None, "confidence": None,
            "ipc_sections": [], "timeline": [],
        }

    # ── 4. Re-rank ────────────────────────────────────────────────────────────
    top_chunks = _rerank(chunks, query, hypothesis)

    context = "\n\n---\n\n".join(c["text"] for c in top_chunks)

    # ── 5. IPC analysis (only if query asks for it) ────────────────────────────
    legal_query  = _is_legal_query(query)
    crime_type   = None
    confidence   = None
    ipc_sections = []

    if legal_query:
        crime_category, conf_score = _classify_crime_from_context(context)
        crime_type   = CATEGORY_LABELS.get(crime_category, "Other")
        confidence   = round(conf_score * 100, 1)
        ipc_sections = _get_ipc_sections_for_category(crime_category, context)

        ipc_ref = "\n".join(
            f"  • IPC {s['section']} — {s['title']}: {s['description'][:100]}..."
            for s in ipc_sections
        )
        system_prompt = (
            "You are an expert criminal investigator and IPC legal analyst.\n"
            "Use the provided evidence to answer the legal question.\n"
            "Cite specific IPC sections and explain WHY they apply based on the evidence.\n"
            "Only use facts present in the retrieved documents."
        )
        user_prompt = (
            f"CASE EVIDENCE (re-ranked, most relevant first):\n{context}\n\n"
            f"RELEVANT IPC SECTIONS:\n{ipc_ref}\n\n"
            f"LEGAL QUERY: {query}"
        )
    else:
        system_prompt = (
            "You are a factual case analysis assistant.\n"
            "Answer the investigator's question using ONLY the provided case evidence.\n"
            "Be direct and specific — cite names, times, locations, and actions.\n"
            "Do NOT mention IPC sections unless the question asks for them.\n"
            "If the evidence does not answer the question, say so clearly."
        )
        user_prompt = (
            f"CASE EVIDENCE (re-ranked, most relevant first):\n{context}\n\n"
            f"QUESTION: {query}"
        )

    # ── 6. Generate final answer ───────────────────────────────────────────────
    answer, is_error = llm_service.generate(system_prompt, user_prompt)

    if is_error:
        chunks_summary = "\n".join(
            f"[{i+1}] {c['meta'].get('source','?')}: {c['text'][:200]}..."
            for i, c in enumerate(top_chunks)
        )
        answer = (
            f"⚠️ LLM unavailable — showing re-ranked retrieved evidence.\n"
            f"{answer}\n\n**Top Retrieved Chunks (re-ranked):**\n{chunks_summary}"
        )

    # ── 7. Format sources ─────────────────────────────────────────────────────
    sources = [
        {
            "chunk_text": c["text"][:150] + "...",
            "score":      round(c["score"], 2),
            "source":     c["meta"].get("source", "Unknown"),
        }
        for c in top_chunks
    ]

    return {
        "answer":       answer,
        "sources":      sources,
        "crime_type":   crime_type,
        "confidence":   confidence,
        "ipc_sections": ipc_sections,
        "timeline":     [],
    }
