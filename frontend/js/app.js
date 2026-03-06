/**
 * ForensicMind — AI Crime Intelligence System
 * Frontend Application Logic
 *
 * Backend API (preserved exactly):
 *   POST /upload   { files, location, year, crime_type }
 *   POST /chat     { query, case_id, advanced_rag }
 *   GET  /evaluate
 */

"use strict";

// ── API Base ──────────────────────────────────────────────────────
const API_BASE = window.location.port === "8000" ? "" : "http://127.0.0.1:8000";

// ── File type → icon mapping ──────────────────────────────────────
function fileIcon(name) {
    const ext = (name || "").split(".").pop().toLowerCase();
    if (ext === "pdf") return "📄";
    if (ext === "txt") return "📝";
    if (["jpg", "jpeg", "png", "gif", "webp"].includes(ext)) return "📷";
    return "📎";
}

function fileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ── State ─────────────────────────────────────────────────────────
let currentCaseId = sessionStorage.getItem("currentCaseId") || null;
let currentCaseMeta = JSON.parse(sessionStorage.getItem("currentCaseMeta") || "null");
let selectedFiles = [];
let isSubmitting = false;
let chatHasMessages = false;

// Accumulated case data for PDF export
const caseReport = {
    messages: [],   // { role, text }
    sources: [],
    ipcSections: [],
    crimeType: null,
    confidence: null,
};

// ── DOM References ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const $ = id => document.getElementById(id);

    const dropZone = $("dropZone");
    const fileInput = $("fileInput");
    const filePreview = $("filePreview");
    const metadataForm = $("metadataForm");
    const caseLocation = $("caseLocation");
    const caseYear = $("caseYear");
    const crimeType = $("crimeType");
    const processBtn = $("processBtn");
    const chatForm = $("chatForm");
    const chatInput = $("chatInput");
    const chatHistory = $("chatHistory");
    const caseIdBadge = $("caseIdBadge");
    const caseIdText = $("caseIdText");
    const systemStatus = $("systemStatus");
    const statusBadge = $("statusBadge");
    const caseSummaryCard = $("caseSummaryCard");
    const caseSummaryBody = $("caseSummaryBody");
    const welcomeState = $("welcomeState");
    const suggestionChips = $("suggestionChips");

    // Restore session state
    if (currentCaseId) {
        showCaseBadge(currentCaseId);
        setStatus("Evidence Processed", "ready");
        if (currentCaseMeta) showCaseSummaryCard(currentCaseMeta);
        showSuggestionChips();
    }

    // ══════════════════════════════════════════════════════════════
    // FILE HANDLING
    // ══════════════════════════════════════════════════════════════

    function addFiles(newFiles) {
        const arr = Array.from(newFiles);
        arr.forEach(f => {
            if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
        });
        renderFilePreview();
        if (selectedFiles.length > 0) dropZone.classList.add("has-files");
    }

    function renderFilePreview() {
        if (selectedFiles.length === 0) {
            filePreview.style.display = "none";
            return;
        }
        filePreview.style.display = "flex";
        filePreview.innerHTML = selectedFiles.map((f, i) => `
      <div class="file-preview-item">
        <span class="file-icon">${fileIcon(f.name)}</span>
        <span class="file-name">${f.name}</span>
        <span class="file-size">${fileSize(f.size)}</span>
      </div>`).join("");
    }

    // Click on drop zone → open file picker
    dropZone.addEventListener("click", e => {
        if (e.target !== fileInput) fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) addFiles(fileInput.files);
    });

    // Drag & Drop
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => { dropZone.classList.remove("drag-over"); });
    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
    });

    // ══════════════════════════════════════════════════════════════
    // UPLOAD / PROCESS EVIDENCE
    // ══════════════════════════════════════════════════════════════

    metadataForm.addEventListener("submit", async e => {
        e.preventDefault();
        if (isSubmitting) return;
        if (selectedFiles.length === 0) { appendSystemMsg("⚠️ Please select at least one evidence file."); return; }

        isSubmitting = true;
        processBtn.disabled = true;
        processBtn.querySelector("span").textContent = "Processing…";
        setStatus("Processing Evidence…", "processing");

        const formData = new FormData();
        selectedFiles.forEach(f => formData.append("files", f));
        formData.append("location", caseLocation.value.trim() || "Unknown");
        formData.append("year", caseYear.value.trim() || new Date().getFullYear());
        formData.append("crime_type", crimeType.value || "other");

        const thinkId = appendThinking([
            "Ingesting evidence files…",
            "Chunking and embedding documents…",
            "Storing in vector database…",
        ]);

        try {
            const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
            const data = await res.json();
            removeThinking(thinkId);

            if (!res.ok) throw new Error(data.detail || "Upload failed");

            currentCaseId = data.case_id;
            sessionStorage.setItem("currentCaseId", currentCaseId);

            const meta = {
                location: caseLocation.value.trim() || "Unknown",
                year: caseYear.value.trim() || "—",
                crimeType: crimeType.options[crimeType.selectedIndex]?.text || "—",
                files: selectedFiles.map(f => f.name),
            };
            currentCaseMeta = meta;
            sessionStorage.setItem("currentCaseMeta", JSON.stringify(meta));

            showCaseBadge(currentCaseId);
            setStatus("Evidence Processed", "ready");
            showCaseSummaryCard(meta);
            showSuggestionChips();

            appendSystemMsg(`✅ Evidence processed successfully. ${selectedFiles.length} file(s) ingested. Ready for investigation.`);

        } catch (err) {
            removeThinking(thinkId);
            appendSystemMsg(`❌ Upload failed: ${err.message}`, "error");
            setStatus("Error", "error");
        } finally {
            isSubmitting = false;
            processBtn.disabled = false;
            processBtn.querySelector("span").textContent = "Process Evidence";
        }
    });

    // ══════════════════════════════════════════════════════════════
    // RAG MODE TOGGLE
    // ══════════════════════════════════════════════════════════════

    let advancedRagMode = false;
    const ragModeToggle = document.getElementById("ragModeToggle");
    const ragModeText = document.getElementById("ragModeText");

    if (ragModeToggle) {
        ragModeToggle.addEventListener("click", () => {
            advancedRagMode = !advancedRagMode;
            ragModeToggle.dataset.mode = advancedRagMode ? "advanced" : "basic";
            ragModeText.textContent = advancedRagMode ? "Advanced RAG" : "Basic RAG";
        });
    }

    // ══════════════════════════════════════════════════════════════
    // CHAT
    // ══════════════════════════════════════════════════════════════

    chatForm.addEventListener("submit", async e => {
        e.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;
        if (!currentCaseId) { appendSystemMsg("⚠️ Please upload and process evidence files first."); return; }

        chatInput.value = "";
        hideSuggestionChips();
        appendUserMsg(msg);
        caseReport.messages.push({ role: "user", text: msg });

        const thinkId = appendThinking([
            "Analyzing evidence…",
            "Retrieving relevant documents…",
            "Mapping IPC legal sections…",
            "Generating investigation report…",
        ]);

        try {
            const res = await fetch(`${API_BASE}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: msg, case_id: currentCaseId, advanced_rag: advancedRagMode }),
            });
            const data = await res.json();
            removeThinking(thinkId);

            if (!res.ok) throw new Error(data.detail || "Chat error");

            // Accumulate report data
            if (data.crime_type) caseReport.crimeType = data.crime_type;
            if (data.confidence) caseReport.confidence = data.confidence;
            if (data.sources) caseReport.sources = [...caseReport.sources, ...data.sources];
            if (data.ipc_sections) caseReport.ipcSections = [...caseReport.ipcSections, ...data.ipc_sections];
            caseReport.messages.push({ role: "assistant", text: data.answer || "" });

            appendChatResponse(data);

        } catch (err) {
            removeThinking(thinkId);
            appendSystemMsg(`❌ ${err.message}`, "error");
        }
    });

    // Suggestion chips — click to auto-send query
    if (suggestionChips) {
        suggestionChips.addEventListener("click", e => {
            const chip = e.target.closest(".chip");
            if (!chip) return;
            const query = chip.dataset.query;
            if (query) {
                chatInput.value = query;
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    }

    // ══════════════════════════════════════════════════════════════
    // RENDER: APPEND MESSAGES
    // ══════════════════════════════════════════════════════════════

    function appendUserMsg(text) {
        chatHasMessages = true;
        if (welcomeState) welcomeState.style.display = "none";
        const div = document.createElement("div");
        div.className = "message user-message";
        div.innerHTML = `
      <div class="message-avatar user-avatar"><i data-lucide="user"></i></div>
      <div class="message-bubble">${escHtml(text)}</div>`;
        chatHistory.appendChild(div);
        scrollChat();
        if (window.lucide) window.lucide.createIcons();
    }

    function appendSystemMsg(text, type = "") {
        chatHasMessages = true;
        if (welcomeState) welcomeState.style.display = "none";
        const div = document.createElement("div");
        div.className = "message system-message";
        const color = type === "error" ? "var(--red)" : "var(--cyan)";
        div.innerHTML = `
      <div class="message-avatar ai-avatar"><i data-lucide="cpu"></i></div>
      <div class="message-bubble" style="border-left-color:${color};">
        <p class="chat-answer">${text}</p>
      </div>`;
        chatHistory.appendChild(div);
        scrollChat();
        if (window.lucide) window.lucide.createIcons();
    }

    function appendChatResponse(data) {
        chatHasMessages = true;
        if (welcomeState) welcomeState.style.display = "none";

        const div = document.createElement("div");
        div.className = "message system-message";

        let html = `
      <div class="message-avatar ai-avatar"><i data-lucide="cpu"></i></div>
      <div class="message-bubble">`;

        // Answer
        html += `<p class="chat-answer">${escHtml(data.answer || "No answer returned.").replace(/\n/g, "<br>")}</p>`;

        // Crime Classification Card
        if (data.crime_type) {
            const conf = data.confidence || 0;
            const confColor = conf >= 75 ? "#00ff9d" : conf >= 50 ? "#f59e0b" : "#f43f5e";
            html += `
      <div class="classification-card">
        <div class="classification-header">
          <i data-lucide="shield-alert"></i> CRIME CLASSIFICATION
        </div>
        <div class="classification-body">
          <div class="crime-type-badge">${data.crime_type}</div>
          <div class="confidence-wrap">
            <span class="confidence-label">Confidence</span>
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill" style="width:${conf}%;background:${confColor};"></div>
            </div>
            <span class="confidence-value" style="color:${confColor};">${conf}%</span>
          </div>
        </div>
      </div>`;
        }

        // IPC Sections
        if (data.ipc_sections && data.ipc_sections.length > 0) {
            html += `<div class="ipc-section-container">
        <p class="section-header-label">⚖️ RECOMMENDED IPC SECTIONS</p>
        <div class="ipc-chips">`;
            data.ipc_sections.forEach(sec => {
                html += `<div class="ipc-chip">
          <div class="ipc-chip-header">
            <span class="ipc-number">IPC §${sec.section}</span>
            <span class="ipc-title">${sec.title}</span>
          </div>
          <p class="ipc-desc">${(sec.description || "").substring(0, 160)}…</p>
        </div>`;
            });
            html += `</div></div>`;
        }

        // Evidence Sources
        if (data.sources && data.sources.length > 0) {
            html += `<div class="sources-section">
        <p class="sources-label">📎 Evidence Sources</p>
        <div class="source-chips">`;
            data.sources.forEach(src => {
                const score = src.score ? ` · ${(src.score * 100).toFixed(0)}% match` : "";
                const icon = fileIcon(src.source || "");
                const snippet = src.chunk_text || src.chunk || "";
                html += `<div class="source-chip">
          <span class="source-file">${icon} ${src.source || "Unknown"}${score}</span>
          ${snippet ? `<span class="source-snippet">${snippet.substring(0, 130)}…</span>` : ""}
        </div>`;
            });
            html += `</div></div>`;
        }

        // Timeline
        if (data.timeline && data.timeline.length > 0) {
            html += `<div class="timeline-section">
        <p class="sources-label">🕐 Case Timeline</p>
        <div class="timeline-list">`;
            data.timeline.forEach(e => {
                html += `<div class="timeline-entry">
          <span class="timeline-time">${e.time || "—"}</span>
          <span class="timeline-event">${e.event || ""}</span>
        </div>`;
            });
            html += `</div></div>`;
        }

        html += `</div>`; // close message-bubble
        div.innerHTML = html;
        chatHistory.appendChild(div);
        scrollChat();
        if (window.lucide) window.lucide.createIcons();
    }

    // ══════════════════════════════════════════════════════════════
    // THINKING ANIMATION (multi-step)
    // ══════════════════════════════════════════════════════════════

    function appendThinking(steps = ["Analyzing…", "Retrieving…", "Generating…"]) {
        const id = "think-" + Date.now();
        const div = document.createElement("div");
        div.id = id;
        div.className = "message system-message";

        const stepsHtml = steps.map((s, i) => `
      <div class="thinking-step" style="animation-delay:${i * 0.25}s;" id="${id}-step${i}">
        <div class="thinking-step-dot"></div>
        ${s}
      </div>`).join("");

        div.innerHTML = `
      <div class="message-avatar ai-avatar"><i data-lucide="cpu"></i></div>
      <div class="thinking-panel">
        ${stepsHtml}
      </div>`;

        chatHistory.appendChild(div);
        scrollChat();
        if (window.lucide) window.lucide.createIcons();

        // Animate steps sequentially
        steps.forEach((_, i) => {
            setTimeout(() => {
                const step = document.getElementById(`${id}-step${i}`);
                if (step) {
                    if (i > 0) {
                        const prev = document.getElementById(`${id}-step${i - 1}`);
                        if (prev) prev.classList.remove("active"), prev.classList.add("done");
                    }
                    step.classList.add("active");
                }
            }, i * 800);
        });

        return id;
    }

    function removeThinking(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // ══════════════════════════════════════════════════════════════
    // EVALUATION METRICS BUTTON
    // ══════════════════════════════════════════════════════════════

    const adminBtn = $("btnAdminOverview");
    if (adminBtn) {
        adminBtn.addEventListener("click", async () => {
            adminBtn.disabled = true;
            adminBtn.querySelector("span").textContent = "Running…";
            const thinkId = appendThinking(["Running evaluation queries…", "Computing Precision & Recall…", "Calculating MRR…"]);

            try {
                const res = await fetch(`${API_BASE}/evaluate`);
                const data = await res.json();
                removeThinking(thinkId);

                if (data.status === "error") { appendSystemMsg(`⚠️ Evaluation failed: ${data.detail}`, "error"); return; }

                const p = (data.precision * 100).toFixed(1);
                const r = (data.recall * 100).toFixed(1);
                const mrr = (data.mrr * 100).toFixed(1);
                const bar = v => v >= 75 ? "#00ff9d" : v >= 50 ? "#f59e0b" : "#f43f5e";

                const div = document.createElement("div");
                div.className = "message system-message";
                div.innerHTML = `
          <div class="message-avatar ai-avatar"><i data-lucide="bar-chart-2"></i></div>
          <div class="message-bubble">
            <div class="eval-panel">
              <div class="eval-header">
                <span class="eval-title">📊 Evaluation Metrics</span>
                <span class="eval-sub">${data.queries_evaluated || 8} test queries</span>
              </div>
              <div class="eval-row">
                <span class="eval-label">Precision</span>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:${p}%;background:${bar(p)};"></div></div>
                <span class="eval-value" style="color:${bar(p)};">${data.precision}</span>
              </div>
              <div class="eval-row">
                <span class="eval-label">Recall</span>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:${r}%;background:${bar(r)};"></div></div>
                <span class="eval-value" style="color:${bar(r)};">${data.recall}</span>
              </div>
              <div class="eval-row">
                <span class="eval-label">MRR</span>
                <div class="eval-bar-bg"><div class="eval-bar-fill" style="width:${mrr}%;background:${bar(mrr)};"></div></div>
                <span class="eval-value" style="color:${bar(mrr)};">${data.mrr}</span>
              </div>
            </div>
          </div>`;
                chatHistory.appendChild(div);
                scrollChat();
                if (window.lucide) window.lucide.createIcons();

            } catch (err) {
                removeThinking(thinkId);
                appendSystemMsg("⚠️ Could not reach /evaluate. Is the backend running?", "error");
            } finally {
                adminBtn.disabled = false;
                adminBtn.querySelector("span").textContent = "Evaluation Metrics";
            }
        });
    }

    // ══════════════════════════════════════════════════════════════
    // EXPORT CASE REPORT (PDF)
    // ══════════════════════════════════════════════════════════════

    const exportBtn = $("btnExportReport");
    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            if (!currentCaseId) {
                appendSystemMsg("⚠️ No active case to export. Please process evidence first.");
                return;
            }
            generatePDF();
        });
    }

    function generatePDF() {
        const { jsPDF } = window.jspdf || {};
        if (!jsPDF) { alert("PDF library not loaded. Check your internet connection."); return; }

        const doc = new jsPDF({ unit: "mm", format: "a4" });
        const pageW = 210;
        const margin = 16;
        const usableW = pageW - margin * 2;
        let y = 20;

        const addLine = (text, size = 10, bold = false, color = [30, 30, 30]) => {
            doc.setFontSize(size);
            doc.setFont("helvetica", bold ? "bold" : "normal");
            doc.setTextColor(...color);
            const lines = doc.splitTextToSize(text, usableW);
            lines.forEach(line => {
                if (y > 270) { doc.addPage(); y = 20; }
                doc.text(line, margin, y);
                y += size * 0.45;
            });
            y += 2;
        };

        const divider = () => {
            doc.setDrawColor(200, 200, 200);
            doc.line(margin, y, pageW - margin, y);
            y += 5;
        };

        // Header
        doc.setFillColor(6, 9, 17);
        doc.rect(0, 0, 210, 28, "F");
        doc.setTextColor(0, 229, 255);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(16);
        doc.text("ForensicMind — AI Crime Intelligence Report", margin, 13);
        doc.setFontSize(9);
        doc.setTextColor(150, 170, 200);
        doc.text(`Generated: ${new Date().toLocaleString()}`, margin, 20);
        doc.text(`Case ID: ${currentCaseId}`, pageW - margin - 60, 20);
        y = 34;

        // Case Summary
        addLine("CASE SUMMARY", 11, true, [0, 100, 130]);
        divider();
        if (currentCaseMeta) {
            addLine(`Location   : ${currentCaseMeta.location}`);
            addLine(`Year       : ${currentCaseMeta.year}`);
            addLine(`Crime Type : ${currentCaseMeta.crimeType}`);
            addLine(`Evidence   : ${(currentCaseMeta.files || []).join(", ")}`);
        }
        y += 4;

        // Crime Classification
        if (caseReport.crimeType) {
            addLine("CRIME CLASSIFICATION", 11, true, [0, 100, 130]);
            divider();
            addLine(`Type       : ${caseReport.crimeType}`);
            addLine(`Confidence : ${caseReport.confidence}%`);
            y += 4;
        }

        // IPC Sections
        const uniqueIpc = [...new Map(caseReport.ipcSections.map(s => [s.section, s])).values()];
        if (uniqueIpc.length > 0) {
            addLine("RECOMMENDED IPC SECTIONS", 11, true, [0, 100, 130]);
            divider();
            uniqueIpc.forEach(sec => {
                addLine(`IPC §${sec.section} — ${sec.title}`, 10, true);
                addLine(sec.description || "", 9, false, [80, 80, 80]);
                y += 2;
            });
            y += 2;
        }

        // Evidence Sources
        const uniqueSrc = [...new Map(caseReport.sources.map(s => [s.source, s])).values()];
        if (uniqueSrc.length > 0) {
            addLine("EVIDENCE SOURCES", 11, true, [0, 100, 130]);
            divider();
            uniqueSrc.forEach(src => {
                const score = src.score ? ` (${(src.score * 100).toFixed(0)}% match)` : "";
                addLine(`• ${src.source}${score}`, 10, true);
                if (src.chunk_text) addLine(src.chunk_text.substring(0, 200), 9, false, [80, 80, 80]);
                y += 2;
            });
            y += 2;
        }

        // Investigation Q&A
        if (caseReport.messages.length > 0) {
            addLine("INVESTIGATION Q&A", 11, true, [0, 100, 130]);
            divider();
            caseReport.messages.forEach(m => {
                const label = m.role === "user" ? "INVESTIGATOR:" : "ForensicMind:";
                addLine(label, 9, true, m.role === "user" ? [0, 80, 120] : [30, 90, 60]);
                addLine(m.text, 9, false, [40, 40, 40]);
                y += 2;
            });
        }

        // Footer
        const pageCount = doc.internal.getNumberOfPages();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(160, 160, 160);
            doc.text(`ForensicMind AI • Page ${i} of ${pageCount} • CONFIDENTIAL`, margin, 290);
        }

        doc.save(`ForensicMind_Case_${currentCaseId}.pdf`);
        appendSystemMsg("📄 Case report exported as PDF successfully.");
    }

    // ══════════════════════════════════════════════════════════════
    // UI HELPERS
    // ══════════════════════════════════════════════════════════════

    function showCaseBadge(caseId) {
        caseIdBadge.style.display = "flex";
        caseIdText.textContent = caseId.substring(0, 20) + (caseId.length > 20 ? "…" : "");
        if (window.lucide) window.lucide.createIcons();
    }

    function setStatus(text, state = "ready") {
        systemStatus.textContent = text;
        statusBadge.className = `badge badge-status ${state}`;
        if (window.lucide) window.lucide.createIcons();
    }

    function showCaseSummaryCard(meta) {
        if (!caseSummaryCard || !meta) return;
        caseSummaryCard.style.display = "block";
        caseSummaryBody.innerHTML = `
      <div class="cs-item"><strong>📍</strong> ${meta.location}</div>
      <div class="cs-item"><strong>📅</strong> ${meta.year}</div>
      <div class="cs-item"><strong>🏷️</strong> ${meta.crimeType}</div>
      <div class="cs-item"><strong>📁</strong> ${(meta.files || []).map(f => `${fileIcon(f)} ${f}`).join(", ")}</div>`;
    }

    function showSuggestionChips() {
        if (!suggestionChips || chatHasMessages) return;
        if (welcomeState) welcomeState.style.display = "none";
        suggestionChips.style.display = "block";
    }

    function hideSuggestionChips() {
        if (suggestionChips) suggestionChips.style.display = "none";
    }

    function scrollChat() { chatHistory.scrollTop = chatHistory.scrollHeight; }

    function escHtml(str) {
        return (str || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // Init icons after everything is rendered
    if (window.lucide) window.lucide.createIcons();
});