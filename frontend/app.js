const API_BASE = "";

// DOM Elements
const statusText = document.getElementById("statusText");
const chunkCount = document.getElementById("chunkCount");
const modelName = document.getElementById("modelName");
const simThreshold = document.getElementById("simThreshold");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadProgress = document.getElementById("uploadProgress");
const uploadResult = document.getElementById("uploadResult");
const clearBtn = document.getElementById("clearBtn");
const chatHistory = document.getElementById("chatHistory");
const emptyState = document.getElementById("emptyState");
const queryForm = document.getElementById("queryForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");

// Initialize
async function loadStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        if (!res.ok) throw new Error("API unreachable");
        const data = await res.json();
        
        statusText.textContent = "Online";
        chunkCount.textContent = `${data.total_chunks_indexed} chunk(s) indexed`;
        modelName.textContent = data.embedding_model;
        simThreshold.textContent = data.similarity_threshold.toFixed(2);
    } catch (err) {
        statusText.textContent = "Offline / Connecting...";
        chunkCount.textContent = "0 chunks indexed";
    }
}

// Upload Handling
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

async function handleFileUpload(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["txt", "pdf"].includes(ext)) {
        showAlert("Only .txt and .pdf files are supported.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadProgress.classList.remove("hidden");
    uploadResult.classList.add("hidden");

    try {
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData,
        });

        const data = await res.json();

        if (!res.ok) {
            showAlert(data.error || "Failed to process document", "error");
            return;
        }

        showAlert(`Indexed <strong>${file.name}</strong> (${data.chunks_indexed} chunks created).`, "success");
        await loadStatus();
    } catch (err) {
        showAlert(`Upload error: ${err.message}`, "error");
    } finally {
        uploadProgress.classList.add("hidden");
        fileInput.value = "";
    }
}

function showAlert(message, type) {
    uploadResult.innerHTML = message;
    uploadResult.className = `alert alert-${type}`;
    uploadResult.classList.remove("hidden");
}

// Clear Vector Index
clearBtn.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to reset the vector store? All indexed document vectors will be deleted.")) {
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/index`, { method: "DELETE" });
        if (res.ok) {
            showAlert("Vector store index reset successfully.", "success");
            await loadStatus();
        }
    } catch (err) {
        showAlert(`Failed to reset index: ${err.message}`, "error");
    }
});

// Query Handling
queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    submitQuery(query);
});

// Sample Chips
document.querySelectorAll(".query-chip").forEach(chip => {
    chip.addEventListener("click", () => {
        const q = chip.getAttribute("data-q");
        queryInput.value = q;
        submitQuery(q);
    });
});

const QUERY_TIMEOUT_MS = 30000; // 30-second hard client-side timeout

async function submitQuery(query) {
    if (emptyState) emptyState.classList.add("hidden");

    appendMessage("user", query);
    queryInput.value = "";
    queryInput.disabled = true;
    sendBtn.disabled = true;

    const loadingElem = appendLoadingMessage();

    // AbortController for hard client-side timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
        controller.abort();
    }, QUERY_TIMEOUT_MS);

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, top_k: 8 }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        const data = await res.json();
        loadingElem.remove();

        if (!res.ok) {
            appendMessage("assistant", `⚠️ ${data.error || "Failed to process query."}`);
            return;
        }

        appendAssistantResponse(data.answer, data.citations);
    } catch (err) {
        clearTimeout(timeoutId);
        loadingElem.remove();
        if (err.name === "AbortError") {
            appendMessage("assistant", "⏱️ Request timed out after 30 seconds. The model may be busy — please try again.");
        } else {
            appendMessage("assistant", `⚠️ Connection error: ${err.message}`);
        }
    } finally {
        queryInput.disabled = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

function appendMessage(sender, text) {
    const msg = document.createElement("div");
    msg.className = `chat-msg ${sender}`;

    const content = document.createElement("div");
    content.className = "msg-content";
    content.textContent = text;

    msg.appendChild(content);
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msg;
}

function appendLoadingMessage() {
    const msg = document.createElement("div");
    msg.className = "chat-msg assistant";

    const content = document.createElement("div");
    content.className = "msg-content";
    content.innerHTML = `<div class="spinner" style="width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 8px;"></div> DocuMind is analyzing relevant document chunks...`;

    msg.appendChild(content);
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msg;
}

function appendAssistantResponse(answer, citations) {
    const msg = document.createElement("div");
    msg.className = "chat-msg assistant";

    const content = document.createElement("div");
    content.className = "msg-content";

    // Answer text
    const textNode = document.createElement("div");
    textNode.innerHTML = answer.replace(/\n/g, "<br>");
    content.appendChild(textNode);

    // Citation cards with number, filename, score, and snippet
    if (citations && citations.length > 0) {
        const citationPanel = document.createElement("div");
        citationPanel.className = "citations-panel";

        const label = document.createElement("div");
        label.className = "citations-label";
        label.textContent = "Sources:";
        citationPanel.appendChild(label);

        citations.forEach(c => {
            const card = document.createElement("div");
            card.className = "citation-card";

            const pageInfo = c.page_number ? ` · Page ${c.page_number}` : "";
            const header = document.createElement("div");
            header.className = "citation-card-header";
            header.innerHTML = `
                <span class="citation-num">[${c.number || "?"}]</span>
                <span class="citation-file">${c.filename || "Document"}${pageInfo}</span>
                <span class="score">${Math.round(c.similarity_score * 100)}% match</span>
            `;
            card.appendChild(header);

            if (c.snippet) {
                const snippet = document.createElement("div");
                snippet.className = "citation-snippet";
                snippet.textContent = c.snippet;
                card.appendChild(snippet);
            }

            citationPanel.appendChild(card);
        });

        content.appendChild(citationPanel);
    }

    msg.appendChild(content);
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Initial status load
loadStatus();

