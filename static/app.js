// ==================== Tab switching ====================
document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const target = tab.dataset.tab;
        document.getElementById("chat-tab").classList.toggle("hidden", target !== "chat");
        document.getElementById("evals-tab").classList.toggle("hidden", target !== "evals");
        document.getElementById("conversations-tab").classList.toggle("hidden", target !== "conversations");
    });
});

// ==================== Helpers ====================
async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(iso) {
    if (!iso) return "";
    return new Date(iso).toLocaleString();
}

// ==================== Chat ====================
let chatHistory = []; // {role, content} for API
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-chat");
const timestampsToggle = document.getElementById("timestamps-toggle");
const thinkingToggle = document.getElementById("thinking-toggle");

function addBubble(role, text, timestamp) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble chat-bubble-${role}`;
    bubble.textContent = text;
    if (timestamp) {
        const ts = document.createElement("div");
        ts.className = "chat-timestamp";
        ts.textContent = timestamp;
        bubble.appendChild(ts);
    }
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
}

function showTyping() {
    const el = document.createElement("div");
    el.className = "chat-typing";
    el.id = "typing-indicator";
    el.textContent = "Thinking...";
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const useTimestamps = timestampsToggle.checked;
    const useThinking = thinkingToggle.checked;

    // Timestamp the message
    const now = new Date();
    const isoTimestamp = now.toISOString();
    const displayTime = now.toLocaleTimeString();

    // Show user bubble (without the timestamp tag)
    addBubble("user", text, useTimestamps ? displayTime : null);
    chatInput.value = "";
    chatInput.style.height = "auto";

    // Build content for API
    let content = text;
    if (useTimestamps) {
        content = `${text}\n\n[timestamp: ${isoTimestamp}]`;
    }

    chatHistory.push({ role: "user", content: content });

    // Disable input while waiting
    sendBtn.disabled = true;
    chatInput.disabled = true;
    showTyping();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                messages: chatHistory,
                use_timestamps: useTimestamps,
                use_thinking: useThinking,
            }),
        });

        hideTyping();

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
            addBubble("assistant", `Error: ${err.detail || res.statusText}`);
            chatHistory.pop(); // remove failed user message
            return;
        }

        const data = await res.json();
        chatHistory.push({ role: "assistant", content: data.response });
        addBubble("assistant", data.response);
    } catch (err) {
        hideTyping();
        addBubble("assistant", `Connection error: ${err.message}`);
        chatHistory.pop();
    } finally {
        sendBtn.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
}

sendBtn.addEventListener("click", sendMessage);

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});

clearBtn.addEventListener("click", () => {
    chatHistory = [];
    chatMessages.innerHTML = "";
});

// ==================== Evals ====================
async function loadEvals() {
    const container = document.getElementById("evals-list");
    try {
        const evals = await fetchJSON("/api/evals");
        if (!evals.length) {
            container.innerHTML = '<div class="empty-state">No eval runs yet.</div>';
            return;
        }
        container.innerHTML = evals.map(ev => `
            <div class="card">
                <div class="card-header" onclick="toggleEval(this, '${ev.eval_id}')">
                    <div>
                        <span class="tag tag-dataset">${escapeHtml(ev.dataset_name || "unknown")}</span>
                        <span class="tag tag-model">${escapeHtml(ev.model || "")}</span>
                        <span class="tag tag-count">${ev.case_count} cases</span>
                    </div>
                    <div style="color:#999;font-size:0.85rem">${formatTime(ev.timestamp)}</div>
                </div>
                <div class="card-body hidden" id="eval-${ev.eval_id}"></div>
            </div>
        `).join("");
    } catch (err) {
        container.innerHTML = `<div class="empty-state">Evals unavailable (DynamoDB not connected)</div>`;
    }
}

async function toggleEval(header, evalId) {
    const body = document.getElementById(`eval-${evalId}`);
    if (!body.classList.contains("hidden")) { body.classList.add("hidden"); return; }
    body.classList.remove("hidden");
    if (body.dataset.loaded) return;
    body.innerHTML = "Loading...";
    try {
        const results = await fetchJSON(`/api/evals/${evalId}`);
        body.innerHTML = results.map(r => {
            const scoreWith = r.score ? r.score.split("|")[0].split(":")[1] : "";
            const scoreWithout = r.score ? r.score.split("|")[1].split(":")[1] : "";
            let expectedHtml = r.expected_answer ? `<div class="expected">Expected: ${escapeHtml(r.expected_answer)}</div>` : "";
            let scoreHtml = scoreWith ? `<span class="score-${scoreWith}">${scoreWith}</span> / <span class="score-${scoreWithout}">${scoreWithout}</span>` : "";
            return `
                <div class="test-case">
                    <div class="test-case-header">
                        <span class="test-case-id">${escapeHtml(r.test_case_id)}</span>
                        ${scoreHtml}
                    </div>
                    ${expectedHtml}
                    <div class="comparison">
                        <div class="comparison-panel panel-with"><h4>With Timestamps</h4>${escapeHtml(r.response_with_time)}</div>
                        <div class="comparison-panel panel-without"><h4>Without Timestamps</h4>${escapeHtml(r.response_without_time)}</div>
                    </div>
                </div>`;
        }).join("");
        body.dataset.loaded = "true";
    } catch (err) {
        body.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
    }
}

// ==================== Conversations ====================
async function loadConversations() {
    const container = document.getElementById("conversations-list");
    try {
        const sessions = await fetchJSON("/api/conversations");
        if (!sessions.length) {
            container.innerHTML = '<div class="empty-state">No saved conversations.</div>';
            return;
        }
        container.innerHTML = sessions.map(s => `
            <div class="card">
                <div class="card-header" onclick="toggleConversation(this, '${s.session_id}')">
                    <div>
                        <span class="tag tag-model">${escapeHtml(s.model || "")}</span>
                        <span class="tag tag-count">${s.message_count} messages</span>
                        <span class="tag ${s.has_timestamps ? 'tag-dataset' : ''}">${s.has_timestamps ? 'timestamps' : 'no timestamps'}</span>
                    </div>
                    <div style="color:#999;font-size:0.85rem">${formatTime(s.first_timestamp)}</div>
                </div>
                <div class="card-body hidden" id="conv-${s.session_id}"></div>
            </div>
        `).join("");
    } catch (err) {
        container.innerHTML = `<div class="empty-state">Conversations unavailable (DynamoDB not connected)</div>`;
    }
}

async function toggleConversation(header, sessionId) {
    const body = document.getElementById(`conv-${sessionId}`);
    if (!body.classList.contains("hidden")) { body.classList.add("hidden"); return; }
    body.classList.remove("hidden");
    if (body.dataset.loaded) return;
    body.innerHTML = "Loading...";
    try {
        const messages = await fetchJSON(`/api/conversations/${sessionId}`);
        body.innerHTML = messages.map(m => `
            <div class="message message-${m.role}">
                <div class="message-role">${escapeHtml(m.role)}</div>
                ${escapeHtml(m.content)}
            </div>
        `).join("");
        body.dataset.loaded = "true";
    } catch (err) {
        body.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
    }
}

// ==================== Init ====================
loadEvals();
loadConversations();
