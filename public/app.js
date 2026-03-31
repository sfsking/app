// app.js
import { pipeline, env, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0";
import { Voy } from "https://cdn.jsdelivr.net/npm/voy-search@0.6.3/dist/voy.js";

// Allow local-only model loads
env.allowLocalModels = true;
env.useBrowserCache = true;

// CONFIG
const MODEL_ID = "/static/models"; // served by Express; contains config.json, tokenizer.json, onnx/, etc.
const MAX_POSITION_EMBEDDINGS = 262144; // from config.json
const MAX_NEW_TOKENS = 512; // be conservative to avoid OOM; you can raise on high-memory devices
const TOP_K = 20;
const TOP_P = 0.95;
const TEMPERATURE = 0.6;
const RAG_CHUNK_TOKENS = 512;
const RAG_TOP_K = 5;

// STATE
let state = {
  loggedIn: false,
  webgpuAdapter: null,
  webgpuDevice: null,
  tokenizer: null,
  qwenGenerator: null,
  embedder: null,
  voyIndex: null,
  embedDim: null,
  ragDocs: [], // { id, text, meta }
  pendingImage: null, // { url, name }
  loading: {
    tokenizer: false,
    embeddings: false,
    qwen: false,
  },
};

// DOM ELEMENTS
const els = {
  loginScreen: document.getElementById("login-screen"),
  appScreen: document.getElementById("app-screen"),
  loginForm: document.getElementById("login-form"),
  usernameInput: document.getElementById("username"),
  passwordInput: document.getElementById("password"),
  loginError: document.getElementById("login-error"),
  chatContainer: document.getElementById("chat-container"),
  userInput: document.getElementById("user-input"),
  sendBtn: document.getElementById("send-btn"),
  imageInput: document.getElementById("image-input"),
  docInput: document.getElementById("doc-input"),
  resetIndexBtn: document.getElementById("reset-index-btn"),
  infoAdapter: document.getElementById("info-adapter"),
  infoArch: document.getElementById("info-arch"),
  infoVram: document.getElementById("info-vram"),
  progressTokenizerPercent: document.getElementById("progress-tokenizer-percent"),
  progressTokenizerBar: document.getElementById("progress-tokenizer-bar"),
  progressEmbeddingsPercent: document.getElementById("progress-embeddings-percent"),
  progressEmbeddingsBar: document.getElementById("progress-embeddings-bar"),
  progressQwenPercent: document.getElementById("progress-qwen-percent"),
  progressQwenBar: document.getElementById("progress-qwen-bar"),
  ragChunks: document.getElementById("rag-chunks"),
  ragDim: document.getElementById("rag-dim"),
};

// AUTH (hardcoded as requested)
els.loginForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const u = els.usernameInput.value.trim();
  const p = els.passwordInput.value.trim();
  if (u === "sfsking" && p === "jericho120") {
    state.loggedIn = true;
    els.loginScreen.classList.add("hidden");
    els.appScreen.classList.remove("hidden");
    initWebGPUAndModels();
  } else {
    els.loginError.textContent = "Invalid credentials.";
  }
});

// WEBGPU INFO & OPTIONAL VRAM TRACKING
async function initWebGPU() {
  if (!navigator.gpu) {
    els.infoAdapter.textContent = "WebGPU not supported";
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    els.infoAdapter.textContent = "No GPU adapter";
    return false;
  }
  const device = await adapter.requestDevice();
  state.webgpuAdapter = adapter;
  state.webgpuDevice = device;

  const info = adapter.info || {};
  els.infoAdapter.textContent = (info.vendor || "") + " " + (info.architecture || "unknown");
  els.infoArch.textContent = info.description || info.device || "-";

  // Optional: integrate webgpu-memory for VRAM estimate
  try {
    if (window.webgpuMemory && typeof window.webgpuMemory.getWebGPUMemoryUsage === "function") {
      const vramUsage = window.webgpuMemory.getWebGPUMemoryUsage(device);
      els.infoVram.textContent = (vramUsage / (1024 * 1024)).toFixed(1) + " MB";
    }
  } catch {
    els.infoVram.textContent = "N/A";
  }
  return true;
}

// PROGRESS BARS
function updateProgress(key, percent) {
  const percentEl = els[`progress${key}Percent`];
  const barEl = els[`progress${key}Bar`];
  if (percentEl && barEl) {
    percentEl.textContent = Math.round(percent) + "%";
    barEl.style.width = Math.min(100, Math.max(0, percent)) + "%";
  }
}

// LOAD MODELS
async function initWebGPUAndModels() {
  const ok = await initWebGPU();
  if (!ok) {
    appendMessage("system", "WebGPU is not available in this browser. Please enable it and reload.");
    return;
  }

  try {
    state.loading.tokenizer = true;
    // We'll use the text-generation pipeline to get tokenizer & model.
    // Transformers.js v3 supports device:'webgpu' and dtype.
    state.loading.qwen = true;
    state.qwenGenerator = await pipeline(
      "text-generation",
      MODEL_ID,
      {
        device: "webgpu",
        dtype: "q4f16", // matches decoder_model_merged_q4f16.onnx
        progress_callback: (progress) => {
          if (progress.status === "downloading") {
            const pct = (progress.progress || 0) * 100;
            // Heuristic: large onnx files -> qwen bar; small ones -> tokenizer
            if (progress.file && progress.file.includes("decoder")) {
              updateProgress("Qwen", pct);
            } else {
              updateProgress("Tokenizer", pct);
            }
          }
        },
      },
    );
    state.tokenizer = state.qwenGenerator.tokenizer;
    state.loading.tokenizer = false;
    state.loading.qwen = false;
    updateProgress("Tokenizer", 100);
    updateProgress("Qwen", 100);

    // Embeddings model (feature-extraction). Choose a lightweight model for browser.
    state.loading.embeddings = true;
    state.embedder = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
      {
        device: "webgpu",
        dtype: "q8",
        progress_callback: (progress) => {
          if (progress.status === "downloading") {
            const pct = (progress.progress || 0) * 100;
            updateProgress("Embeddings", pct);
          }
        },
      },
    );
    state.loading.embeddings = false;
    updateProgress("Embeddings", 100);

    // Initialize a Voy index (empty at first)
    state.voyIndex = new Voy({ embeddings: [] });
    updateRagStats();

    appendMessage("system", "Models loaded (WebGPU). You can now chat, upload images, or add documents for RAG.");
  } catch (err) {
    console.error(err);
    appendMessage("system", "Error loading models: " + err.message);
  }
}

// RAG CHUNKING (simple recursive by characters)
function chunkText(text, maxChunkChars = 1000, overlap = 150) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + maxChunkChars, text.length);
    if (end < text.length) {
      const lastSpace = text.lastIndexOf(" ", end);
      if (lastSpace > start + maxChunkChars / 2) end = lastSpace;
    }
    chunks.push(text.slice(start, end).trim());
    start = end - overlap;
    if (start < 0) start = 0;
  }
  return chunks.filter(Boolean);
}

// INDEX DOCUMENTS
async function indexDocuments(files) {
  if (!state.embedder || !state.voyIndex) {
    appendMessage("system", "Embeddings model not ready yet.");
    return;
  }

  appendMessage("system", `Indexing ${files.length} document(s)…`);

  for (const file of files) {
    const text = await file.text();
    const chunks = chunkText(text);
    const toEmbed = chunks.map((c) => c.replace(/\s+/g, " ").trim());

    const outputs = await state.embedder(toEmbed, { pooling: "mean", normalize: true });
    const embeddings = outputs.tolist ? outputs.tolist() : Array.from(outputs);

    if (!state.embedDim && Array.isArray(embeddings[0])) {
      state.embedDim = embeddings[0].length;
    }

    const items = embeddings.map((vec, i) => ({
      id: String(state.ragDocs.length + i),
      text: chunks[i],
      meta: { file: file.name, chunkIndex: i },
      embeddings: vec,
    }));

    for (const item of items) {
      state.ragDocs.push(item);
    }

    // Rebuild Voy index with updated embeddings
    const resource = { embeddings: state.ragDocs };
    state.voyIndex = new Voy(resource);
  }

  updateRagStats();
  appendMessage("system", `Indexing complete. ${state.ragDocs.length} chunks stored in local Voy index.`);
}

function updateRagStats() {
  els.ragChunks.textContent = state.ragDocs.length;
  els.ragDim.textContent = state.embedDim != null ? String(state.embedDim) : "–";
}

// RAG RETRIEVAL
async function retrieveContext(query, topK = RAG_TOP_K) {
  if (!state.embedder || !state.voyIndex || state.ragDocs.length === 0) return [];
  const qEmb = await state.embedder(query, { pooling: "mean", normalize: true });
  const qVec = qEmb.tolist ? qEmb.tolist()[0] : Array.from(qEmb)[0];
  const result = state.voyIndex.search(qVec, topK);
  const hits = (result.neighbors || []).map((n) => state.ragDocs[Number(n.id)]);
  return hits.filter(Boolean);
}

// IMAGE HANDLING
els.imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  state.pendingImage = { url, name: file.name };
  appendMessage("user", `[Image: ${file.name}]`, url);
  // Reset input
  els.imageInput.value = "";
});

// DOCUMENT UPLOAD FOR RAG
els.docInput.addEventListener("change", (e) => {
  const files = Array.from(e.target.files);
  if (files.length === 0) return;
  indexDocuments(files);
  els.docInput.value = "";
});

// RESET INDEX & MODELS
els.resetIndexBtn.addEventListener("click", async () => {
  state.ragDocs = [];
  state.voyIndex = new Voy({ embeddings: [] });
  state.embedDim = null;
  updateRagStats();
  appendMessage("system", "RAG index cleared. Models remain loaded.");
});

// BUILD MESSAGES WITH CHAT TEMPLATE & RAG CONTEXT
async function buildMessages(userText, includeImage = false) {
  const messages = [
    {
      role: "system",
      content: "You are Qwen3.5, a helpful multimodal assistant with access to retrieved context via RAG. Use the <think> tags to show your reasoning, then answer concisely.",
    },
  ];

  // Retrieve RAG context if any
  const hits = await retrieveContext(userText);
  let contextText = "";
  if (hits.length > 0) {
    contextText =
      "Below are some relevant excerpts from your knowledge base:\n" +
      hits
        .map(
          (d, i) =>
            `[${i + 1}] ${d.meta.file || "(unknown)"} — chunk ${d.meta.chunkIndex}\n${d.text}`,
        )
        .join("\n\n");
    messages.push({
      role: "system",
      content: "Use the following context when answering. If it doesn't help, say so.\n\n" + contextText,
    });
  }

  const userContent = includeImage && state.pendingImage
    ? [
        { type: "image", image: state.pendingImage.url },
        { type: "text", text: userText },
      ]
    : userText;

  messages.push({ role: "user", content: userContent });

  // Include past assistant messages (not full history to stay within context)
  // TODO: optionally maintain a small rolling history window here.

  return messages;
}

// RENDER MESSAGES
function appendMessage(role, text, imageUrl = null) {
  const div = document.createElement("div");
  div.className = `message max-w-3xl mx-auto ${role === "user" ? "text-right" : "text-left"}`;
  const bubble = document.createElement("div");
  bubble.className =
    "inline-block rounded-xl px-4 py-2.5 text-sm leading-relaxed text-left break-words " +
    (role === "user"
      ? "bg-accent text-slate-900 rounded-tr-sm"
      : role === "system"
      ? "bg-slate-800 text-slate-300 border border-slate-700"
      : "bg-slate-800 border border-slate-700 text-slate-200 rounded-tl-sm");

  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.className = "rounded-lg mb-2 max-h-64 object-contain bg-slate-900/50";
    img.alt = "user-uploaded";
    bubble.appendChild(img);
  }

  if (role === "assistant" || role === "system") {
    // Render <think> tags as collapsible, then Markdown for the rest
    const html = renderThinkAndMarkdown(text);
    bubble.innerHTML = html;
    bubble.querySelectorAll(".think-summary").forEach((el) => {
      el.parentElement.addEventListener("click", () => {
        el.parentElement.classList.toggle("open");
      });
    });
  } else {
    bubble.textContent = text;
  }

  div.appendChild(bubble);
  els.chatContainer.appendChild(div);
  els.chatContainer.scrollTop = els.chatContainer.scrollHeight;

  if (role === "assistant" || role === "system") {
    bubble.querySelectorAll("pre code").forEach((block) => {
      if (window.hljs) hljs.highlightElement(block);
    });
  }
}

function renderThinkAndMarkdown(text) {
  // Detect <think>...</think> blocks
  const parts = [];
  let lastIndex = 0;
  const regex = /
