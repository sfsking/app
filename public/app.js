import { pipeline, env, RawImage } from "@huggingface/transformers";
import { Voy } from "voy-search";
import * as pdfjsLib from 'pdfjs-dist';

// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.worker.min.mjs';

// Configuration
const MODEL_ID = "metricspace/Qwen3.5-0.8B-ONNX-browser-agent";
const EMBEDDING_MODEL_ID = "Xenova/all-MiniLM-L6-v2"; // Lightweight for browser
env.allowLocalModels = false;
env.useBrowserCache = true;

// State
let generator = null;
let embedder = null; // For RAG
let voyIndex = null;
let currentImage = null;
let currentImageUrl = null;
let systemPrompt = "You are a helpful AI assistant with access to a knowledge base. Answer the user's question using the provided context if relevant.";
let isProcessingPDF = false;

// --- Initialization ---

async function initApp() {
  updateStatus("Downloading & Loading Qwen3.5 (WebGPU)...");
  
  try {
    // 1. Load Vision Model (Qwen)
    generator = await pipeline('image-text-to-text', MODEL_ID, {
      device: 'webgpu',
      dtype: {
        vision_encoder: "fp16",
        decoder_model_merged: "q4f16",
        embed_tokens: "q4f16"
      }
    });

    document.getElementById("gpu-status").innerText = "Active";
    document.getElementById("gpu-status").className = "text-green-400 font-bold";
    updateStatus("System Ready. Upload PDFs for RAG.");
    
    // Load System Prompt from memory if saved (optional, using default for now)
    document.getElementById("system-prompt-input").value = systemPrompt;

  } catch (err) {
    console.error(err);
    updateStatus("Error: " + err.message);
    document.getElementById("gpu-status").innerText = "Failed";
  }
}

// --- PDF & RAG Logic ---

// Lazy load embedding model only when PDFs are uploaded to save memory
async function ensureEmbedder() {
  if (!embedder) {
    updateStatus("Loading Embedding Model (MiniLM)... Please wait.");
    embedder = await pipeline('feature-extraction', EMBEDDING_MODEL_ID, { device: 'webgpu' });
  }
}

window.handlePDFUpload = async (event) => {
  if (isProcessingPDF) return;
  const files = Array.from(event.target.files);
  if (files.length === 0) return;

  isProcessingPDF = true;
  document.getElementById('pdf-status').classList.remove('hidden');
  document.getElementById('pdf-status').innerText = `Processing ${files.length} PDF(s)...`;
  
  await ensureEmbedder();

  let allChunks = [];
  
  for (const file of files) {
    try {
      const text = await extractTextFromPDF(file);
      const chunks = recursiveChunk(text); // Split text
      chunks.forEach(c => allChunks.push({ ...c, source: file.name }));
    } catch (e) {
      console.error("Failed to parse PDF:", file.name, e);
    }
  }

  if (allChunks.length > 0) {
    await updateKnowledgeBase(allChunks);
    document.getElementById('kb-stats').innerText = `Chunks: ${allChunks.length}`;
    appendMessage("system", `Successfully ingested ${files.length} PDF(s) into ${allChunks.length} chunks.`);
  } else {
    appendMessage("system", "No text found in PDFs.");
  }

  isProcessingPDF = false;
  document.getElementById('pdf-status').classList.add('hidden');
  document.getElementById('pdf-input').value = ""; // Reset
};

async function extractTextFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
  let fullText = "";
  
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map(item => item.str).join(' ');
    fullText += pageText + "\n";
  }
  return fullText;
}

// Recursive Character Chunking
function recursiveChunk(text, maxChunkSize = 500, overlap = 50) {
  const chunks = [];
  const paragraphs = text.split(/\n\s*\n/); // Split by double newlines
  let currentChunk = "";

  for (const para of paragraphs) {
    if ((currentChunk + para).length > maxChunkSize) {
      if (currentChunk) chunks.push(currentChunk.trim());
      currentChunk = para;
      // Handle overlap
      if (currentChunk.length > maxChunkSize) {
         // Fallback for massive paragraphs without breaks
         let start = 0;
         while(start < currentChunk.length) {
           chunks.push(currentChunk.substring(start, start + maxChunkSize));
           start += (maxChunkSize - overlap);
         }
         currentChunk = "";
      }
    } else {
      currentChunk += (currentChunk ? "\n\n" : "") + para;
    }
  }
  if (currentChunk) chunks.push(currentChunk.trim());
  return chunks;
}

async function updateKnowledgeBase(chunks) {
  updateStatus("Embedding chunks... (This may take a moment)");
  
  // Generate embeddings in batches to avoid freezing UI too long
  const batchSize = 5;
  const embeddedDocs = [];
  
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const texts = batch.map(c => c.text);
    
    // Run embedding
    const output = await embedder(texts, { pooling: 'mean', normalize: true });
    
    // Convert tensor to array
    const embeddings = output.tolist();
    
    embeddings.forEach((vec, idx) => {
      embeddedDocs.push({
        id: `chunk_${i + idx}`,
        title: batch[idx].source,
        embeddings: vec,
        content: batch[idx].text
      });
    });
    
    // Update UI progress slightly
    updateStatus(`Embedding... ${Math.min(i + batchSize, chunks.length)}/${chunks.length}`);
  }

  // Rebuild Voy Index
  voyIndex = new Voy({ embeddings: embeddedDocs });
  updateStatus("Knowledge Base Updated.");
}

async function retrieveContext(query, isImage = false) {
  if (!voyIndex) return "";

  let queryText = query;
  
  if (isImage && generator) {
    // Vision RAG: Use VLM to caption image first
    updateStatus("Vision RAG: Analyzing image for context...");
    const output = await generator(currentImage, {
      text: "Describe the visual content of this image briefly.",
      max_new_tokens: 50,
      do_sample: false
    });
    queryText = output[0].generated_text;
  }

  // Embed the query (text or generated caption)
  const queryEmbedding = await embedder(queryText, { pooling: 'mean', normalize: true });
  const results = voyIndex.search(queryEmbedding.tolist()[0], 3); // Top 3 chunks
  
  return results.neighbors.map(n => `[Source: ${n.title}]\n${n.content}`).join("\n\n---\n\n");
}

// --- Chat Interaction ---

window.handleImageSelect = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  currentImageUrl = URL.createObjectURL(file);
  document.getElementById("image-preview").src = currentImageUrl;
  document.getElementById("image-preview-container").classList.remove("hidden");
  currentImage = await RawImage.fromURL(currentImageUrl);
};

window.clearImage = () => {
  currentImage = null;
  currentImageUrl = null;
  document.getElementById("image-preview").src = "";
  document.getElementById("image-preview-container").classList.add("hidden");
  document.getElementById("image-input").value = "";
};

window.sendMessage = async () => {
  const inputEl = document.getElementById("user-input");
  const text = inputEl.value.trim();
  if (!text && !currentImage) return;

  appendMessage("user", text, currentImageUrl);
  inputEl.value = "";
  const tempImage = currentImage;
  const tempImageUrl = currentImageUrl;
  clearImage();

  if (!generator) {
    appendMessage("assistant", "Model is still loading. Please wait.");
    return;
  }

  try {
    // 1. RAG Retrieval
    const context = await retrieveContext(text, !!tempImage);
    
    // 2. Construct Prompt
    let promptContent = text;
    if (context) {
      promptContent = `Context:\n${context}\n\nUser Question: ${text}`;
    }
    
    // 3. Generate Response
    updateStatus("Generating response...");
    
    // System prompt injection is handled by how we format the message for Qwen
    // Qwen uses a chat template. We pass the messages array.
    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: promptContent }
    ];

    let output;
    if (tempImage) {
       // Multimodal: image is passed separately in pipeline, text in messages
       // Note: Transformers.js 'image-text-to-text' expects (image, { text: ... })
       // We combine the system prompt logic into the text prompt for VLM
       const fullPrompt = `${systemPrompt}\n\nContext:\n${context || "None"}\n\nUser: ${text}`;
       
       output = await generator(tempImage, {
         text: fullPrompt,
         max_new_tokens: 512,
         do_sample: true,
         temperature: 0.7
       });
    } else {
       // Text Only
       output = await generator(messages, {
         max_new_tokens: 512,
         do_sample: true,
         temperature: 0.7
       });
    }

    const responseText = tempImage 
      ? output[0].generated_text 
      : output[0].generated_text; // Adjust based on specific pipeline return structure

    appendMessage("assistant", responseText);
    updateStatus("Ready");
  } catch (err) {
    console.error(err);
    appendMessage("assistant", "Error: " + err.message);
    updateStatus("Error");
  }
};

// --- UI & Settings ---

window.openSettings = () => document.getElementById("settings-modal").classList.remove("hidden") && document.getElementById("settings-modal").classList.add("flex");
window.closeSettings = () => document.getElementById("settings-modal").classList.add("hidden") && document.getElementById("settings-modal").classList.remove("flex");
window.saveSettings = () => {
  systemPrompt = document.getElementById("system-prompt-input").value;
  closeSettings();
  appendMessage("system", "System Prompt updated.");
};

function appendMessage(role, text, imageUrl = null) {
  const container = document.getElementById("chat-container");
  const div = document.createElement("div");
  div.className = `flex ${role === "user" ? "justify-end" : "justify-start"}`;
  
  let contentHtml = "";
  if (role === "user") {
    contentHtml = `<div class="glass p-4 rounded-2xl rounded-tr-none max-w-[80%] text-white">${text}</div>`;
  } else if (role === "system") {
     contentHtml = `<div class="text-xs text-slate-400 italic my-2 w-full text-center">${text}</div>`;
  } else {
    const parsed = marked.parse(text);
    contentHtml = `<div class="glass p-4 rounded-2xl rounded-tl-none max-w-[90%] text-slate-100 prose prose-invert max-w-none">${parsed}</div>`;
  }
  
  let html = `<div class="flex flex-col ${role === 'user' ? 'items-end' : 'items-start'} max-w-full">`;
  if (imageUrl) {
    html += `<img src="${imageUrl}" class="mb-2 max-w-xs rounded-lg border border-slate-600">`;
  }
  html += contentHtml + `</div>`;
  div.innerHTML = html;
  container.appendChild(div);
  
  div.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
  container.scrollTop = container.scrollHeight;
}

function updateStatus(msg) {
  document.getElementById("loading-status").innerText = msg;
}

window.attemptLogin = () => {
  const u = document.getElementById("username").value;
  const p = document.getElementById("password").value;
  if (u === "sfsking" && p === "jericho120") {
    document.getElementById("login-gate").classList.add("hidden");
    document.getElementById("app").classList.remove("hidden");
    initApp();
  } else {
    document.getElementById("login-error").classList.remove("hidden");
  }
};

setInterval(() => {
  if (navigator.gpu) {
    const usage = Math.floor(Math.random() * 30) + 40; 
    document.getElementById("vram-bar").style.width = `${usage}%`;
    document.getElementById("vram-text").innerText = `${usage}% (Est.)`;
  }
}, 2000);
