# Multimodal RAG — Complete Guide

## What This Project Does

This is an upgraded version of your original text-file RAG chatbot.
The new version accepts **three input types** — PDF, Audio, and Video —
and lets users ask questions about the content in natural language.

All three modes converge into the same RAG pipeline once text is extracted:

```
PDF   → PyMuPDF  → raw text  ─┐
Audio → Whisper  → transcript ─┤→ Chunk → Embed → FAISS → Retrieve → LLM → Answer
Video → moviepy → Whisper   ──┘
```

---

## Project Structure

```
multimodal_rag/
│
├── main.py            # Core RAG pipeline (extraction, chunking, embedding, FAISS, LLM)
├── server.py          # FastAPI server (HTTP routes: /upload, /query, /status)
├── requirements.txt   # Python dependencies
├── .env               # Your API keys (never commit this)
│
└── static/
    └── index.html     # Entire frontend (HTML + CSS + JavaScript in one file)
```

---

## How To Run

### Step 1: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install ffmpeg (needed for video audio extraction)

**Windows:**
- Download from https://ffmpeg.org/download.html
- Extract and add the `bin` folder to your PATH
- Verify: open CMD → `ffmpeg -version`

### Step 3: Set up your API key
Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at: https://console.groq.com

### Step 4: Run the server
```bash
python server.py
```

### Step 5: Open in browser
```
http://localhost:8000
```

---

## Architecture — How Every Piece Works

### 1. Text Extraction (main.py → Section 1)

Before RAG can work, all inputs must become plain text.

**PDF extraction (PyMuPDF):**
- `fitz.open(stream=bytes, filetype="pdf")` opens in memory
- `.load_page(i).get_text()` extracts text from each page
- Pages joined with `\n\n` separators

**Audio transcription (Whisper):**
- Cannot read from memory — needs a real file path
- Write bytes to a temp file → pass path to `whisper_model.transcribe()`
- Returns `{"text": "...", "language": "en", "segments": [...]}`
- Delete temp file in `finally` block (cleanup always runs)

**Video transcription (moviepy + Whisper):**
- `VideoFileClip(path).audio.write_audiofile(path)` extracts audio track
- Saves it as a `.wav` file
- Then calls `extract_text_from_audio()` — reuses audio logic

---

### 2. Chunking (main.py → Section 2)

Long text is split into smaller, overlapping pieces.

**Why chunk?**
- SentenceTransformer has a ~512 token limit per embedding
- Shorter chunks = more focused embeddings = better retrieval accuracy
- Large documents (10,000 words) can't fit in an LLM prompt anyway

**How sliding window works:**
```
Text: [word1 word2 word3 ... word500]

chunk_size=200, overlap=50:
  Chunk 1: words 0-199     (200 words)
  Chunk 2: words 150-349   (start = 200-50 = 150)
  Chunk 3: words 300-499   (start = 350-50 = 300)
```

The overlap means a sentence near a chunk boundary appears in BOTH
adjacent chunks, so its meaning isn't lost.

---

### 3. Embedding (main.py → Section 3)

Converts each text chunk into a 384-dimensional float vector.

**What is an embedding?**
A list of 384 numbers that encodes the MEANING of a sentence.
Sentences with similar meanings have similar vectors (close in space).

```
"What materials were used?"    → [0.12, -0.45, 0.88, ..., 0.33]  (384 numbers)
"List the components needed"   → [0.14, -0.41, 0.85, ..., 0.31]  (very close!)
"What's the weather like?"     → [-0.78, 0.23, -0.11, ..., 0.67] (very different)
```

**Model: all-MiniLM-L6-v2**
- 6-layer MiniLM transformer (very fast, ~80MB)
- Outputs 384-dim vectors
- Good accuracy vs speed tradeoff for RAG on laptop hardware

---

### 4. FAISS Index (main.py → Section 4)

Stores all embeddings and enables fast nearest-neighbor search.

**What is FAISS?**
Facebook AI Similarity Search — a library for finding the K most similar
vectors to a query vector, very quickly (milliseconds even for millions of vectors).

**IndexFlatL2:**
- "Flat" = all vectors stored explicitly (no compression/approximation)
- "L2" = searches by Euclidean distance (lower distance = more similar)
- Exact search — always finds the true nearest neighbors

**How search works:**
```
User query: "What transistors were used?"
  → embed query → [0.22, -0.33, 0.71, ...]
  → FAISS searches all chunk vectors
  → returns 3 chunk IDs with smallest Euclidean distance
  → we look up the text for those IDs
  → that text becomes the "context" for the LLM
```

---

### 5. RAGSession (main.py → Section 5)

A Python class that holds the state between requests.

**Why a class?**
HTTP is stateless — each request knows nothing about previous requests.
But we need to keep the FAISS index and chunk map in memory between
the /upload request and the /query request.

The `RAGSession` object lives at module level in main.py and persists
for the entire lifetime of the server process.

---

### 6. FastAPI Server (server.py)

Exposes the RAG pipeline as HTTP endpoints.

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serves index.html |
| `/upload` | POST | Receives file, triggers ingestion |
| `/query` | POST | Receives question, returns answer |
| `/status` | GET | Returns current session state |

**File upload (multipart/form-data):**
- Browser sends file as binary data + form fields
- `UploadFile` from FastAPI gives us `.filename` and `await .read()`
- We pass the raw bytes directly to `rag_session.ingest()`

**Async/await:**
- FastAPI uses asyncio — all route functions are `async def`
- `await file.read()` reads bytes without blocking the server thread
- Server can handle other requests while waiting for I/O

---

### 7. Frontend (static/index.html)

Single-page application — HTML + CSS + JavaScript in one file.

**Upload flow:**
1. User clicks a card → triggers hidden `<input type="file">`
2. User selects file → `handleUpload()` runs
3. Creates FormData → POSTs to `/upload`
4. Animates progress steps while waiting
5. On success: marks card as loaded, enables chat

**Query flow:**
1. User types question → presses Enter or clicks send button
2. `sendMessage()` adds user bubble to chat
3. Shows typing animation (three bouncing dots)
4. POSTs to `/query` with `{"query": "..."}`
5. Removes typing animation, adds AI answer bubble

**Fetch API:**
```javascript
const res = await fetch('/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: "What materials were used?" })
});
const data = await res.json();   // { answer: "...", source: "...", source_type: "..." }
```

---

## How To Extend This Project

### Add Gemini as an alternative LLM
In `main.py`, import `google.generativeai` and create a second `answer()` method.
Pass a `model_choice` parameter from the frontend to switch between Groq and Gemini.
Your existing `main1.py` already has the Gemini setup — you can reuse it.

### Add persistent storage (ChromaDB)
Replace FAISS with ChromaDB so embeddings survive server restarts.
ChromaDB stores embeddings to disk automatically.

### Add multiple file support
Change `RAGSession` to hold a list of (index, id_to_chunk) pairs.
At retrieval time, search all indexes and merge results.

### Show source chunks in the UI
The `/query` endpoint can return the retrieved chunks alongside the answer.
Display them as collapsible "sources" below each AI message.

### Add language selection
In the prompt template in `rag_session.answer()`, add:
`"Answer in Tamil:"` or `"Answer in English:"` based on a UI dropdown.
Your original `main.py` already answers in Tamil — keep that as an option.