"""
===============================================================
  MULTIMODAL RAG — server.py
  FastAPI backend: HTTP routes connecting the UI to main.py
===============================================================
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import rag_session


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_EXT_TO_SOURCE_TYPE: dict[str, str] = {
    ".pdf":  "pdf",
    ".mp3":  "audio", ".wav":  "audio", ".m4a": "audio",
    ".ogg":  "audio", ".flac": "audio", ".aac": "audio", ".wma": "audio",
    ".mp4":  "video", ".mkv":  "video", ".avi": "video",
    ".mov":  "video", ".webm": "video", ".flv": "video", ".wmv": "video",
}


def detect_source_type(filename: str) -> str | None:
    """Derive source_type from file extension. Returns None if unrecognised."""
    ext = os.path.splitext(filename)[1].lower()
    return _EXT_TO_SOURCE_TYPE.get(ext)


# ═══════════════════════════════════════════════════════════════════════════════
#  APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Multimodal RAG API",
    description="Upload PDF, Audio, or Video and ask questions about the content.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_type: str = Form(None),   # optional override; auto-detected if absent
):
    """
    Receive an uploaded file, transcribe / parse it, build the FAISS index.

    WHY Form(None) AND NOT A PLAIN DEFAULT:
      When a route declares File(...), FastAPI treats the whole request body as
      multipart/form-data.  Any additional parameter declared WITHOUT Form(...)
      is expected on the query string instead — so a frontend that sends
      source_type as a form field is silently ignored and the value falls back
      to the default (which was "pdf" in the original code, breaking every
      video and audio upload).

      Form(None) tells FastAPI to read source_type from the multipart body.
      If the frontend omits it, we fall back to auto-detection from the
      file extension, so the frontend never needs to send it at all.
    """
    resolved = source_type or detect_source_type(file.filename or "")

    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot detect file type for '{file.filename}'. "
                "Supported: .pdf | .mp3 .wav .m4a .ogg .flac .aac .wma "
                "| .mp4 .mkv .avi .mov .webm .flv .wmv"
            ),
        )

    if resolved not in {"pdf", "audio", "video"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type '{resolved}'. Must be pdf, audio, or video.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        rag_session.ingest(
            file_bytes=file_bytes,
            filename=file.filename,
            source_type=resolved,
        )
        return JSONResponse({
            "message":       "File processed successfully",
            "filename":      file.filename,
            "source_type":   resolved,
            "chunks_indexed": rag_session.index.ntotal,
        })

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/query")
async def query_endpoint(body: QueryRequest):
    """Run the RAG pipeline and return an answer."""
    if rag_session.index is None:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded yet. Please upload a PDF, audio, or video first.",
        )

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = rag_session.answer(query)
        return JSONResponse({
            "answer":      answer,
            "source":      rag_session.source_name,
            "source_type": rag_session.source_type,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/status")
async def get_status():
    """Return whether a document is loaded and ready."""
    if rag_session.index is None:
        return {"ready": False, "source": None, "source_type": None, "chunks": 0}

    return {
        "ready":       True,
        "source":      rag_session.source_name,
        "source_type": rag_session.source_type,
        "chunks":      rag_session.index.ntotal,
    }


@app.get("/transcript")
async def get_transcript():
    """
    Return the full raw transcript extracted from the uploaded file.

    USE THIS TO DEBUG transcription quality.
    If the transcript here is empty or garbled, the problem is in the audio
    itself (silent video, unsupported language, severe background noise) rather
    than in the RAG pipeline.

    RESPONSE:
      { "source": "lecture.mp4",
        "source_type": "video",
        "word_count": 1842,
        "transcript": "Today we are going to talk about …" }
    """
    if rag_session.transcript is None:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded yet.",
        )

    return JSONResponse({
        "source":      rag_session.source_name,
        "source_type": rag_session.source_type,
        "word_count":  len(rag_session.transcript.split()),
        "transcript":  rag_session.transcript,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)