"""
===============================================================
  MULTIMODAL RAG — main.py
  Core pipeline: ingest → process → embed → index → retrieve → generate

  NOW SUPPORTS VISUAL ANALYSIS:
    Videos are analysed both by:
      1. Audio transcription  (Groq Whisper large-v3)
      2. Visual frame analysis (Groq llama-4-scout vision LLM)

    Key frames are extracted at regular intervals via ffmpeg,
    encoded as base64 JPEG, and described by the vision model.
    The visual descriptions are merged with the audio transcript
    before chunking — so questions about on-screen text, slides,
    diagrams, gestures, scenes, or anything visual are answerable.
===============================================================

TRANSCRIPTION ENGINE:
  Audio and video transcription uses Groq's Whisper API
  (model: whisper-large-v3) instead of a local Whisper model.

  WHY GROQ WHISPER INSTEAD OF LOCAL WHISPER-BASE:
    - whisper-large-v3 is dramatically more accurate on real-world audio
      (background noise, accents, low volume, fast speech, technical terms).
    - whisper-base — the smallest variant — frequently returns empty strings,
      hallucinated repetitions, or near-silence on anything that isn't clean
      studio-quality speech.  That is why the chat showed only the filename.
    - No local GPU computation required; Groq inference is faster than CPU.
    - The same Groq client already used for the LLM handles transcription too
      — no new credentials or packages needed.

  FILE SIZE HANDLING:
    Groq's audio endpoint accepts files up to 25 MB.
    A 16 kHz mono WAV (as extracted by ffmpeg) uses ~1.9 MB per minute.
    Videos longer than ~12 minutes produce a WAV over 25 MB.
    In that case ffmpeg splits the WAV into ≤10-minute segments, each segment
    is transcribed individually, and the results are joined in order.
===============================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import glob
import base64
import shutil
import tempfile
import subprocess

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import faiss
import fitz                                            # PyMuPDF
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
# One Groq client handles both LLM chat completions and Whisper transcription.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Embedding model ───────────────────────────────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Groq audio limits ─────────────────────────────────────────────────────────
GROQ_AUDIO_LIMIT_BYTES = 24 * 1024 * 1024   # 24 MB (1 MB below the 25 MB ceiling)
SEGMENT_SECONDS        = 600                 # split into 10-minute chunks if oversized

# ── Visual frame extraction settings ─────────────────────────────────────────
# Extract 1 frame every N seconds for visual analysis.
# Lower = more detail but more Groq API calls and slower ingestion.
# For a 5-min lecture video, FRAME_INTERVAL=10 gives ~30 frames — a good balance.
FRAME_INTERVAL         = 10   # seconds between sampled frames
MAX_FRAMES             = 60   # hard cap to avoid runaway API usage on very long videos

# Vision model: llama-4-scout supports image inputs on Groq
VISION_MODEL           = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Cross-platform ffmpeg detection ───────────────────────────────────────────
_ffmpeg_env = os.getenv("FFMPEG_PATH")
FFMPEG_PATH = (
    _ffmpeg_env
    if _ffmpeg_env and os.path.isfile(_ffmpeg_env)
    else shutil.which("ffmpeg") or "ffmpeg"
)

_ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
if _ffmpeg_dir and _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

if not shutil.which("ffmpeg") and not os.path.isfile(FFMPEG_PATH):
    raise EnvironmentError(
        "ffmpeg not found. Install it and add it to your PATH, "
        "or set FFMPEG_PATH=/full/path/to/ffmpeg in your .env file."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: TRANSCRIPTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ffmpeg(*args: str) -> None:
    """
    Run ffmpeg with the given arguments.
    Raises RuntimeError with ffmpeg's own stderr message on failure.
    """
    result = subprocess.run(
        [FFMPEG_PATH, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n" + result.stderr.decode("utf-8", errors="ignore")
        )


def _run_ffmpeg_output(args: list[str]) -> str:
    """Run ffmpeg/ffprobe and return its stderr output (where stream info lives)."""
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stderr.decode("utf-8", errors="ignore")


def _check_audio_stream(vid_path: str) -> bool:
    """Return True if the video contains at least one audio stream."""
    output = _run_ffmpeg_output([FFMPEG_PATH, "-i", vid_path])
    return "Audio:" in output


def _get_video_duration(vid_path: str) -> float:
    """
    Return video duration in seconds by parsing ffmpeg's stderr output.
    Returns 0.0 if duration cannot be determined.
    """
    output = _run_ffmpeg_output([FFMPEG_PATH, "-i", vid_path])
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", output)
    if not match:
        return 0.0
    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return h * 3600 + m * 60 + s


def _call_groq_whisper(audio_bytes: bytes, display_name: str, language: str = "en") -> str:
    """
    Send audio bytes to Groq's whisper-large-v3 endpoint.
    response_format="text" returns the transcript as a plain Python string.

    language="en" is forced to avoid auto-detect failures on short clips.
    Remove or set to None if your content is not English.
    """
    result = client.audio.transcriptions.create(
        file=(display_name, audio_bytes),
        model="whisper-large-v3",
        response_format="text",
        language=language,
    )
    # Groq returns a str when response_format="text"
    return result if isinstance(result, str) else result.text


def _transcribe_wav(wav_path: str, display_name: str) -> str:
    """
    Transcribe a WAV file (16 kHz, mono, PCM) via Groq Whisper.

    Files within 24 MB are sent in a single request.
    Larger files are split by ffmpeg into ≤10-minute segments, each
    transcribed separately, then concatenated.

    PARAMETERS
      wav_path     : absolute path to the WAV file on disk
      display_name : filename shown to the Groq API (for format detection)

    RETURNS
      Full transcript as a single string.
    """
    size = os.path.getsize(wav_path)

    if size <= GROQ_AUDIO_LIMIT_BYTES:
        with open(wav_path, "rb") as f:
            return _call_groq_whisper(f.read(), display_name)

    # ── Oversized: split into segments ───────────────────────────────────────
    print(f"WAV is {size / 1024 / 1024:.1f} MB — splitting into "
          f"{SEGMENT_SECONDS // 60}-min segments …")

    seg_dir     = tempfile.mkdtemp()
    seg_pattern = os.path.join(seg_dir, "seg_%04d.wav")

    try:
        _run_ffmpeg(
            "-y", "-i", wav_path,
            "-f", "segment",
            "-segment_time", str(SEGMENT_SECONDS),
            "-c", "copy",          # copy PCM stream; no re-encode needed
            seg_pattern,
        )

        seg_files = sorted(glob.glob(os.path.join(seg_dir, "seg_*.wav")))
        if not seg_files:
            raise RuntimeError("ffmpeg produced no segment files.")

        parts = []
        for i, seg in enumerate(seg_files):
            print(f"  Segment {i + 1}/{len(seg_files)} …")
            with open(seg, "rb") as f:
                parts.append(_call_groq_whisper(f.read(), f"seg_{i:04d}.wav"))

        return " ".join(parts)

    finally:
        for seg in glob.glob(os.path.join(seg_dir, "*.wav")):
            try:
                os.remove(seg)
            except OSError:
                pass
        try:
            os.rmdir(seg_dir)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1B: VISUAL FRAME ANALYSIS  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_frames(vid_path: str, interval: int = FRAME_INTERVAL,
                    max_frames: int = MAX_FRAMES) -> list[tuple[float, bytes]]:
    """
    Extract JPEG frames from a video at regular intervals using ffmpeg.

    STRATEGY:
      ffmpeg -vf fps=1/N selects 1 frame every N seconds.
      Frames are output as individual JPEG files in a temp directory,
      then loaded into memory as bytes and cleaned up.

    RETURNS
      List of (timestamp_seconds, jpeg_bytes) tuples, capped at max_frames.
      If the video has no video stream or is very short, returns [].
    """
    duration = _get_video_duration(vid_path)
    if duration < 1.0:
        print("Video too short for frame extraction.")
        return []

    # Compute actual number of frames that will be extracted
    expected_frames = int(duration / interval)
    if expected_frames == 0:
        expected_frames = 1

    # Adjust interval upward if we'd exceed max_frames
    actual_interval = interval
    if expected_frames > max_frames:
        actual_interval = int(duration / max_frames)
        print(f"Video is long ({duration:.0f}s) — adjusting frame interval "
              f"to {actual_interval}s to stay under {max_frames} frames.")

    frame_dir = tempfile.mkdtemp()
    frame_pattern = os.path.join(frame_dir, "frame_%04d.jpg")

    try:
        _run_ffmpeg(
            "-y", "-i", vid_path,
            "-vf", f"fps=1/{actual_interval}",
            "-q:v", "3",           # JPEG quality (2=best, 31=worst)
            "-vframes", str(max_frames),
            frame_pattern,
        )

        frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
        results = []

        for i, fpath in enumerate(frame_files[:max_frames]):
            timestamp = i * actual_interval
            with open(fpath, "rb") as f:
                results.append((float(timestamp), f.read()))

        print(f"Extracted {len(results)} frames at {actual_interval}s intervals.")
        return results

    except RuntimeError as e:
        # If video has no video stream (e.g. audio-only mp4), gracefully return []
        print(f"Frame extraction skipped: {e}")
        return []

    finally:
        for fpath in glob.glob(os.path.join(frame_dir, "*.jpg")):
            try:
                os.remove(fpath)
            except OSError:
                pass
        try:
            os.rmdir(frame_dir)
        except OSError:
            pass


def _describe_frame(jpeg_bytes: bytes, timestamp: float, filename: str) -> str:
    """
    Send a single JPEG frame to Groq's vision LLM and get a detailed description.

    The prompt asks the model to describe everything visible: text, diagrams,
    people, objects, UI elements, slides, scenes — so that the description
    becomes rich RAG context.

    RETURNS
      A string like:
        "[Visual at 0:30] The slide shows a title 'Introduction to RAG' with
         a bullet list: 'Retrieval', 'Augmentation', 'Generation'. A bar chart
         on the right compares accuracy scores across four models."
    """
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    time_label = f"{minutes}:{seconds:02d}"

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"This is a frame from a video file named '{filename}' "
                                f"captured at timestamp {time_label}.\n\n"
                                "Describe EVERYTHING visible in this frame in detail:\n"
                                "- Any text, titles, headings, bullet points, captions\n"
                                "- Diagrams, charts, graphs, tables, code snippets\n"
                                "- People, gestures, facial expressions, body language\n"
                                "- Objects, equipment, environment, setting\n"
                                "- UI elements, applications, websites shown on screen\n"
                                "- Slides, whiteboards, posters, documents\n\n"
                                "Be specific and thorough. This description will be used "
                                "to answer questions about the video content."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.1,
        )
        description = response.choices[0].message.content.strip()
        return f"[Visual at {time_label}] {description}"

    except Exception as e:
        print(f"  Vision API error at {time_label}: {e}")
        return f"[Visual at {time_label}] Frame could not be described: {str(e)}"


def analyze_video_frames(vid_path: str, filename: str) -> str:
    """
    Extract frames from the video and describe each one using the vision LLM.

    RETURNS
      A single string containing all visual descriptions, separated by newlines.
      Returns "" if no frames could be extracted.
    """
    frames = _extract_frames(vid_path)
    if not frames:
        return ""

    print(f"Analyzing {len(frames)} frames with vision model …")
    descriptions = []

    for i, (timestamp, jpeg_bytes) in enumerate(frames):
        print(f"  Frame {i + 1}/{len(frames)} (t={timestamp:.0f}s) …")
        desc = _describe_frame(jpeg_bytes, timestamp, filename)
        descriptions.append(desc)

    visual_text = "\n\n".join(descriptions)
    print(f"Visual analysis complete — {len(descriptions)} frame descriptions.")
    return visual_text


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from all pages of a PDF using PyMuPDF.
    fitz.open(stream=...) reads directly from bytes — no temp file needed.
    """
    doc   = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [doc.load_page(n).get_text() for n in range(len(doc))]
    doc.close()
    return "\n\n".join(pages)


def extract_text_from_audio(file_bytes: bytes, filename: str) -> str:
    """
    Transcribe an audio file (mp3, wav, m4a, ogg, flac, aac, wma, opus, webm)
    to text using Groq Whisper.

    Files within 24 MB are sent directly to Groq without touching disk.
    Larger files are decoded to a normalised 16 kHz mono WAV by ffmpeg first,
    then handled by _transcribe_wav() which performs segment splitting.
    """
    if len(file_bytes) <= GROQ_AUDIO_LIMIT_BYTES:
        return _call_groq_whisper(file_bytes, filename)

    # Slow path: decode to normalised WAV first
    ext         = os.path.splitext(filename)[1].lower()
    fd_in,  in_path  = tempfile.mkstemp(suffix=ext)
    fd_wav, wav_path = tempfile.mkstemp(suffix=".wav")

    try:
        os.close(fd_in)
        os.close(fd_wav)

        with open(in_path, "wb") as f:
            f.write(file_bytes)

        _run_ffmpeg(
            "-y", "-i", in_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            wav_path,
        )
        return _transcribe_wav(wav_path, filename)

    finally:
        for p in (in_path, wav_path):
            if os.path.exists(p):
                os.remove(p)


def extract_text_from_video(file_bytes: bytes, filename: str) -> str:
    """
    Extract BOTH audio transcript AND visual frame descriptions from a video.

    PIPELINE:
      video bytes → temp file
        ├─ AUDIO BRANCH ──────────────────────────────────────────────────────
        │    → ffmpeg: strip video, resample → 16 kHz mono WAV
        │    → _transcribe_wav(): Groq Whisper (with auto-segmentation if needed)
        │    → audio transcript string
        │
        └─ VISUAL BRANCH (NEW) ───────────────────────────────────────────────
             → ffmpeg: extract 1 JPEG frame every FRAME_INTERVAL seconds
             → each frame → Groq vision LLM → detailed text description
             → visual descriptions string

      Both outputs are concatenated into a single rich text block:
        === AUDIO TRANSCRIPT ===
        <whisper output>

        === VISUAL CONTENT ===
        [Visual at 0:00] <description>
        [Visual at 0:10] <description>
        …

    This combined text is then chunked and indexed by the RAG pipeline,
    so questions about on-screen content, slides, diagrams, text overlays,
    or anything else visible in the video are all answerable.

    GRACEFUL DEGRADATION:
      - If the video has no audio → transcript = "" (no error raised here)
      - If the video has no video stream → visual = "" (frame extraction skipped)
      - If BOTH are empty → ValueError is raised in ingest()
    """
    ext              = os.path.splitext(filename)[1].lower()
    fd_vid, vid_path = tempfile.mkstemp(suffix=ext)
    fd_aud, wav_path = tempfile.mkstemp(suffix=".wav")

    try:
        os.close(fd_vid)
        os.close(fd_aud)

        with open(vid_path, "wb") as f:
            f.write(file_bytes)

        # ── AUDIO BRANCH ─────────────────────────────────────────────────────
        audio_transcript = ""
        has_audio = _check_audio_stream(vid_path)

        if has_audio:
            print("Audio stream detected — transcribing …")
            try:
                _run_ffmpeg(
                    "-y", "-i", vid_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    wav_path,
                )
                audio_transcript = _transcribe_wav(wav_path, filename)
                print(f"Audio transcript: {len(audio_transcript.split())} words.")
            except Exception as e:
                print(f"Audio transcription failed: {e}")
        else:
            print("No audio stream detected — skipping transcription.")

        # ── VISUAL BRANCH ─────────────────────────────────────────────────────
        print("Extracting and analyzing video frames …")
        visual_text = analyze_video_frames(vid_path, filename)

        # ── COMBINE ──────────────────────────────────────────────────────────
        parts = []

        if audio_transcript.strip():
            parts.append(f"=== AUDIO TRANSCRIPT ===\n{audio_transcript.strip()}")

        if visual_text.strip():
            parts.append(f"=== VISUAL CONTENT ===\n{visual_text.strip()}")

        return "\n\n".join(parts)

    finally:
        for p in (vid_path, wav_path):
            if os.path.exists(p):
                os.remove(p)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: TEXT CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """
    Split text into overlapping word-level windows.

    chunk_size=200, overlap=50 means each chunk is 200 words and shares its
    last 50 words with the next chunk — preserving context at boundaries.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> np.ndarray:
    """Convert text to a 384-dim float32 vector (required by FAISS)."""
    return embedding_model.encode(text, convert_to_numpy=True).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: FAISS INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(chunks: list):
    """
    Embed all chunks and store them in a FAISS IndexFlatL2.

    Returns (index, id_to_chunk) where id_to_chunk maps integer FAISS IDs
    back to the original chunk text strings.
    """
    print(f"Building FAISS index for {len(chunks)} chunks …")

    embeddings  = [get_embedding(c) for c in chunks]
    matrix      = np.array(embeddings)
    index       = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    id_to_chunk = {i: c for i, c in enumerate(chunks)}

    print(f"FAISS index ready — {index.ntotal} vectors.")
    return index, id_to_chunk


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: RAG SESSION
# ═══════════════════════════════════════════════════════════════════════════════

class RAGSession:
    """
    Holds all state for the active document session.

    Attributes:
      index        FAISS IndexFlatL2 (None until a file is ingested)
      id_to_chunk  {int: str} mapping vector IDs → chunk text
      source_type  "pdf" | "audio" | "video"
      source_name  original uploaded filename
      transcript   full raw extracted text (exposed via /transcript for debugging)
                   For video: includes both audio transcript and visual descriptions.
    """

    def __init__(self):
        self.index       = None
        self.id_to_chunk = {}
        self.source_type = None
        self.source_name = None
        self.transcript  = None

    def ingest(self, file_bytes: bytes, filename: str, source_type: str):
        """Extract text → chunk → embed → index. Stores results on self."""
        print(f"\nIngesting {source_type}: {filename}")

        if source_type == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif source_type == "audio":
            text = extract_text_from_audio(file_bytes, filename)
        elif source_type == "video":
            text = extract_text_from_video(file_bytes, filename)
        else:
            raise ValueError(f"Unknown source_type '{source_type}'.")

        if not text or len(text.split()) < 10:
            raise ValueError(
                f"Extraction returned only {len(text.split()) if text else 0} words — "
                "effectively empty.\n"
                "Possible causes:\n"
                "  1. The video has no audio AND no recognisable visual content\n"
                "  2. The audio contains only music (no speech)\n"
                "  3. The speech is too quiet or heavily distorted\n"
                "  4. The clip is too short (< 2 seconds)\n"
                "Try GET /transcript to inspect the raw output."
            )

        print(f"Total extracted content: {len(text.split())} words.")
        print(f"Preview:\n{text[:600]!r}\n")

        self.transcript  = text
        chunks           = chunk_text(text)
        print(f"Created {len(chunks)} chunks.")

        self.index, self.id_to_chunk = build_faiss_index(chunks)
        self.source_type = source_type
        self.source_name = filename
        print(f"Done — {self.index.ntotal} vectors indexed.\n")

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        Return the top_k most relevant chunks as a single string.
        top_k raised from 3 to 5 to capture both audio and visual chunks.
        """
        if self.index is None:
            raise RuntimeError("No document ingested yet.")

        q_vec    = np.array([get_embedding(query)])
        actual_k = min(top_k, self.index.ntotal)
        _, I     = self.index.search(q_vec, actual_k)

        return "\n\n---\n\n".join(
            self.id_to_chunk[i] for i in I[0] if i != -1
        )

    def answer(self, query: str) -> str:
        """Full RAG pipeline: retrieve → augment prompt → Groq LLM → answer."""
        context = self.retrieve(query)

        prompt = (
            f"You are a helpful assistant. Answer the user's question using ONLY "
            f"the context below — which contains content extracted from a "
            f"{self.source_type} file named '{self.source_name}'.\n\n"
            f"The context may include:\n"
            f"  - AUDIO TRANSCRIPT: spoken words transcribed from the video\n"
            f"  - VISUAL CONTENT: descriptions of what was visible on screen at each timestamp\n\n"
            f"Use both sources of information to give the most complete answer possible.\n"
            f"If the context is insufficient to answer, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM error: {e}")
            return f"Error generating answer: {str(e)}"


# ── Module-level singleton ────────────────────────────────────────────────────
rag_session = RAGSession()