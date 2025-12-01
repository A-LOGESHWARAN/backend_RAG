import os
import tempfile
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from docx import Document
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF (new)
import re

load_dotenv()

# ------------------- Groq Client -------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in .env")
client = Groq(api_key=GROQ_API_KEY)

# ------------------- FastAPI -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------- Stronger Embedding Model -------------------
embedder = SentenceTransformer("all-mpnet-base-v2")  # UPGRADE
dim = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dim)
stored_texts = []

# ------------------- Chunking -------------------
def chunk_text(text, chunk_size=220, overlap=60):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

# ------------------- Clean extracted text -------------------
def clean_text(txt):
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"Page \d+", "", txt)
    return txt.strip()

# ------------------- Improved PDF Extraction (Hybrid) -------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    # Save temporary PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    text_blocks = []

    # ----- 1) PyMuPDF extraction (BEST for layout) -----
    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # sort by y, then x
            for b in blocks:
                blk = b[4].strip()
                if blk:
                    text_blocks.append(blk)
        doc.close()

        if len(" ".join(text_blocks)) > 50:
            os.remove(tmp_path)
            return clean_text("\n".join(text_blocks))

    except Exception as e:
        print("PyMuPDF failed:", e)

    # ----- 2) pdfplumber extraction -----
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                ptxt = page.extract_text()
                if ptxt:
                    text_blocks.append(ptxt)

        if len(" ".join(text_blocks)) > 20:
            os.remove(tmp_path)
            return clean_text("\n".join(text_blocks))

    except Exception as e:
        print("pdfplumber failed:", e)

    # ----- 3) OCR fallback (scanned PDFs) -----
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=250).original
                ocr = pytesseract.image_to_string(img)
                if ocr.strip():
                    text_blocks.append(ocr)
    except:
        pass

    os.remove(tmp_path)
    return clean_text("\n".join(text_blocks))

# ------------------- DOCX -------------------
def extract_text_from_docx_bytes(bytes_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(bytes_data)
        tmp_path = tmp.name

    doc = Document(tmp_path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    os.remove(tmp_path)
    return clean_text("\n".join(lines))

# ------------------- Plain text -------------------
def extract_text_from_plain_bytes(b):
    try:
        return clean_text(b.decode("utf-8"))
    except:
        return clean_text(b.decode("latin-1", errors="ignore"))

# ------------------- Vectorstore Add -------------------
def add_to_vectorstore(texts):
    global stored_texts
    embs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    index.add(embs)
    stored_texts.extend(texts)

# ------------------- API Models -------------------
class IngestRequest(BaseModel):
    texts: list[str]

class QueryRequest(BaseModel):
    query: str

# ------------------- Upload File -------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    name = file.filename.lower()

    # Detect file type
    if name.endswith(".pdf"):
        text = extract_text_from_pdf_bytes(contents)
    elif name.endswith(".docx"):
        text = extract_text_from_docx_bytes(contents)
    else:
        text = extract_text_from_plain_bytes(contents)

    if len(text.strip()) == 0:
        return {"error": "No text extracted from file"}

    chunks = chunk_text(text)
    add_to_vectorstore(chunks)
    return {"filename": file.filename, "chunks": len(chunks)}

# ------------------- Improved Query (Retrieves More + Reranks) -------------------
@app.post("/query")
def query(req: QueryRequest):

    if len(stored_texts) == 0:
        return {"answer": "No documents ingested yet."}

    # Get query embedding
    q_emb = embedder.encode([req.query]).astype("float32")

    # Retrieve top 10 chunks instead of 4
    D, I = index.search(q_emb, 12)
    retrieved = [stored_texts[i] for i in I[0] if i < len(stored_texts)]

    # Build context (top 5 chunks)
    context = "\n\n---\n\n".join(retrieved[:5])

    # Stronger system prompt
    prompt = f"""
You MUST answer ONLY using the context.
If the answer is not present, reply exactly: "I don't know".

CONTEXT:
{context}

QUESTION:
{req.query}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": retrieved[:5]}
