## Gemini 2.5 Pro RAG + Codegen for Manim

All code lives in `rag/`. This builds a real RAG pipeline over the Manim docs and uses Gemini 2.5 Pro via `google.genai` to generate executable Manim animations from a plain-text prompt.

### Setup
- Python 3.10+
- Set `GOOGLE_API_KEY` in your environment. Optionally set `GEMINI_MODEL` (default: `gemini-2.5-pro`).

```powershell
# Windows PowerShell
$env:GOOGLE_API_KEY="YOUR_KEY"
$env:GEMINI_MODEL="gemini-2.5-pro"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r rag\requirements.txt
```

### 1) Build the vector index from docs
This will crawl the site and persist a Chroma DB under `rag/docs_index/`.

```powershell
python rag\scrape_and_chunk_docs.py --base-url https://docs.manim.community/ --max-pages 1500 --collection manim-docs
```

Notes:
- The scraper is same-origin and path-prefixed to the base URL. It stores URL + title metadata.
- Chunking uses character windows with overlap suitable for retrieval.

### 2) Generate Manim code from a prompt (with RAG)
```powershell
python rag\generate_animation.py --query "Visualize chain rule with moving tangents" --collection manim-docs --k 6 --out output.py
```

This will:
- Embed the query, retrieve top-k doc chunks
- Build a grounded prompt: “Use ONLY the APIs shown in retrieved chunks”
- Call Gemini 2.5 Pro via `google.genai`
- Extract the Python/JS code block
- Validate syntax (Python `ast.parse` or JS via `esprima`)
- If errors, send code + error back to Gemini for up to 3 fixes
- Print final code and save to `--out` (default `output.py` or `animation.js`)

### Environment variables
- `GOOGLE_API_KEY` (required)
- `GEMINI_MODEL` (optional; examples: `gemini-2.5-pro`, `gemini-2.5-pro-preview-06-05`)

### Folder layout
- `rag/scrape_and_chunk_docs.py`: crawler, chunker, indexer
- `rag/generate_animation.py`: RAG+codegen+validation loop
- `rag/rag_utils/`: modules for scraping, chunking, embeddings, vectorstore, prompt building, validation
- `rag/docs_index/`: persistent Chroma DB

### Tip
To run generated Manim code, ensure Manim is installed in your environment:
```powershell
pip install manim
# then run (example)
manim output.py SceneClassName -p -ql
```