#!/usr/bin/env python
import argparse
import os
import re
import sys
import subprocess
from typing import List, Tuple, Optional

from google import genai
from google.genai import types

from rag_utils.embeddings import LocalHuggingFaceEmbeddingFunction
from rag_utils.vectorstore import get_or_create_collection, similarity_search
from rag_utils.prompt_builder import build_codegen_prompt
from rag_utils.code_validator import (
    validate_python_code,
    validate_js_code,
    validate_manim_runtime,
    is_manim_code,
)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


CODE_BLOCK_RE = re.compile(r"```(python|py|javascript|js)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_first_code_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = CODE_BLOCK_RE.search(text)
    if not match:
        return None, None
    lang = match.group(1).lower() if match.group(1) else None
    code = match.group(2).strip()
    # Normalize language tag
    if lang in {"py"}:
        lang = "python"
    if lang in {"js"}:
        lang = "javascript"
    return lang, code


def call_gemini(model: str, prompt: str, api_key: Optional[str] = None) -> str:
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment.")
    print(f"[Stage] Initializing Gemini client and calling model: {model}")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        ),
    )
    print(resp)
    text = resp.text or ""
    print("[Model Response] Begin\n" + text + "\n[Model Response] End")
    return text


def refine_with_error(model: str, code: str, error_text: str) -> str:
    instruction = (
        "The following code failed to validate. Fix the error and return ONLY a single corrected code block.\n"
        "Do not add explanations. Preserve the intent.\n\n"
        f"Error:\n{error_text}\n\nCode:\n```\n{code}\n```"
    )
    return call_gemini(model, instruction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and validate Manim/JS animation code via Gemini with RAG")
    parser.add_argument("--query", required=True, help="User intent, e.g. 'Visualize chain rule'")
    parser.add_argument("--persist-path", default=os.path.join("rag", "docs_index"))
    parser.add_argument("--collection", default="manim-docs")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="Output file path. Defaults to output.py or animation.js")
    parser.add_argument("--preview", action="store_true", help="If Manim code, open a preview window after validation")
    parser.add_argument(
        "--quality",
        default="ql",
        choices=["ql", "qm", "qh"],
        help="Manim render quality for preview: ql (low), qm (medium), qh (high)",
    )

    args = parser.parse_args()

    # Load embeddings + vector store
    print("[Stage] Loading embeddings and vector store...")
    ef = LocalHuggingFaceEmbeddingFunction(model_name=DEFAULT_EMBED_MODEL)
    collection = get_or_create_collection(args.persist_path, args.collection, ef)

    # Retrieve top-k chunks
    print(f"[Stage] Performing similarity search (k={args.k})...")
    results = similarity_search(collection, query=args.query, k=args.k)

    if not results.documents:
        print("No results from the vector store. Did you run scrape_and_chunk_docs.py?", file=sys.stderr)
        sys.exit(1)

    print("[Stage] Building grounded prompt...")
    prompt = build_codegen_prompt(user_query=args.query, retrieved_docs=results)

    # Initial generation
    print("[Stage] Requesting initial code generation from Gemini...")
    raw = call_gemini(args.model, prompt)
    print("[Stage] Extracting first code block from model response...")
    lang, code = extract_first_code_block(raw)
    print(f"[Info] Detected language: {lang or 'unknown'}; code present: {bool(code)}")

    # Retry if code block missing
    retries = 0
    while (not code) and retries < 2:
        print(f"[Stage] No code block found. Retrying ({retries+1}/2) with stricter instructions...")
        raw = call_gemini(args.model, (
            "Return ONLY a single code block. No commentary.\n\n" + prompt
        ))
        print("[Stage] Extracting first code block from retry response...")
        lang, code = extract_first_code_block(raw)
        print(f"[Info] Detected language: {lang or 'unknown'}; code present: {bool(code)}")
        retries += 1

    if not code:
        print("Model did not return a code block.")
        sys.exit(2)

    # Validate with retries (syntax + runtime for Manim)
    val_retries = 0
    while val_retries < 3:
        print(f"[Stage] Validation attempt {val_retries+1}/3...")
        if (lang or "python") == "python":
            print("[Stage] Python syntax validation...")
            ok, err = validate_python_code(code)
            if ok:
                out_path = args.out or "output.py"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                # If it's Manim code, render to validate runtime
                if is_manim_code(code):
                    print("[Stage] Detected Manim code. Attempting runtime render validation...")
                    rok, rerr = validate_manim_runtime(out_path, code)
                    if rok:
                        print(code)
                        print(f"\nRendered OK. Saved: {out_path}")
                        # Optional preview render
                        if args.preview:
                            try:
                                from rag_utils.code_validator import _extract_manim_scene_classes
                                scenes = _extract_manim_scene_classes(code)
                                scene = scenes[0] if scenes else None
                                if scene:
                                    print("[Stage] Launching Manim preview...")
                                    quality_flag = f"-{args.quality}"
                                    cmd = [
                                        sys.executable,
                                        "-m",
                                        "manim",
                                        out_path,
                                        scene,
                                        quality_flag,
                                        "-p",
                                    ]
                                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                    if proc.returncode != 0:
                                        print("[Warning] Preview render failed. You can run it manually:")
                                        print(f"manim {out_path} {scene} {quality_flag} -p")
                                else:
                                    print("[Info] No Scene detected for preview.")
                            except Exception as e:
                                print(f"[Warning] Failed to launch preview: {e}")
                        sys.exit(0)
                    else:
                        print("[Error] Manim runtime validation failed. Sending error back to model for fix.")
                        err = rerr
                else:
                    print(code)
                    print(f"\nSaved: {out_path}")
                    sys.exit(0)
            else:
                print("[Error] Python syntax validation failed. Sending error back to model for fix.")
                fix_raw = refine_with_error(args.model, code, err)
                print("[Stage] Extracting code block from fix response...")
                nlang, ncode = extract_first_code_block(fix_raw)
                if ncode:
                    lang, code = nlang or lang, ncode
        else:  # javascript
            print("[Stage] JavaScript syntax validation...")
            ok, err = validate_js_code(code)
            if ok:
                out_path = args.out or "animation.js"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print(code)
                print(f"\nSaved: {out_path}")
                sys.exit(0)
            else:
                print("[Error] JavaScript validation failed. Sending error back to model for fix.")
                fix_raw = refine_with_error(args.model, code, err)
                print("[Stage] Extracting code block from fix response...")
                nlang, ncode = extract_first_code_block(fix_raw)
                if ncode:
                    lang, code = nlang or lang, ncode
        val_retries += 1

    print("Failed to produce valid code after retries.")
    print(code)
    sys.exit(3)