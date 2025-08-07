#!/usr/bin/env python
import argparse
import os
import sys
import time
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from rag_utils.chunking import chunk_text
from rag_utils.vectorstore import get_or_create_collection, add_documents
from rag_utils.embeddings import LocalHuggingFaceEmbeddingFunction

DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def same_origin_and_prefix(url: str, base: str) -> bool:
    u, b = urlparse(url), urlparse(base)
    if (u.scheme, u.netloc) != (b.scheme, b.netloc):
        return False
    # stay under the base path prefix
    return u.path.startswith(b.path)


def clean_html_to_text(html: str) -> str:
    # Prefer readable text over raw HTML. Use bs4 .get_text("\n")
    soup = BeautifulSoup(html, "html.parser")
    # Remove nav/footer/script/style
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text("\n")
    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([l for l in lines if l])


def fetch(url: str, timeout: int = 20) -> tuple[str, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "manim-rag-indexer/1.0"})
    r.raise_for_status()
    return r.text, r.url


def extract_links(html: str, current_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(current_url, href)
        abs_url, _ = urldefrag(abs_url)
        links.append(abs_url)
    return links


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def crawl_and_index(base_url: str, persist_path: str, collection_name: str, max_pages: int, chunk_size: int, overlap: int):
    os.makedirs(persist_path, exist_ok=True)

    ef = LocalHuggingFaceEmbeddingFunction(model_name=DEFAULT_EMBED_MODEL)
    collection = get_or_create_collection(persist_path, collection_name, ef)

    seen = set()
    queue = [base_url]
    stored = 0

    pbar = tqdm(total=max_pages, desc="Crawling")
    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if not same_origin_and_prefix(url, base_url):
            continue
        try:
            html, final_url = fetch(url)
        except Exception:
            continue

        if not same_origin_and_prefix(final_url, base_url):
            continue

        title = extract_title(html)
        text = clean_html_to_text(html)
        if not text.strip():
            pbar.update(1)
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        documents = []
        metadatas = []
        ids = []
        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"url": final_url, "title": title, "chunk_index": idx})
            ids.append(f"{final_url}#chunk={idx}")
        if documents:
            add_documents(collection, ids=ids, documents=documents, metadatas=metadatas)
            stored += len(documents)

        # enqueue new links
        try:
            links = extract_links(html, final_url)
        except Exception:
            links = []
        for link in links:
            if link not in seen and same_origin_and_prefix(link, base_url):
                queue.append(link)

        pbar.update(1)
        time.sleep(0.05)
    pbar.close()
    print(f"Indexed pages: {len(seen)}; chunks stored: {stored}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl docs and build a Chroma vector index")
    parser.add_argument("--base-url", required=True, help="Base docs URL, e.g. https://docs.manim.community/")
    parser.add_argument("--persist-path", default=os.path.join("rag", "docs_index"))
    parser.add_argument("--collection", default="manim-docs")
    parser.add_argument("--max-pages", type=int, default=1500)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)

    args = parser.parse_args()

    crawl_and_index(
        base_url=args.base_url,
        persist_path=args.persist_path,
        collection_name=args.collection,
        max_pages=args.max_pages,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )