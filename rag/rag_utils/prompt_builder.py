from typing import List
from rag_utils.vectorstore import SearchResults


SYSTEM_INSTRUCTION = (
    "You are an expert Manim (Python) and front-end animation (Anime.js) engineer.\n"
    "Generate clean, runnable, pedagogically strong animations.\n"
    "Constraints:\n"
    "- Use ONLY the APIs/classes shown in the retrieved documentation chunks.\n"
    "- Prefer Manim (Python) when docs are from Manim site; otherwise you may use Anime.js.\n"
    "- Return a SINGLE fenced code block only. No commentary.\n"
    "- For Manim: define a single Scene subclass (e.g., class Demo(Scene): ...) with construct().\n"
    "- Keep imports minimal and from the shown APIs.\n"
    "- Prefer explicit durations and easing for animations.\n"
)


def format_context(results: SearchResults, max_chars: int = 6000) -> str:
    lines: List[str] = []
    total = 0
    for doc, meta in zip(results.documents, results.metadatas):
        src = meta.get("url", "")
        title = meta.get("title", "")
        header = f"[Source] {title} — {src}"
        snippet = doc.strip()
        chunk = f"{header}\n{snippet}\n---\n"
        if total + len(chunk) > max_chars:
            break
        lines.append(chunk)
        total += len(chunk)
    return "\n".join(lines)


def build_codegen_prompt(user_query: str, retrieved_docs: SearchResults) -> str:
    context = format_context(retrieved_docs)
    user = (
        f"Task: {user_query}\n\n"
        "Using ONLY the documentation excerpts below, produce a rich, detailed animation.\n"
        "Requirements for Manim output:\n"
        "- Create a single Scene subclass with construct().\n"
        "- Include multiple steps, labels, and on-screen guides so the concept is visually clear.\n"
        "- Use camera framing (e.g., self.play(self.camera.frame.animate.move_to(...))) when useful.\n"
        "- Use Transform/ReplacementTransform and animations with run_time and rate_functions for clarity.\n"
        "- Add titles, subtitles, and explanatory text or braces/arrows to highlight parts.\n"
        "- Keep runtime around 10–25 seconds with several sequential animations.\n"
        "- Do not include commentary; only code.\n"
        "If you choose JS/Anime.js instead of Manim:\n"
        "- Provide minimal, complete HTML + JS in one code block.\n"
        "- Include visible labels and a sequence of animations.\n"
        "\nContext:\n" + context + "\n"
        "Return only one fenced code block labeled with the language (python or javascript)."
    )
    # Combine into a single text since google.genai accepts string and system via config
    return SYSTEM_INSTRUCTION + "\n\n" + user