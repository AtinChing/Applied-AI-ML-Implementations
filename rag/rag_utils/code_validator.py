import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

import esprima  # type: ignore


def validate_python_code(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Python syntax error: {e}"
    except Exception as e:
        return False, f"Python parse error: {e}"


def validate_js_code(code: str) -> Tuple[bool, str]:
    try:
        esprima.parseScript(code, tolerant=True)
        return True, ""
    except Exception as e:
        return False, f"JavaScript parse error: {e}"


# -------- Manim runtime validation --------

_MANIM_IMPORT_RE = re.compile(r"\bfrom\s+manim\s+import\b|\bimport\s+manim\b")


def _extract_manim_scene_classes(code: str) -> List[str]:
    """Return names of classes that inherit from Scene (best-effort)."""
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    scene_classes: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                # class X(Scene):
                if isinstance(base, ast.Name) and base.id == "Scene":
                    scene_classes.append(node.name)
                    break
                # class X(manim.Scene) or something.Scene
                if isinstance(base, ast.Attribute) and base.attr == "Scene":
                    scene_classes.append(node.name)
                    break
    return scene_classes


def is_manim_code(code: str) -> bool:
    if _MANIM_IMPORT_RE.search(code):
        return True
    scenes = _extract_manim_scene_classes(code)
    return len(scenes) > 0


def validate_manim_runtime(py_file_path: str, code: str, timeout_seconds: int = 180) -> Tuple[bool, str]:
    """Render the first detected Scene using manim CLI in a temp media dir.

    Returns (ok, error_message).
    """
    scenes = _extract_manim_scene_classes(code)
    if not scenes:
        return False, "No Manim Scene subclass found to render."
    scene = scenes[0]

    temp_media_dir = tempfile.mkdtemp(prefix="manim_media_")
    try:
        cmd = [
            sys.executable,
            "-m",
            "manim",
            py_file_path,
            scene,
            "-ql",  # quick, low quality
            "--disable_caching",
            "--media_dir",
            temp_media_dir,
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        if proc.returncode == 0:
            return True, ""
        else:
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            # Truncate large logs
            if len(out) > 8000:
                out = out[-8000:]
            return False, f"Manim runtime error (exit {proc.returncode}):\n{out}"
    except subprocess.TimeoutExpired:
        return False, "Manim runtime timed out."
    except FileNotFoundError:
        return False, "Manim not found. Ensure 'manim' is installed in this environment."
    except Exception as e:
        return False, f"Manim runtime failed: {e}"
    finally:
        try:
            shutil.rmtree(temp_media_dir, ignore_errors=True)
        except Exception:
            pass