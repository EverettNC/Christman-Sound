#!/usr/bin/env python3
"""
Christman Universal Preflight
The Christman AI Project

The most thorough automated module diagnostic ever built.
Drop into any Python project and run it.

Usage:
  python christman_preflight.py                    <- Full scan, Python + Frontend
  python christman_preflight.py --autofix          <- Auto-install all missing pip packages
  python christman_preflight.py --dir /path        <- Scan a specific directory
  python christman_preflight.py --skip mod1,mod2   <- Skip specific modules
  python christman_preflight.py --json             <- JSON output only
  python christman_preflight.py --no-color         <- Plain text output
  python christman_preflight.py --quiet            <- Suppress verbose traceback
  python christman_preflight.py --no-frontend      <- Skip .tsx/.ts/.css/.js scanning

© 2026 Everett Nathaniel Christman & The Christman AI Project
Luma Cognify AI — "How can we help you love yourself more?"
Patent Pending TCAP-2026-001
"""

import ast
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


# ── ANSI Color System ─────────────────────────────────────────────────────────

class C:
    _enabled = True
    RED      = "\033[91m"
    GREEN    = "\033[92m"
    YELLOW   = "\033[93m"
    BLUE     = "\033[94m"
    MAGENTA  = "\033[95m"
    CYAN     = "\033[96m"
    WHITE    = "\033[97m"
    ORANGE   = "\033[38;5;208m"
    DIM      = "\033[2m"
    BOLD     = "\033[1m"
    RESET    = "\033[0m"

    @classmethod
    def disable(cls):
        cls._enabled = False
        for attr in ["RED","GREEN","YELLOW","BLUE","MAGENTA","CYAN",
                     "WHITE","ORANGE","DIM","BOLD","RESET"]:
            setattr(cls, attr, "")

    @classmethod
    def r(cls, text, *codes):
        if not cls._enabled:
            return text
        return "".join(codes) + str(text) + cls.RESET


def c(text, *codes):
    return C.r(text, *codes)


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
   ██████╗██╗  ██╗██████╗ ██╗███████╗████████╗███╗   ███╗ █████╗ ███╗   ██╗
  ██╔════╝██║  ██║██╔══██╗██║██╔════╝╚══██╔══╝████╗ ████║██╔══██╗████╗  ██║
  ██║     ███████║██████╔╝██║███████╗   ██║   ██╔████╔██║███████║██╔██╗ ██║
  ██║     ██╔══██║██╔══██╗██║╚════██║   ██║   ██║╚██╔╝██║██╔══██║██║╚██╗██║
  ╚██████╗██║  ██║██║  ██║██║███████║   ██║   ██║ ╚═╝ ██║██║  ██║██║ ╚████║
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
"""

def print_banner(project_dir: str):
    print()
    print(c(BANNER, C.CYAN, C.BOLD))
    print(c("  U N I V E R S A L   P R E F L I G H T   D I A G N O S T I C S", C.WHITE, C.BOLD))
    print(c("  The Christman AI Project  ·  Luma Cognify AI  ·  Patent Pending TCAP-2026-001", C.DIM))
    print(c('  "How can we help you love yourself more?"', C.GREEN))
    print()
    print(c("  ┌─────────────────────────────────────────────────────────────────────┐", C.DIM))
    print(c("  │  TARGET  ", C.DIM) + c(f"{project_dir:<60}", C.CYAN) + c("│", C.DIM))
    print(c("  └─────────────────────────────────────────────────────────────────────┘", C.DIM))
    print()


# ── Progress Bar ──────────────────────────────────────────────────────────────

def progress_bar(current, total, name, passed, failed, width=40):
    pct    = current / total if total else 0
    filled = int(width * pct)
    bar    = c("█" * filled, C.GREEN) + c("░" * (width - filled), C.DIM)
    status = c(f"✅ {passed}", C.GREEN) + c(" │ ", C.DIM) + c(f"❌ {failed}", C.RED)
    print(f"  [{bar}] {c(f'{int(pct*100):3d}%', C.CYAN)}  {status}  {c(str(name)[-35:], C.DIM)}", end="\r", flush=True)

def clear_progress():
    print(" " * 120, end="\r")


# ── Failure Categories ────────────────────────────────────────────────────────

class FailureType(str, Enum):
    SYNTAX_ERROR     = "SYNTAX_ERROR"
    MISSING_PIP      = "MISSING_PIP"
    MISSING_INTERNAL = "MISSING_INTERNAL"
    CIRCULAR_IMPORT  = "CIRCULAR_IMPORT"
    IMPORT_ERROR     = "IMPORT_ERROR"
    RUNTIME_ERROR    = "RUNTIME_ERROR"
    FILE_NOT_FOUND   = "FILE_NOT_FOUND"
    FRONTEND_ERROR   = "FRONTEND_ERROR"
    UNKNOWN          = "UNKNOWN"

CATEGORY_COLORS = {
    FailureType.SYNTAX_ERROR:     C.RED,
    FailureType.CIRCULAR_IMPORT:  C.MAGENTA,
    FailureType.MISSING_PIP:      C.YELLOW,
    FailureType.MISSING_INTERNAL: C.ORANGE,
    FailureType.FILE_NOT_FOUND:   C.ORANGE,
    FailureType.IMPORT_ERROR:     C.CYAN,
    FailureType.RUNTIME_ERROR:    C.RED,
    FailureType.FRONTEND_ERROR:   C.MAGENTA,
    FailureType.UNKNOWN:          C.DIM,
}

CATEGORY_ICONS = {
    FailureType.SYNTAX_ERROR:     "🔴",
    FailureType.CIRCULAR_IMPORT:  "🔁",
    FailureType.MISSING_PIP:      "📦",
    FailureType.MISSING_INTERNAL: "🔍",
    FailureType.FILE_NOT_FOUND:   "🗂",
    FailureType.IMPORT_ERROR:     "⚠️",
    FailureType.RUNTIME_ERROR:    "💥",
    FailureType.FRONTEND_ERROR:   "🎨",
    FailureType.UNKNOWN:          "❓",
}

CATEGORY_LABELS = {
    FailureType.SYNTAX_ERROR:     "SYNTAX ERRORS       Fix the file first. Nothing else runs until this is clean.",
    FailureType.CIRCULAR_IMPORT:  "CIRCULAR IMPORTS    Two modules locked in a death grip. Break the cycle.",
    FailureType.MISSING_PIP:      "MISSING PIP PKGS    Not installed in this environment. One command fixes all.",
    FailureType.MISSING_INTERNAL: "MISSING INTERNAL    Module expected but the file does not exist on disk.",
    FailureType.FILE_NOT_FOUND:   "FILE NOT FOUND      Referenced file was deleted or moved.",
    FailureType.IMPORT_ERROR:     "IMPORT ERRORS       Symbol does not exist where expected.",
    FailureType.RUNTIME_ERROR:    "RUNTIME ERRORS      Crashes the moment Python loads it.",
    FailureType.FRONTEND_ERROR:   "FRONTEND ERRORS     Structural issues in .tsx/.ts/.css/.js files.",
    FailureType.UNKNOWN:          "UNKNOWN             Run without --quiet to diagnose.",
}

CATEGORY_ORDER = [
    FailureType.SYNTAX_ERROR, FailureType.CIRCULAR_IMPORT,
    FailureType.FRONTEND_ERROR, FailureType.MISSING_PIP,
    FailureType.MISSING_INTERNAL, FailureType.FILE_NOT_FOUND,
    FailureType.IMPORT_ERROR, FailureType.RUNTIME_ERROR,
    FailureType.UNKNOWN,
]


@dataclass
class ModuleResult:
    name:          str
    passed:        bool
    filepath:      str                   = ""
    load_time_ms:  float                 = 0.0
    failure_type:  Optional[FailureType] = None
    error_msg:     str                   = ""
    root_cause:    str                   = ""
    what_to_do:    str                   = ""
    pip_package:   str                   = ""
    traceback_str: str                   = ""
    is_frontend:   bool                  = False


# ── pip + internal maps ───────────────────────────────────────────────────────

PIP_MAP = {
    "cv2":"opencv-python","sklearn":"scikit-learn","PIL":"Pillow",
    "dotenv":"python-dotenv","yaml":"PyYAML","bs4":"beautifulsoup4",
    "usb":"pyusb","serial":"pyserial","wx":"wxPython","gi":"PyGObject",
    "Crypto":"pycryptodome","OpenSSL":"pyOpenSSL","jwt":"PyJWT",
    "attr":"attrs","pkg_resources":"setuptools","google.cloud":"google-cloud",
    "azure":"azure-sdk","boto3":"boto3","botocore":"botocore",
    "faster_whisper":"faster-whisper","speech_recognition":"SpeechRecognition",
    "sounddevice":"sounddevice","soundfile":"soundfile","librosa":"librosa",
    "pyaudio":"PyAudio","playsound":"playsound","gtts":"gTTS","pygame":"pygame",
    "torch":"torch","tensorflow":"tensorflow","keras":"keras",
    "transformers":"transformers","diffusers":"diffusers",
    "sentence_transformers":"sentence-transformers","spacy":"spacy",
    "nltk":"nltk","textgrid":"textgrid","networkx":"networkx","sympy":"sympy",
    "qiskit":"qiskit","pandas":"pandas","numpy":"numpy","scipy":"scipy",
    "matplotlib":"matplotlib","seaborn":"seaborn","plotly":"plotly",
    "joblib":"joblib","aiohttp":"aiohttp","websockets":"websockets",
    "httpx":"httpx","anthropic":"anthropic","openai":"openai",
    "fastapi":"fastapi","uvicorn":"uvicorn","flask":"flask",
    "flask_cors":"flask-cors","flask_sqlalchemy":"flask-sqlalchemy",
    "fastmcp":"fastmcp","pydantic":"pydantic","sqlalchemy":"sqlalchemy",
    "alembic":"alembic","pymongo":"pymongo","redis":"redis","celery":"celery",
    "newspaper":"newspaper4k","autopep8":"autopep8","jsonschema":"jsonschema",
    "cryptography":"cryptography","paramiko":"paramiko","docker":"docker",
    "kubernetes":"kubernetes","aioschedule":"aioschedule","tiktoken":"tiktoken",
    "requests":"requests","pyttsx3":"pyttsx3","deepgram":"deepgram-sdk",
    "pyperclip":"pyperclip","rich":"rich","typer":"typer","click":"click",
    "speechbrain":"speechbrain",
}

KNOWN_INTERNAL = {
    "brain","brain_common_events","brain_ferrari_v1","derek_brain",
    "brockston_core","Brockston_Brain_CC1","brockston_module_loader",
    "brockston_boot","brockston_reality_check","brockston_education_system",
    "brain_combined","cognitive_cortex","local_reasoning_engine",
    "knowledge_engine","memory_engine","conversation_engine","intent_engine",
    "crisis_detection","provider_router","perplexity_service","tone_manager",
    "reasoning_reflective_planner","reasoning_intent","reasoning_reasoner",
    "reasoning_cortex_types","family_coordinator","ultimateev",
    "alphavox_knowledge_engine","config","utils","events","database",
    "app_init","web_crawler","nlp_module","memory_mesh","memory_mesh_bridge",
    "tts_bridge","soul_forge_bridge","bridge","voice_analysis_service",
    "christman_tone_engine_v2","emotion_service","emotion","store","indexer",
    "memory_rag","embodiment","backend","speech","src","test_code",
    "routes","loop","answer","executor","learning_analytics","learning_journey",
    "learning_service","learning_routes","voice_cortex","alphavox_app",
    "behavior_capture","nonverbal_engine","behavioral_interpreter",
    "color_scheme_routes","play_readiness","self_modifying_code",
    "ai_learning_engine","research_module","models","db",
    "json_guardian","stillhere","interpreter","tier7_steg","timbre",
    "voice_stack","alphavox_ultimate_voice",
}


# ── Package context resolver ──────────────────────────────────────────────────

def resolve_package_context(filepath: str) -> Tuple[str, str]:
    """
    Walk up the directory tree to find the package root.
    Returns (dotted_module_name, package_root_dir).

    If the file lives inside a package (directory chain with __init__.py files),
    load it WITH its package context so relative imports resolve correctly.
    If it is a standalone file, load it as-is with a safe unique name.
    """
    parts     = []
    directory = os.path.dirname(os.path.abspath(filepath))
    stem      = os.path.splitext(os.path.basename(filepath))[0]

    while os.path.exists(os.path.join(directory, "__init__.py")):
        parts.append(os.path.basename(directory))
        directory = os.path.dirname(directory)

    if not parts:
        # Standalone file — original behavior
        return f"_cpf_{stem}", directory

    parts.reverse()
    dotted = ".".join(parts) + "." + stem
    return dotted, directory


# ── Python syntax check ───────────────────────────────────────────────────────

def check_syntax(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        ast.parse(source, filename=filepath)
        return None
    except SyntaxError as e:
        snippet = e.text.strip() if e.text else ""
        return f"Line {e.lineno}: {e.msg}  →  {snippet}"
    except Exception as e:
        return str(e)


# ── Root cause classifier ─────────────────────────────────────────────────────

def extract_root_cause(tb: str, error: Exception, project_dir: str) -> Tuple[str, str, str, FailureType]:
    err_str = str(error)

    if any(s in err_str.lower() for s in ["circular import","partially initialized module","most likely due to a circular import"]):
        m   = re.search(r"module '([^']+)'", err_str)
        mod = m.group(1) if m else "unknown"
        return (f"Circular import involving '{mod}'",
                f"Find which module imports '{mod}' and vice versa. Move shared code to a third module or use lazy imports.",
                "", FailureType.CIRCULAR_IMPORT)

    if isinstance(error, ModuleNotFoundError):
        missing = getattr(error, "name", None) or ""
        if not missing:
            m       = re.search(r"No module named '([^']+)'", err_str)
            missing = m.group(1) if m else err_str
        root = missing.split(".")[0]

        # ── Exists as a .py file that failed to load ──────────────────
        if os.path.exists(os.path.join(project_dir, f"{root}.py")):
            return (f"'{root}' exists on disk but failed to load — it has its own broken import",
                    f"Run preflight on {root}.py directly and fix its imports first.",
                    "", FailureType.IMPORT_ERROR)

        # ── Exists as a directory (local package) — NOT a pip package ─
        if os.path.isdir(os.path.join(project_dir, root)):
            return (
                f"'{root}' is a local package directory — not a pip package",
                f"Install with: pip install -e . from the project root, or check PYTHONPATH.",
                "", FailureType.MISSING_INTERNAL,
            )

        # ── Known internal single-word subpackage names ───────────────
        # These are directory names inside SDK packages, never pip packages
        INTERNAL_SUBPACKAGES = {
            "audio", "engines", "timbre", "synthesis", "tone", "utils",
            "integration", "nonverbal", "music", "voice_stack",
            "christman_voice_sdk", "christman_sound", "CHRISTMAN_EAR_CANAL",
            "base_synthesizer", "speech_recognition_engine", "audio_processor",
        }
        if root in INTERNAL_SUBPACKAGES:
            return (
                f"'{root}' is an internal subpackage — not a pip package",
                f"This is a relative import issue. Run: pip install -e . from the SDK root.",
                "", FailureType.MISSING_INTERNAL,
            )

        # ── Preflight's own temp module names ─────────────────────────
        if root.startswith("_cpf_"):
            return (
                f"'{root}' is a preflight internal name — this is a package context issue",
                f"The file lives inside a package but is being loaded without package context. Check __init__.py chain.",
                "", FailureType.IMPORT_ERROR,
            )

        if root in KNOWN_INTERNAL:
            return (f"Internal module '{root}' is missing — not a pip package",
                    f"Create or restore '{root}.py' in the project directory.",
                    "", FailureType.MISSING_INTERNAL)
        if root in PIP_MAP:
            pkg = PIP_MAP[root]
            return (f"pip package '{root}' not installed  (pip name: {pkg})", f"pip install {pkg}", pkg, FailureType.MISSING_PIP)
        return (f"Module '{root}' not found — likely a pip package", f"pip install {root}", root, FailureType.MISSING_PIP)

    if isinstance(error, ImportError):
        if "cannot import name" in err_str:
            m = re.search(r"cannot import name '([^']+)' from '([^']+)'", err_str)
            if m:
                sym, src = m.group(1), m.group(2)
                return (f"'{sym}' does not exist in '{src}'",
                        f"grep -r '{sym}' --include='*.py' — it may be renamed, moved, or never defined.",
                        "", FailureType.IMPORT_ERROR)
            return (err_str, "Check the import — the symbol may have been renamed.", "", FailureType.IMPORT_ERROR)
        if "No such file or directory" in err_str:
            return (err_str, "A file referenced during import does not exist.", "", FailureType.FILE_NOT_FOUND)
        return (err_str, "Check the import statement at the top of this module.", "", FailureType.IMPORT_ERROR)

    if isinstance(error, FileNotFoundError):
        return (err_str, "A file this module opens at load time does not exist. Check hardcoded paths.", "", FailureType.FILE_NOT_FOUND)

    if isinstance(error, AttributeError):
        return (f"Attribute error at module load: {err_str}",
                "A dependency is loading as None. Check module-level code that runs outside functions.",
                "", FailureType.RUNTIME_ERROR)

    return (f"{type(error).__name__}: {err_str}",
            "Module crashes on load. Run without --quiet to see the full traceback.",
            "", FailureType.RUNTIME_ERROR)


# ── Python module tester ──────────────────────────────────────────────────────

def test_module(name: str, filepath: str, project_dir: str) -> ModuleResult:
    start      = time.perf_counter()
    syntax_err = check_syntax(filepath)
    if syntax_err:
        return ModuleResult(
            name=name, filepath=filepath, passed=False,
            load_time_ms=round((time.perf_counter()-start)*1000, 1),
            failure_type=FailureType.SYNTAX_ERROR,
            error_msg=syntax_err,
            root_cause=f"Syntax error — {syntax_err}",
            what_to_do="Fix the syntax error at the indicated line. Common causes: missing colon, unclosed bracket, bad indentation, mismatched quotes.",
        )
    try:
        original_argv = sys.argv.copy()
        sys.argv      = ["christman_preflight.py"]

        # Resolve package context so relative imports work correctly
        dotted_name, pkg_root = resolve_package_context(filepath)
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)

        # Pre-load parent packages so relative imports chain correctly
        parts = dotted_name.split(".")
        for i in range(1, len(parts)):
            parent_name = ".".join(parts[:i])
            if parent_name not in sys.modules:
                parent_path = os.path.join(pkg_root, *parts[:i], "__init__.py")
                if os.path.exists(parent_path):
                    pspec = importlib.util.spec_from_file_location(parent_name, parent_path)
                    pmod  = importlib.util.module_from_spec(pspec)
                    sys.modules[parent_name] = pmod
                    try:
                        pspec.loader.exec_module(pmod)
                    except Exception:
                        pass  # Best effort — parent failure should not block child check

        spec = importlib.util.spec_from_file_location(dotted_name, filepath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[dotted_name] = mod

        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = original_argv
            sys.modules.pop(dotted_name, None)

        return ModuleResult(
            name=name, filepath=filepath, passed=True,
            load_time_ms=round((time.perf_counter()-start)*1000, 1),
        )
    except SystemExit as e:
        sys.argv = original_argv
        return ModuleResult(
            name=name, filepath=filepath, passed=True,
            load_time_ms=round((time.perf_counter()-start)*1000, 1),
            error_msg=f"Module attempted system exit (code: {e.code}) — likely a CLI tool",
        )
    except Exception as e:
        sys.argv = original_argv
        tb = traceback.format_exc()
        root_cause, what_to_do, pip_pkg, ftype = extract_root_cause(tb, e, project_dir)
        return ModuleResult(
            name=name, filepath=filepath, passed=False,
            load_time_ms=round((time.perf_counter()-start)*1000, 1),
            failure_type=ftype, error_msg=str(e).split("\n")[0][:160],
            root_cause=root_cause, what_to_do=what_to_do,
            pip_package=pip_pkg, traceback_str=tb,
        )


# ── Frontend file checker ─────────────────────────────────────────────────────

FRONTEND_EXTENSIONS    = {".tsx", ".ts", ".css", ".js"}
FRONTEND_SKIP_PATTERNS = {".min.js", ".min.css", ".bundle.js", ".chunk.js", ".d.ts"}


def check_frontend_syntax(filepath: str) -> List[str]:
    errors = []
    try:
        source = open(filepath, "r", encoding="utf-8", errors="replace").read()
    except Exception as e:
        return [f"Could not read file: {e}"]
    if not source.strip():
        return []

    # Balance check outside strings and comments
    in_single = in_double = in_template = in_comment = in_block = False
    curly = paren = square = 0
    i = 0
    while i < len(source):
        ch      = source[i]
        next_ch = source[i+1] if i+1 < len(source) else ""
        if in_block:
            if ch == "*" and next_ch == "/": in_block = False; i += 2; continue
            i += 1; continue
        if in_comment:
            if ch == "\n": in_comment = False
            i += 1; continue
        if in_single:
            if ch == "\\": i += 2; continue
            if ch == "'": in_single = False
            i += 1; continue
        if in_double:
            if ch == "\\": i += 2; continue
            if ch == '"': in_double = False
            i += 1; continue
        if in_template:
            if ch == "\\": i += 2; continue
            if ch == "`": in_template = False
            i += 1; continue
        if ch == "/" and next_ch == "/": in_comment = True; i += 2; continue
        if ch == "/" and next_ch == "*": in_block = True; i += 2; continue
        if   ch == "'": in_single   = True
        elif ch == '"': in_double   = True
        elif ch == "`": in_template = True
        elif ch == "{": curly  += 1
        elif ch == "}": curly  -= 1
        elif ch == "(": paren  += 1
        elif ch == ")": paren  -= 1
        elif ch == "[": square += 1
        elif ch == "]": square -= 1
        i += 1

    if curly  != 0: errors.append(f"Mismatched curly braces {{ }}: net {curly:+d}")
    if paren  != 0: errors.append(f"Mismatched parentheses ( ): net {paren:+d}")
    if square != 0: errors.append(f"Mismatched square brackets [ ]: net {square:+d}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".ts", ".tsx"):
        if re.search(r"export\s+default\s*;", source):
            errors.append("export default; — missing value (empty export)")
        if ext == ".tsx" and "export default" in source and "return" not in source and "=>" not in source:
            errors.append("TSX component appears to have no return statement")

    return errors


def test_frontend_file(name: str, filepath: str) -> ModuleResult:
    start   = time.perf_counter()
    errors  = check_frontend_syntax(filepath)
    elapsed = round((time.perf_counter()-start)*1000, 1)
    if errors:
        return ModuleResult(
            name=name, filepath=filepath, passed=False,
            load_time_ms=elapsed, failure_type=FailureType.FRONTEND_ERROR,
            error_msg=errors[0], root_cause="; ".join(errors),
            what_to_do="Fix the structural issue(s) in this frontend file.",
            is_frontend=True,
        )
    return ModuleResult(name=name, filepath=filepath, passed=True, load_time_ms=elapsed, is_frontend=True)


# ── Discovery ─────────────────────────────────────────────────────────────────

SKIP_DIRS = {
    "venv", "__pycache__", "node_modules", ".git",
    "dist", "build", ".next", "coverage", ".turbo",
    "out", ".cache", "disabled_old_voice",
    "archive_learning_prototypes", "lstm_models", "Self_Improver",
}

ALWAYS_SKIP = {
    "christman_preflight","christman_preflight0","christman_preflight2",
    "christman_preflight3","derek_preflight","full_preflight",
    "fix_derek_ferrari","start_derek_ferrari","Brockston_Brain_CC1",
    "bridge","brockston_cortex","derek_mcp_server","derek_free_api",
    "normalize_indent","repair_app","silent_count","reclaim_modules",
    "flatten_repo","mass_fix","final_polisher","non_vital_sweep",
    "add_footer","add_license_headers","email_url_fixer",
    "aggressive_lint_eliminator","advanced_markdown_fixer",
    "markdown_lint_fixer","generate_patent_visuals",
    "generate_admin_hash","color_scheme_generator",
    "compare_environments","comprehensive_module_scan",
    "capacity_verification_report","complete_system_verification",
    "debug_startup","exact_count","final_count","find_stubs",
    "fix_critical","check_module_warnings","truth_audit",
    "verify_no_derek","setup_complete_environment",
    "create_test_lstm_models","train_lstm_model","train_models",
    "train_nonverbal_models","reclaim_modules","repo_guard",
}


def discover_modules(project_dir: str, skip: set, include_frontend: bool = True) -> List[Tuple[str, str]]:
    modules   = []
    root_path = os.path.abspath(project_dir)
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS and not d.startswith(".")]
        rel_root = os.path.relpath(root, root_path)
        for f in sorted(files):
            filepath = os.path.join(root, f)
            ext      = os.path.splitext(f)[1].lower()
            if f.endswith(".py"):
                stem = f[:-3]
                if stem not in skip:
                    modules.append((stem, filepath))
            elif include_frontend and ext in FRONTEND_EXTENSIONS:
                if any(f.endswith(p) for p in FRONTEND_SKIP_PATTERNS):
                    continue
                display = os.path.join(rel_root, f) if rel_root != "." else f
                modules.append((display, filepath))
    return modules


# ── Report ────────────────────────────────────────────────────────────────────

def divider(label="", color=None, width=72):
    if label:
        pad  = width - len(label) - 6
        line = f"  ╠══ {label} {'═' * max(0,pad)}╣"
    else:
        line = "  " + "═" * width
    print(c(line, color or C.DIM))


def render_report(results: List[ModuleResult], quiet: bool, project_dir: str):
    passed   = [r for r in results if r.passed]
    failed   = [r for r in results if not r.passed]
    py_total = sum(1 for r in results if not r.is_frontend)
    fe_total = sum(1 for r in results if r.is_frontend)
    total    = len(results)
    elapsed  = sum(r.load_time_ms for r in results)
    pct      = round(len(passed)/total*100) if total else 0

    if failed:
        by_cat = {}
        for r in failed:
            by_cat.setdefault(r.failure_type or FailureType.UNKNOWN, []).append(r)
        for ftype in CATEGORY_ORDER:
            group = by_cat.get(ftype)
            if not group: continue
            col   = CATEGORY_COLORS.get(ftype, C.DIM)
            icon  = CATEGORY_ICONS.get(ftype, "❓")
            label = CATEGORY_LABELS.get(ftype, ftype.value)
            print()
            divider(f"{icon}  {label}  [{len(group)}]", color=col)
            for r in group:
                tag = c(" [FE]", C.MAGENTA) if r.is_frontend else ""
                print()
                print(f"  {c('❌', C.RED)}  {c(r.name, C.WHITE, C.BOLD):<50}{tag}  {c(f'{r.load_time_ms}ms', C.DIM)}")
                print(f"     {c('FILE:', C.DIM)}  {c(r.filepath, C.DIM)}")
                print(f"     {c('WHAT:', C.DIM)}  {c(r.root_cause, col)}")
                print(f"     {c('FIX: ', C.DIM)}  {c(r.what_to_do, C.CYAN)}")
                if r.pip_package:
                    print(f"     {c('CMD: ', C.DIM)}  {c(f'pip install {r.pip_package}', C.GREEN)}")
                if not quiet and r.traceback_str:
                    print(c("     ── TRACEBACK ──────────────────────────────────────────────", C.DIM))
                    for line in r.traceback_str.strip().split("\n"):
                        print(c(f"     {line}", C.DIM))

    print()
    divider(f"✅  LOADED  [{len(passed)}]", color=C.GREEN)
    print()
    cols = 3
    for i, r in enumerate(passed):
        tag   = c(" [FE]", C.MAGENTA) if r.is_frontend else ""
        entry = f"  {c('✅', C.GREEN)} {c(r.name, C.WHITE):<36}{tag} {c(f'[{r.load_time_ms}ms]', C.DIM)}"
        print(entry, end="\n" if (i+1) % cols == 0 else "  ")
    if len(passed) % cols != 0:
        print()

    unique_pkgs = sorted(set(r.pip_package for r in failed if r.pip_package))
    if unique_pkgs:
        print()
        divider("📦  ONE COMMAND TO RESTORE ALL MISSING PACKAGES", color=C.YELLOW)
        print()
        print(f"  {c('pip install ' + ' '.join(unique_pkgs), C.GREEN, C.BOLD)}")

    print()
    print(c("  ╔══════════════════════════════════════════════════════════════════════╗", C.DIM))
    print(c("  ║  MISSION REPORT                                                      ║", C.WHITE))
    print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))

    def row(label, value, val_color=C.WHITE):
        clean = str(value)
        pad   = max(0, 68 - len(label) - len(clean))
        print(c("  ║  ", C.DIM) + c(label, C.DIM) + c(clean, val_color) + " "*pad + c("║", C.DIM))

    row("PROJECT          : ", os.path.basename(project_dir), C.CYAN)
    row("TARGET           : ", project_dir[:55], C.DIM)
    row("TOTAL MODULES    : ", str(total), C.WHITE)
    row("  Python         : ", f"{py_total}  ({sum(1 for r in passed if not r.is_frontend)} passed)", C.CYAN)
    if fe_total > 0:
        row("  Frontend        : ", f"{fe_total}  ({sum(1 for r in passed if r.is_frontend)} passed)", C.MAGENTA)
    row("LOADED           : ", f"{len(passed)}  ({pct}%)", C.GREEN)
    row("FAILED           : ", str(len(failed)), C.RED if failed else C.GREEN)
    row("LOAD TIME        : ", f"{elapsed:.0f}ms total", C.DIM)

    by_cat2 = {}
    for r in failed:
        ft = r.failure_type or FailureType.UNKNOWN
        by_cat2[ft] = by_cat2.get(ft, 0) + 1

    if by_cat2:
        print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))
        print(c("  ║  FAILURE BREAKDOWN                                                   ║", C.WHITE))
        print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))
        for ftype in CATEGORY_ORDER:
            if ftype in by_cat2:
                col  = CATEGORY_COLORS.get(ftype, C.DIM)
                icon = CATEGORY_ICONS.get(ftype, "❓")
                row(f"  {icon}  {ftype.value:<22}", str(by_cat2[ftype]), col)

    print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))

    if not failed:   status, scol = "🟢  ALL SYSTEMS GO — ready to fly.", C.GREEN
    elif pct >= 90:  status, scol = f"🟡  DEPLOYMENT READY — {pct}% loaded. Minor issues only.", C.YELLOW
    elif pct >= 80:  status, scol = f"🟡  MOSTLY READY — {pct}% loaded. Fix dependencies above.", C.YELLOW
    elif pct >= 50:  status, scol = f"🟠  DEGRADED — {pct}% loaded. Significant work needed.", C.ORANGE
    else:            status, scol = f"🔴  NOT READY — {pct}% loaded. Major dependencies missing.", C.RED

    print(c("  ║  ", C.DIM) + c(f"STATUS  :  {status:<59}", scol) + c("║", C.DIM))
    print(c("  ╚══════════════════════════════════════════════════════════════════════╝", C.DIM))
    print()
    print(c('  "How can we help you love yourself more?"', C.GREEN, C.BOLD))
    print(c("  © 2026 Everett Nathaniel Christman & The Christman AI Project", C.DIM))
    print()


def render_json(results, project_dir):
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    print(json.dumps({
        "project": project_dir,
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate_pct": round(len(passed)/len(results)*100) if results else 0,
        "python_total": sum(1 for r in results if not r.is_frontend),
        "frontend_total": sum(1 for r in results if r.is_frontend),
        "modules": [{"name":r.name,"filepath":r.filepath,"passed":r.passed,"is_frontend":r.is_frontend,"load_time_ms":r.load_time_ms,"failure_type":r.failure_type.value if r.failure_type else None,"error":r.error_msg,"root_cause":r.root_cause,"what_to_do":r.what_to_do,"pip_package":r.pip_package} for r in results],
        "pip_install_command": ("pip install " + " ".join(sorted(set(r.pip_package for r in failed if r.pip_package)))) if any(r.pip_package for r in failed) else "",
    }, indent=2))


# ── Autofix ───────────────────────────────────────────────────────────────────

def autofix_pip(results):
    import subprocess
    pkgs = sorted(set(r.pip_package for r in results if not r.passed and r.pip_package))
    if not pkgs:
        print(c("\n  ✅ No missing pip packages to install.", C.GREEN))
        return
    print(c(f"\n  📦 Installing {len(pkgs)} missing package(s)...", C.YELLOW, C.BOLD))
    result = subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs, capture_output=False)
    if result.returncode == 0:
        print(c("\n  ✅ All packages installed. Re-run preflight to verify.", C.GREEN, C.BOLD))
    else:
        print(c("\n  ⚠️  Some packages may have failed. Check output above.", C.YELLOW))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Christman Universal Preflight — Python + Frontend")
    parser.add_argument("--dir",         default=".")
    parser.add_argument("--skip",        default="")
    parser.add_argument("--quiet",       action="store_true")
    parser.add_argument("--json",        action="store_true")
    parser.add_argument("--no-color",    action="store_true")
    parser.add_argument("--no-frontend", action="store_true")
    parser.add_argument("--autofix",     action="store_true")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        C.disable()

    project_dir      = os.path.abspath(args.dir)
    include_frontend = not args.no_frontend

    if not os.path.isdir(project_dir):
        print(f"ERROR: {project_dir} is not a directory.")
        sys.exit(2)

    sys.path.insert(0, project_dir)
    skip    = ALWAYS_SKIP | {s.strip() for s in args.skip.split(",") if s.strip()}
    modules = discover_modules(project_dir, skip, include_frontend=include_frontend)

    py_count = sum(1 for _, fp in modules if fp.endswith(".py"))
    fe_count = len(modules) - py_count

    if not args.json:
        print_banner(project_dir)
        summary = f"  Scanning {len(modules)} files"
        if include_frontend and fe_count > 0:
            summary += f"  ({py_count} Python  +  {fe_count} Frontend)"
        print(c(summary + "...", C.DIM))
        print()

    results  = []
    n_passed = n_failed = 0

    for i, (name, filepath) in enumerate(modules, 1):
        ext    = os.path.splitext(filepath)[1].lower()
        result = test_frontend_file(name, filepath) if ext in FRONTEND_EXTENSIONS else test_module(name, filepath, project_dir)
        results.append(result)
        if result.passed: n_passed += 1
        else:             n_failed += 1
        if not args.json:
            progress_bar(i, len(modules), name, n_passed, n_failed)

    if not args.json:
        clear_progress()

    if args.json:
        render_json(results, project_dir)
    else:
        render_report(results, args.quiet, project_dir)
        if args.autofix:
            autofix_pip(results)

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()

# ==============================================================================
# © 2026 Everett Nathaniel Christman & The Christman AI Project
# Luma Cognify AI — "How can we help you love yourself more?"
# Patent Pending TCAP-2026-001
# Cardinal Rule 1: It has to actually work.
# Cardinal Rule 6: Fail loud.
# Cardinal Rule 13: Every failure tells you exactly what to do next.
# ==============================================================================
