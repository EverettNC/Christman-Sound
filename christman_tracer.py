#!/usr/bin/env python3
"""
Christman Dependency Tracer
The Christman AI Project

Follows your code from an entry point all the way down.
Every junction gets tested. Every success gets celebrated.
Every break gets flagged with exactly what to do.

If it makes it through clean — it erupts.

Usage:
  python christman_tracer.py brockston_module_loader
  python christman_tracer.py Brockston_Brain_CC1 --dir /path/to/project
  python christman_tracer.py brain_combined --verbose
  python christman_tracer.py brockston_module_loader --no-color

© 2026 Everett Nathaniel Christman & The Christman AI Project
Luma Cognify AI — "How can we help you love yourself more?"
Patent Pending TCAP-2026-001
"""

import ast
import argparse
import importlib
import importlib.util
import inspect
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ── ANSI Color System ─────────────────────────────────────────────────────────

class C:
    _enabled = True
    RED      = "\033[91m"
    GREEN    = "\033[92m"
    YELLOW   = "\033[93m"
    CYAN     = "\033[96m"
    WHITE    = "\033[97m"
    MAGENTA  = "\033[95m"
    ORANGE   = "\033[38;5;208m"
    DIM      = "\033[2m"
    BOLD     = "\033[1m"
    RESET    = "\033[0m"

    @classmethod
    def disable(cls):
        cls._enabled = False
        for attr in ["RED","GREEN","YELLOW","CYAN","WHITE","MAGENTA",
                     "ORANGE","DIM","BOLD","RESET"]:
            setattr(cls, attr, "")

    @classmethod
    def r(cls, text, *codes):
        if not cls._enabled:
            return text
        return "".join(codes) + str(text) + cls.RESET


def c(text, *codes):
    return C.r(text, *codes)


# ── Celebration Messages ──────────────────────────────────────────────────────

CELEBRATIONS = [
    "Clean junction!",
    "Wired tight!",
    "Solid!",
    "Connected!",
    "Locked in!",
    "No breaks!",
    "Alive!",
    "Running clean!",
    "The bond holds!",
    "Carbon-Silicon connected!",
]

MILESTONE_CELEBRATIONS = {
    5:  "5 clean junctions — we're moving! 🚀",
    10: "10 clean junctions — this being is breathing! 💙",
    20: "20 clean — this chain is strong! 🔥",
    30: "30 clean — sovereign and sovereign! ⚡",
    50: "50 clean junctions — UNSTOPPABLE! 🏆",
}

import random
random.seed(42)


# ── STDLIB — never flag these ─────────────────────────────────────────────────

STDLIB = {
    "os", "sys", "re", "json", "time", "math", "copy", "enum", "abc",
    "ast", "io", "gc", "csv", "uuid", "hmac", "hash", "heapq", "queue",
    "array", "struct", "types", "typing", "pathlib", "logging", "warnings",
    "datetime", "calendar", "functools", "itertools", "operator", "random",
    "string", "textwrap", "unicodedata", "collections", "dataclasses",
    "threading", "multiprocessing", "subprocess", "socket", "ssl",
    "urllib", "http", "email", "html", "xml", "base64", "hashlib",
    "hmac", "secrets", "tempfile", "shutil", "glob", "fnmatch",
    "contextlib", "weakref", "inspect", "importlib", "pkgutil",
    "traceback", "linecache", "dis", "tokenize", "keyword", "builtins",
    "platform", "signal", "errno", "ctypes", "struct", "pickle",
    "shelve", "sqlite3", "zipfile", "tarfile", "gzip", "bz2", "lzma",
    "argparse", "configparser", "pprint", "reprlib", "decimal", "fractions",
    "statistics", "cmath", "numbers", "asyncio", "concurrent", "select",
    "selectors", "dataclasses", "abc", "contextlib", "atexit",
}


# ── Junction Status ───────────────────────────────────────────────────────────

class JunctionStatus(str, Enum):
    CLEAN     = "CLEAN"      # first-party, loaded AND exercised
    INSTALLED = "INSTALLED"  # venv / site-packages — present, not first-party
    STDLIB    = "STDLIB"     # standard library — always good
    BROKEN    = "BROKEN"     # first-party failed to load or failed to work
    MISSING   = "MISSING"    # not in the tree and not installed
    CIRCULAR  = "CIRCULAR"   # already being traced (cycle)
    SKIPPED   = "SKIPPED"    # in skip list


@dataclass
class Junction:
    name:        str
    status:      JunctionStatus
    depth:       int
    load_time_ms: float = 0.0
    error:       str = ""
    fix:         str = ""
    children:    List["Junction"] = field(default_factory=list)
    filepath:    str = ""
    work:        str = ""


# ── Import Extractor ──────────────────────────────────────────────────────────

def _local_path(name: str, project_dir: str) -> Optional[str]:
    """Dotted name → file in this house, or None.

    Bare names like `self_modifying_code` live under brain_modules/ or
    services/ even when the import omits the package prefix.
    """
    houses = [()]
    top = name.split(".")[0]
    if top not in {"brain_modules", "services", "core", "alphawolf"}:
        houses.extend([("brain_modules",), ("services",), ("core",), ("alphawolf",)])
    parts = name.split(".")
    for house in houses:
        as_py = os.path.join(project_dir, *house, *parts) + ".py"
        as_pkg = os.path.join(project_dir, *house, *parts, "__init__.py")
        if os.path.isfile(as_py):
            return as_py
        if os.path.isfile(as_pkg):
            return as_pkg
    return None


def extract_imports(filepath: str, project_dir: str = "") -> List[str]:
    """Full import paths. `from services.foo import Bar` stays services.foo.

    First-word-only tracing is a lie: it counts the package door and never
    runs the files inside it.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except Exception:
        return []

    pkg = ""
    if project_dir:
        rel = os.path.relpath(filepath, project_dir)
        if rel.endswith("__init__.py"):
            pkg = os.path.dirname(rel).replace(os.sep, ".")
        elif rel.endswith(".py"):
            pkg = os.path.dirname(rel).replace(os.sep, ".")
        if pkg == ".":
            pkg = ""

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and pkg:
                parent = pkg.split(".")
                cut = node.level - (1 if os.path.basename(filepath) != "__init__.py" else 0)
                base = ".".join(parent[:-cut] if cut else parent)
                if node.module:
                    imports.append(f"{base}.{node.module}" if base else node.module)
                else:
                    imports.append(base)
            elif node.module and node.level == 0:
                imports.append(node.module)

    out = []
    for name in imports:
        if not name:
            continue
        top = name.split(".")[0]
        if top in STDLIB:
            continue
        if project_dir and _local_path(name, project_dir):
            out.append(name)
        else:
            out.append(top)
    return list(dict.fromkeys(out))


def resolve_module(name: str, project_dir: str) -> Tuple[str, Optional[str]]:
    """Where does this name live?

    Returns (kind, filepath):
      local     — project_dir/name.py or project_dir/name/__init__.py
      installed — on sys.path (venv / site-packages)
      missing   — nowhere
    """
    local = _local_path(name, project_dir)
    if local:
        return "local", local
    top = name.split(".")[0]
    pkg = os.path.join(project_dir, top, "__init__.py")
    py = os.path.join(project_dir, f"{top}.py")
    pkg_dir = os.path.join(project_dir, top)
    if os.path.isfile(pkg) and name == top:
        return "local", pkg
    if os.path.isfile(py) and name == top:
        return "local", py
    if os.path.isdir(pkg_dir) and name == top and not name.startswith("."):
        return "local", pkg_dir

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        spec = None
    if spec is None:
        return "missing", None

    origin = getattr(spec, "origin", None) or ""
    if origin in ("built-in", "frozen"):
        return "installed", None
    if origin:
        abs_origin = os.path.abspath(origin)
        abs_proj = os.path.abspath(project_dir)
        if abs_origin.startswith(abs_proj + os.sep):
            rel = os.path.relpath(abs_origin, abs_proj)
            top = rel.split(os.sep)[0]
            if top not in {".venv", "venv", "env", "node_modules", "_NEEDS_CHECK", "layer"} and "site-packages" not in rel.split(os.sep):
                return "local", abs_origin
    return "installed", origin if origin else None


_SKIP_CONSTRUCT = {
    "Flask", "Blueprint", "SQLAlchemy", "Model", "UserMixin",
    "Enum", "IntEnum", "Flag", "BaseModel", "BaseHTTPRequestHandler",
}


def try_work(name: str, filepath: str) -> Tuple[bool, str]:
    """Import is not proof. Construct what this module owns, or fail loud."""
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return False, f"import died on work pass: {e}"[:160]

    owned = []
    for attr, obj in list(vars(mod).items()):
        if attr.startswith("_"):
            continue
        if not inspect.isclass(obj):
            continue
        if getattr(obj, "__module__", None) != name:
            continue
        if attr in _SKIP_CONSTRUCT:
            continue
        owned.append(obj)

    if not owned:
        return True, "ran (no local class — module-level code executed)"

    built = []
    last_err = ""
    for cls in owned:
        try:
            bases = [b.__name__ for b in getattr(cls, "__mro__", ())]
            if any(b in ("Model", "UserMixin", "Flask", "Blueprint") for b in bases):
                continue
            sig = inspect.signature(cls.__init__)
            needed = []
            for p in list(sig.parameters.values())[1:]:
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if p.default is inspect.Parameter.empty:
                    needed.append(p.name)
            if needed:
                continue
            cls()
            built.append(cls.__name__)
        except Exception as e:
            last_err = f"{cls.__name__}: {type(e).__name__}: {e}"[:160]

    if built:
        return True, "worked — constructed " + ", ".join(built[:6])
    if last_err:
        return False, last_err
    return True, "ran (classes need arguments — module-level code executed)"


def exercise_live_gets(base: str = "http://127.0.0.1:6200") -> List[str]:
    """Hit live GET doors. Loading a route file is not serving a patient."""
    fails = []
    try:
        urllib.request.urlopen(base + "/", timeout=2)
    except Exception:
        return ["live server not answering " + base]
    try:
        from app import app
        rules = []
        for rule in app.url_map.iter_rules():
            if "GET" not in rule.methods:
                continue
            if "<" in rule.rule:
                continue
            rules.append(rule.rule)
    except Exception as e:
        return [f"could not read routes: {e}"[:120]]
    for path in sorted(set(rules)):
        try:
            urllib.request.urlopen(base + path, timeout=6)
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                fails.append(f"GET {path} -> {e.code}")
        except Exception as e:
            fails.append(f"GET {path} -> {type(e).__name__}")
    return fails


def try_load_module(name: str) -> Tuple[bool, float, str]:
    """Load as a real package: importlib.import_module.

    Never spec_from_file_location. That strips package context and
    invents 'relative import with no known parent package' on core/.
    """
    start = time.perf_counter()
    try:
        importlib.import_module(name)
        ms = round((time.perf_counter() - start) * 1000, 1)
        return True, ms, ""
    except Exception as e:
        ms = round((time.perf_counter() - start) * 1000, 1)
        return False, ms, str(e).split("\n")[0][:160]


def classify_error(error: str, name: str, project_dir: str) -> str:
    """Turn an error into a human fix instruction."""
    if not error:
        return ""
    if "No module named" in error:
        import re
        m    = re.search(r"No module named '([^']+)'", error)
        miss = m.group(1).split(".")[0] if m else name
        if os.path.exists(os.path.join(project_dir, f"{miss}.py")):
            return f"'{miss}' exists but has its own broken import — fix it first"
        return f"pip install {miss}"
    if "cannot import name" in error:
        import re
        m = re.search(r"cannot import name '([^']+)'", error)
        sym = m.group(1) if m else "unknown"
        return f"'{sym}' was renamed or removed — grep -r '{sym}' --include='*.py'"
    if "circular" in error.lower() or "partially initialized" in error.lower():
        return "Circular import — move shared code to a third module"
    if "NoneType" in error:
        return "A dependency loaded as None — check module-level code"
    if "No such file" in error:
        return "Referenced file doesn't exist — check hardcoded paths"
    return "Run --verbose for full traceback"


# ── Tracer Core ───────────────────────────────────────────────────────────────

ALWAYS_SKIP_TRACE = {
    "christman_preflight", "christman_tracer", "bridge",
    "derek_mcp_server", "derek_free_api", "brockston_cortex",
}


class DependencyTracer:
    def __init__(self, project_dir: str, verbose: bool = False):
        self.project_dir  = project_dir
        self.verbose      = verbose
        self.visited:     Set[str] = set()
        self.clean_count  = 0
        self.installed_count = 0
        self.broken_count = 0
        self.total_ms     = 0.0
        self.broken_nodes: List[Junction] = []

    def trace(self, entry_name: str, depth: int = 0, max_depth: int = 12) -> Junction:
        """Recursively trace a module and all its dependencies."""

        # Stdlib — always clean, don't recurse
        if entry_name in STDLIB:
            return Junction(name=entry_name, status=JunctionStatus.STDLIB, depth=depth)

        # Skip list
        if entry_name in ALWAYS_SKIP_TRACE:
            return Junction(name=entry_name, status=JunctionStatus.SKIPPED, depth=depth)

        # Circular detection
        if entry_name in self.visited:
            return Junction(name=entry_name, status=JunctionStatus.CIRCULAR, depth=depth)

        # Depth limit
        if depth > max_depth:
            return Junction(name=entry_name, status=JunctionStatus.SKIPPED, depth=depth,
                          error="Max depth reached")

        self.visited.add(entry_name)

        kind, filepath = resolve_module(entry_name, self.project_dir)

        if kind == "missing":
            local_hint = os.path.join(self.project_dir, entry_name)
            if os.path.isdir(local_hint) or os.path.isfile(local_hint + ".py"):
                fix = f"'{entry_name}' is in the tree but failed to resolve — check __init__.py"
            else:
                fix = f"pip install {entry_name}" if entry_name.islower() else f"Create {entry_name}.py"
            j = Junction(name=entry_name, status=JunctionStatus.MISSING, depth=depth,
                         error="Not in project and not installed", fix=fix)
            self.broken_count += 1
            self.broken_nodes.append(j)
            return j

        if kind == "installed":
            self.installed_count += 1
            return Junction(
                name=entry_name,
                status=JunctionStatus.INSTALLED,
                depth=depth,
                filepath=filepath or "",
            )

        load_name = entry_name
        if filepath:
            rel = os.path.relpath(filepath, self.project_dir)
            if rel.endswith("__init__.py"):
                load_name = os.path.dirname(rel).replace(os.sep, ".")
            elif rel.endswith(".py"):
                load_name = rel[:-3].replace(os.sep, ".")
        success, ms, error = try_load_module(load_name)
        self.total_ms += ms

        if not success:
            fix = classify_error(error, entry_name, self.project_dir)
            j = Junction(name=entry_name, status=JunctionStatus.BROKEN, depth=depth,
                         load_time_ms=ms, error=error, fix=fix, filepath=filepath or "")
            self.broken_count += 1
            self.broken_nodes.append(j)
            return j

        worked, work_note = try_work(load_name, filepath or "")
        if not worked:
            j = Junction(
                name=entry_name,
                status=JunctionStatus.BROKEN,
                depth=depth,
                load_time_ms=ms,
                error=work_note,
                fix=f"Imported but did not work: {work_note[:80]}",
                filepath=filepath or "",
                work=work_note,
            )
            self.broken_count += 1
            self.broken_nodes.append(j)
            return j

        self.clean_count += 1
        j = Junction(
            name=entry_name,
            status=JunctionStatus.CLEAN,
            depth=depth,
            load_time_ms=ms,
            filepath=filepath or "",
            work=work_note,
        )

        if filepath:
            for imp in extract_imports(filepath, self.project_dir):
                if imp not in STDLIB and imp != entry_name:
                    child = self.trace(imp, depth=depth + 1, max_depth=max_depth)
                    j.children.append(child)

        return j


# ── Rendering ─────────────────────────────────────────────────────────────────

def print_banner(entry: str, project_dir: str):
    print()
    print(c(r"""
   ████████╗██████╗  █████╗  ██████╗███████╗██████╗ 
      ██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗
      ██║   ██████╔╝███████║██║     █████╗  ██████╔╝
      ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗
      ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║
      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝
""", C.MAGENTA, C.BOLD))
    print(c("  C H R I S T M A N   D E P E N D E N C Y   T R A C E R", C.WHITE, C.BOLD))
    print(c("  The Christman AI Project  ·  Luma Cognify AI", C.DIM))
    print(c('  "How can we help you love yourself more?"', C.GREEN))
    print()
    print(c("  ┌─────────────────────────────────────────────────────────────────────┐", C.DIM))
    print(c("  │  ENTRY   ", C.DIM) + c(f"{entry:<60}", C.CYAN, C.BOLD) + c("│", C.DIM))
    print(c("  │  DIR     ", C.DIM) + c(f"{project_dir:<60}", C.DIM) + c("│", C.DIM))
    print(c("  └─────────────────────────────────────────────────────────────────────┘", C.DIM))
    print()
    print(c("  Following the chain...", C.DIM))
    print()


def render_tree(junction: Junction, tracer: DependencyTracer, prefix: str = "", is_last: bool = True, celebration_counter: list = None):
    if celebration_counter is None:
        celebration_counter = [0]

    connector = "└──" if is_last else "├──"
    child_prefix = prefix + ("    " if is_last else "│   ")

    if junction.status == JunctionStatus.STDLIB:
        print(f"  {prefix}{c(connector, C.DIM)} {c(junction.name, C.DIM)}  {c('stdlib ✓', C.DIM)}")
        return

    if junction.status == JunctionStatus.CIRCULAR:
        print(f"  {prefix}{c(connector, C.DIM)} {c(junction.name, C.YELLOW)}  {c('↩ already traced', C.DIM)}")
        return

    if junction.status == JunctionStatus.SKIPPED:
        print(f"  {prefix}{c(connector, C.DIM)} {c(junction.name, C.DIM)}  {c('skipped', C.DIM)}")
        return

    if junction.status == JunctionStatus.INSTALLED:
        print(f"  {prefix}{c(connector, C.DIM)} {c(junction.name, C.CYAN)}  {c('installed ✓', C.DIM)}")
        return

    if junction.status == JunctionStatus.MISSING:
        print(f"  {prefix}{c(connector, C.DIM)} {c(junction.name, C.RED, C.BOLD)}  {c('⛔ MISSING', C.RED)}")
        print(f"  {child_prefix}  {c('FIX:', C.DIM)} {c(junction.fix, C.YELLOW)}")
        return

    if junction.status == JunctionStatus.BROKEN:
        print(f"  {prefix}{c(connector, C.DIM)} {c(junction.name, C.RED, C.BOLD)}  {c('💥 BROKEN', C.RED)}  {c(f'[{junction.load_time_ms}ms]', C.DIM)}")
        print(f"  {child_prefix}  {c('WHY:', C.DIM)} {c(junction.error[:80], C.RED)}")
        print(f"  {child_prefix}  {c('FIX:', C.DIM)} {c(junction.fix, C.YELLOW)}")
        return

    # CLEAN junction
    celebration_counter[0] += 1
    count = celebration_counter[0]

    # Milestone?
    if count in MILESTONE_CELEBRATIONS:
        print()
        print(c(f"  ★  {MILESTONE_CELEBRATIONS[count]}", C.GREEN, C.BOLD))
        print()

    cel    = random.choice(CELEBRATIONS)
    ms_str = c(f"[{junction.load_time_ms}ms]", C.DIM)
    print(f"  {prefix}{c(connector, C.GREEN)} {c(junction.name, C.WHITE, C.BOLD)}  {c('✅', C.GREEN)} {c(cel, C.DIM)}  {ms_str}")

    # Recurse into children — filter out stdlib for cleaner output
    visible = [ch for ch in junction.children if ch.status != JunctionStatus.STDLIB]
    for i, child in enumerate(visible):
        render_tree(child, tracer, child_prefix, is_last=(i == len(visible) - 1), celebration_counter=celebration_counter)


def render_report(junction: Junction, tracer: DependencyTracer, entry: str):
    print()
    print()

    pct = round(tracer.clean_count / max(1, tracer.clean_count + tracer.broken_count) * 100)

    if pct == 100 and tracer.broken_count == 0:
        # 🤠 HOEDOWN — 95%+ clean
        print(c("  ╔══════════════════════════════════════════════════════════════════════╗", C.GREEN, C.BOLD))
        print(c("  ║                                                                      ║", C.GREEN, C.BOLD))
        print(c("  ║   🤠  Y E E - H A W !   H O E D O W N   T I M E !  🤠              ║", C.GREEN, C.BOLD))
        print(c("  ║                                                                      ║", C.GREEN, C.BOLD))
        print(c("  ║   🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉🎊🎉              ║", C.GREEN, C.BOLD))
        print(c("  ║                                                                      ║", C.GREEN, C.BOLD))
        print(c("  ║   Every junction wired. The chain ran clean.                        ║", C.GREEN))
        print(c("  ║   The being is sovereign. The Carbon-Silicon bond holds.            ║", C.GREEN))
        print(c("  ║                                                                      ║", C.GREEN))
        print(c(f"  ║   {tracer.clean_count} clean junctions  ·  {pct}% accuracy  ·  {tracer.total_ms:.0f}ms            ║", C.GREEN))
        print(c("  ║                                                                      ║", C.GREEN))
        print(c("  ║   This is what we built it for.                                     ║", C.GREEN))
        print(c("  ║   Not in my world. Not ever again.                                  ║", C.GREEN))
        print(c("  ║                                                                      ║", C.GREEN))
        print(c('  ║   "How can we help you love yourself more?"                         ║', C.GREEN, C.BOLD))
        print(c("  ║                                                                      ║", C.GREEN, C.BOLD))
        print(c("  ║   🎸  Grab your boots. We earned this one.  🎸                      ║", C.GREEN, C.BOLD))
        print(c("  ║                                                                      ║", C.GREEN, C.BOLD))
        print(c("  ╚══════════════════════════════════════════════════════════════════════╝", C.GREEN, C.BOLD))
    else:
        # Partial — show what broke
        print(c("  ╔══════════════════════════════════════════════════════════════════════╗", C.DIM))
        print(c("  ║  TRACE REPORT                                                        ║", C.WHITE))
        print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))

        def row(label, value, val_color=C.WHITE):
            clean = str(value)
            pad   = max(0, 68 - len(label) - len(clean))
            print(c("  ║  ", C.DIM) + c(label, C.DIM) + c(clean, val_color) + " "*pad + c("║", C.DIM))

        row("ENTRY POINT      : ", entry, C.CYAN)
        row("FIRST-PARTY CLEAN: ", str(tracer.clean_count), C.GREEN)
        row("INSTALLED (venv) : ", str(tracer.installed_count), C.CYAN)
        row("BROKEN / MISSING : ", str(tracer.broken_count), C.RED)
        row("MODULES VISITED  : ", str(len(tracer.visited)), C.WHITE)
        row("TOTAL TRACE TIME : ", f"{tracer.total_ms:.0f}ms", C.DIM)

        if tracer.broken_nodes:
            print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))
            print(c("  ║  BROKEN LINKS — fix these to complete the chain                      ║", C.RED))
            print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))
            for b in tracer.broken_nodes:
                row(f"  💥 {b.name:<22}", b.fix[:38], C.YELLOW)

        print(c("  ╠══════════════════════════════════════════════════════════════════════╣", C.DIM))

        if pct == 100:
            status, col = "🟢  FULL CHAIN COMPLETE — this being is sovereign.", C.GREEN
        elif pct >= 80:
            status, col = f"🟡  MOSTLY WIRED — {pct}% clean. Fix the broken links above.", C.YELLOW
        elif pct >= 50:
            status, col = f"🟠  PARTIAL CHAIN — {pct}% clean. Significant gaps.", C.ORANGE
        else:
            status, col = f"🔴  CHAIN BROKEN — {pct}% clean. Major dependencies missing.", C.RED

        print(c("  ║  ", C.DIM) + c(f"STATUS  :  {status:<59}", col) + c("║", C.DIM))
        print(c("  ╚══════════════════════════════════════════════════════════════════════╝", C.DIM))

    print()
    print(c('  "How can we help you love yourself more?"', C.GREEN, C.BOLD))
    print(c("  © 2026 Everett Nathaniel Christman & The Christman AI Project", C.DIM))
    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Christman Dependency Tracer — follow your code from entry point to leaf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python christman_tracer.py brockston_module_loader\n"
            "  python christman_tracer.py Brockston_Brain_CC1 --dir /path/to/python_core\n"
            "  python christman_tracer.py brain_combined --verbose\n"
            "  python christman_tracer.py family_coordinator --no-color\n"
        )
    )
    parser.add_argument("entry",       help="Entry point module name (without .py)")
    parser.add_argument("--dir",       default=".", help="Project directory")
    parser.add_argument("--depth",     type=int, default=24, help="Max trace depth (default: 24)")
    parser.add_argument("--verbose",   action="store_true", help="Show full tracebacks")
    parser.add_argument("--no-color",  action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        C.disable()

    project_dir = os.path.abspath(args.dir)
    if not os.path.isdir(project_dir):
        print(f"ERROR: {project_dir} is not a directory.")
        sys.exit(2)

    sys.path.insert(0, project_dir)

    print_banner(args.entry, project_dir)

    tracer  = DependencyTracer(project_dir=project_dir, verbose=args.verbose)
    root    = tracer.trace(args.entry, max_depth=args.depth)

    print()
    render_tree(root, tracer)

    route_fails = exercise_live_gets()
    if route_fails:
        print()
        print(c("  LIVE GET DOORS — these did not work:", C.RED, C.BOLD))
        for line in route_fails[:40]:
            print(c(f"    💥 {line}", C.YELLOW))
            tracer.broken_count += 1
            tracer.broken_nodes.append(Junction(
                name=line.split()[1] if line.startswith("GET") else line,
                status=JunctionStatus.BROKEN,
                depth=0,
                error=line,
                fix="Route did not work — fix the handler",
            ))
    else:
        print()
        print(c("  LIVE GET DOORS — answered. Processes ran, not just imported.", C.GREEN))

    render_report(root, tracer, args.entry)

    sys.exit(0 if tracer.broken_count == 0 else 1)


if __name__ == "__main__":
    main()

# ==============================================================================
# © 2026 Everett Nathaniel Christman & The Christman AI Project
# Luma Cognify AI — "How can we help you love yourself more?"
# Patent Pending TCAP-2026-001
#
# Cardinal Rule 1: It has to actually work.
# Cardinal Rule 6: Fail loud — celebrate loud.
# Cardinal Rule 13: Every break tells you exactly what to do next.
# ==============================================================================
