#!/usr/bin/env python3
"""
Git contributor statistics: line counts per contributor, top languages, totals, percentages.
"""

import subprocess
import re
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

EXCLUDE_PATTERNS = [
    r"__pycache__",
    r"\.pyc$",
    r"[/\\]venv[/\\]",        # venv anywhere in path
    r"[/\\]\.venv[/\\]",
    r"[/\\]env[/\\]",
    r"node_modules/",
    r"\.git/",
    r"\.eggs/",
    r"[/\\]dist[/\\]",
    r"[/\\]build[/\\]",
    r"[/\\]target[/\\]",      # Rust/Java build dirs
    r"\.cargo/registry",
    r"Cargo\.lock$",
    r"package-lock\.json$",
    r"yarn\.lock$",
    r"bun\.lockb$",
    r"\.DS_Store$",
    r"\.idea/",
    r"\.vscode/",
    # Generated / artifact files
    r"[/\\]artifacts[/\\]",   # experiment outputs (gilgamesh sweeps, raster data)
    r"weights\.json$",         # serialised model weights
    r"site-packages[/\\]",     # anything inside a venv's site-packages
]

EXT_TO_LANG = {
    # Systems / embedded
    "rs":      "Rust",
    "c":       "C",
    "h":       "C",
    "cpp":     "C++",
    "hpp":     "C++",
    "ino":     "Arduino (C++)",
    # HDL / schematics
    "v":       "Verilog",
    "sv":      "SystemVerilog",
    "kicad_sch": "KiCad Schematic",
    "kicad_pcb": "KiCad PCB",
    "kicad_pro": "KiCad Project",
    # Python
    "py":      "Python",
    "ipynb":   "Jupyter Notebook",
    # Web / scripts
    "js":      "JavaScript",
    "ts":      "TypeScript",
    "tsx":     "TypeScript",
    "jsx":     "JavaScript",
    "html":    "HTML",
    "css":     "CSS",
    "sh":      "Shell",
    "bash":    "Shell",
    "zsh":     "Shell",
    # Data / config
    "toml":    "TOML",
    "json":    "JSON",
    "yaml":    "YAML",
    "yml":     "YAML",
    "xml":     "XML",
    "csv":     "CSV",
    # Docs
    "md":      "Markdown",
    "txt":     "Text",
    "tex":     "LaTeX",
    # Net / SPICE
    "net":     "Netlist",
    "asc":     "SPICE",
    "sp":      "SPICE",
    "cir":     "SPICE",
}

AUTHOR_ALIASES = {
    # normalise known aliases to a display name
    "158038827+etm-code@users.noreply.github.com": "Eoghan Collins",
    "thecubeplayzgames@gmail.com": "Jake",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_exclude(path: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, path):
            return True
    return False


def file_language(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return EXT_TO_LANG.get(ext, "Other")


def normalise_author(email: str) -> str:
    return AUTHOR_ALIASES.get(email.lower(), email)


# ── Data collection ───────────────────────────────────────────────────────────

def collect_stats():
    """
    Stream `git log --numstat` line-by-line to avoid buffering the whole output.
    Accumulates lines-added per author per language (deletions ignored).
    """
    cmd = [
        "git", "log",
        "--pretty=format:COMMIT|%ae",
        "--numstat",
        "--diff-filter=ACDMRT",   # skip renames/copies to avoid double-counting
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, errors="replace", bufsize=1,
    )

    # author -> language -> lines_added
    contrib: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    commit_counts: dict[str, int] = defaultdict(int)

    current_author = None
    commits_seen = 0

    for line in proc.stdout:          # type: ignore[union-attr]
        line = line.rstrip("\n")

        if line.startswith("COMMIT|"):
            email = line[7:]
            current_author = normalise_author(email)
            commit_counts[current_author] += 1
            commits_seen += 1
            if commits_seen % 10 == 0:
                print(f"\r  processed {commits_seen} commits…", end="", flush=True)
            continue

        if not line.strip() or current_author is None:
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added_s, _, path = parts
        if added_s == "-":          # binary file
            continue
        if should_exclude(path):
            continue

        try:
            added = int(added_s)
        except ValueError:
            continue

        contrib[current_author][file_language(path)] += added

    proc.wait()
    print(f"\r  processed {commits_seen} commits total.    ")
    return contrib, commit_counts


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_lines(n: int) -> str:
    return f"{n:,}"


def bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_table(contrib, commit_counts):
    # totals per author
    author_totals = {a: sum(langs.values()) for a, langs in contrib.items()}
    grand_total   = sum(author_totals.values())

    # sort by lines descending
    ranked = sorted(author_totals.items(), key=lambda x: x[1], reverse=True)

    W = 72
    print("╔" + "═" * W + "╗")
    print(f"║{'  GIT CONTRIBUTION REPORT':^{W}}║")
    print("╠" + "═" * W + "╣")

    for author, total in ranked:
        pct = (total / grand_total * 100) if grand_total else 0
        commits = commit_counts[author]

        print(f"║  {'Author:':<10} {author:<{W-14}}║")
        print(f"║  {'Commits:':<10} {commits:<6}  Lines added: {fmt_lines(total):<10} ({pct:.1f}%)  ║")
        print(f"║  {bar(pct):<{W-4}}  ║")

        # top 3 languages
        langs = contrib[author]
        top3 = sorted(langs.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (lang, lines) in enumerate(top3, 1):
            lpct = (lines / total * 100) if total else 0
            tag  = f"#{i}"
            print(f"║    {tag:<3} {lang:<22} {fmt_lines(lines):>8} lines  ({lpct:5.1f}%)     ║")

        print("╠" + "─" * W + "╣" if author != ranked[-1][0] else "╠" + "═" * W + "╣")

    # grand total row
    print(f"║  {'TOTAL LINES ADDED:':<30} {fmt_lines(grand_total):>{W-32}}║")
    print(f"║  {'(excluding pycache, venv, build dirs, lockfiles, etc.)':<{W}}║")
    print("╚" + "═" * W + "╝")

    # language breakdown across whole repo
    lang_totals: dict[str, int] = defaultdict(int)
    for langs in contrib.values():
        for lang, n in langs.items():
            lang_totals[lang] += n

    print()
    print("╔" + "═" * W + "╗")
    print(f"║{'  REPO-WIDE LANGUAGE BREAKDOWN':^{W}}║")
    print("╠" + "═" * W + "╣")
    for lang, lines in sorted(lang_totals.items(), key=lambda x: x[1], reverse=True):
        lpct = (lines / grand_total * 100) if grand_total else 0
        print(f"║  {lang:<25} {fmt_lines(lines):>10} lines   {bar(lpct, 14)}  {lpct:5.1f}%  ║")
    print("╚" + "═" * W + "╝")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Scanning git history…\n")
    contrib, commit_counts = collect_stats()
    print_table(contrib, commit_counts)
