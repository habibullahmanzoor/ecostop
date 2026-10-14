"""Static LaTeX lint (no TeX engine needed): brace balance, begin/end pairing by name, and per-row column counts in every
tabular (the usual cause of 'Extra alignment tab' errors)."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ecostop_paper_v2.tex"
t = open(path, encoding="utf-8").read()
body = re.sub(r"(?<!\\)%.*", "", t)          # strip comments
problems = 0

# 1. brace balance (ignoring escaped braces)
clean = body.replace("\\{", "").replace("\\}", "")
if clean.count("{") != clean.count("}"):
    print(f"BRACE IMBALANCE: {clean.count('{')} open vs {clean.count('}')} close"); problems += 1

# 2. begin/end pairing with a stack
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, name = m.groups()
    if kind == "begin":
        stack.append(name)
    else:
        if not stack or stack[-1] != name:
            print(f"MISMATCHED \\end{{{name}}} (open: {stack[-3:]})"); problems += 1
            if name in stack:
                while stack and stack.pop() != name:
                    pass
        else:
            stack.pop()
if stack:
    print("UNCLOSED environments:", stack); problems += 1


# 3. tabular column counts
def ncols(spec):
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    n, i = 0, 0
    while i < len(spec):
        c = spec[i]
        if c in "lrc":
            n += 1; i += 1
        elif c in "pmb":
            n += 1
            j = spec.index("{", i); depth = 0
            for k in range(j, len(spec)):
                depth += spec[k] == "{"; depth -= spec[k] == "}"
                if depth == 0:
                    i = k + 1; break
        else:
            i += 1
    return n


for m in re.finditer(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}", body, flags=re.S):
    spec, inner = m.group(1), m.group(2)
    n = ncols(spec)
    rows, depth, cur, i = [], 0, "", 0
    while i < len(inner):                       # split at top-level \ only (not inside {...} such as \shortstack)
        ch = inner[i]
        if ch == "{" and inner[i - 1:i] != "\\":
            depth += 1
        elif ch == "}" and inner[i - 1:i] != "\\":
            depth -= 1
        if depth == 0 and inner.startswith("\\\\", i):
            rows.append(cur); cur = ""; i += 2; continue
        cur += ch; i += 1
    rows = [r for r in rows + [cur] if r.strip()]
    for r in rows:
        r2 = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule(\([lr]+\))?\{[^}]*\})", "", r).strip()
        if not r2 or r2.startswith("\\multicolumn") and False:
            continue
        # count cells: '&' separators, with \multicolumn{k}{..}{..} spanning k columns
        cells = re.split(r"(?<!\\)&", r2)
        span = 0
        for c in cells:
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}", c)
            span += int(mc.group(1)) if mc else 1
        if span != n:
            print(f"COLUMN MISMATCH: spec has {n} columns, row spans {span}: {r2[:90]!r}"); problems += 1

# 4. stray control characters / broken refs
for pat, msg in ((r"\\S\n", "line break after \\S (broken \\ref)"), (r"\nef\{", "line starting with 'ef{' (broken \\ref)"),
                 (r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "control character")):
    if re.search(pat, t):
        print("FOUND:", msg); problems += 1
print("lint:", "OK, no problems found" if not problems else f"{problems} problem(s)")
