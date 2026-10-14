"""Embed every \\input{tables/*.tex} table into the paper and make all tabulars fit the page.

- replaces each \\input{tables/NAME.tex} with the file's content,
- wraps every tabular in adjustbox{max width=...} (shrinks only if too wide; width is the column for
  table, the text width for table*),
- uses \\footnotesize inside table floats,
- adds \\usepackage{adjustbox}.
Idempotent: running it again changes nothing.
"""
import os
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "ecostop_paper_v2.tex"
t = open(SRC, encoding="utf-8").read()

# 1. inline \input{tables/...}
def inline(m):
    path = m.group(1)
    return open(path, encoding="utf-8").read().strip("\n")

t = re.sub(r"\\input\{(tables/[^}]+\.tex)\}", inline, t)

# 2. package
if "\\usepackage{adjustbox}" not in t:
    t = t.replace("\\usepackage{amsthm}", "\\usepackage{amsthm}\n\\usepackage{adjustbox}", 1)

# 3. footnotesize inside table floats
def fix_float(m):
    return m.group(0).replace("\\small", "\\footnotesize")

t = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", fix_float, t, flags=re.S)

# 4. wrap each tabular
out, pos = [], 0
for m in re.finditer(r"\\begin\{tabular\}.*?\\end\{tabular\}", t, flags=re.S):
    before = t[: m.start()]
    star = before.rfind("\\begin{table*}")
    plain = before.rfind("\\begin{table}")
    width = "\\textwidth" if star > plain else "\\columnwidth"
    # skip if already wrapped
    if before.rstrip().endswith("max width=" + width + "}") or before.rstrip().endswith("\\begin{adjustbox}{max width=\\columnwidth}") \
            or before.rstrip().endswith("\\begin{adjustbox}{max width=\\textwidth}"):
        continue
    out.append((m.start(), m.end(), "\\begin{adjustbox}{max width=%s}\n%s\n\\end{adjustbox}" % (width, m.group(0))))
res, last = [], 0
for s, e, rep in out:
    res.append(t[last:s]); res.append(rep); last = e
res.append(t[last:])
t = "".join(res)

open(SRC, "w", encoding="utf-8").write(t)
print("inputs left:", len(re.findall(r"\\input\{", t)),
      "| tabulars:", t.count("\\begin{tabular}"), "| wrapped:", t.count("\\begin{adjustbox}"))
