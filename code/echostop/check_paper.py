"""Static integrity check of the LaTeX draft (no LaTeX engine needed)."""
import os
import re
import sys

tex = sys.argv[1] if len(sys.argv) > 1 else "ecostop_paper_v2.tex"
t = open(tex, encoding="utf-8").read()
bib = open("references.bib", encoding="utf-8").read()
keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
cited = set()
for m in re.findall(r"\\cite[tp]?\{([^}]+)\}", t):
    cited |= {k.strip() for k in m.split(",")}
print("cited:", len(cited), "| missing in bib:", sorted(cited - keys))
labels = set(re.findall(r"\\label\{([^}]+)\}", t))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", t))
print("refs without label:", sorted(refs - labels))
print("labels never referenced:", sorted(labels - refs))
for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", t) + re.findall(r"\\input\{([^}]+)\}", t):
    print(("OK      " if os.path.exists(f) else "MISSING ") + f)
print("placeholders:", re.findall(r"%%\w+%%|\\textbf\{\[[^\]]*\]\}", t))
print("unused bib keys:", sorted(keys - cited))
b = t.count("\\begin{"), t.count("\\end{")
print("begin/end counts:", b)
