"""Bundle the paper for Overleaf: main.tex + references.bib + exactly the figures/tables it includes,
in the folder layout the LaTeX expects. Output: overleaf_upload/ and overleaf_upload.zip."""
import os
import re
import shutil
import zipfile

SRC = "ecostop_paper_v2.tex"
tex = open(SRC, encoding="utf-8").read()
need = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex) + re.findall(r"\\input\{([^}]+)\}", tex)))
out = "overleaf_upload"
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out + "/figures")
shutil.copy(SRC, out + "/main.tex")
shutil.copy("references.bib", out + "/references.bib")
for f in need:
    assert os.path.exists(f), "missing " + f
    shutil.copy(f, out + "/" + f)
if os.path.exists("overleaf_upload.zip"):
    os.remove("overleaf_upload.zip")
with zipfile.ZipFile("overleaf_upload.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(out):
        for fn in files:
            p = os.path.join(root, fn)
            z.write(p, os.path.relpath(p, out))
print("files in zip:")
for n in sorted(zipfile.ZipFile("overleaf_upload.zip").namelist()):
    print("  ", n)
print("size KB:", os.path.getsize("overleaf_upload.zip") // 1024)
