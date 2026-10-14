"""Build the public release package (release/ and release/ecostop_release.zip): benchmark, frozen protocols, raw logs of the
confirmatory study, analysis code, result tables, figure sources and a SHA-256 manifest. Does not upload anything.
usage: python -m echostop.make_release"""
import glob
import hashlib
import json
import os
import shutil
import zipfile

ROOT = "release"
CODE = sorted(glob.glob("echostop/*.py"))
PROTOCOLS = ["b1_protocol.json", "c1_protocol.json", "c1_manifest.json", "c1_analysis_plan.md", "c1_framework_plan.md",
             "SLM_PLAN.md"]
TABLES = sorted(glob.glob("figures/tierA/*.csv")) + sorted(glob.glob("figures/tierA/*.json"))
FIGS = ["figures/" + f for f in ("fig_regime.pdf", "fig_pooled.pdf", "fig_pool_domain.pdf", "fig_c1.pdf", "fig_validity_splits.pdf",
        "fig_frontier_oof.pdf", "fig_b1_fresh_risk.pdf", "fig_trajectories.pdf", "figure1.pdf", "figure2.pdf", "figure_framework_v3.pdf")]
RAW = {"c1": "logs/c1", "b1": "logs/b1", "slm_pilot": "logs/slm/s0"}
EXTRA_LOGS = ["logs/c1_supervisor.log", "logs/c1_gpu_health.csv", "logs/c1_gpu_health_2.csv", "logs/slm_gpu_health.csv",
              "logs/slm_supervisor.log", "reference_verification.csv"]


def copy(src, dst):
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    n = 0
    n += copy("ecostop_bench/runs.csv.gz", f"{ROOT}/ecostop_bench/runs.csv.gz")
    for f in ("bench.py", "README.md", "export_data.py"):
        n += copy(f"ecostop_bench/{f}", f"{ROOT}/ecostop_bench/{f}")
    for f in PROTOCOLS:
        n += copy(f, f"{ROOT}/protocols/{f}")
    n += copy("protocols_DEVIATIONS_AND_DECISIONS.md", f"{ROOT}/protocols/DEVIATIONS_AND_DECISIONS.md")
    for f in CODE:
        n += copy(f, f"{ROOT}/code/{f}")
    n += copy("requirements.txt", f"{ROOT}/code/requirements.txt")
    n += copy("matplotlibrc", f"{ROOT}/code/matplotlibrc")
    n += copy("references.bib", f"{ROOT}/paper/references.bib")
    n += copy("ecostop_paper_v2.tex", f"{ROOT}/paper/main.tex")
    for f in TABLES:
        n += copy(f, f"{ROOT}/results/{os.path.basename(f)}")
    for f in FIGS:
        n += copy(f, f"{ROOT}/figures/{os.path.basename(f)}")
    # repository-layout copy so that the analysis scripts run unchanged: raw dev/fresh/C1 logs and the small trace caches
    for d in ("logs/pilot", "logs/calibration_tabular", "logs/calibration_openml", "logs/nlp_pilot", "logs/b1", "logs/b2", "logs/b4",
              "logs/c1", "logs/step_decay_test", "logs/slm"):
        for f in glob.glob(f"{d}/**/*", recursive=True):
            if os.path.isfile(f) and f.endswith((".csv", ".json")):
                n += copy(f, f"{ROOT}/repo/{f}")
    for f in ("cache/traces.pkl", "cache/lcpfn_traces.pkl", "cache/b1_lcpfn_traces.pkl", "cache/c1_lcpfn_traces.pkl", "cache/tierA_mats_all.pkl"):
        n += copy(f, f"{ROOT}/repo/{f}")
    for f in ("b1_protocol.json", "c1_protocol.json", "c1_manifest.json", "c1_analysis_plan.md", "matplotlibrc", "requirements.txt",
              "figures/tierA/tier1_scope_flags.csv"):
        n += copy(f, f"{ROOT}/repo/{f}")
    for f in CODE:
        n += copy(f, f"{ROOT}/repo/{f}")
    n += copy("echostop/__init__.py", f"{ROOT}/repo/echostop/__init__.py")
    for name, d in RAW.items():
        for f in glob.glob(f"{d}/**/*", recursive=True):
            if os.path.isfile(f) and f.endswith((".csv", ".json")):
                n += copy(f, f"{ROOT}/raw/{name}/{os.path.relpath(f, d)}")
    for f in EXTRA_LOGS:
        n += copy(f, f"{ROOT}/raw/{os.path.basename(f)}")
    readme = open("cache/release_readme.md", encoding="utf8").read() if os.path.exists("cache/release_readme.md") else "# EcoStop release\n"
    open(f"{ROOT}/README.md", "w", encoding="utf8").write(readme)
    cff = open("cache/release_citation.cff", encoding="utf8").read() if os.path.exists("cache/release_citation.cff") else ""
    if cff:
        open(f"{ROOT}/CITATION.cff", "w", encoding="utf8").write(cff)
    # manifest
    lines, total = [], 0
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            fp = os.path.join(dp, f)
            if f == "SHA256SUMS.txt":
                continue
            h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            lines.append(f"{h}  {os.path.relpath(fp, ROOT).replace(os.sep, '/')}")
            total += os.path.getsize(fp)
    lines.sort(key=lambda s: s.split("  ")[1])
    open(f"{ROOT}/SHA256SUMS.txt", "w", encoding="utf8").write("\n".join(lines) + "\n")
    with zipfile.ZipFile(f"{ROOT}/ecostop_release.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(ROOT):
            for f in fs:
                if f == "ecostop_release.zip":
                    continue
                fp = os.path.join(dp, f)
                z.write(fp, os.path.relpath(fp, ROOT))
    print(f"{len(lines)} files, {total / 1e6:.1f} MB raw, zip {os.path.getsize(f'{ROOT}/ecostop_release.zip') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
