"""Freeze B5c: matched cosine counterparts of the B1 step-decay arm."""
import hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "b5c_protocol.json"
if OUT.exists():
    sys.exit("REFUSING: b5c_protocol.json already exists")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
proto = {
  "study": "B5c matched cosine counterparts of the B1 step-decay arm",
  "question": ("In the fresh test the discontinuous schedule families also used different architectures. "
               "Holding dataset, architecture, optimiser, learning rate, weight decay, batch size and "
               "horizon fixed, do the cosine counterparts of the four B1 step-decay jobs carry lower "
               "risk at the same frozen threshold?"),
  "matched_fields_source": "logs/b1/vision/<dataset>_resnet18_step_seed0.meta.json",
  "arm": {"model": "resnet18", "schedule": "cosine (CosineAnnealingLR, T_max=epochs)",
          "opt": "sgd nesterov momentum 0.9", "lr": 0.1, "weight_decay": 5e-4,
          "epochs": 40, "batch_size": 256},
  "datasets": ["dtd", "oxford_pet", "semeion"],
  "seeds": [0, 1, 2],
  "fourth_pair": ("fashion_mnist: dev:fashion_mnist_resnet18 (cosine, same loader and seed) against "
                  "b1:fashion_mnist_resnet18_step; included as a fourth matched pair drawn from the "
                  "development corpus, labelled as such"),
  "evaluation": {"tolerance_eps_pp": 0.5, "gammas": {"rho_0.15": 0.14, "rho_0.20": 0.46},
                 "patience_k": 2, "plateau_gate": "schedule_relative_safeguard(40) = 5",
                 "gate": "declared-drop gate of the paper; inert on cosine by construction"},
  "predictions_fixed_before_the_run": {
     "P1": "cosine counterparts realise lower risk than their step-decay partners at the same threshold",
     "P2": "the gate is inert on all of them, since cosine declares no drop",
     "falsification": "if the cosine counterparts violate at a rate comparable to their step partners, the fresh-test split is an architecture effect and the stratification claim fails"},
  "hashes": {"echostop/b5_confound.py": sha(ROOT/"echostop/b5_confound.py"),
             "echostop/data.py": sha(ROOT/"echostop/data.py"),
             "echostop/calibration.py": sha(ROOT/"echostop/calibration.py")},
  "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
OUT.write_text(json.dumps(proto, indent=2), encoding="utf-8")
print("frozen", OUT.name, proto["frozen_utc"])
