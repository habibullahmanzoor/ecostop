"""Freeze the B5 protocol BEFORE any run. Refuses to overwrite an existing freeze."""
import hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "b5_protocol.json"
if OUT.exists():
    sys.exit(f"REFUSING: {OUT.name} already exists (protocol is frozen)")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

protocol = {
    "study": "B5 schedule-confound",
    "question": ("With architecture, dataset, optimiser, epochs and batch size held fixed, does the "
                 "learning-rate schedule family alone reproduce the risk split seen in B1, where the "
                 "discontinuous families were also different architectures?"),
    "fixed": {"dataset": "cifar10", "model": "resnet18", "epochs": 40, "batch_size": 256,
              "lr": 0.1, "weight_decay": 5e-4, "optimizer": "SGD nesterov momentum 0.9",
              "seeds": [0, 1, 2], "val_split": "echostop.data.get_dataset (corpus construction)"},
    "arms": {
        "cosine":   {"scheduler": "CosineAnnealingLR", "T_max": 40, "scheduled_events": []},
        "step":     {"scheduler": "MultiStepLR", "milestones": [20, 30], "gamma": 0.1,
                     "scheduled_events": [20, 30]},
        "onecycle": {"scheduler": "OneCycleLR", "max_lr": 0.1, "total_steps": 40,
                     "pct_start": 0.3, "div_factor": 10.0, "final_div_factor": 100.0,
                     "scheduled_events": [12]}
    },
    "evaluation": {
        "tolerance_eps_pp": 0.5,
        "gamma_frozen": {"rho_0.15": 0.14, "rho_0.20": 0.46},
        "gamma_source": "certified on the 142-job development corpus (paper, RQ1)",
        "patience_k": 2,
        "plateau_gate": "schedule_relative_safeguard(40) = 5",
        "gate": {"definition": "do not stop at epoch t when a scheduled event t_d satisfies "
                               "t_d - h <= t <= t_d",
                 "h_epochs": 2,
                 "h_source": "fixed a priori, not tuned on B5"},
        "primary_endpoint": "per-schedule violation rate at gamma=0.46 (rho=0.20)",
        "secondary": ["violation rate at gamma=0.14", "energy-weighted saving per schedule",
                      "gate on/off at gamma=0.46"]
    },
    "predictions_fixed_before_the_run": {
        "P1": "with the confound removed, the schedule still separates risk: cosine inside rho, step and one-cycle above it",
        "P2": "the gate reduces step and one-cycle risk without changing cosine, because cosine declares no event",
        "falsification": "if all three schedules land within rho, the B1 split was an architecture effect and the paper's central claim fails"
    },
    "hashes": {
        "echostop/b5_confound.py": sha(ROOT / "echostop/b5_confound.py"),
        "echostop/calibration.py": sha(ROOT / "echostop/calibration.py"),
        "echostop/methods.py": sha(ROOT / "echostop/methods.py"),
        "echostop/energy.py": sha(ROOT / "echostop/energy.py"),
    },
    "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
OUT.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
print("frozen ->", OUT.name)
print("trainer sha:", protocol["hashes"]["echostop/b5_confound.py"][:16], "...")
print("utc:", protocol["frozen_utc"])
