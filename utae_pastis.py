"""Train/eval the UTAE baseline on PASTIS (uni- or multi-modal).

Vendored UTAE model + official PASTIS_Dataset live in utae/. Mirrors the OlmoEarth
finetune harness (config YAML, AdamW, CE w/ ignore_index, per-epoch val eval,
best-by-mIoU checkpoint, CSV log) but uses the official dataloader (full time series +
dates) which UTAE needs.

Runs in the base env (torch 2.12), NOT env_olmo:
    source env_login.sh        # or: module load ...; source env/bin/activate
    python -u utae_pastis.py --config configs/utae_s2.yaml

Folds follow the PASTIS benchmark: train=1,2,3  val=4  test=5.
"""
import argparse
import csv
import os
from dataclasses import dataclass, asdict, fields as dataclass_fields
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# NOTE: we compute metrics from a confusion matrix in plain torch rather than using
# torchmetrics. torchmetrics eagerly imports torchvision, and this env has torch 2.12
# with a torchvision built for torch 2.7 (the only one in the cluster wheelhouse), so
# importing it raises "torchvision::nms does not exist". UTAE needs neither.

from utae.dataloader import PASTIS_Dataset
from utae.collate import pad_collate
from utae.fusion import build_model

NUM_CLASSES = 20
IGNORE_INDEX = 19          # PASTIS void label
ALLOWED_SATS = ("S2", "S1A", "S1D")
RESULTS_CSV = "utae_pastis.csv"
CSV_FIELDS = ["timestamp", "checkpoint", "modalities", "fusion",
              "test_miou", "test_overall_acc", "test_macro_acc", "test_macro_f1"]


UTAE_DEFAULTS_YAML = os.path.join(os.path.dirname(__file__), "configs", "utae_defaults.yaml")


@dataclass
class UTAEConfig:
    # REQUIRED architecture field (no default): modalities. fusion is required when
    # multimodal (>1 sat); for uni-modal it's unused.
    modalities: "list[str] | None" = None
    fusion: "str | None" = None    # early | late (required if >1 modality)
    # tuning knobs (defaults from configs/utae_defaults.yaml)
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 0
    data_root: str = "data/PASTIS-R"

    def __post_init__(self):
        if not self.modalities:
            raise ValueError("Required config field not set (give via --set or YAML): modalities")
        bad = [s for s in self.modalities if s not in ALLOWED_SATS]
        if bad:
            raise ValueError(f"modalities {bad} not in {list(ALLOWED_SATS)}")
        if len(self.modalities) > 1:
            if self.fusion not in ("early", "late"):
                raise ValueError("multimodal run requires fusion=early|late (set via --set/YAML)")
        elif self.fusion is None:
            self.fusion = "early"  # unused for uni-modal; set a value so run_name is stable

    @property
    def run_name(self) -> str:
        mods = "".join(s.lower() for s in self.modalities)
        tag = mods if len(self.modalities) == 1 else f"{mods}_{self.fusion}"
        return f"utae_{tag}_lr{self.lr:g}_ep{self.epochs}"

    @property
    def ckpt_path(self) -> str:
        return f"checkpoints/{self.run_name}_best.pt"


_UTAE_TYPES = {f.name: str(f.type) for f in dataclass_fields(UTAEConfig)}


def _coerce(name, value):
    if not isinstance(value, str):
        return value
    t = _UTAE_TYPES.get(name, "str")
    if "bool" in t:
        return value.strip().lower() in ("true", "1", "yes")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    if "list" in t:
        return [v for v in value.split(",") if v]
    return value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> UTAEConfig:
    """utae_defaults.yaml + optional --config YAML + --set key=value (CLI wins)."""
    data: dict = {}
    if os.path.exists(UTAE_DEFAULTS_YAML):
        with open(UTAE_DEFAULTS_YAML) as f:
            data.update(yaml.safe_load(f) or {})
    if path:
        with open(path) as f:
            data.update(yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        data[k] = _coerce(k, v)
    unknown = set(data) - set(UTAEConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)}; allowed: {sorted(UTAEConfig.__dataclass_fields__)}")
    return UTAEConfig(**data)


def make_loader(cfg: UTAEConfig, folds, shuffle):
    ds = PASTIS_Dataset(folder=cfg.data_root, norm=True, target="semantic",
                        folds=folds, sats=cfg.modalities)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=pad_collate)


def _subset(d, mods):
    return {k: d[k] for k in mods}


def _forward(model, batch, device, mods):
    (data, dates), target = batch
    data = {k: v.to(device) for k, v in _subset(data, mods).items()}
    dates = {k: v.to(device) for k, v in _subset(dates, mods).items()}
    target = target.to(device)
    return model(data, dates), target


def _metrics_from_confmat(cm):
    """cm: (C,C) confusion matrix, rows=true, cols=pred (void row/col already excluded).
    Returns overall_acc (micro), macro_acc (mean recall), miou (macro), macro_f1."""
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    present = cm.sum(1) > 0           # classes that appear in ground truth
    overall_acc = tp.sum() / cm.sum().clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    iou = tp / (tp + fp + fn).clamp(min=1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    macro_acc = recall[present].mean()
    miou = iou[present].mean()
    macro_f1 = f1[present].mean()
    return {"miou": miou.item(), "overall_acc": overall_acc.item(),
            "macro_acc": macro_acc.item(), "macro_f1": macro_f1.item()}


@torch.no_grad()
def evaluate(model, loader, device, mods):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for batch in loader:
        logits, target = _forward(model, batch, device, mods)
        pred = logits.argmax(1).cpu().flatten()
        lab = target.cpu().flatten()
        valid = lab != IGNORE_INDEX
        cm += torch.bincount(lab[valid] * NUM_CLASSES + pred[valid],
                             minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    # drop the void class row/col from macro averages
    keep = [c for c in range(NUM_CLASSES) if c != IGNORE_INDEX]
    return _metrics_from_confmat(cm[keep][:, keep])


def _append_csv(row):
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(cfg: UTAEConfig):
    print(f"Run: {cfg.run_name}")
    print(f"Config: {asdict(cfg)}")
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(cfg, folds=[1, 2, 3], shuffle=True)
    val_loader = make_loader(cfg, folds=[4], shuffle=False)
    test_loader = make_loader(cfg, folds=[5], shuffle=False)

    model = build_model(cfg.modalities, cfg.fusion, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {len(cfg.modalities)} sat(s), fusion={cfg.fusion if len(cfg.modalities)>1 else 'n/a'}, {n_params:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_miou = float("-inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(cfg.epochs):
        model.train()
        last = float("nan")
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False):
            logits, target = _forward(model, batch, device, cfg.modalities)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        val = evaluate(model, val_loader, device, cfg.modalities)
        print(f"epoch {epoch+1}/{cfg.epochs} | train_loss {last:.4f} | val miou {val['miou']:.4f} | {val}")
        if val["miou"] > best_miou:
            best_miou = val["miou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, cfg.ckpt_path)
    test = evaluate(model, test_loader, device, cfg.modalities)
    print(f"\nBEST val miou {best_miou:.4f}")
    print(f"TEST {test}")
    _append_csv({
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M"),
        "checkpoint": os.path.basename(cfg.ckpt_path),
        "modalities": "+".join(cfg.modalities),
        "fusion": cfg.fusion if len(cfg.modalities) > 1 else "n/a",
        "test_miou": round(test["miou"], 6),
        "test_overall_acc": round(test["overall_acc"], 6),
        "test_macro_acc": round(test["macro_acc"], 6),
        "test_macro_f1": round(test["macro_f1"], 6),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UTAE baseline on PASTIS.")
    parser.add_argument("--config", default=None,
                        help="Optional YAML overriding configs/utae_defaults.yaml.")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value",
                        help="Override fields, e.g. --set modalities=S2,S1A fusion=late epochs=32")
    args = parser.parse_args()
    main(load_config(args.config, getattr(args, "set")))

# env.sh
# python -u utae_pastis.py --config configs/utae_s2.yaml          # ~63 mIoU target