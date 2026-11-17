"""Train ResNet-18 on one split, keep the best checkpoint, evaluate the test set once.

    python -m src.train --fold 0 --seed 1 --pretrained true
    python -m src.train --fold 0 --seed 1 --pretrained false
    python -m src.train --fold 0 --seed 1 --pretrained true --smoke   # quick check + time estimate
    python -m src.train --fold 0 --seed 1 --pretrained true --config configs/train_aug.yaml       # Change A
    python -m src.train --fold 0 --seed 1 --pretrained true --config configs/train_valactor.yaml  # Change B

Optional config keys (absent in configs/train.yaml, so the baseline is unchanged):
    run_tag                 output folder suffix instead of pretrained/scratch
    train.augmentation      dict of training-only augmentation settings (src/data/augment.py)
    validation              {mode: holdout_actor, val_actor_by_fold: {...}}: one training actor = validation
A folder that already contains DONE is never overwritten.

Output folder (default models/fold{k}_seed{s}_{run_tag | pretrained | scratch}/):
    best.pt            best checkpoint (by validation macro-F1)
    settings.json      every setting, seeds, library versions, GPU
    epochs.csv         per epoch: train loss, val loss, val accuracy, val macro-F1, seconds
    test_predictions.csv  clip_id, true / predicted label and the 7 probabilities
    metrics.json       test metrics (computed once, with best.pt), shortcut baselines, timing
    DONE               written last; its presence means the run finished
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.augment import LogMelAugment  # noqa: E402
from src.data.splits import fold_statistics  # noqa: E402
from src.data.torch_dataset import SpectrogramDataset, split_paths  # noqa: E402
from src.evaluate import accuracy_by, classification_metrics, length_baseline, top_confusions  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402
from src.utils import REPO_ROOT, load_config, logmel_dir  # noqa: E402


def set_all_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def holdout_actor_split(split_df: pd.DataFrame, val_actor: str) -> pd.DataFrame:
    """Change B: all clips of `val_actor` become validation, all clips of the other training actors train.

    The test actor (split == "test") is untouched.
    """
    d = split_df.copy()
    trainval = d["split"].isin(["train", "val"])
    assert val_actor in set(d.loc[trainval, "actor"]), f"{val_actor} is not a training actor in this fold"
    d.loc[trainval, "split"] = np.where(d.loc[trainval, "actor"] == val_actor, "val", "train")
    assert not (set(d.loc[d["split"] == "test", "actor"]) & set(d.loc[d["split"] != "test", "actor"]))
    return d


def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("use true or false")


@torch.no_grad()
def predict(model, loader, device, amp: bool, channels_last: bool):
    model.eval()
    logits_all, y_all, loss_sum = [], [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        loss_sum += ce(out.float(), y).item()
        logits_all.append(out.float().cpu())
        y_all.append(y.cpu())
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    return logits, y, loss_sum / len(y)


def train_one_epoch(model, loader, opt, scaler, device, amp, channels_last, max_batches=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, n, losses = 0.0, 0, []
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = ce(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
        losses.append(loss.item())
    return total / n, losses


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0", help="0..3, or 'random'")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pretrained", type=str2bool, default=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "train.yaml"))
    ap.add_argument("--smoke", action="store_true", help="few batches + one timed epoch, no outputs kept")
    args = ap.parse_args(argv)
    fold = args.fold if args.fold == "random" else int(args.fold)

    tcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pcfg = load_config(REPO_ROOT / tcfg["preprocess_config"])
    if tcfg.get("logmel_dir"):   # planted-tone experiment: train / evaluate on a separate, modified cache
        pcfg["paths"]["logmel_dir"] = (REPO_ROOT / tcfg["logmel_dir"]).resolve()
    tr_cfg, op_cfg = tcfg["train"], tcfg["optim"]
    styles = pcfg["labels"]["styles"]
    tag = tcfg.get("run_tag") or ("pretrained" if args.pretrained else "scratch")
    out = Path(args.out) if args.out else REPO_ROOT / tcfg["models_dir"] / f"fold{fold}_seed{args.seed}_{tag}"
    if not args.smoke and (out / "DONE").exists():
        sys.exit(f"{out} already holds a finished run; refusing to overwrite it (use another run_tag or --out)")

    assert torch.cuda.is_available(), "CUDA GPU not available"
    device = torch.device("cuda")
    amp, cl = bool(tr_cfg["mixed_precision"]), bool(tr_cfg["channels_last"])
    set_all_seeds(args.seed, tr_cfg["deterministic"])

    t0 = time.time()
    # Split: the fold's CSV, or (Change B) one training actor held out entirely as validation.
    split_df = pd.read_csv(split_paths(pcfg, fold)[0])
    stats, val_actor = None, None
    vcfg = tcfg.get("validation") or {"mode": "fold_csv"}
    if vcfg["mode"] == "holdout_actor":
        val_actor = vcfg["val_actor_by_fold"][fold]
        split_df = holdout_actor_split(split_df, val_actor)
        index = pd.read_csv(logmel_dir(pcfg) / "index.csv")
        stats, _ = fold_statistics(pcfg, split_df, index)  # TRAIN (2 actors) only, real frames only
    elif vcfg["mode"] != "fold_csv":
        sys.exit(f"unknown validation.mode {vcfg['mode']}")
    fills = None
    if tcfg.get("logmel_dir"):   # the modified cache changes the TRAIN statistics: recompute them (same method)
        stats, fills = fold_statistics(pcfg, split_df, pd.read_csv(logmel_dir(pcfg) / "index.csv"))
    aug_cfg = tr_cfg.get("augmentation", "none")
    augment = None
    if isinstance(aug_cfg, dict):
        augment = LogMelAugment(aug_cfg, pcfg["spectrogram"]["floor_db"], seed=args.seed)
    elif aug_cfg != "none":
        sys.exit(f"unknown train.augmentation {aug_cfg}")
    ds = {p: SpectrogramDataset(pcfg, fold, p, split_df=split_df, stats=stats,
                                augment=augment if p == "train" else None) for p in ("train", "val", "test")}
    print(f"[data] fold {fold}: " + ", ".join(f"{p} {len(d)}" for p, d in ds.items())
          + f" | loaded in {time.time() - t0:.0f} s", flush=True)
    g = torch.Generator().manual_seed(args.seed)
    bs, nw = tr_cfg["batch_size"], tr_cfg["num_workers"]
    train_loader = DataLoader(ds["train"], batch_size=bs, shuffle=True, generator=g, num_workers=nw,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(ds["val"], batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(ds["test"], batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)

    model = build_resnet18(tcfg["model"]["n_classes"], tcfg["model"]["in_channels"], args.pretrained).to(device)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    lr = op_cfg["lr_pretrained"] if args.pretrained else op_cfg["lr_scratch"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=op_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr_cfg["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if args.smoke:
        x, y = next(iter(train_loader))
        print(f"[smoke] batch x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)}, x mean {x.mean():.3f} std {x.std():.3f}")
        _, losses = train_one_epoch(model, train_loader, opt, scaler, device, amp, cl, max_batches=40)
        print(f"[smoke] loss first 5 batches {np.mean(losses[:5]):.3f} -> last 5 batches {np.mean(losses[-5:]):.3f}")
        t = time.time()
        train_one_epoch(model, train_loader, opt, scaler, device, amp, cl)
        logits, yv, vloss = predict(model, val_loader, device, amp, cl)
        torch.cuda.synchronize()
        ep = time.time() - t
        print(f"[smoke] logits {tuple(logits.shape)}; one full epoch (train + val) = {ep:.1f} s; "
              f"max {tr_cfg['max_epochs']} epochs = {ep * tr_cfg['max_epochs'] / 60:.1f} min upper bound")
        return

    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "fold": fold, "seed": args.seed, "pretrained": args.pretrained, "lr": lr,
        "seeds_fixed": {"python": args.seed, "numpy": args.seed, "torch": args.seed, "dataloader_shuffle": args.seed},
        "train_config": tcfg, "config_file": str(Path(args.config).resolve().relative_to(REPO_ROOT)),
        "run_tag": tag, "split_file": str(split_paths(pcfg, fold)[0].relative_to(REPO_ROOT)),
        "validation": {"mode": vcfg["mode"], "val_actor": val_actor,
                       "actors": {p: sorted(d.items["actor"].unique()) for p, d in ds.items()}},
        "n_clips": {p: len(d) for p, d in ds.items()},
        "normalisation": {"mean_db": ds["train"].mean, "std_db": ds["train"].std},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "gpu": torch.cuda.get_device_name(0), "command": " ".join(sys.argv),
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if fills is not None:
        (out / "norm_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        (out / "fill_values.json").write_text(json.dumps(fills, indent=2), encoding="utf-8")
    if val_actor is not None:
        split_df.to_csv(out / "split_used.csv", index=False)

    log, best_f1, best_epoch, bad = [], -1.0, -1, 0
    t_train = time.time()
    for epoch in range(1, tr_cfg["max_epochs"] + 1):
        t = time.time()
        tr_loss, _ = train_one_epoch(model, train_loader, opt, scaler, device, amp, cl)
        logits, yv, vloss = predict(model, val_loader, device, amp, cl)
        sched.step()
        pv = logits.argmax(1).numpy()
        vacc = float((pv == yv).mean())
        vf1 = float(f1_score(yv, pv, average="macro", labels=list(range(len(styles))), zero_division=0))
        secs = time.time() - t
        log.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vloss, "val_accuracy": vacc,
                    "val_macro_f1": vf1, "lr": opt.param_groups[0]["lr"], "seconds": round(secs, 2)})
        pd.DataFrame(log).to_csv(out / "epochs.csv", index=False)
        improved = vf1 > best_f1
        if improved:
            best_f1, best_epoch, bad = vf1, epoch, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        print(f"epoch {epoch:2d}  train_loss {tr_loss:.4f}  val_loss {vloss:.4f}  val_acc {vacc:.4f}  "
              f"val_f1 {vf1:.4f}  {secs:5.1f}s{'  *best' if improved else ''}", flush=True)
        if bad >= tr_cfg["early_stopping_patience"]:
            print(f"early stopping: no better val macro-F1 for {bad} epochs")
            break
    train_seconds = time.time() - t_train

    # Test set: used once, with the best checkpoint.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    logits, yt, tloss = predict(model, test_loader, device, amp, cl)
    prob = torch.softmax(logits, 1).numpy()
    pt = prob.argmax(1)
    pred = pd.DataFrame({"clip_id": ds["test"].clip_ids, "y_true": yt, "y_pred": pt})
    for i, s in enumerate(styles):
        pred[f"prob_{s}"] = prob[:, i].round(6)
    pred.to_csv(out / "test_predictions.csv", index=False)

    manifest = pd.read_csv(pcfg["paths"]["processed_dir"] / "manifest.csv", keep_default_na=False)
    pm = pred.join(manifest.set_index("clip_id")[["style", "was_cropped", "actor"]], on="clip_id")
    m = classification_metrics(yt, pt, styles)
    metrics = {
        "fold": fold, "seed": args.seed, "pretrained": args.pretrained,
        "test_actor": sorted(pm["actor"].unique()), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "epochs_run": len(log), "test_loss": tloss, **m,
        "accuracy_by_was_cropped": accuracy_by(pm, "was_cropped"),
        "accuracy_by_style": accuracy_by(pm, "style"),
        "top_confusions": top_confusions(m["confusion_counts"], styles),
        "baselines": length_baseline(manifest, split_df, styles, args.seed),
        "timing": {"seconds_per_epoch_mean": float(np.mean([r["seconds"] for r in log])),
                   "train_seconds_total": round(train_seconds, 1), "gpu": settings["gpu"]},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "DONE").write_text("finished\n", encoding="utf-8")
    print(f"[test] accuracy {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  (best epoch {best_epoch})")


if __name__ == "__main__":
    main()
