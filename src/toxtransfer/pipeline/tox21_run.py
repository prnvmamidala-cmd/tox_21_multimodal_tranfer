"""Within-Tox21 chemical baselines under scaffold and random splits.

Biological signatures are not invented here. If a signature matrix is absent,
this module records that the multimodal Tox21 arm was not run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

from toxtransfer.data.tox21 import NR_TASKS, SR_TASKS, TASKS, prepare_tox21
from toxtransfer.evaluation.metrics import classification_metrics
from toxtransfer.features.preprocess import fit_preprocess, transform


def run_tox21(cfg: dict, root: Path, output_dir: Path) -> Path:
    frame = prepare_tox21(root / cfg["data"]["tox21_path"])
    X = frame.attrs["fingerprints"]
    groups = frame["scaffold"].to_numpy()
    n_splits = int(cfg["transfer"].get("tox21_n_splits", 5))
    min_class = int(cfg["transfer"].get("tox21_min_class", 20))
    continuous = np.zeros(X.shape[1], dtype=bool)
    rows = []
    fold_sets = {
        "scaffold_oof": list(GroupKFold(n_splits=n_splits).split(X, groups=groups)),
        "random_oof": list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X)),
    }
    for phase, folds in fold_sets.items():
        for task_i, task in enumerate(TASKS):
            y = frame[task].to_numpy(dtype=float)
            oof = np.full(len(y), np.nan)
            for train, test in folds:
                tr = train[np.isfinite(y[train])]
                te = test[np.isfinite(y[test])]
                if len(np.unique(y[tr])) < 2 or len(te) == 0:
                    continue
                state = fit_preprocess(X[tr], continuous, clip=False, scale=True)
                model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
                model.fit(transform(X[tr], state), y[tr])
                oof[te] = model.predict_proba(transform(X[te], state))[:, 1]
            labeled = np.isfinite(y) & np.isfinite(oof)
            for i in np.where(labeled)[0]:
                rows.append(
                    {
                        "task": task,
                        "family": "NR" if task in NR_TASKS else "SR",
                        "phase": phase,
                        "inchikey_or_row": int(i),
                        "y_true": float(y[i]),
                        "y_pred": float(oof[i]),
                        "scaffold": groups[i],
                    }
                )
            print(f"tox21 {phase} {task}", flush=True)
    pred = pd.DataFrame(rows)
    path = output_dir / "predictions" / "tox21_predictions.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(path, index=False, compression="gzip")
    summary = _summarize(pred, min_class)
    summary["n_compounds"] = int(len(frame))
    summary["n_labels"] = int(frame[TASKS].notna().sum().sum())
    summary["bio_signatures"] = "not_run"
    summary["bio_reason"] = (
        "Chemical Checker signatures were not regenerated. The signaturizer "
        "stack was not executed in this run; see reports/signaturizer_probe.json."
    )
    (output_dir / "metrics" / "tox21_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return path


def _summarize(pred: pd.DataFrame, min_class: int) -> dict:
    rows = []
    for (phase, task), sub in pred.groupby(["phase", "task"]):
        y = sub["y_true"].to_numpy()
        if (y == 1).sum() < min_class or (y == 0).sum() < min_class:
            continue
        metrics = classification_metrics(y, sub["y_pred"].to_numpy())
        metrics.update({"phase": phase, "task": task, "family": sub["family"].iloc[0]})
        rows.append(metrics)
    table = pd.DataFrame(rows)
    out = {"tasks": rows}
    for phase in table["phase"].unique() if len(table) else []:
        piece = table.loc[table["phase"] == phase]
        out[phase] = {
            "macro_auroc": float(piece["auroc"].mean()),
            "macro_auprc": float(piece["auprc"].mean()),
            "nr_macro_auroc": float(piece.loc[piece["family"] == "NR", "auroc"].mean()),
            "sr_macro_auroc": float(piece.loc[piece["family"] == "SR", "auroc"].mean()),
            "n_tasks": int(len(piece)),
        }
    if "scaffold_oof" in out and "random_oof" in out:
        out["random_minus_scaffold_macro_auroc"] = (
            out["random_oof"]["macro_auroc"] - out["scaffold_oof"]["macro_auroc"]
        )
    out["note"] = (
        "NR versus SR is a comparison of assay families under the same split, "
        "not a label-aligned transfer. Random-split AUROC is an optimistic "
        "reference that allows scaffold leakage."
    )
    return out
