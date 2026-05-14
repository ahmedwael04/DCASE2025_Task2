#!/usr/bin/env python
"""Run the final AutoTrash clustered-k6 normalization/PCA ablation."""

from __future__ import annotations

import argparse
import csv
import copy
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compute_dev_metrics import compute_metrics
from src.pipeline import get_bank_path
from src.utils.file_utils import load_config


RUNS = [
    {
        "name": "clustered_k6_none",
        "score_norm": "none",
        "pca_enabled": False,
    },
    {
        "name": "clustered_k6_mean_ratio",
        "score_norm": "mean_ratio",
        "pca_enabled": False,
    },
    {
        "name": "clustered_k6_zscore",
        "score_norm": "zscore",
        "pca_enabled": False,
    },
    {
        "name": "clustered_k6_robust_zscore",
        "score_norm": "robust_zscore",
        "pca_enabled": False,
    },
    {
        "name": "clustered_k6_zscore_pca128",
        "score_norm": "zscore",
        "pca_enabled": True,
        "pca_components": 128,
    },
]


def _set_final_config(base_cfg: dict, run: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["mode"] = "clip"
    cfg["pipeline"]["auto_rebuild_bank"] = True
    cfg["pipeline"]["cleanup_stale_banks"] = True

    cfg.setdefault("memory_bank", {})
    cfg["memory_bank"]["mode"] = "clustered"
    cfg["memory_bank"]["num_clusters"] = 6
    cfg["memory_bank"]["cluster_method"] = "kmeans"
    cfg["memory_bank"]["cluster_score_mode"] = "nearest_cluster_knn"
    cfg["memory_bank"]["cluster_score_normalization"] = run["score_norm"]
    cfg["memory_bank"]["cluster_score_eps"] = float(cfg["memory_bank"].get("cluster_score_eps", 1e-6))
    cfg["memory_bank"]["cleanup_stale_banks"] = True

    cfg.setdefault("embedding_postprocess", {})
    cfg["embedding_postprocess"].setdefault("pca_whitening", {})
    cfg["embedding_postprocess"]["pca_whitening"]["enabled"] = bool(run["pca_enabled"])
    cfg["embedding_postprocess"]["pca_whitening"]["n_components"] = int(run.get("pca_components", 128))
    cfg["embedding_postprocess"]["pca_whitening"]["whiten"] = True
    cfg["embedding_postprocess"]["pca_whitening"]["l2_after"] = True

    cfg.setdefault("augmentation", {})
    cfg["augmentation"]["enabled"] = False
    cfg["augmentation"]["num_views_test"] = 0

    cfg.setdefault("detector", {})
    cfg["detector"]["distance"] = "cosine"
    cfg["detector"].setdefault("temporal", {})
    cfg["detector"]["temporal"]["enabled"] = False

    cfg.setdefault("threshold", {})
    cfg["threshold"]["method"] = "percentile"
    cfg["threshold"]["percentile"] = 90
    cfg["threshold"]["sweep_enabled"] = False
    return cfg


def cleanup_stale_autotrash_banks(bank_dir: Path) -> list[Path]:
    bank_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {
        "memory_bank_AutoTrash_clip_clustered_k6.pt",
        "memory_bank_AutoTrash_clip_clustered_k6_norm_mean_ratio.pt",
        "memory_bank_AutoTrash_clip_clustered_k6_norm_zscore.pt",
        "memory_bank_AutoTrash_clip_clustered_k6_norm_robust_zscore.pt",
        "memory_bank_AutoTrash_clip_clustered_k6_norm_zscore_pca128.pt",
    }
    patterns = [
        "memory_bank_AutoTrash_window*.pt",
        "memory_bank_AutoTrash_clip_clustered_k3*.pt",
        "memory_bank_AutoTrash_clip_clustered_k4*.pt",
        "memory_bank_AutoTrash_clip_clustered_k8*.pt",
        "memory_bank_AutoTrash*source*.pt",
        "memory_bank_AutoTrash*target*.pt",
        "memory_bank_AutoTrash*dual*.pt",
        "memory_bank_AutoTrash.pt",
        "memory_bank.pt",
        "*AutoTrash*threshold*sweep*.pt",
        "*AutoTrash*threshold*sweep*.pkl",
        "*AutoTrash*threshold*sweep*.npy",
        "*AutoTrash*threshold*sweep*.npz",
    ]

    removed: list[Path] = []
    for pattern in patterns:
        for path in sorted(bank_dir.glob(pattern)):
            if not path.is_file() or path.name in keep_names:
                continue
            print(f"Removed stale bank/cache: {path}")
            path.unlink()
            removed.append(path)
    if not removed:
        print("No stale AutoTrash banks/caches to clean.")
    return removed


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def copy_run_csvs(cfg: dict, machine: str, run_name: str, result_dir: Path) -> tuple[str, str]:
    csv_dir = REPO_ROOT / cfg["logging"]["csv_out_dir"]
    run_csv_dir = result_dir / "csv"
    run_csv_dir.mkdir(parents=True, exist_ok=True)

    anomaly_src = csv_dir / f"anomaly_score_{machine}_section_00_test.csv"
    decision_src = csv_dir / f"decision_result_{machine}_section_00_test.csv"
    anomaly_dst = run_csv_dir / f"{run_name}_anomaly_score_{machine}_section_00_test.csv"
    decision_dst = run_csv_dir / f"{run_name}_decision_result_{machine}_section_00_test.csv"

    if anomaly_src.exists():
        shutil.copy2(anomaly_src, anomaly_dst)
        print(f"Copied run anomaly CSV: {anomaly_dst}")
    else:
        print(f"WARNING: missing anomaly CSV to copy: {anomaly_src}")

    if decision_src.exists():
        shutil.copy2(decision_src, decision_dst)
        print(f"Copied run decision CSV: {decision_dst}")
    else:
        print(f"WARNING: missing decision CSV to copy: {decision_src}")

    return str(anomaly_dst), str(decision_dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--machine", default="AutoTrash")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    result_dir = REPO_ROOT / "results" / "final_ablation"
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = result_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = REPO_ROOT / base_cfg["logging"]["bank_out"]

    cleanup_stale_autotrash_banks(bank_dir)

    rows = []
    for run in RUNS:
        cfg = _set_final_config(base_cfg, run)
        config_path = cfg_dir / f"{args.machine}_{run['name']}.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        pca_cfg = cfg["embedding_postprocess"]["pca_whitening"]
        bank_path = get_bank_path(
            bank_dir,
            args.machine,
            "clip",
            "clustered",
            6,
            run["score_norm"],
            pca_cfg["n_components"] if pca_cfg["enabled"] else None,
        )

        print("\nFINAL ATTEMPT CONFIG")
        print(f"- run: {run['name']}")
        print("- pipeline mode: clip")
        print("- memory bank mode: clustered")
        print("- num clusters: 6")
        print(f"- score normalization mode: {run['score_norm']}")
        print(f"- PCA whitening: {'enabled' if pca_cfg['enabled'] else 'disabled'}")
        if pca_cfg["enabled"]:
            print(f"- PCA components: {pca_cfg['n_components']}")
        print("- threshold percentile: 90")
        print("- distance metric: cosine")
        print(f"- kNN K: {cfg.get('detector', {}).get('k', cfg.get('model', {}).get('k', 3))}")
        print(f"- bank path: {bank_path}")

        if bank_path.exists():
            print(f"Reusing existing bank: {bank_path}")
        else:
            run_command(
                [
                    sys.executable,
                    "scripts/train_knn.py",
                    "--config",
                    str(config_path),
                    "--machine",
                    args.machine,
                    "--stage",
                    "both",
                    "--pipeline-mode",
                    "clip",
                ]
            )
        run_command(
            [
                sys.executable,
                "scripts/infer.py",
                "--config",
                str(config_path),
                "--machine",
                args.machine,
                "--stage",
                "eval_data",
                "--pipeline-mode",
                "clip",
            ]
        )
        anomaly_csv, decision_csv = copy_run_csvs(cfg, args.machine, run["name"], result_dir)

        metric_status = "ok"
        metrics_error = ""
        try:
            metrics = compute_metrics(str(config_path), machine_filter=args.machine, return_results=True)[args.machine]
        except RuntimeError as exc:
            metrics_error = str(exc)
            if "was not found under" not in metrics_error:
                raise
            metric_status = "unavailable_no_labeled_dev_machine"
            print(
                "WARNING: labeled dev metrics are unavailable for this machine. "
                "Continuing the ablation and writing NaN metrics."
            )
            print(f"  reason: {metrics_error}")
            metrics = {
                "auc_all": math.nan,
                "auc_source": math.nan,
                "auc_target": math.nan,
                "pauc": math.nan,
                "official_score": math.nan,
            }
        row = {
            "run": run["name"],
            "bank_path": str(bank_path),
            "anomaly_csv": anomaly_csv,
            "decision_csv": decision_csv,
            "auc_all": metrics["auc_all"] / 100.0,
            "auc_source": metrics["auc_source"] / 100.0,
            "auc_target": metrics["auc_target"] / 100.0,
            "pauc": metrics["pauc"] / 100.0,
            "official_score": metrics["official_score"] / 100.0,
            "metric_status": metric_status,
            "metrics_error": metrics_error,
        }
        rows.append(row)

    out_csv = result_dir / f"{args.machine}_final_ablation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "bank_path",
                "anomaly_csv",
                "decision_csv",
                "auc_all",
                "auc_source",
                "auc_target",
                "pauc",
                "official_score",
                "metric_status",
                "metrics_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved final comparison to: {out_csv}")
    scored_rows = [row for row in rows if math.isfinite(float(row["official_score"]))]
    if scored_rows:
        best = sorted(scored_rows, key=lambda r: (r["official_score"], r["auc_target"], r["pauc"]), reverse=True)[0]
        print("Best run by official score, then AUC target, then pAUC:")
        print(
            f"- {best['run']}: official={best['official_score']:.4f}, "
            f"auc_target={best['auc_target']:.4f}, pauc={best['pauc']:.4f}"
        )
    else:
        print(
            "No labeled metrics were available locally, so no best run can be selected by "
            "official score/AUC target/pAUC. Use the generated CSV submissions with the "
            "official scorer or add labeled AutoTrash dev data to compute metrics here."
        )


if __name__ == "__main__":
    main()
