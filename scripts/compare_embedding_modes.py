#!/usr/bin/env python
"""Compare BEATs embedding pooling modes with the fixed AutoTrash detector."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # When executed as `python scripts/compare_embedding_modes.py`.
    from compute_dev_metrics import compute_metrics
except ModuleNotFoundError:
    # When imported as a module (repo root on sys.path).
    from scripts.compute_dev_metrics import compute_metrics
from src.pipeline import get_bank_path
from src.utils.file_utils import load_config


def _configure_console_encoding() -> None:
    """Best-effort: avoid UnicodeEncode/Decode issues on Windows consoles (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_console_encoding()


EMBEDDING_MODES = [
    "last_layer_mean",
    "last_layer_cls",
    "last_layer_mean_std",
    "last4_layers_mean",
    "middle_layer_mean",
    "middle_layer_mean_std",
]


METRIC_FIELDS = [
    "AUC(all)",
    "AUC(source)",
    "AUC(target)",
    "pAUC",
    "precision(source)",
    "precision(target)",
    "recall(source)",
    "recall(target)",
    "F1(source)",
    "F1(target)",
    "official_score",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def fixed_embedding_config(base_cfg: dict, embedding_mode: str) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["mode"] = "clip"
    cfg["pipeline"]["auto_rebuild_bank"] = True
    cfg["pipeline"]["cleanup_stale_banks"] = False

    cfg.setdefault("memory_bank", {})
    cfg["memory_bank"]["mode"] = "clustered"
    cfg["memory_bank"]["num_clusters"] = 6
    cfg["memory_bank"]["cluster_method"] = "kmeans"
    cfg["memory_bank"]["cluster_score_mode"] = "nearest_cluster_knn"
    cfg["memory_bank"]["cluster_score_normalization"] = "none"
    cfg["memory_bank"]["cleanup_stale_banks"] = False

    cfg.setdefault("embedding_postprocess", {})
    cfg["embedding_postprocess"].setdefault("pca_whitening", {})
    cfg["embedding_postprocess"]["pca_whitening"]["enabled"] = False

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

    cfg.setdefault("model", {})
    cfg["model"]["embedding_mode"] = embedding_mode
    cfg["model"]["normalize"] = True

    cfg.setdefault("logging", {})
    cfg["logging"]["embedding_mode_subdirs"] = True
    return cfg


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    # Ensure Python subprocesses and our decoding agree on UTF-8.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def copy_mode_csvs(cfg: dict, machine: str, embedding_mode: str, result_dir: Path) -> tuple[str, str]:
    csv_dir = REPO_ROOT / cfg["logging"]["csv_out_dir"] / embedding_mode
    run_csv_dir = result_dir / "csv" / embedding_mode
    run_csv_dir.mkdir(parents=True, exist_ok=True)

    anomaly_src = csv_dir / f"anomaly_score_{machine}_section_00_test.csv"
    decision_src = csv_dir / f"decision_result_{machine}_section_00_test.csv"
    anomaly_dst = run_csv_dir / anomaly_src.name
    decision_dst = run_csv_dir / decision_src.name

    if anomaly_src.exists():
        shutil.copy2(anomaly_src, anomaly_dst)
    else:
        print(f"WARNING: missing anomaly CSV to copy: {anomaly_src}")

    if decision_src.exists():
        shutil.copy2(decision_src, decision_dst)
    else:
        print(f"WARNING: missing decision CSV to copy: {decision_src}")

    return str(anomaly_dst), str(decision_dst)


def nan_metrics() -> dict[str, float]:
    return {
        "auc_all": math.nan,
        "auc_source": math.nan,
        "auc_target": math.nan,
        "pauc": math.nan,
        "precision_source": math.nan,
        "precision_target": math.nan,
        "recall_source": math.nan,
        "recall_target": math.nan,
        "f1_source": math.nan,
        "f1_target": math.nan,
        "official_score": math.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--machine", default="AutoTrash")
    ap.add_argument("--stage", default="both", choices=["dev_data", "eval_data", "both"])
    ap.add_argument("--skip-infer", action="store_true", help="Only build banks and compute dev metrics.")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    result_dir = REPO_ROOT / "results" / "embedding_modes"
    cfg_dir = result_dir / "configs"
    log_dir = result_dir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = REPO_ROOT / base_cfg["logging"]["bank_out"]

    rows: list[dict[str, object]] = []
    for embedding_mode in EMBEDDING_MODES:
        cfg = fixed_embedding_config(base_cfg, embedding_mode)
        config_path = cfg_dir / f"{args.machine}_{embedding_mode}.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        bank_path = get_bank_path(
            bank_dir,
            args.machine,
            "clip",
            "clustered",
            6,
            "none",
            None,
            embedding_mode,
        )
        log_path = log_dir / f"{args.machine}_{embedding_mode}.log"

        with log_path.open("w", encoding="utf-8") as log_fp:
            tee = Tee(sys.stdout, log_fp)
            old_stdout = sys.stdout
            sys.stdout = tee
            try:
                print("\nEMBEDDING MODE EXPERIMENT")
                print(f"- embedding mode: {embedding_mode}")
                print("- pipeline mode: clip")
                print("- memory bank mode: clustered")
                print("- num clusters: 6")
                print("- distance metric: cosine")
                print("- threshold method: percentile")
                print("- threshold percentile: 90")
                print(f"- bank path: {bank_path}")

                run_command(
                    [
                        sys.executable,
                        "scripts/train_knn.py",
                        "--config",
                        str(config_path),
                        "--machine",
                        args.machine,
                        "--stage",
                        args.stage,
                        "--pipeline-mode",
                        "clip",
                        "--embedding-mode",
                        embedding_mode,
                    ],
                )
                if not args.skip_infer:
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
                            "--embedding-mode",
                            embedding_mode,
                        ],
                    )
                anomaly_csv, decision_csv = copy_mode_csvs(cfg, args.machine, embedding_mode, result_dir)

                metric_status = "ok"
                metrics_error = ""
                try:
                    metrics = compute_metrics(
                        str(config_path),
                        machine_filter=args.machine,
                        return_results=True,
                        embedding_mode_override=embedding_mode,
                    )[args.machine]
                except RuntimeError as exc:
                    metric_status = "unavailable"
                    metrics_error = str(exc)
                    print(f"WARNING: metrics unavailable for {embedding_mode}: {metrics_error}")
                    metrics = nan_metrics()

                row = {
                    "embedding_mode": embedding_mode,
                    "AUC(all)": metrics["auc_all"],
                    "AUC(source)": metrics["auc_source"],
                    "AUC(target)": metrics["auc_target"],
                    "pAUC": metrics["pauc"],
                    "precision(source)": metrics["precision_source"],
                    "precision(target)": metrics["precision_target"],
                    "recall(source)": metrics["recall_source"],
                    "recall(target)": metrics["recall_target"],
                    "F1(source)": metrics["f1_source"],
                    "F1(target)": metrics["f1_target"],
                    "official_score": metrics["official_score"],
                    "bank_path": str(bank_path),
                    "anomaly_csv": anomaly_csv,
                    "decision_csv": decision_csv,
                    "log_path": str(log_path),
                    "metric_status": metric_status,
                    "metrics_error": metrics_error,
                }
                rows.append(row)
            finally:
                sys.stdout = old_stdout

    out_csv = result_dir / f"{args.machine}_embedding_mode_comparison.csv"
    fieldnames = ["embedding_mode", *METRIC_FIELDS, "bank_path", "anomaly_csv", "decision_csv", "log_path", "metric_status", "metrics_error"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved embedding-mode comparison to: {out_csv}")
    print("embedding_mode,AUC(all),AUC(source),AUC(target),pAUC,official_score")
    for row in rows:
        print(
            f"{row['embedding_mode']},"
            f"{row['AUC(all)']:.2f},"
            f"{row['AUC(source)']:.2f},"
            f"{row['AUC(target)']:.2f},"
            f"{row['pAUC']:.2f},"
            f"{row['official_score']:.2f}"
        )


if __name__ == "__main__":
    main()
