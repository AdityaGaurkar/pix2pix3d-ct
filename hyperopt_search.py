#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Continuous hyperparameter runner for pix2pix3d-ct.

Search dimensions:
- learning rates (P2P_LRS)
- grid (P2P_GRID)
- L_weights (P2P_L_WEIGHTS)

Notes:
- train.py already logs each trial to MLflow.
- In continuous mode this script runs until you stop it (Ctrl+C).
"""
import argparse
import csv
import itertools
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime


METRIC_RE = re.compile(
    r"\[PSNR:\s*([^\]]+)\]\s*\[SSIM:\s*([^\]]+)\]\s*\[NMSE:\s*([^\]]+)\](?:\s*\[DICE:\s*([^\]]+)\])?"
)
MODEL_DIR_RE = re.compile(r"^model_(\d+)$")


def _parse_space(raw, expected_len, cast_type):
    vals = json.loads(raw)
    if not isinstance(vals, list):
        raise ValueError("Search space must be a JSON list.")
    out = []
    for item in vals:
        if not isinstance(item, (list, tuple)) or len(item) != expected_len:
            raise ValueError("Each item must have {} values: {}".format(expected_len, item))
        out.append(tuple(cast_type(x) for x in item))
    if not out:
        raise ValueError("Search space is empty.")
    return out


def _list_model_dirs(models_root):
    result = {}
    if not os.path.isdir(models_root):
        return result
    for name in os.listdir(models_root):
        m = MODEL_DIR_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        result[idx] = os.path.join(models_root, name)
    return result


def _parse_infer_metrics(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        return None

    model_name = os.path.basename(model_dir.rstrip(os.sep))
    preferred = os.path.join(model_dir, "{}_infer".format(model_name), "log.txt")
    candidates = [preferred]
    if not os.path.exists(preferred):
        for name in os.listdir(model_dir):
            if name.endswith("_infer"):
                candidates.append(os.path.join(model_dir, name, "log.txt"))
    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        return None

    log_path = max(candidates, key=os.path.getmtime)
    psnr_vals, ssim_vals, nmse_vals, dice_vals = [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            m = METRIC_RE.search(line)
            if not m:
                continue
            psnr_vals.append(float(m.group(1)))
            ssim_vals.append(float(m.group(2)))
            nmse_vals.append(float(m.group(3)))
            if m.group(4) is not None:
                dice_vals.append(float(m.group(4)))
    if not psnr_vals:
        return None

    n = len(psnr_vals)
    return {
        "n_cases": n,
        "psnr_mean": sum(psnr_vals) / n,
        "ssim_mean": sum(ssim_vals) / n,
        "nmse_mean": sum(nmse_vals) / n,
        "dice_mean": (sum(dice_vals) / len(dice_vals)) if dice_vals else None,
        "log_path": log_path,
    }


def _objective_value(metrics_dict, objective):
    if metrics_dict is None:
        return None
    if objective == "psnr":
        return metrics_dict["psnr_mean"]
    if objective == "ssim":
        return metrics_dict["ssim_mean"]
    if objective == "nmse":
        return metrics_dict["nmse_mean"]
    if objective == "dice":
        return metrics_dict["dice_mean"]
    raise ValueError("Unknown objective '{}'".format(objective))


def _is_better(candidate, best, objective):
    if best is None:
        return True
    if objective == "nmse":
        return candidate < best
    return candidate > best


def _neighbor_combo(combo, lrs_space, grid_space, lw_space, rng):
    lrs, grid, l_weights = combo
    i = lrs_space.index(lrs)
    j = grid_space.index(grid)
    k = lw_space.index(l_weights)

    def _pick_neighbor(idx, n):
        choices = [idx]
        if idx > 0:
            choices.append(idx - 1)
        if idx + 1 < n:
            choices.append(idx + 1)
        return rng.choice(choices)

    ni = _pick_neighbor(i, len(lrs_space))
    nj = _pick_neighbor(j, len(grid_space))
    nk = _pick_neighbor(k, len(lw_space))
    return (lrs_space[ni], grid_space[nj], lw_space[nk])


def _select_combo(mode, combos, all_combos, best_row, lrs_space, grid_space, lw_space, rng, explore_prob, trial_idx):
    if mode == "continuous":
        if best_row is None or rng.random() < explore_prob:
            return rng.choice(all_combos)

        best_combo = (
            tuple(best_row["lrs"]),
            tuple(best_row["grid"]),
            tuple(best_row["L_weights"]),
        )
        return _neighbor_combo(best_combo, lrs_space, grid_space, lw_space, rng)

    # Finite modes
    return combos[trial_idx - 1]


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for pix2pix3d-ct.")
    parser.add_argument(
        "--train-script",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
        help="Path to train.py",
    )
    parser.add_argument(
        "--models-root",
        default="/home/cet/pix2pix/models",
        help="Root folder that contains model_<n> outputs.",
    )
    parser.add_argument(
        "--lrs",
        default="[[0.00018, 0.1], [0.00015, 0.05], [0.0001, 0.0]]",
        help="JSON list of [lr_ini, lr_decay] candidates.",
    )
    parser.add_argument(
        "--grids",
        default="[[64, 64, 64], [64, 64, 32], [128, 128, 32]]",
        help="JSON list of [gx, gy, gz] candidates.",
    )
    parser.add_argument(
        "--l-weights",
        default="[[1, 100], [1, 150], [1, 200]]",
        help="JSON list of [adv_weight, recon_weight] candidates.",
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "random", "continuous"],
        default="continuous",
        help="grid = finite exhaustive product, random = finite random sample, continuous = run until stopped.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit number of trials. Required for random mode.",
    )
    parser.add_argument(
        "--objective",
        choices=["ssim", "psnr", "nmse", "dice"],
        default="dice",
        help="Metric used for improvement tracking.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for all trials (P2P_EPOCHS).",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["all", "final"],
        default="final",
        help="Checkpoint policy for hyperopt trials: all = existing behavior, final = only final weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--explore-prob",
        type=float,
        default=0.30,
        help="Continuous mode: probability of random exploration vs local neighborhood.",
    )
    args = parser.parse_args()

    lrs_space = _parse_space(args.lrs, expected_len=2, cast_type=float)
    grid_space = _parse_space(args.grids, expected_len=3, cast_type=int)
    lw_space = _parse_space(args.l_weights, expected_len=2, cast_type=float)

    all_combos = list(itertools.product(lrs_space, grid_space, lw_space))
    rng = random.Random(args.seed)

    if args.mode == "random":
        if args.max_trials is None:
            raise ValueError("--max-trials is required when --mode=random")
        rng.shuffle(all_combos)
        combos = all_combos[: args.max_trials]
    elif args.mode == "grid":
        combos = all_combos[: args.max_trials] if args.max_trials is not None else all_combos
    else:
        combos = None

    if args.mode != "continuous" and not combos:
        raise ValueError("No trial combinations to evaluate.")

    os.makedirs(args.models_root, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(args.models_root, "hyperopt_results_{}.csv".format(run_stamp))
    best_json = os.path.join(args.models_root, "hyperopt_best_{}.json".format(run_stamp))

    print("Total combos available:", len(all_combos))
    if args.mode == "continuous":
        print("Trials to run: infinite (stop with Ctrl+C)")
    else:
        print("Trials to run:", len(combos))
    print("Objective:", args.objective)
    print("Train script:", args.train_script)
    print("Models root:", args.models_root)

    fieldnames = [
        "trial",
        "return_code",
        "model_id",
        "model_dir",
        "lrs",
        "grid",
        "L_weights",
        "objective",
        "objective_value",
        "n_cases",
        "psnr_mean",
        "ssim_mean",
        "nmse_mean",
        "dice_mean",
        "infer_log_path",
    ]

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    rows = []
    best_row = None
    best_obj = None

    trial_idx = 0
    try:
        while True:
            trial_idx += 1
            if args.mode != "continuous" and trial_idx > len(combos):
                break

            lrs, grid, l_weights = _select_combo(
                args.mode,
                combos,
                all_combos,
                best_row,
                lrs_space,
                grid_space,
                lw_space,
                rng,
                args.explore_prob,
                trial_idx,
            )

            total_label = "?" if args.mode == "continuous" else str(len(combos))
            print("\n=== Trial {}/{} ===".format(trial_idx, total_label))
            print("lrs={}, grid={}, L_weights={}".format(lrs, grid, l_weights))

            before = _list_model_dirs(args.models_root)

            env = os.environ.copy()
            env["P2P_LRS"] = "{},{}".format(lrs[0], lrs[1])
            env["P2P_GRID"] = "{},{},{}".format(grid[0], grid[1], grid[2])
            env["P2P_L_WEIGHTS"] = "{},{}".format(l_weights[0], l_weights[1])
            if args.epochs is not None:
                env["P2P_EPOCHS"] = str(args.epochs)
            if args.checkpoint_mode == "final":
                env["P2P_MODEL_INTERVAL"] = "0"
                env["P2P_SAVE_TEMP_WEIGHTS"] = "0"
                env["P2P_SAVE_FINAL_WEIGHTS"] = "1"

            proc = subprocess.run([sys.executable, args.train_script], env=env)
            return_code = int(proc.returncode)

            after = _list_model_dirs(args.models_root)
            new_ids = sorted(set(after.keys()) - set(before.keys()))
            model_id = new_ids[-1] if new_ids else (max(after.keys()) if after else None)
            model_dir = after.get(model_id) if model_id is not None else None
            metrics = _parse_infer_metrics(model_dir) if (return_code == 0 and model_dir) else None
            obj_val = _objective_value(metrics, args.objective)

            row = {
                "trial": trial_idx,
                "return_code": return_code,
                "model_id": model_id,
                "model_dir": model_dir,
                "lrs": list(lrs),
                "grid": list(grid),
                "L_weights": list(l_weights),
                "objective": args.objective,
                "objective_value": obj_val,
                "n_cases": metrics["n_cases"] if metrics else None,
                "psnr_mean": metrics["psnr_mean"] if metrics else None,
                "ssim_mean": metrics["ssim_mean"] if metrics else None,
                "nmse_mean": metrics["nmse_mean"] if metrics else None,
                "dice_mean": metrics["dice_mean"] if metrics else None,
                "infer_log_path": metrics["log_path"] if metrics else None,
            }
            rows.append(row)

            if obj_val is not None and _is_better(obj_val, best_obj, args.objective):
                best_obj = obj_val
                best_row = row
                print("New best ({}) = {}".format(args.objective, best_obj))

            print("return_code:", return_code)
            print("model_dir:", model_dir)
            if metrics:
                print(
                    "means -> PSNR={:.6f}, SSIM={:.6f}, NMSE={:.6f}".format(
                        metrics["psnr_mean"], metrics["ssim_mean"], metrics["nmse_mean"]
                    )
                )
                if metrics["dice_mean"] is not None:
                    print("mean DICE={:.6f}".format(metrics["dice_mean"]))
            else:
                print("No inference metrics found for this trial.")

            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            with open(best_json, "w") as f:
                json.dump({"best_trial": best_row, "all_trials": len(rows)}, f, indent=2)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\nSaved results:", results_csv)
    print("Saved best:", best_json)
    if best_row:
        print("Best trial:", best_row["trial"])
        print("Best model:", best_row["model_dir"])
        print("Best objective value:", best_row["objective_value"])
        print(
            "Best params: lrs={}, grid={}, L_weights={}".format(
                best_row["lrs"], best_row["grid"], best_row["L_weights"]
            )
        )
    else:
        print("No successful trial produced inference metrics.")


if __name__ == "__main__":
    main()
