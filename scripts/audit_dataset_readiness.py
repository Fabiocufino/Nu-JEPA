from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.sparse_preprocessing import extract_reco_hits, hits_to_sparse_event, summarize_active_patches
from src.evaluation.latent_space import classify_neutrino
from src.masking.block3d_masking import Block3DMaskGenerator
from src.training.config import load_config


def _as_paths(values: str | Sequence[str]) -> List[Path]:
    if isinstance(values, str):
        return [Path(values)]
    return [Path(value) for value in values]


def _list_files(paths: Sequence[Path], recursive: bool) -> List[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(path for root in paths for path in root.glob(pattern))


def _quantiles(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"min": None, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _histogram(values: Iterable[int], max_key: int = 20) -> Dict[str, int]:
    hist = Counter(int(value) for value in values)
    out = {str(key): int(hist.get(key, 0)) for key in range(max_key + 1)}
    overflow = sum(count for key, count in hist.items() if key > max_key)
    out[f">{max_key}"] = int(overflow)
    return out


def _label_row(npz: Any) -> Dict[str, object]:
    pdg = int(npz["in_neutrino_pdg"].item()) if "in_neutrino_pdg" in npz else 0
    is_cc = bool(npz["is_cc"].item()) if "is_cc" in npz else False
    event_id = int(npz["event_id"].item()) if "event_id" in npz else -1
    run_number = int(npz["run_number"].item()) if "run_number" in npz else -1
    energy = float(npz["in_neutrino_energy"].item()) if "in_neutrino_energy" in npz else float("nan")
    return {
        "event_id": event_id,
        "run_number": run_number,
        "in_neutrino_pdg": pdg,
        "is_cc": is_cc,
        "flavour_class": classify_neutrino(pdg, is_cc),
        "in_neutrino_energy": energy,
    }


def audit(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_config(args.config)

    data_cfg = dict(cfg["data"])
    if args.dataset_path:
        dataset_paths = [Path(path) for path in args.dataset_path]
    else:
        dataset_paths = _as_paths(data_cfg["dataset_path"])
    recursive = bool(args.recursive if args.recursive is not None else data_cfg.get("recursive", False))
    files = _list_files(dataset_paths, recursive)
    population = len(files)
    if args.max_events is not None:
        files = files[: args.max_events]
    if args.sample_events is not None and args.sample_events < len(files):
        rng = random.Random(args.seed)
        files = rng.sample(files, args.sample_events)
        files = sorted(files)

    data_cfg.pop("dataset_path", None)
    data_cfg["recursive"] = recursive
    detector_size = data_cfg["detector_size"]
    patch_size = data_cfg["patch_size"]
    mask_cfg = dict(cfg["masking"])
    grid = tuple((int(d) + int(p) - 1) // int(p) for d, p in zip(detector_size, patch_size))
    masker = Block3DMaskGenerator(grid_size=grid, **mask_cfg)

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    counts: Counter[str] = Counter()
    raw_hits: List[int] = []
    voxel_counts: List[int] = []
    patch_counts: List[int] = []
    context_counts: List[int] = []
    target_counts: List[int] = []
    energy_sums: List[float] = []
    label_counts: Counter[str] = Counter()
    start = time.time()

    for idx, path in enumerate(files, start=1):
        try:
            with np.load(path, allow_pickle=True) as npz:
                label = _label_row(npz)
                hits = extract_reco_hits(npz, data_cfg["reco_hits_key"])
        except Exception as exc:
            counts["read_errors"] += 1
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        try:
            event = hits_to_sparse_event(
                hits,
                detector_size=detector_size,
                coord_mode=data_cfg["coord_mode"],
                energy_scale=data_cfg["energy_scale"],
                energy_transform=data_cfg["energy_transform"],
                path=str(path),
                event_id=int(label["event_id"]),
            )
            patches = summarize_active_patches(
                event,
                detector_size=detector_size,
                patch_size=patch_size,
                min_voxels_per_patch=data_cfg["min_voxels_per_patch"],
            )
            mask = masker(patches.patch_coords, patches.patch_ids, patches.energy)
        except Exception as exc:
            counts["processing_errors"] += 1
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        n_raw = int(hits.shape[0])
        n_voxels = int(event.coords.shape[0])
        n_patches = int(patches.patch_ids.numel())
        n_context = int(mask.context_patch_ids.numel())
        n_target = int(mask.target_patch_ids.numel())
        energy_sum = float(event.energy.sum().item())
        raw_hits.append(n_raw)
        voxel_counts.append(n_voxels)
        patch_counts.append(n_patches)
        context_counts.append(n_context)
        target_counts.append(n_target)
        energy_sums.append(energy_sum)
        label_counts[str(label["flavour_class"])] += 1

        if n_raw == 0:
            counts["empty_reco_hits"] += 1
        if n_voxels == 0:
            counts["zero_valid_voxels"] += 1
        if n_patches == 0:
            counts["zero_active_patches"] += 1
        if mask.valid:
            counts["trainable_events"] += 1
        else:
            counts[f"untrainable_{mask.reason}"] += 1

        rows.append(
            {
                "path": str(path),
                **label,
                "raw_hits": n_raw,
                "valid_voxels": n_voxels,
                "active_patches": n_patches,
                "context_patches": n_context,
                "target_patches": n_target,
                "energy_sum": energy_sum,
                "trainable": bool(mask.valid),
                "mask_reason": mask.reason,
            }
        )
        if args.progress_every and idx % args.progress_every == 0:
            elapsed = time.time() - start
            print(f"[audit] processed={idx}/{len(files)} elapsed_s={elapsed:.1f}", flush=True)

    valid_rows = len(rows)
    sampled_fraction = valid_rows / population if population else 0.0
    trainable = int(counts["trainable_events"])
    trainable_fraction = trainable / valid_rows if valid_rows else 0.0
    summary: Dict[str, object] = {
        "dataset_paths": [str(path) for path in dataset_paths],
        "population_files": population,
        "audited_files": len(files),
        "valid_processed_events": valid_rows,
        "sampled_fraction_of_population": sampled_fraction,
        "elapsed_seconds": time.time() - start,
        "detector_size": detector_size,
        "patch_size": patch_size,
        "patch_grid": grid,
        "max_patch_tokens_per_event": int(grid[0] * grid[1] * grid[2]),
        "min_trainable_active_patches": int(mask_cfg["min_context_patches"] + mask_cfg["min_target_patches"]),
        "counts": {key: int(value) for key, value in counts.items()},
        "trainable_fraction": trainable_fraction,
        "estimated_trainable_events_in_population": int(round(trainable_fraction * population)) if population else 0,
        "raw_hits": _quantiles(raw_hits),
        "valid_voxels": _quantiles(voxel_counts),
        "active_patches": _quantiles(patch_counts),
        "context_patches": _quantiles(context_counts),
        "target_patches": _quantiles(target_counts),
        "energy_sum": _quantiles(energy_sums),
        "active_patch_histogram_0_20": _histogram(patch_counts, 20),
        "label_counts": {key: int(value) for key, value in label_counts.items()},
        "errors": errors[: args.max_errors],
        "num_errors_total": len(errors),
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].keys()) if rows else [
                "path",
                "event_id",
                "run_number",
                "in_neutrino_pdg",
                "is_cc",
                "flavour_class",
                "in_neutrino_energy",
                "raw_hits",
                "valid_voxels",
                "active_patches",
                "context_patches",
                "target_patches",
                "energy_sum",
                "trainable",
                "mask_reason",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NPZ event readiness for NuJEPA pretraining.")
    parser.add_argument("--config", default="configs/nu_jepa_3dcal_12k_robust.yaml")
    parser.add_argument("--dataset-path", action="append", help="Override config data.dataset_path. Repeat for multiple roots.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-events", type=int, default=None, help="Use the first N files after sorting.")
    parser.add_argument("--sample-events", type=int, default=None, help="Randomly sample N files after optional max-events.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    summary = audit(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
