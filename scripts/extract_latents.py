from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.faser_npz_dataset import FASERCalNPZDataset, faser_sparse_collate
from src.evaluation.latent_space import (
    attach_reductions,
    classification_probe,
    load_event_labels,
    make_reductions,
    representation_checks,
    save_plotly_latent_plots,
    write_json,
)
from src.models.nu_jepa import NuJEPA
from src.training.config import load_config
from src.training.train_nu_jepa import resolve_device


def _load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if "path" in ckpt and "model" not in ckpt:
        ckpt = torch.load(ckpt["path"], map_location="cpu")
    return ckpt


def extract_latents(args: argparse.Namespace) -> None:
    ckpt = _load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config")
    if cfg is None:
        cfg = load_config(args.config)

    data_cfg = dict(cfg["data"])
    dataset_path = data_cfg.pop("dataset_path")
    if args.max_events is not None:
        data_cfg["max_events"] = args.max_events
    dataset = FASERCalNPZDataset(dataset_path, **data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=faser_sparse_collate,
        drop_last=False,
    )

    device = resolve_device(args.device)
    model = NuJEPA(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    latents: List[np.ndarray] = []
    paths: List[str] = []
    patch_counts: List[int] = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            out = model.encode_events(batch)
            event_latents = out["event_latents"].detach().cpu().numpy()
            if not np.isfinite(event_latents).all():
                raise RuntimeError(f"Non-finite latent detected at extraction batch {step}")
            latents.append(event_latents)
            paths.extend(batch["paths"])
            patch_counts.extend([int(summary.patch_ids.numel()) for summary in batch["patches"]])
            if step % args.log_every == 0:
                print(f"extracted batch={step} events={len(paths)}")

    embeddings = np.concatenate(latents, axis=0)
    labels = load_event_labels(paths)
    labels["num_active_patches"] = patch_counts

    scratch_dir = Path(args.scratch_output_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    npz_path = scratch_dir / "event_latents.npz"
    csv_path = scratch_dir / "event_latents_labels.csv"
    checks_path = scratch_dir / "latent_checks.json"
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        paths=np.asarray(paths),
        flavour_class=labels["flavour_class"].to_numpy(),
        in_neutrino_pdg=labels["in_neutrino_pdg"].to_numpy(),
        is_cc=labels["is_cc"].to_numpy(),
        event_id=labels["event_id"].to_numpy(),
    )

    reductions = make_reductions(embeddings, seed=args.seed, include_tsne=not args.skip_tsne)
    df = attach_reductions(labels, reductions)
    df.to_csv(csv_path, index=False)
    plot_paths = save_plotly_latent_plots(df, args.plot_output_dir)

    checks = representation_checks(embeddings, labels)
    checks["classification_probe"] = classification_probe(embeddings, labels, seed=args.seed)
    checks["checkpoint"] = args.checkpoint
    checks["checkpoint_step"] = int(ckpt.get("step", -1))
    checks["latent_npz"] = str(npz_path)
    checks["latent_csv"] = str(csv_path)
    checks["plot_paths"] = plot_paths
    write_json(str(checks_path), checks)

    print(f"wrote {npz_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {checks_path}")
    for path in plot_paths:
        print(f"wrote {path}")
    print("checks:", checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/scratch/fcufino/nu_jepa/checkpoints/latest.pt")
    parser.add_argument("--config", default="configs/nu_jepa_3dcal_baseline.yaml")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE and write PCA/UMAP outputs only.")
    parser.add_argument("--scratch-output-dir", default="/scratch/fcufino/nu_jepa/logs/latent_space")
    parser.add_argument("--plot-output-dir", default="plots/latent_space")
    args = parser.parse_args()
    extract_latents(args)


if __name__ == "__main__":
    main()
