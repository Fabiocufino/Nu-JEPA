# Nu-JEPA for FASERCal 3DCal

Sparse 3DCal-only Nu-JEPA/I-JEPA-style pretraining for FASERCal event data.

Large data and outputs are intentionally kept outside the repository:

- checkpoints: `/scratch/fcufino/nu_jepa/checkpoints/`
- CSV logs and intermediate arrays: `/scratch/fcufino/nu_jepa/logs/`
- visualizations: project-local `plots/` or scratch

## Data Contract

The local approved sample is `/scratch/fcufino/events_v8.0_1000`. Do not use local
`/scratch/fcufino/events_v8.0_200*` or `/scratch/fcufino/events_v8.0_300*`
without explicit approval.

- `reco_hits` is read as a sparse `(N, >=5)` array.
- columns `0:3` are voxel coordinates, column `3` is module id when present, and column `4` is deposited energy.
- sparse coordinates/features are kept sparse until patch/token aggregation.
- if `spconv` is installed, `SparsePatchEmbed` can use a sparse 3D convolution with `kernel_size=stride=patch_size`.
- a pure PyTorch fallback aggregates active voxels into active patch tokens.

## Masking Design

Masking is done in patch space after sparse voxelization:

1. Active voxels are assigned to patch coordinates with integer floor division by `patch_size`.
2. Only patches containing at least `min_voxels_per_patch` active voxels are eligible.
3. Target blocks are sampled as 3D boxes in patch coordinates.
4. A proposed target block is accepted only if it contains active patch tokens.
5. Context tokens are active patches that are not target patches.
6. Context and target masks are disjoint by construction.

Events with fewer than `min_context_patches + min_target_patches` active patches are marked non-trainable and filtered before the forward pass.

## Setup

```bash
cd /home/fcufino/FabioFaserDL/projects/Nu-JEPA
python -m pip install -r requirements.txt
python -m pytest tests
```

`spconv` is optional. If it is unavailable, the model uses the pure PyTorch aggregated patch embedding path.

## Training

Single GPU:

```bash
python -m src.training.train_nu_jepa \
  --config configs/nu_jepa_3dcal_12k_robust.yaml
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m src.training.train_nu_jepa \
  --config configs/nu_jepa_3dcal_12k_robust.yaml
```

Small smoke run:

```bash
python -m src.training.train_nu_jepa \
  --config configs/nu_jepa_3dcal_baseline.yaml \
  --max-steps 5
```

TensorBoard is a compact time-series dashboard:

```bash
tensorboard --logdir /scratch/fcufino/nu_jepa --host 0.0.0.0 --port 6006
```

Useful scalar groups are loss, optimizer, data health, masking, throughput, and GPU memory. Full metadata is written as files under the run log directory.

## Readiness And Capacity

Audit a new dataset before long training:

```bash
python scripts/audit_dataset_readiness.py \
  --config configs/nu_jepa_3dcal_12k_robust.yaml \
  --dataset-path /path/to/events \
  --sample-events 50000 \
  --output-json /scratch/fcufino/nu_jepa/logs_remote/readiness_50k.json \
  --output-csv /scratch/fcufino/nu_jepa/logs_remote/readiness_50k.csv
```

Benchmark candidate model/batch sizes:

```bash
python scripts/benchmark_training_batch.py \
  --config configs/nu_jepa_3dcal_12k_robust.yaml \
  --batch-size 16
```

The current detector and patch setup gives at most `4 x 4 x 20 = 320` patch tokens per event. Memory risk for a large remote sample comes mostly from dense voxel tails before patch aggregation, so audit voxel-count p95/p99/max before scaling to 1M events.

## Latent-Space Evaluation

Extract event-level latents and make Plotly classification views:

```bash
python scripts/extract_latents.py \
  --checkpoint /scratch/fcufino/nu_jepa/checkpoints/latest.pt \
  --max-events 1000 \
  --scratch-output-dir /scratch/fcufino/nu_jepa/logs/latent_space \
  --plot-output-dir plots/latent_space
```

This writes embeddings and numeric checks under scratch, and interactive Plotly HTML files under `plots/latent_space/`.

## Notebooks

The notebooks in `notebooks/` are exploratory:

- `00_inspect_npz_events.ipynb`
- `01_debug_3dcal_masks.ipynb`

They load events from `/scratch/fcufino/events_v8.0_1000/`, inspect `reco_hits`, build sparse patches, generate context/target masks, and produce Plotly 3D views.
