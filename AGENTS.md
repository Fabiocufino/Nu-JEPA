# Nu-JEPA Agent Handoff

This repository is a sparse 3DCal Nu-JEPA pretraining baseline. Be conservative: inspect the data path, run readiness checks, and write new outputs to new scratch directories before starting long jobs.

## Dataset Rules

- Current approved local training sample: `/scratch/fcufino/events_v8.0_1000`.
- Do not use local `/scratch/fcufino/events_v8.0_200*` or `/scratch/fcufino/events_v8.0_300*` directories unless Fabio explicitly says to use them.
- A future 1M-event dataset is expected to live on another machine. Do not assume its sparsity or labels match the 12k sample.
- Before training on any new dataset, run `scripts/audit_dataset_readiness.py` and check:
  - read/processing errors are zero or understood;
  - trainable fraction is high enough;
  - active patch counts fit the 320-token grid cap;
  - voxel-count p95/p99/max are compatible with the intended batch size;
  - class counts are not too imbalanced for the downstream classification goal.

Example remote audit:

```bash
python scripts/audit_dataset_readiness.py \
  --config configs/nu_jepa_3dcal_12k_robust.yaml \
  --dataset-path /path/to/approved/events \
  --sample-events 50000 \
  --output-json /scratch/fcufino/nu_jepa/logs_remote/readiness_50k.json \
  --output-csv /scratch/fcufino/nu_jepa/logs_remote/readiness_50k.csv
```

## Masking Semantics

The training target is defined in active patch space. With the current config:

- `min_context_patches = 2`
- `min_target_patches = 1`
- minimum trainable active patches per event is therefore `3`

Events with fewer than 3 active patches are marked non-trainable by the mask generator. Training filters them before the forward pass. In DDP, if any rank has no trainable events in its local batch, all ranks skip that batch together to avoid all-reduce hangs.

## Training Outputs

For every serious run, use a unique output/checkpoint/log directory. Do not reuse a previous run directory unless you intentionally want to overwrite `train_metrics.csv`.

The trainer writes, on rank 0:

- CSV metrics: `<log_dir>/train_metrics.csv`
- TensorBoard events: `<tensorboard_dir>` or `<log_dir>/tensorboard`
- resolved config: `<log_dir>/config_resolved.yaml`
- model summary and parameter counts: `<log_dir>/model_summary.txt`
- run metadata: `<log_dir>/run_info.json`
- checkpoints: `<checkpoint_dir>/nu_jepa_step_*.pt`
- latest checkpoint pointer: `<checkpoint_dir>/latest.pt`

TensorBoard is intentionally a compact time-series dashboard, not the full archive. The full metadata still lives in `config_resolved.yaml`, `model_summary.txt`, `run_info.json`, and `train_metrics.csv`.

TensorBoard scalar groups:

- `01_loss/train`
- `02_optimization/learning_rate`
- `02_optimization/grad_norm`
- `03_data_health/valid_events`
- `03_data_health/skipped_events`
- `03_data_health/trainable_event_fraction`
- `04_masking/context_patches`
- `04_masking/target_patches`
- `04_masking/context_energy_fraction`
- `04_masking/target_energy_fraction`
- `05_throughput/events_per_second`
- `05_throughput/seconds_per_step`
- `06_gpu/memory_allocated_mb`
- `06_gpu/memory_peak_allocated_mb`

Launch TensorBoard:

```bash
tensorboard --logdir /scratch/fcufino/nu_jepa --host 0.0.0.0 --port 6006
```

## Single-GPU Training

```bash
python -m src.training.train_nu_jepa \
  --config configs/nu_jepa_3dcal_12k_robust.yaml
```

## Multi-GPU Training

Use `torchrun`; the trainer auto-detects `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m src.training.train_nu_jepa \
  --config configs/nu_jepa_3dcal_12k_robust.yaml
```

For a real 1M run, make a new config or copy the robust config and change at minimum:

- `data.dataset_path`
- `training.output_dir`
- `training.checkpoint_dir`
- `training.log_dir`
- `training.tensorboard_dir`
- optionally `training.batch_size` after benchmarking voxel density

## Capacity Notes

The current detector/patch setup gives a maximum of `4 x 4 x 20 = 320` patch tokens per event. The transformer memory is bounded by this number, not by the total number of files.

The expensive part for very dense events is the sparse voxel input before patch aggregation. Local stress tests showed:

- current 128-dim/4-layer model, batch 16, 320 patches/event, 1 voxel/patch: under 0.5 GB allocated GPU memory;
- current model, batch 16, 320 patches/event, 64k voxels/event: about 40 GB allocated GPU memory;
- candidate 256-dim/6-layer model, batch 16, 320 patches/event, 1 voxel/patch: under 1 GB allocated GPU memory.

So the remote 1M dataset must be audited for voxel-count tails before choosing batch size.

## Evaluation

Current UMAP/classification outputs for the robust 12k checkpoint are under:

- `plots/latent_space_12k_robust/`
- `/scratch/fcufino/nu_jepa/logs_12k_robust/latent_space/`

The classification probe is weak at this stage. Do not overclaim separation from UMAP plots. Use the probe metrics and class counts from `latent_checks.json`.

## Quick Checks Before Long Runs

```bash
python -m compileall -q src scripts tests

python scripts/benchmark_training_batch.py \
  --config configs/nu_jepa_3dcal_12k_robust.yaml \
  --batch-size 16

python scripts/benchmark_training_batch.py \
  --config configs/nu_jepa_3dcal_12k_robust.yaml \
  --batch-size 16 \
  --embed-dim 256 \
  --depth 6 \
  --num-heads 8
```

If `pytest` is installed, run `python -m pytest tests`. If it is not installed, do not claim the pytest suite passed.
