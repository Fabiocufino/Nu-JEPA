#!/usr/bin/env bash
set -euo pipefail

cd /home/fcufino/FabioFaserDL/projects/Nu-JEPA
mkdir -p /scratch/fcufino/nu_jepa/checkpoints /scratch/fcufino/nu_jepa/logs
python -m src.training.train_nu_jepa --config configs/nu_jepa_3dcal_baseline.yaml "$@"
