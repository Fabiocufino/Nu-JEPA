from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping


class CSVLogger:
    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict[str, object]) -> None:
        self._writer.writerow({key: row.get(key, "") for key in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


TENSORBOARD_SCALAR_TAGS = {
    "loss": "01_loss/train",
    "lr": "02_optimization/learning_rate",
    "grad_norm": "02_optimization/grad_norm",
    "valid_events": "03_data_health/valid_events",
    "skipped_events": "03_data_health/skipped_events",
    "trainable_event_fraction": "03_data_health/trainable_event_fraction",
    "num_context_patches": "04_masking/context_patches",
    "num_target_patches": "04_masking/target_patches",
    "context_energy_fraction": "04_masking/context_energy_fraction",
    "target_energy_fraction": "04_masking/target_energy_fraction",
    "events_per_second": "05_throughput/events_per_second",
    "seconds_per_step": "05_throughput/seconds_per_step",
    "gpu_memory_allocated_mb": "06_gpu/memory_allocated_mb",
    "gpu_memory_peak_allocated_mb": "06_gpu/memory_peak_allocated_mb",
}


def log_tensorboard_scalars(writer: object, row: Mapping[str, object], step: int) -> None:
    for key, tag in TENSORBOARD_SCALAR_TAGS.items():
        value = row.get(key)
        if value in (None, ""):
            continue
        writer.add_scalar(tag, float(value), step)
