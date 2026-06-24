from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE


def classify_neutrino(pdg: int, is_cc: bool) -> str:
    apdg = abs(int(pdg))
    if not bool(is_cc):
        return "NC"
    if apdg == 12:
        return "CC nue"
    if apdg == 14:
        return "CC numu"
    if apdg == 16:
        return "CC nutau"
    return f"CC pdg={pdg}"


def load_event_labels(paths: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            pdg = int(npz["in_neutrino_pdg"].item()) if "in_neutrino_pdg" in npz else 0
            is_cc = bool(npz["is_cc"].item()) if "is_cc" in npz else False
            is_tau = bool(npz["is_tau"].item()) if "is_tau" in npz else False
            energy = float(npz["in_neutrino_energy"].item()) if "in_neutrino_energy" in npz else np.nan
            event_id = int(npz["event_id"].item()) if "event_id" in npz else -1
            run_number = int(npz["run_number"].item()) if "run_number" in npz else -1
            reco_hits = npz["reco_hits"] if "reco_hits" in npz else np.empty((0, 0))
        rows.append(
            {
                "path": str(path),
                "file": Path(path).name,
                "run_number": run_number,
                "event_id": event_id,
                "in_neutrino_pdg": pdg,
                "is_cc": is_cc,
                "is_tau": is_tau,
                "flavour_class": classify_neutrino(pdg, is_cc),
                "signed_pdg": str(pdg),
                "in_neutrino_energy": energy,
                "num_reco_hits": int(reco_hits.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def make_reductions(embeddings: np.ndarray, seed: int = 17, include_tsne: bool = True) -> Dict[str, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf")
    n_samples = embeddings.shape[0]
    scaled = StandardScaler().fit_transform(embeddings)
    reductions: Dict[str, np.ndarray] = {}
    pca3 = PCA(n_components=min(3, embeddings.shape[1]), random_state=seed).fit_transform(scaled)
    if pca3.shape[1] < 3:
        pca3 = np.pad(pca3, ((0, 0), (0, 3 - pca3.shape[1])))
    reductions["pca3"] = pca3
    pca2 = pca3[:, :2]
    reductions["pca2"] = pca2
    if include_tsne and n_samples >= 10:
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        reductions["tsne2"] = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
        ).fit_transform(scaled)
    try:
        import umap

        reductions["umap2"] = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, n_samples // 20)),
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(scaled)
        reductions["umap3"] = umap.UMAP(
            n_components=3,
            n_neighbors=min(30, max(5, n_samples // 20)),
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(scaled)
    except Exception as exc:
        print(f"[latent_space] UMAP skipped: {type(exc).__name__}: {exc}")
    return reductions


def attach_reductions(labels: pd.DataFrame, reductions: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = labels.copy()
    for name, values in reductions.items():
        for dim in range(values.shape[1]):
            df[f"{name}_{dim + 1}"] = values[:, dim]
    return df


def save_plotly_latent_plots(df: pd.DataFrame, output_dir: str) -> List[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hover = ["file", "event_id", "in_neutrino_pdg", "is_cc", "in_neutrino_energy", "num_reco_hits"]
    written: List[str] = []

    fig = px.scatter_3d(
        df,
        x="pca3_1",
        y="pca3_2",
        z="pca3_3",
        color="flavour_class",
        symbol="is_cc",
        hover_data=hover,
        title="NuJEPA event latent space: PCA 3D by neutrino class",
    )
    path = out_dir / "latent_pca3d_by_flavour.html"
    fig.write_html(path)
    written.append(str(path))

    if {"umap2_1", "umap2_2"}.issubset(df.columns):
        fig = px.scatter(
            df,
            x="umap2_1",
            y="umap2_2",
            color="flavour_class",
            symbol="is_cc",
            hover_data=hover,
            title="NuJEPA event latent space: UMAP 2D by neutrino class",
        )
        path = out_dir / "latent_umap2d_by_flavour.html"
        fig.write_html(path)
        written.append(str(path))

    if {"umap3_1", "umap3_2", "umap3_3"}.issubset(df.columns):
        fig = px.scatter_3d(
            df,
            x="umap3_1",
            y="umap3_2",
            z="umap3_3",
            color="flavour_class",
            symbol="is_cc",
            hover_data=hover,
            title="NuJEPA event latent space: UMAP 3D by neutrino class",
        )
        path = out_dir / "latent_umap3d_by_flavour.html"
        fig.write_html(path)
        written.append(str(path))

    if {"tsne2_1", "tsne2_2"}.issubset(df.columns):
        fig = px.scatter(
            df,
            x="tsne2_1",
            y="tsne2_2",
            color="flavour_class",
            symbol="is_cc",
            hover_data=hover,
            title="NuJEPA event latent space: t-SNE 2D by neutrino class",
        )
        path = out_dir / "latent_tsne2d_by_flavour.html"
        fig.write_html(path)
        written.append(str(path))
    return written


def representation_checks(embeddings: np.ndarray, labels: pd.DataFrame) -> Dict[str, object]:
    norms = np.linalg.norm(embeddings, axis=1)
    class_counts = labels["flavour_class"].value_counts().to_dict()
    covariance = np.cov(embeddings.T)
    eigvals = np.linalg.eigvalsh(covariance)
    return {
        "num_events": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "all_finite": bool(np.isfinite(embeddings).all()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "feature_std_min": float(embeddings.std(axis=0).min()),
        "feature_std_mean": float(embeddings.std(axis=0).mean()),
        "feature_std_max": float(embeddings.std(axis=0).max()),
        "cov_eig_min": float(eigvals.min()),
        "cov_eig_max": float(eigvals.max()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
    }


def classification_probe(embeddings: np.ndarray, labels: pd.DataFrame, seed: int = 17) -> Dict[str, object]:
    y = labels["flavour_class"].to_numpy()
    counts = labels["flavour_class"].value_counts()
    keep_classes = counts[counts >= 5].index
    keep = labels["flavour_class"].isin(keep_classes).to_numpy()
    if keep.sum() < 20 or len(keep_classes) < 2:
        return {"skipped": True, "reason": "Need at least two classes with >=5 samples each."}

    x = embeddings[keep]
    y = y[keep]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    probes = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "knn_7": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(7, len(y_train)))),
    }
    out: Dict[str, object] = {
        "skipped": False,
        "classes_used": sorted(str(c) for c in keep_classes),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
    }
    for name, model in probes.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        out[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
    return out


def write_json(path: str, data: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
