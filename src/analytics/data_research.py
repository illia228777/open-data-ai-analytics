from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


FEATURE_COLS = ["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "MAKE_YEAR"]
CURRENT_YEAR = 2022
N_CLUSTERS = 2


def preprocess_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Rows before cleaning: {len(df)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=FEATURE_COLS)

    work = work[(work["MAKE_YEAR"] >= 1950) & (work["MAKE_YEAR"] <= CURRENT_YEAR)]
    work = work[(work["CAPACITY"] >= 0) & (work["OWN_WEIGHT"] >= 0) & (work["TOTAL_WEIGHT"] >= 0)]

    work["vehicle_age"] = CURRENT_YEAR - work["MAKE_YEAR"]

    print(f"Rows after cleaning: {len(work)}")

    return work


def find_optimal_k(X_scaled, k_min=2, k_max=6):
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        print(f"Trying k={k}...")

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette score: {round(score, 4)}/n")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Optimal number of clusters (silhouette): {best_k}")
    print(f"Best silhouette score: {round(best_score, 4)}")

    return best_k


def fit_kmeans_clusters(df_prepared: pd.DataFrame) -> pd.DataFrame:
    features = df_prepared[["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "vehicle_age"]]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    sample = X[:20000]

    optimal_k = N_CLUSTERS if N_CLUSTERS else find_optimal_k(sample)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df_prepared = df_prepared.copy()
    df_prepared["cluster"] = kmeans.fit_predict(X)

    return df_prepared


def run(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)

    df_prepared = preprocess_for_clustering(df)
    df_clustered = fit_kmeans_clusters(df_prepared)

    print("\nCluster sizes:")
    print(df_clustered["cluster"].value_counts())

    print("\nCluster means:")
    print(
        df_clustered.groupby("cluster")[["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "vehicle_age"]]
        .mean()
    )


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data-research", help="Perform exploratory data research")
    p.add_argument("--input", required=True, type=Path, help="Path to processed dataset (.parquet)")
    p.set_defaults(func=run)