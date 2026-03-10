from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CURRENT_YEAR = 2022

PLATE_COL = "N_REG_NEW"
OWNER_TYPE_COL = "PERSON"
BODY_TYPE_COL = "BODY"
FUEL_COL = "FUEL"

NUMERIC_COLS = ["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "MAKE_YEAR"]

REGION_CODE_MAP = {
    "AA": "Kyiv City", "KA": "Kyiv City",
    "AB": "Vinnytsia", "KB": "Vinnytsia",
    "AC": "Volyn", "KC": "Volyn",
    "AE": "Dnipropetrovsk", "KE": "Dnipropetrovsk",
    "AH": "Donetsk", "KH": "Donetsk",
    "AI": "Kyiv Region", "KI": "Kyiv Region",
    "AK": "Crimea", "KK": "Crimea",
    "AM": "Zhytomyr", "KM": "Zhytomyr",
    "AO": "Zakarpattia", "KO": "Zakarpattia",
    "AP": "Zaporizhzhia", "KP": "Zaporizhzhia",
    "AT": "Ivano-Frankivsk", "KT": "Ivano-Frankivsk",
    "AX": "Kharkiv", "KX": "Kharkiv",
    "BA": "Kirovohrad", "HA": "Kirovohrad",
    "BB": "Luhansk", "HB": "Luhansk",
    "BC": "Lviv", "HC": "Lviv",
    "BE": "Mykolaiv", "HE": "Mykolaiv",
    "BH": "Odesa", "HH": "Odesa",
    "BI": "Poltava", "HI": "Poltava",
    "BK": "Rivne", "HK": "Rivne",
    "BM": "Sumy", "HM": "Sumy",
    "BO": "Ternopil", "HO": "Ternopil",
    "BT": "Kherson", "HT": "Kherson",
    "BX": "Khmelnytskyi", "HX": "Khmelnytskyi",
    "CA": "Cherkasy", "IA": "Cherkasy",
    "CB": "Chernihiv", "IB": "Chernihiv",
    "CE": "Chernivtsi", "IE": "Chernivtsi",
}

CYR_TO_LAT = str.maketrans({
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "І": "I",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
})


def extract_region_from_plate(plate: object) -> str | None:
    if pd.isna(plate):
        return None

    value = str(plate).strip().upper()
    value = value.translate(CYR_TO_LAT)
    value = "".join(ch for ch in value if ch.isalnum())

    if len(value) < 2:
        return None

    prefix = value[:2]
    return REGION_CODE_MAP.get(prefix)


def preprocess_for_research(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Rows before cleaning: {len(df)}")

    required_cols = NUMERIC_COLS + [PLATE_COL, OWNER_TYPE_COL, BODY_TYPE_COL, FUEL_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()

    for col in NUMERIC_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["region"] = work[PLATE_COL].apply(extract_region_from_plate)

    work = work.dropna(subset=["MAKE_YEAR", "region", OWNER_TYPE_COL, BODY_TYPE_COL, FUEL_COL])
    work = work[(work["MAKE_YEAR"] >= 1950) & (work["MAKE_YEAR"] <= CURRENT_YEAR)]
    work = work[
        (work["CAPACITY"] >= 0) &
        (work["OWN_WEIGHT"] >= 0) &
        (work["TOTAL_WEIGHT"] >= 0)
    ]

    work["vehicle_age"] = CURRENT_YEAR - work["MAKE_YEAR"]

    for col in ["region", OWNER_TYPE_COL, BODY_TYPE_COL, FUEL_COL]:
        work[col] = work[col].astype(str).str.strip()

    print(f"Rows after cleaning: {len(work)}")
    return work


def analyze_regional_differences(df: pd.DataFrame) -> None:
    print("\n=== 1. Regional differences ===")

    for target_col in [BODY_TYPE_COL, FUEL_COL]:
        print(f"\nAnalysis: region vs {target_col}")

        contingency = pd.crosstab(df["region"], df[target_col])

        chi2, p_value, dof, _ = chi2_contingency(contingency)

        print(f"chi2 = {chi2:.4f}")
        print(f"p-value = {p_value:.8f}")
        print(f"dof = {dof}")

        if p_value < 0.05:
            print("There are statistically significant regional differences.")
        else:
            print("No statistically significant regional differences were found.")

        print("\nShares by region:")
        shares = contingency.div(contingency.sum(axis=1), axis=0)
        print(shares.head(10))


def find_optimal_k(X, k_min=2, k_max=6) -> int:
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        print(f"Trying k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"Silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nOptimal number of clusters: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")
    return best_k


def fit_kmeans_clusters(df_prepared: pd.DataFrame) -> pd.DataFrame:
    numeric_features = ["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "vehicle_age"]
    categorical_features = [BODY_TYPE_COL, FUEL_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X = preprocessor.fit_transform(df_prepared)
    sample = X[:20000] if len(df_prepared) > 20000 else X

    optimal_k = find_optimal_k(sample, 2, 6)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

    result = df_prepared.copy()
    result["cluster"] = kmeans.fit_predict(X)

    return result


def analyze_clusters(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 2. Natural structural clusters ===")

    clustered = fit_kmeans_clusters(df)

    print("\nCluster sizes:")
    print(clustered["cluster"].value_counts().sort_index())

    print("\nCluster means:")
    print(
        clustered.groupby("cluster")[["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "vehicle_age"]]
        .mean()
        .round(2)
    )

    print(f"\nDominant {BODY_TYPE_COL} by cluster:")
    print(
        clustered.groupby("cluster")[BODY_TYPE_COL]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else None)
    )

    print(f"\nDominant {FUEL_COL} by cluster:")
    print(
        clustered.groupby("cluster")[FUEL_COL]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else None)
    )

    return clustered


def analyze_owner_type_differences(df: pd.DataFrame) -> None:
    print("\n=== 3. Differences between individuals and legal entities ===")

    for target_col in [BODY_TYPE_COL, FUEL_COL]:
        print(f"\nAnalysis: {OWNER_TYPE_COL} vs {target_col}")

        contingency = pd.crosstab(df[OWNER_TYPE_COL], df[target_col])
        chi2, p_value, dof, _ = chi2_contingency(contingency)

        print(f"chi2 = {chi2:.4f}")
        print(f"p-value = {p_value:.8f}")
        print(f"dof = {dof}")

        if p_value < 0.05:
            print("The structure differs in a statistically significant way.")
        else:
            print("No statistically significant difference was found.")

        shares = contingency.div(contingency.sum(axis=1), axis=0)
        print("\nShares:")
        print(shares)

    owner_values = df[OWNER_TYPE_COL].dropna().unique()

    if len(owner_values) == 2:
        group_a, group_b = owner_values[0], owner_values[1]

        df_a = df[df[OWNER_TYPE_COL] == group_a]
        df_b = df[df[OWNER_TYPE_COL] == group_b]

        for col in ["vehicle_age", "CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT"]:
            a = df_a[col].dropna()
            b = df_b[col].dropna()

            if len(a) > 0 and len(b) > 0:
                stat, p_value = mannwhitneyu(a, b, alternative="two-sided")

                print(f"\nMann-Whitney for {col}")
                print(f"{group_a} median = {a.median():.2f}")
                print(f"{group_b} median = {b.median():.2f}")
                print(f"statistic = {stat:.4f}")
                print(f"p-value = {p_value:.8f}")

                if p_value < 0.05:
                    print("The difference is statistically significant.")
                else:
                    print("No statistically significant difference was found.")
    else:
        print(
            "\nExactly 2 owner groups were expected for the numerical comparison. "
            "Check the values in OWNER_TYPE_COL."
        )


def run(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)

    df_prepared = preprocess_for_research(df)

    analyze_regional_differences(df_prepared)
    clustered = analyze_clusters(df_prepared)
    analyze_owner_type_differences(df_prepared)

    print("\nClustered sample:")
    print(
        clustered[
            ["region", OWNER_TYPE_COL, BODY_TYPE_COL, FUEL_COL, "vehicle_age", "cluster"]
        ].head(10)
    )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        clustered.to_parquet(args.output / "clustered_vehicles.parquet", index=False)

        cluster_summary = (
            clustered.groupby("cluster")[["CAPACITY", "OWN_WEIGHT", "TOTAL_WEIGHT", "vehicle_age"]]
            .mean()
            .round(2)
            .reset_index()
        )
        cluster_summary.to_csv(args.output / "cluster_summary.csv", index=False)

        print(f"\nArtifacts saved to: {args.output}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data-research", help="Perform exploratory data research")
    p.add_argument("--input", required=True, type=Path, help="Path to processed dataset (.parquet)")
    p.add_argument("--output", type=Path, help="Directory to save output artifacts")
    p.set_defaults(func=run)