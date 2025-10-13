#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BONUS — Feature Engineering avanzado (con datasets externos):
- Lee 'synthetic_housing.csv' y 'ames_mini.csv' desde rutas configurables
- PolynomialFeatures (grado 2) en columnas numéricas
- RFE con RandomForestRegressor (selección top-k)
- One-Hot Encoding de Neighborhood (Ames)
- Pipeline reproducible: ColumnTransformer + Pipeline + CV

Rutas por defecto:
    data/synthetic_housing.csv
    data/ames_mini.csv

Cómo ejecutar:
    python bonus_feature_engineering.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

# ------------------------ CONFIGURACIÓN ------------------------

RANDOM_SEED = 42

PATH_SYNTHETIC = "data/synthetic_housing.csv"
PATH_AMES = "data/ames_mini.csv"

# Esquema esperado para cada dataset
REQUIRED_SYNTH_COLS = [
    "price",
    "sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "garage_spaces",
    "lot_size",
    "distance_to_city",
    "school_rating",
    "crime_rate",
]

REQUIRED_AMES_COLS = [
    "SalePrice",
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "LotArea",
    "Neighborhood",
]

# Columnas numéricas para polinomios en sintético y Ames
POLY_COLS_SYNTH = ["sqft", "lot_size", "property_age", "school_rating"]
POLY_COLS_AMES = ["GrLivArea", "LotArea", "property_age", "GarageCars"]

np.random.seed(RANDOM_SEED)


# ------------------------ UTILIDADES ------------------------

def load_csv(path: str, required_cols: List[str]) -> pd.DataFrame:
    """Carga un CSV y valida columnas requeridas."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}\n"
            f"Verificá la ruta o actualizá PATH_* en el script."
        )
    df = pd.read_csv(path)
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(
            f"El archivo '{path}' no contiene las columnas requeridas:\n"
            f"  Faltan: {missing}\n"
            f"  Presentes: {sorted(df.columns.tolist())}"
        )
    return df


def engineer_features_synthetic(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """Feature engineering base para el dataset sintético."""
    dfe = df.copy()

    # saneo mínimo de no-negatividad para variables clave
    for c in ["price", "sqft", "lot_size", "distance_to_city"]:
        dfe[c] = np.abs(dfe[c].astype(float))

    # Ratios / proporciones
    dfe["price_per_sqft"] = dfe["price"] / dfe["sqft"]
    dfe["sqft_per_bedroom"] = dfe["sqft"] / dfe["bedrooms"]
    dfe["build_density"] = dfe["sqft"] / dfe["lot_size"]
    dfe["price_per_bedroom"] = dfe["price"] / dfe["bedrooms"]

    # Temporales
    dfe["property_age"] = current_year - dfe["year_built"]
    dfe["age_category"] = pd.cut(
        dfe["property_age"], bins=[-1, 5, 20, np.inf],
        labels=["Nueva", "Moderna", "Antigua"]
    )
    dfe["is_new_property"] = (dfe["property_age"] <= 5).astype(int)

    # Transformaciones matemáticas
    dfe["log_price"] = np.log(dfe["price"])
    dfe["sqrt_sqft"] = np.sqrt(dfe["sqft"])
    dfe["sqft_squared"] = dfe["sqft"] ** 2

    # Scores compuestos
    dfe["luxury_score"] = (
        0.5 * (dfe["price_per_sqft"] / dfe["price_per_sqft"].max())
        + 0.3 * (dfe["sqft"] / dfe["sqft"].max())
        + 0.2 * (dfe["garage_spaces"] / max(1, dfe["garage_spaces"].max()))
    )
    dfe["location_score"] = (
        0.4 * (1 - dfe["distance_to_city"] / dfe["distance_to_city"].max())
        + 0.4 * (dfe["school_rating"] / dfe["school_rating"].max())
        + 0.2 * (1 - dfe["crime_rate"] / dfe["crime_rate"].max())
    )
    return dfe


def add_polynomial_features(
    df_in: pd.DataFrame,
    cols: List[str],
    degree: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    """Agrega columnas polinómicas (grado=2) para columnas numéricas seleccionadas."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df_in[cols])
    names = poly.get_feature_names_out(cols)
    df_poly = pd.DataFrame(X_poly, columns=names, index=df_in.index)
    return pd.concat([df_in, df_poly], axis=1), list(names)


def rfe_select(
    dfe_poly: pd.DataFrame,
    target: str,
    n_features_to_select: int = 10,
) -> Tuple[List[str], float, float]:
    """RFE con RandomForestRegressor; devuelve columnas seleccionadas y R² full vs selected."""
    # tomar solo numéricas, excluir target
    num_cols = dfe_poly.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    X = dfe_poly[num_cols].fillna(0.0)
    y = dfe_poly[target].astype(float)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    base = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
    selector = RFE(base, n_features_to_select=n_features_to_select)
    selector.fit(Xtr, ytr)

    selected_cols = list(X.columns[selector.support_])

    base.fit(Xtr, ytr)
    yhat_full = base.predict(Xte)
    r2_full = r2_score(yte, yhat_full)

    base.fit(Xtr[selected_cols], ytr)
    yhat_sel = base.predict(Xte[selected_cols])
    r2_sel = r2_score(yte, yhat_sel)

    return selected_cols, r2_full, r2_sel


def expand_ames_with_noise(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """Amplía Ames con remuestreo y ruido suave (para evaluación offline)."""
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(df), size=n)
    base = df.iloc[idx].reset_index(drop=True).copy()

    def jitter(x, pct=0.05):
        return x * (1 + rng.normal(0.0, pct, size=len(x)))

    base["GrLivArea"] = np.clip(jitter(base["GrLivArea"], 0.08), 400, None)
    base["LotArea"] = np.clip(jitter(base["LotArea"], 0.08), 2000, None)
    base["SalePrice"] = np.clip(jitter(base["SalePrice"], 0.10), 50_000, None)

    # derivadas clave
    base["price_per_sqft"] = base["SalePrice"] / base["GrLivArea"]
    base["property_age"] = 2025 - base["YearBuilt"]
    base["space_efficiency"] = base["GrLivArea"] / base["LotArea"]
    return base


def simple_ols_with_onehot(ames_df: pd.DataFrame) -> float:
    """Modelo lineal simple con One-Hot(Neighborhood) — devuelve R² CV promedio."""
    target = "SalePrice"
    base_feats = ["price_per_sqft", "property_age"]
    cat = ["Neighborhood"]

    ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
    Z = ohe.fit_transform(ames_df[cat])
    zcols = ohe.get_feature_names_out(cat)
    Zdf = pd.DataFrame(Z, columns=zcols, index=ames_df.index)

    X = pd.concat([ames_df[base_feats], Zdf], axis=1)
    y = ames_df[target].astype(float)

    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    r2_cv = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return float(np.mean(r2_cv))


def build_and_eval_pipeline(ames_df: pd.DataFrame) -> dict:
    """Pipeline completo: (Scaler+Poly) sobre numéricas + OneHot en cat + RandomForest."""
    target = "SalePrice"
    num_features = ["GrLivArea", "LotArea", "property_age", "GarageCars"]
    cat_features = ["Neighborhood"]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ]
    )
    categorical_transformer = OneHotEncoder(
        drop="first", sparse=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X = ames_df[num_features + cat_features]
    y = ames_df[target].astype(float)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    metrics = {
        "r2_test": float(r2_score(yte, yhat)),
        "mae_test": float(mean_absolute_error(yte, yhat)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_total": int(len(ames_df)),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    metrics["r2_cv_mean"] = float(np.mean(cv_r2))
    metrics["r2_cv_std"] = float(np.std(cv_r2))
    return metrics


# ------------------------ MAIN ------------------------

def main():
    # ---------- A) SINTÉTICO: carga + FE + Poly + RFE ----------
    try:
        print(f">> Cargando dataset sintético desde: {PATH_SYNTHETIC}")
        synth = load_csv(PATH_SYNTHETIC, REQUIRED_SYNTH_COLS)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(">> Engineer features (sintético)…")
    dfe = engineer_features_synthetic(synth, current_year=2025)
    print(f"   - Dimensiones: base={synth.shape}  con_features={dfe.shape}")

    print(f">> PolynomialFeatures grado=2 en columnas: {POLY_COLS_SYNTH}")
    # asegurar presencia de columnas requeridas para poly
    missing_poly = sorted(set(POLY_COLS_SYNTH) - set(dfe.columns))
    if missing_poly:
        raise ValueError(f"Faltan columnas para polinomios (sintético): {missing_poly}")
    dfe_poly, poly_names = add_polynomial_features(dfe, POLY_COLS_SYNTH, degree=2)
    print(f"   - Nuevas columnas polinómicas: {len(poly_names)}  | Total cols: {dfe_poly.shape[1]}")

    print(">> RFE (RandomForest) — top-10 features (sintético)…")
    selected_cols, r2_full, r2_sel = rfe_select(dfe_poly, target="price", n_features_to_select=10)
    print(f"   - TOP-10 seleccionadas:\n     {selected_cols}")
    print(f"   - R² (todas): {r2_full:.4f} | R² (seleccionadas): {r2_sel:.4f}")

    # ---------- B) AMES: carga + expansión + OneHot + Pipeline ----------
    try:
        print(f"\n>> Cargando Ames mini desde: {PATH_AMES}")
        ames_mini = load_csv(PATH_AMES, REQUIRED_AMES_COLS)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(">> Expansión con ruido suave para evaluación (Ames)…")
    ames = expand_ames_with_noise(ames_mini, n=400)
    print(f"   - Ames expandido: {ames.shape}")

    print(">> OLS + OneHot(Neighborhood) — R² CV promedio…")
    r2_ols = simple_ols_with_onehot(ames)
    print(f"   - R² OLS (CV mean): {r2_ols:.4f}")

    print(">> Pipeline (Scaler+Poly en num, OneHot en cat) + RandomForest (Ames)…")
    # Validar columnas necesarias para polinomios en Ames
    for c in POLY_COLS_AMES:
        if c not in ames.columns:
            if c == "property_age":
                ames["property_age"] = 2025 - ames["YearBuilt"]
            else:
                raise ValueError(f"Falta columna requerida en Ames para polinomios: '{c}'")

    metrics = build_and_eval_pipeline(ames)
    print(
        "   - Métricas:\n"
        f"     R² test     : {metrics['r2_test']:.4f}\n"
        f"     MAE test    : {metrics['mae_test']:.2f}\n"
        f"     R² CV (mean): {metrics['r2_cv_mean']:.4f} ± {metrics['r2_cv_std']:.4f}\n"
        f"     n_total/train/test: {metrics['n_total']}/{metrics['n_train']}/{metrics['n_test']}"
    )

    print("\n✅ BONUS terminado correctamente.")


if __name__ == "__main__":
    main()
