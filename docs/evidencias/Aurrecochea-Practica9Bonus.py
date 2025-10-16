#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Práctica 09 — Próximos Pasos (Aplicación a otro dataset)
Dataset: Ames Housing (OpenML, "house_prices") -> Clasificación binaria: SalePrice > mediana
Autoría: Alexia + equipo
Requisitos recomendados:
    pip install scikit-learn category-encoders shap joblib graphviz
"""
from __future__ import annotations
import os
import json
import time
import math
import joblib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

# Encoders adicionales
from category_encoders import TargetEncoder, CatBoostEncoder, BinaryEncoder, HashingEncoder

# SHAP opcional
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)


# =========================
# Utilidades generales
# =========================
@dataclass
class EvalResult:
    tag: str
    accuracy: float
    auc: float
    f1: float
    fit_time: float
    n_features: Optional[int] = None
    model_path: Optional[str] = None
    notes: str = ""


def summarize_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
    }


def classify_cardinality(df: pd.DataFrame, cat_cols: List[str], low=10, mid=50) -> Tuple[List[str], List[str], List[str]]:
    low_c, mid_c, high_c = [], [], []
    for c in cat_cols:
        n = df[c].astype("object").nunique(dropna=True)
        if n <= low:
            low_c.append(c)
        elif n <= mid:
            mid_c.append(c)
        else:
            high_c.append(c)
    return low_c, mid_c, high_c


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (simple) para monitoreo de drift.
    expected: distribución base (train), actual: nueva (production/test).
    """
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    qs = np.linspace(0, 1, bins + 1)
    cuts = expected.quantile(qs).values
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a_counts = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))


def export_results_table(results: List[EvalResult], path_csv: str):
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(path_csv, index=False)
    print(f"[OK] Resultados guardados en {path_csv}")


def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =========================
# Carga dataset (Ames)
# =========================
def load_ames(as_frame: bool = True) -> pd.DataFrame:
    """
    Carga Ames Housing desde OpenML. Si falla, indicamos fallback local.
    """
    try:
        Xy = fetch_openml(name="house_prices", version=1, as_frame=True)
        df = Xy.frame.copy()
        # target original: SalePrice (regresión)
        # lo convertimos en clasificación: > mediana
        df = df.dropna(subset=["SalePrice"])
        median_price = df["SalePrice"].median()
        df["target"] = (df["SalePrice"] > median_price).astype(int)
        return df
    except Exception as e:
        print("[WARN] No se pudo descargar desde OpenML:", e)
        print("       Usá un CSV local con las columnas de Ames y la columna SalePrice.")
        print("       Ejemplo: df = pd.read_csv('ames.csv'); df['target'] = (df['SalePrice'] > df['SalePrice'].median()).astype(int)")
        raise


# =========================
# Columnas y preprocesado
# =========================
def split_types(df: pd.DataFrame, target_col="target") -> Tuple[List[str], List[str]]:
    feat_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feat_cols if c not in num_cols]
    return num_cols, cat_cols


def make_branched_preprocessor(
    low_card: List[str],
    high_card: List[str],
    num_cols: List[str],
    smoothing: float = 20.0,
) -> ColumnTransformer:
    """
    Pipeline branched:
      - One-Hot en baja cardinalidad
      - Target Encoding (CatBoostEncoder por defecto) en alta cardinalidad
      - StandardScaler en numéricas
    """
    onehot_branch = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    target_branch = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Podés alternar entre TargetEncoder y CatBoostEncoder
        ("tenc", CatBoostEncoder(a=smoothing, b=smoothing)),  # prior/post smoothing
        # Alternativa:
        # ("tenc", TargetEncoder(smoothing=smoothing))
    ])

    numeric_branch = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("low_card", onehot_branch, low_card),
            ("high_card", target_branch, high_card),
            ("num", numeric_branch, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return pre


# =========================
# Experimentos de encoding
# =========================
def run_experiment(
    tag: str,
    preprocessor: ColumnTransformer,
    model,
    X_train, y_train, X_test, y_test,
    export_dir="artifacts"
) -> EvalResult:
    os.makedirs(export_dir, exist_ok=True)
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", model)
    ])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        # fallback para modelos sin predict_proba
        y_proba = pipe.decision_function(X_test)
        # estandarizamos a [0,1] vía min-max si es necesario
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    metrics = summarize_metrics(y_test, y_pred, y_proba)
    n_features = None
    try:
        n_features = pipe.named_steps["pre"].get_feature_names_out().shape[0]
    except Exception:
        pass

    model_path = os.path.join(export_dir, f"model_{tag.replace(' ', '_')}.joblib")
    joblib.dump(pipe, model_path)

    print(f"[{tag}] Acc={metrics['Accuracy']:.4f} | AUC={metrics['ROC_AUC']:.4f} | F1={metrics['F1']:.4f} | time={fit_time:.2f}s | feats={n_features}")
    return EvalResult(tag=tag, accuracy=metrics["Accuracy"], auc=metrics["ROC_AUC"], f1=metrics["F1"], fit_time=fit_time, n_features=n_features, model_path=model_path)


def build_and_compare_encoders(
    df: pd.DataFrame, target_col="target", export_dir="artifacts"
) -> List[EvalResult]:
    num_cols, cat_cols = split_types(df, target_col)
    low_c, mid_c, high_c = classify_cardinality(df, cat_cols, low=10, mid=50)

    X = df[num_cols + cat_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelos a evaluar
    logreg = LogisticRegression(max_iter=1000, n_jobs=None)
    hgb = HistGradientBoostingClassifier(random_state=42)

    results: List[EvalResult] = []

    # 1) One-Hot en todas las categóricas (baseline interpretable)
    pre_onehot_all = ColumnTransformer(transformers=[
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols)
    ], remainder="drop", verbose_feature_names_out=True)

    results.append(run_experiment("OneHot_All + LogReg", pre_onehot_all, logreg, X_train, y_train, X_test, y_test, export_dir))
    results.append(run_experiment("OneHot_All + HGB", pre_onehot_all, hgb, X_train, y_train, X_test, y_test, export_dir))

    # 2) Binary Encoding (reduce dimensión log2(N))
    pre_binary = ColumnTransformer(transformers=[
        ("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("be", BinaryEncoder(cols=cat_cols))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols)
    ], remainder="drop", verbose_feature_names_out=True)

    results.append(run_experiment("Binary + LogReg", pre_binary, logreg, X_train, y_train, X_test, y_test, export_dir))
    results.append(run_experiment("Binary + HGB", pre_binary, hgb, X_train, y_train, X_test, y_test, export_dir))

    # 3) Hashing Encoding (streaming-friendly, dimensionalidad fija)
    hash_dim = 64  # ajustable
    pre_hash = ColumnTransformer(transformers=[
        ("hash", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("he", HashingEncoder(cols=cat_cols, n_components=hash_dim, drop_invariant=False))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols)
    ], remainder="drop", verbose_feature_names_out=True)

    results.append(run_experiment("Hash64 + LogReg", pre_hash, logreg, X_train, y_train, X_test, y_test, export_dir))
    results.append(run_experiment("Hash64 + HGB", pre_hash, hgb, X_train, y_train, X_test, y_test, export_dir))

    # 4) CatBoost/Target Encoding solo en alta cardinalidad + One-Hot en baja
    pre_branched = make_branched_preprocessor(low_card=low_c + mid_c, high_card=high_c, num_cols=num_cols, smoothing=20.0)
    results.append(run_experiment("Branched(CBEnc)+LogReg", pre_branched, logreg, X_train, y_train, X_test, y_test, export_dir))
    results.append(run_experiment("Branched(CBEnc)+HGB", pre_branched, hgb, X_train, y_train, X_test, y_test, export_dir))

    # 5) GridSearch de smoothing (CatBoostEncoder a,b) dentro del branched
    grid_pipe = Pipeline(steps=[
        ("pre", make_branched_preprocessor(low_c + mid_c, high_c, num_cols, smoothing=10.0)),
        ("clf", HistGradientBoostingClassifier(random_state=42))
    ])

    # Los parámetros del encoder dentro del ColumnTransformer se referencian por nombre:
    # pre__high_card__tenc__a  y  pre__high_card__tenc__b
    param_grid = {
        "pre__high_card__tenc__a": [1, 5, 10, 20, 50, 100],
        "pre__high_card__tenc__b": [1, 5, 10, 20, 50, 100],
        "clf__max_depth": [None, 6, 10],
        "clf__learning_rate": [0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(grid_pipe, param_grid, cv=cv, n_jobs=-1, scoring="roc_auc", refit=True, verbose=0)
    t0 = time.time()
    gs.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = gs.predict(X_test)
    y_proba = gs.predict_proba(X_test)[:, 1]
    m = summarize_metrics(y_test, y_pred, y_proba)
    try:
        n_feats = gs.best_estimator_.named_steps["pre"].get_feature_names_out().shape[0]
    except Exception:
        n_feats = None

    model_path = os.path.join(export_dir, "model_BestGrid_BranchedCBEnc_HGB.joblib")
    joblib.dump(gs.best_estimator_, model_path)
    results.append(EvalResult(
        tag="BestGrid Branched(CBEnc)+HGB",
        accuracy=m["Accuracy"], auc=m["ROC_AUC"], f1=m["F1"],
        fit_time=fit_time, n_features=n_feats, model_path=model_path,
        notes=f"best_params={gs.best_params_}"
    ))
    print(f"[BestGrid] AUC={m['ROC_AUC']:.4f} | params={gs.best_params_} | time={fit_time:.2f}s | feats={n_feats}")

    # Guardamos también el reporte de CV
    cv_report = pd.DataFrame(gs.cv_results_)
    cv_report.to_csv(os.path.join(export_dir, "gridsearch_cv_results.csv"), index=False)

    # Reporte texto del mejor modelo
    print_section("Reporte del mejor modelo (GridSearch - Branched CBEnc + HGB)")
    print(classification_report(y_test, y_pred, digits=4))

    # SHAP opcional (árboles): resumen rápido
    if SHAP_AVAILABLE:
        print_section("SHAP (resumen rápido)")
        explainer = shap.Explainer(gs.best_estimator_.named_steps["clf"])
        # Obtenemos X_test transformado para SHAP
        X_test_trans = gs.best_estimator_.named_steps["pre"].transform(X_test)
        try:
            shap_values = explainer(X_test_trans[:200])  # muestreo por rapidez
            # Guardamos valores medios absolutos por feature
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": [f for f in range(X_test_trans.shape[1])],
                "mean_abs_shap": mean_abs
            }).sort_values("mean_abs_shap", ascending=False)
            shap_df.head(30).to_csv(os.path.join(export_dir, "shap_top_features.csv"), index=False)
            print("[OK] SHAP exportado (shap_top_features.csv)")
        except Exception as e:
            print("[WARN] SHAP no pudo correr:", e)

    # Monitoreo de drift básico (PSI) sobre probabilidades
    psi_score = psi(
        expected=gs.predict_proba(X_train)[:, 1],
        actual=gs.predict_proba(X_test)[:, 1],
        bins=10
    )
    with open(os.path.join(export_dir, "monitoring.json"), "w") as f:
        json.dump({"psi_proba_train_vs_test": float(psi_score)}, f, indent=2)
    print(f"[Monitor] PSI(prob_train vs prob_test) = {psi_score:.4f} (<=0.1 estable, 0.1-0.25 leve, >0.25 alerta)")

    return results


def export_pipeline_diagram(estimator: Pipeline, path="artifacts/pipeline.txt"):
    """
    Exporta una representación textual del pipeline (útil para documentación).
    """
    from sklearn import set_config
    os.makedirs(os.path.dirname(path), exist_ok=True)
    set_config(display="diagram")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(estimator))
    print(f"[OK] Diagrama del pipeline exportado a {path}")


def main():
    print_section("Cargando dataset: Ames Housing (OpenML)")
    df = load_ames()
    print(f"Shape={df.shape} | target rate={(df['target']==1).mean():.3f}")

    results = build_and_compare_encoders(df, target_col="target", export_dir="artifacts")
    export_results_table(results, "artifacts/resultados_modelos.csv")

    # Persistimos también el mejor por AUC de la tanda rápida (no el grid)
    best_auc = max(results, key=lambda r: r.auc)
    print_section(f"Mejor (rápido) por AUC: {best_auc.tag} | AUC={best_auc.auc:.4f}")
    print(f"Modelo guardado en: {best_auc.model_path}")

    # Exportar diagrama textual de ese pipeline
    try:
        pipe_loaded: Pipeline = joblib.load(best_auc.model_path)
        export_pipeline_diagram(pipe_loaded, "artifacts/pipeline_diagrama.txt")
    except Exception as e:
        print("[WARN] No se pudo exportar diagrama del pipeline:", e)

    print_section("FIN — Próximos pasos implementados")
    print("* Branched Pipeline con CatBoost/Target Encoding y One-Hot")
    print("* GridSearch del smoothing (a,b) y HGB hyperparams")
    print("* Comparación con Binary/Hash/One-Hot")
    print("* SHAP opcional y monitoreo PSI")
    print("* Artefactos guardados en ./artifacts")


if __name__ == "__main__":
    main()
