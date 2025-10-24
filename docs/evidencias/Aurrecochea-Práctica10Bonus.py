# ============================================================
# Importación de librerías
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import json
import os

# ============================================================
# Carga del dataset Ames Housing
# ============================================================

df = pd.read_csv("AmesHousing.csv")
# Variable objetivo y predictores
target = "SalePrice"
X = df.drop(columns=[target])
y = df[target]

# Eliminamos columnas no numéricas o las convertimos temporalmente
X = X.select_dtypes(include=[np.number]).fillna(0)

print(f"Dimensiones iniciales: {X.shape}")

# ============================================================
# División de datos y estandarización
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Definición de los módulos individuales
# ============================================================

# PCA (mantiene el 80% de la varianza)
pca = PCA(n_components=0.8, random_state=42)

# Filter Method: Mutual Information (top 30 features)
selector_mi = SelectKBest(mutual_info_regression, k=30)

# Wrapper Method: RFE con Random Forest
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rfe = RFE(rf_model, n_features_to_select=20, step=1)

# Embedded Method: LassoCV
lasso = LassoCV(cv=5, random_state=42, n_alphas=100)

# ============================================================
# Pipeline híbrido
# ============================================================

hybrid_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.8, random_state=42)),
    ("select_mi", SelectKBest(mutual_info_regression, k=30)),
    ("rfe", RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=20)),
    ("lasso", LassoCV(cv=5, random_state=42)),
])

# ============================================================
# Entrenamiento y evaluación
# ============================================================

print("Entrenando pipeline híbrido...")

hybrid_pipeline.fit(X_train, y_train)
y_pred = hybrid_pipeline.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# ============================================================
#  Comparación con otros métodos individuales
# ============================================================

def evaluate_model(name, model):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return {"Método": name, "RMSE": rmse, "R²": r2}

models = [
    ("PCA", Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.8)), ("rf", rf_model)])),
    ("MutualInfo", Pipeline([("scaler", StandardScaler()), ("mi", selector_mi), ("rf", rf_model)])),
    ("RFE", Pipeline([("scaler", StandardScaler()), ("rfe", rfe), ("rf", rf_model)])),
    ("Lasso", Pipeline([("scaler", StandardScaler()), ("lasso", lasso)]))
]

results = [evaluate_model(name, model) for name, model in models]
results.append({"Método": "Híbrido (PCA+MI+RFE+Lasso)", "RMSE": rmse, "R²": r2})

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("R²", ascending=False)
df_results.reset_index(drop=True, inplace=True)

print("\n Resultados comparativos:\n")
print(df_results)

# ============================================================
# Ranking de importancia de variables (Random Forest)
# ============================================================

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10,6))
plt.barh(importances["Feature"], importances["Importance"])
plt.title("Top 20 Features - Random Forest Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("artifacts/importance_plot.png", dpi=150)
plt.show()

# ============================================================
# 9️Guardado de artefactos
# ============================================================

os.makedirs("artifacts", exist_ok=True)

# Pipeline completo
joblib.dump(hybrid_pipeline, "artifacts/hybrid_feature_selection_pipeline.joblib")

# Resultados de comparación
df_results.to_csv("artifacts/metrics_comparison.csv", index=False)

# Ranking de features
importances.to_csv("artifacts/feature_ranking_comparativo.csv", index=False)

# Reporte JSON resumido
report = {
    "RMSE": float(rmse),
    "R2": float(r2),
    "BestModel": "HybridPipeline",
    "FeaturesSelected": int(importances.shape[0]),
}
with open("artifacts/model_hybrid_report.json", "w") as f:
    json.dump(report, f, indent=4)

# Diagrama textual del pipeline
with open("artifacts/pipeline_diagram.txt", "w") as f:
    f.write(str(hybrid_pipeline))

print("\n Artefactos guardados en carpeta /artifacts")
