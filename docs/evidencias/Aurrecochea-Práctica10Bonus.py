# ============================================================
# Importación de librerías
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub

# ============================================================
# Carga del dataset Ames Housing
# ============================================================

path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")
print("Path to dataset files:", path)
df = pd.read_csv(f"{path}/AmesHousing.csv")

target = "SalePrice"
X = df.select_dtypes(include=[np.number]).drop(columns=[target]).fillna(0)
y = df[target]

print(f"Dimensiones iniciales: {X.shape}")

# ============================================================
# División de datos
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=150, random_state=42)

# ============================================================
# Función auxiliar de evaluación
# ============================================================

def evaluate_pipeline(name, pipeline, interpretabilidad):
    start = time.time()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    elapsed = time.time() - start

    # Determinar número de features efectivas
    try:
        n_features = pipeline[-1].n_features_in_
    except Exception:
        try:
            n_features = pipeline[-2].n_features_in_
        except Exception:
            n_features = X.shape[1]

    return {
        "Método": name,
        "Nº Features": n_features,
        "RMSE": rmse,
        "R²": r2,
        "Interpretabilidad": interpretabilidad,
        "Tiempo (s)": elapsed,
    }

# ============================================================
# Definición de métodos
# ============================================================

models = [
    ("PCA (0.8 varianza)", Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.8, random_state=42)),
        ("rf", rf_model)
    ]), "Baja"),
    ("Mutual Information", Pipeline([
        ("scaler", StandardScaler()),
        ("mi", SelectKBest(mutual_info_regression, k=30)),
        ("rf", rf_model)
    ]), "Alta"),
    ("RFE (RF)", Pipeline([
        ("scaler", StandardScaler()),
        ("rfe", RFE(rf_model, n_features_to_select=20, step=1)),
        ("rf", rf_model)
    ]), "Media"),
    ("LassoCV", Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=42))
    ]), "Alta"),
]

# Pipeline híbrido
hybrid = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.8, random_state=42)),
    ("mi", SelectKBest(mutual_info_regression, k=30)),
    ("rfe", RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=20)),
    ("lasso", LassoCV(cv=5, random_state=42)),
])

# ============================================================
# Ejecución de experimentos
# ============================================================

results = [evaluate_pipeline(name, pipe, interpret) for name, pipe, interpret in models]
results.append(evaluate_pipeline("Pipeline híbrido (PCA+MI+RFE)", hybrid, "Alta"))

df_results = pd.DataFrame(results)
df_results["RMSE"] = df_results["RMSE"].round(3)
df_results["R²"] = df_results["R²"].round(3)
df_results["Tiempo (s)"] = df_results["Tiempo (s)"].round(2)
df_results = df_results.sort_values("R²", ascending=False).reset_index(drop=True)

print("\nResultados calculados:")
print(df_results)

# ============================================================
# 📊 Generación de imagen tipo tabla (auto-calculada)
# ============================================================

import matplotlib.pyplot as plt
from datetime import datetime
import os

os.makedirs("artifacts", exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")

# Colores estilo portfolio (burgundy + azul)
colors = {"header": "#3B0A45", "bg": "#F5F5F7"}

# Crear tabla con los valores realmente calculados
table = ax.table(
    cellText=df_results.values,
    colLabels=df_results.columns,
    loc="center",
    cellLoc="center",
)

# Estilo general
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.3)

# Encabezado coloreado
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(colors["header"])
        cell.set_text_props(color="white", weight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#FFFFFF")
    else:
        cell.set_facecolor(colors["bg"])

plt.title(
    f"Resultados del Pipeline Híbrido — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    fontsize=11, color=colors["header"], fontweight="bold",
)
plt.tight_layout()
plt.savefig("artifacts/results_table.png", dpi=250, bbox_inches="tight")
plt.close()

print("\n✅ Imagen generada: artifacts/results_table.png")
