# === PRACTICA 11 — BONUS ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# === 1. Dataset Público: Ventas Minoristas (Daily Female Births) ===
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
df = pd.read_csv(url)
df.columns = ["date", "value"]
df["date"] = pd.to_datetime(df["date"])

# Simulamos múltiples clientes para detección individual
np.random.seed(42)
clientes = [f"C{i}" for i in range(1, 6)]
df_full = pd.concat([
    df.assign(cliente=c, value=df["value"] * np.random.uniform(0.8, 1.2))
    for c in clientes
])
df_full = df_full.reset_index(drop=True)

# === 2. Feature Engineering Temporal ===
def add_temporal_features(data):
    data = data.copy()
    data["dayofweek"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    for lag in [1, 3, 7]:
        data[f"lag_{lag}"] = data.groupby("cliente")["value"].shift(lag)
    data["rolling_mean_7"] = data.groupby("cliente")["value"].transform(lambda x: x.rolling(7).mean())
    data["rolling_std_7"] = data.groupby("cliente")["value"].transform(lambda x: x.rolling(7).std())
    return data

df_full = add_temporal_features(df_full).dropna()

# === 3. Shocks exógenos: feriados y promociones ===
feriados = pd.to_datetime(["1959-02-14", "1959-07-04", "1959-12-25"])
df_full["feriado"] = df_full["date"].isin(feriados).astype(int)
df_full["promocion"] = (df_full["date"].dt.day % 15 == 0).astype(int)

# === 4. Detección de anomalías ===
df_full["is_anomaly"] = df_full.apply(
    lambda r: (r["value"] > r["rolling_mean_7"] + 2 * r["rolling_std_7"])
    if not np.isnan(r["rolling_std_7"]) else False,
    axis=1
)

# === 5. Comparación de estrategias ===
features_lag = ["lag_1", "lag_3", "lag_7", "dayofweek", "month", "feriado", "promocion"]
features_win = ["rolling_mean_7", "rolling_std_7", "dayofweek", "month", "feriado", "promocion"]

cliente_demo = "C1"
df_c = df_full[df_full["cliente"] == cliente_demo].dropna()

split_date = "1959-09-01"
train, test = df_c[df_c["date"] < split_date], df_c[df_c["date"] >= split_date]
y_train, y_test = train["value"], test["value"]

# === Escalado ===
scaler = StandardScaler()
X_train_lag = scaler.fit_transform(train[features_lag])
X_test_lag = scaler.transform(test[features_lag])
X_train_win = scaler.fit_transform(train[features_win])
X_test_win = scaler.transform(test[features_win])

# === 6. Modelos XGBoost ===
def fit_xgb(X_train, y_train, X_test, y_test, label):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"RMSE {label}: {rmse:.3f}")
    return model, pred

xgb_lag, pred_lag = fit_xgb(X_train_lag, y_train, X_test_lag, y_test, "Lag features")
xgb_win, pred_win = fit_xgb(X_train_win, y_train, X_test_win, y_test, "Window features")

# === 7. Visualización ===
plt.figure(figsize=(10, 5))
plt.plot(test["date"], y_test, label="Real")
plt.plot(test["date"], pred_lag, label="Pred XGB (lag)")
plt.plot(test["date"], pred_win, label="Pred XGB (window)")
plt.title("Predicción Temporal — Comparativa de Estrategias")
plt.legend()
plt.tight_layout()
plt.show()

# === 8. Exportación para dashboard ===
df_full.to_csv("predicciones_temporales_livianas.csv", index=False)
print("Archivo exportado: predicciones_temporales_livianas.csv")
