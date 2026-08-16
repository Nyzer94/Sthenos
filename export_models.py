import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
csv_path = os.path.join(parent_dir, "dataset_poutre_brut.csv")

df = pd.read_csv(csv_path)
df['Sigma_max_MPa'] = df['Sigma_max_MPa'] / 1e4

X = df[['Longueur_mm', 'Force_N']].values
y = df[['Delta_max_mm', 'Sigma_max_MPa']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
y_train_s = scaler_y.fit_transform(y_train)

# 1. Modèle MLP
mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train_s)

# 2. Modèle Polynomial
poly_delta = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('ridge', Ridge(alpha=1.0))])
poly_sigma = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=3)), ('ridge', Ridge(alpha=1.0))])
poly_delta.fit(X_train, y_train[:, 0])
poly_sigma.fit(X_train, y_train[:, 1])

# 3. Modèle Random Forest
rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
rf.fit(X_train, y_train)

# Export dans un dictionnaire sérialisé
artifacts = {
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'mlp': mlp,
    'poly_delta': poly_delta,
    'poly_sigma': poly_sigma,
    'rf': rf,
    'X_test': X_test,
    'y_test': y_test
}

export_path = os.path.join(current_dir, "models_artifacts.joblib")
joblib.dump(artifacts, export_path)
print(f"✅ Tous les modèles et scalers ont été exportés avec succès dans : {export_path}")