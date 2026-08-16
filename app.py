import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Poutre IA - Surrogate Model Nastran",
    layout="wide"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
artifacts_path = os.path.join(current_dir, "models_artifacts.joblib")

@st.cache_resource
def load_models():
    return joblib.load(artifacts_path)

try:
    artifacts = load_models()
except Exception as e:
    st.error(f"Erreur de chargement des modèles. Lance d'abord export_models.py ! ({e})")
    st.stop()

scaler_X = artifacts['scaler_X']
scaler_y = artifacts['scaler_y']
mlp_model = artifacts['mlp']
poly_delta = artifacts['poly_delta']
poly_sigma = artifacts['poly_sigma']
rf_model = artifacts['rf']

# ==============================================================================
# NAVIGATION SIDEBAR
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    [
        "🚀 Démonstrateur Temps Réel", 
        "📊 Métriques & Validation", 
        "💻 Pipeline & Code Source", 
        "📄 Rapport & Théorie"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("**Projet Poutre IA**\nMétamodèle pour le calcul de structures\n*IPSA - Aéronautique*")

# ==============================================================================
# PAGE 1 : DÉMONSTRATEUR TEMPS RÉEL
# ==============================================================================
if page == "🚀 Démonstrateur Temps Réel":
    st.title("🚀 Démonstrateur IA en Temps Réel (Surrogate Model)")
    st.markdown("Prédisez instantanément la réponse structurelle d'une poutre sans exécuter de simulation éléments finis MSC Nastran.")

    col_in1, col_in2, col_in3 = st.columns([1.2, 1.2, 1])

    with col_in1:
        st.subheader("1. Paramètres Géométriques & Charges")
        # Double contrôle : Slider ET champ numérique direct
        L_val = st.number_input("Longueur $L$ (mm)", min_value=500.0, max_value=2500.0, value=1500.0, step=10.0)
        L_slider = st.slider("Ajustement rapide Longueur", 500.0, 2500.0, float(L_val), key="l_slide")
        
        # Synchronisation slider/number
        if L_slider != L_val:
            L_val = L_slider

    with col_in2:
        st.subheader(" ")
        st.write("")
        F_val = st.number_input("Force appliquée $F$ (N)", min_value=1000.0, max_value=10000.0, value=5000.0, step=50.0)
        F_slider = st.slider("Ajustement rapide Force", 1000.0, 10000.0, float(F_val), key="f_slide")
        
        if F_slider != F_val:
            F_val = F_slider

    with col_in3:
        st.subheader("2. Moteur d'IA")
        chosen_model = st.selectbox(
            "Modèle prédictif :",
            ["Réseau de Neurones (MLP)", "Régression Polynomiale (Deg 3)", "Forêt Aléatoire (Random Forest)"]
        )

    # --- INFÉRENCE TEMPS RÉEL ---
    t_start = time.perf_counter_ns()

    if chosen_model == "Réseau de Neurones (MLP)":
        x_scaled = scaler_X.transform([[L_val, F_val]])
        y_pred_scaled = mlp_model.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        d_pred, s_pred = y_pred[0], y_pred[1]

    elif chosen_model == "Régression Polynomiale (Deg 3)":
        d_pred = poly_delta.predict([[L_val, F_val]])[0]
        s_pred = poly_sigma.predict([[L_val, F_val]])[0]

    else:
        y_pred = rf_model.predict([[L_val, F_val]])[0]
        d_pred, s_pred = y_pred[0], y_pred[1]

    t_inference_us = (time.perf_counter_ns() - t_start) / 1000.0  # en microsecondes
    t_nastran_est_s = 2.5  # Temps moyen d'un run Nastran SOL 101
    speed_up = (t_nastran_est_s * 1e6) / max(t_inference_us, 1.0)

    st.markdown("---")
    st.subheader("📈 Résultats de l'Inférence")

    res1, res2, res3, res4 = st.columns(4)
    res1.metric("Flèche Max (Δ)", f"{d_pred:.2f} mm")
    res2.metric("Contrainte Max (Σ)", f"{s_pred:.2f} MPa")
    res3.metric("Temps de Calcul IA", f"{t_inference_us:.1f} µs")
    res4.metric("Gain de Vitesse (Speed-Up)", f"× {speed_up:,.0f}")

    # Visualisation 2D dynamique de la poutre déformée
    st.markdown("### 🔍 Vue de la Déformée en Temps Réel")
    
    x_beam = np.linspace(0, L_val, 100)
    # Profil de flexion cubique normalisé par d_pred
    y_beam = - d_pred * (3 * (x_beam / L_val)**2 - (x_beam / L_val)**3) / 2.0

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x_beam, np.zeros_like(x_beam), 'k--', label="Poutre non déformée", alpha=0.5)
    
    # Couleur du profil en fonction de la contrainte
    cmap = plt.cm.coolwarm
    norm_stress = min(max(s_pred / 500.0, 0.0), 1.0)
    beam_color = cmap(norm_stress)

    ax.plot(x_beam, y_beam, color=beam_color, lw=4, label=f"Déformée IA (Δmax = {d_pred:.1f} mm)")
    ax.plot(0, 0, marker='s', color='black', markersize=12, label="Encastrement")
    ax.annotate(f"F = {F_val:.0f} N", xy=(L_val, y_beam[-1]), xytext=(L_val, y_beam[-1] + 30),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                ha='center', fontweight='bold', color='red')

    ax.set_ylim(-350, 60)
    ax.set_xlim(-50, 2600)
    ax.set_xlabel("Longueur (mm)")
    ax.set_ylabel("Flèche (mm)")
    ax.set_title(f"Visualisation de la déformée structurelle | Niveau de contrainte : {s_pred:.1f} MPa", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left')
    st.pyplot(fig)

# ==============================================================================
# PAGE 2 : MÉTRIQUES & VALIDATION
# ==============================================================================
elif page == "📊 Métriques & Validation":
    st.title("📊 Métriques & Comparaison des Modèles")
    st.markdown("Évaluation rigoureuse des 3 modèles sur les **60 cas de test indépendants** (20% du dataset).")

    metrics_csv = os.path.join(parent_dir, "ETAPE 7", "tableau_comparatif_modeles.csv")
    if os.path.exists(metrics_csv):
        df_met = pd.read_csv(metrics_csv)
        st.dataframe(df_met.style.highlight_max(subset=['R² (Δ)', 'R² (Σ)'], color='#90ee90'), use_container_width=True)
    else:
        st.warning("Tableau comparatif non trouvé. Exécute d'abord validate_models.py dans l'Étape 7.")

    st.markdown("---")
    st.subheader("Galerie des Graphiques de Validation")

    g1, g2 = st.columns(2)
    img_dir = os.path.join(parent_dir, "ETAPE 7")
    
    with g1:
        p1 = os.path.join(img_dir, "1_parity_plots.png")
        if os.path.exists(p1):
            st.image(p1, caption="1. Parity Plots (Prédit vs Réel Nastran)")
        p3 = os.path.join(img_dir, "3_mlp_learning_curves.png")
        if os.path.exists(p3):
            st.image(p3, caption="3. Courbes d'apprentissage du MLP")

    with g2:
        p2 = os.path.join(img_dir, "2_residual_plots.png")
        if os.path.exists(p2):
            st.image(p2, caption="2. Analyse des Résidus")
        p4 = os.path.join(img_dir, "4_error_maps_2d.png")
        if os.path.exists(p4):
            st.image(p4, caption="4. Cartes d'erreur 2D dans l'espace (L, F)")

# ==============================================================================
# PAGE 3 : PIPELINE & CODE SOURCE
# ==============================================================================
elif page == "💻 Pipeline & Code Source":
    st.title("💻 Architecture & Pipeline du Projet")
    
    st.markdown("""
    ### 🔄 Pipeline de Données et d'Entraînement :
    1. **CAO / Maillage** : Génération du modèle de référence sous MSC Patran (`Hex20_12mm.txt`).
    2. **DoE (Plan d'Expériences)** : Latin Hypercube Sampling (LHS) 2D sur $$L \in [500, 2500]\text{ mm}$$ et $$F \in [1000, 16000]\text{ N}$$.
    3. **Automatisation FEA** : Scripting Python modifiant les cartes `GRID` et `FORCE` des fichiers `.bdf`, puis résolution par MSC Nastran (SOL 101).
    4. **Parsing & Nettoyage** : Extraction automatique des déplacements $$T_2$$ et contraintes de Von Mises des `.f06`.
    5. **Métamodélisation** : Entraînement de MLP, Régression Polynomiale et Random Forest.
    """)

    st.markdown("---")
    st.subheader("Extrait du code d'inférence IA")
    st.code("""
# Inférence directe avec le Multi-Layer Perceptron (MLP)
x_scaled = scaler_X.transform([[longueur_mm, force_n]])
y_pred_scaled = mlp_model.predict(x_scaled)
delta_max, sigma_max = scaler_y.inverse_transform(y_pred_scaled)[0]
    """, language="python")

# ==============================================================================
# PAGE 4 : RAPPORT & THÉORIE
# ==============================================================================
elif page == "📄 Rapport & Théorie":
    st.title("📄 Rapport Technique & Formulation Mathématique")

    st.markdown("""
    ### 📚 Rappel Théorique (Flexion des Poutres)
    Pour une poutre encastrée-libre soumise à un effort ponctuel $F$ en son extrémité :
    
    * **Flèche maximale théorique :**
    $$ \delta_{\max} = \frac{F \cdot L^3}{3 E I} $$
    
    * **Contrainte maximale de flexion :**
    $$ \sigma_{\max} = \frac{M_f \cdot y_{\max}}{I} = \frac{F \cdot L \cdot (h/2)}{I} $$
    
    * **Facteur d'accélération numérique :**
    $$ \text{Speed-Up} = \frac{T_{\text{Nastran}}}{T_{\text{IA}}} \approx \frac{2.5\text{ s}}{50\text{ }\mu\text{s}} \approx 50\,000\times $$
    """)