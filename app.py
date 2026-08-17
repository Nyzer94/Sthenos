import os
import time
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==============================================================================
# CONFIGURATION GLOBALE ET RECHERCHE ROBUSTE DE FICHIERS
# ==============================================================================

st.set_page_config(
    page_title="STHENOS — Surrogate Model",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

def find_file(*relative_paths):
    """Recherche un fichier parmi plusieurs chemins relatifs possibles."""
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    possible_roots = [
        current_file_dir,
        os.path.dirname(current_file_dir),
        os.path.dirname(os.path.dirname(current_file_dir)),
        os.getcwd(),
        os.path.join(os.getcwd(), "Projet Python Poutre"),
        os.path.join(os.getcwd(), ".."),
    ]
    for rel in relative_paths:
        for root in possible_roots:
            path = os.path.normpath(os.path.join(root, rel))
            if os.path.exists(path):
                return path
    return None

# Custom CSS — Style bureau d'études (sobre, lisible, professionnel)
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem;
        font-weight: 500;
    }
    h1 {
        font-weight: 700;
        font-size: 1.6rem;
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
    }
    h2, h3 {
        font-weight: 600;
        color: #1e293b;
    }
    [data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    .sidebar-brand {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc !important;
        letter-spacing: 0.05em;
    }
    .sidebar-info {
        font-size: 0.78rem;
        color: #94a3b8 !important;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CHARGEMENT DES ARTEFACTS (MODÈLES ET SCALERS)
# ==============================================================================

artifacts_path = find_file(
    os.path.join("ETAPE 8", "models_artifacts.joblib"),
    "models_artifacts.joblib",
    os.path.join("Projet Python Poutre", "ETAPE 8", "models_artifacts.joblib")
)

@st.cache_resource
def load_models(path):
    return joblib.load(path)

if artifacts_path and os.path.exists(artifacts_path):
    try:
        artifacts = load_models(artifacts_path)
        scaler_X   = artifacts['scaler_X']
        scaler_y   = artifacts['scaler_y']
        mlp_model  = artifacts['mlp']
        poly_delta = artifacts['poly_delta']
        poly_sigma = artifacts['poly_sigma']
        rf_model   = artifacts['rf']
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        st.stop()
else:
    st.error("Artefact `models_artifacts.joblib` introuvable. Exécutez `export_models.py` d'abord.")
    st.stop()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

st.sidebar.markdown('<p class="sidebar-brand">STHENOS</p>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<p class="sidebar-info">'
    'Métamodèle d\'apprentissage automatique<br>'
    'pour le calcul de structures<br><br>'
    'IPSA — Ingénierie Aéronautique<br>'
    'Spécialité Mécanique & Structures'
    '</p>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Démonstrateur temps réel",
        "Métriques & validation",
        "Architecture & pipeline",
        "Rapport technique",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p class="sidebar-info">'
    'Domaine de validité :<br>'
    '• Longueur L ∈ [500 – 2 500] mm<br>'
    '• Force F ∈ [1 000 – 16 000] N'
    '</p>',
    unsafe_allow_html=True
)

# ==============================================================================
# PAGE 1 — DÉMONSTRATEUR TEMPS RÉEL
# ==============================================================================

if page == "Démonstrateur temps réel":

    st.title("Démonstrateur — Prédiction structurelle en temps réel")
    st.markdown(
        "Prédiction instantanée du comportement d'une poutre encastrée-libre "
        "par métamodélisation IA (substitution du solveur MSC Nastran SOL 101)."
    )

    col_in1, col_in2, col_in3 = st.columns([1.2, 1.2, 1])

    with col_in1:
        st.markdown("**Paramètres géométriques**")
        L_val = st.number_input(
            "Longueur L (mm)",
            min_value=500.0, max_value=2500.0, value=1500.0, step=10.0
        )
        L_slider = st.slider(
            "Ajustement rapide (L)",
            500.0, 2500.0, float(L_val), key="l_slide"
        )
        if L_slider != L_val:
            L_val = L_slider

    with col_in2:
        st.markdown("**Chargement appliqué**")
        F_val = st.number_input(
            "Force F (N)",
            min_value=1000.0, max_value=10000.0, value=5000.0, step=50.0
        )
        F_slider = st.slider(
            "Ajustement rapide (F)",
            1000.0, 10000.0, float(F_val), key="f_slide"
        )
        if F_slider != F_val:
            F_val = F_slider

    with col_in3:
        st.markdown("**Moteur prédictif**")
        chosen_model = st.selectbox(
            "Sélection du modèle",
            [
                "Réseau de neurones (MLP)",
                "Régression polynomiale (degré 3)",
                "Forêt aléatoire (Random Forest)",
            ]
        )

    # Inférence
    t_start = time.perf_counter_ns()

    if chosen_model == "Réseau de neurones (MLP)":
        x_scaled      = scaler_X.transform([[L_val, F_val]])
        y_pred_scaled = mlp_model.predict(x_scaled)
        y_pred        = scaler_y.inverse_transform(y_pred_scaled)[0]
        d_pred, s_pred = y_pred[0], y_pred[1]

    elif chosen_model == "Régression polynomiale (degré 3)":
        d_pred = poly_delta.predict([[L_val, F_val]])[0]
        s_pred = poly_sigma.predict([[L_val, F_val]])[0]

    else:
        y_pred = rf_model.predict([[L_val, F_val]])[0]
        d_pred, s_pred = y_pred[0], y_pred[1]

    t_inference_us = (time.perf_counter_ns() - t_start) / 1000.0
    t_nastran_est_s = 2.5
    speed_up = (t_nastran_est_s * 1e6) / max(t_inference_us, 1.0)

    st.markdown("---")
    st.markdown("**Résultats numériques de l'inférence**")

    res1, res2, res3, res4 = st.columns(4)
    res1.metric("Flèche maximale (Δ)", f"{d_pred:.2f} mm")
    res2.metric("Contrainte de Von Mises (Σ)", f"{s_pred:.2f} MPa")
    res3.metric("Temps d'inférence IA", f"{t_inference_us:.1f} µs")
    res4.metric("Facteur d'accélération", f"× {speed_up:,.0f}")

    # Visualisation matplotlib
    st.markdown("**Visualisation de la déformée structurelle**")

    x_beam = np.linspace(0, L_val, 100)
    y_beam = -d_pred * (3 * (x_beam / L_val)**2 - (x_beam / L_val)**3) / 2.0

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.color': '#e2e8f0',
    })

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafbfc')

    ax.plot(
        x_beam, np.zeros_like(x_beam),
        color='#94a3b8', linestyle='--', linewidth=1.0,
        label="Géométrie non déformée"
    )

    cmap = plt.cm.coolwarm
    norm_stress = min(max(s_pred / 500.0, 0.0), 1.0)
    beam_color  = cmap(norm_stress)

    ax.plot(
        x_beam, y_beam,
        color=beam_color, linewidth=3.5,
        label=f"Déformée IA (Δmax = {d_pred:.1f} mm)"
    )
    ax.plot(0, 0, marker='s', color='#0f172a', markersize=10, label="Encastrement")
    ax.annotate(
        f"F = {F_val:.0f} N",
        xy=(L_val, y_beam[-1]),
        xytext=(L_val, y_beam[-1] + 30),
        arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1.5, headwidth=7),
        ha='center', fontsize=9, color='#ef4444', fontweight='600'
    )

    ax.set_ylim(-350, 60)
    ax.set_xlim(-50, 2600)
    ax.set_xlabel("Position axiale (mm)", fontsize=9, color='#475569')
    ax.set_ylabel("Flèche (mm)", fontsize=9, color='#475569')
    ax.set_title(
        f"Profil de flexion — Contrainte Von Mises max : {s_pred:.1f} MPa",
        fontsize=10, color='#0f172a', fontweight='600', pad=10
    )
    ax.tick_params(labelsize=8, colors='#64748b')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================================================
# PAGE 2 — MÉTRIQUES & VALIDATION
# ==============================================================================

elif page == "Métriques & validation":

    st.title("Métriques d'évaluation et validation des modèles")
    st.markdown(
        "Évaluation rigoureuse des performances prédictives sur **60 cas de test indépendants** "
        "(20 % du jeu de données global de 296 points)."
    )

    # Chargement robuste du fichier CSV comparatif
    metrics_csv = find_file(
        os.path.join("ETAPE 7", "tableau_comparatif_modeles.csv"),
        "tableau_comparatif_modeles.csv",
        os.path.join("..", "ETAPE 7", "tableau_comparatif_modeles.csv"),
        os.path.join("Projet Python Poutre", "ETAPE 7", "tableau_comparatif_modeles.csv")
    )

    if metrics_csv and os.path.exists(metrics_csv):
        df_met = pd.read_csv(metrics_csv)
        st.markdown("**Tableau comparatif des métriques de précision**")
        st.dataframe(df_met, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Le fichier `tableau_comparatif_modeles.csv` est introuvable. "
            "Exécutez le script `validates_models.py` dans le dossier Étape 7 pour le régénérer."
        )

    st.markdown("---")
    st.markdown("**Galerie des graphiques d'évaluation métrologique**")

    p1 = find_file(os.path.join("ETAPE 7", "1_parity_plots.png"), "1_parity_plots.png", "ETAPE 7/1_parity_plots.png")
    p2 = find_file(os.path.join("ETAPE 7", "2_residual_plots.png"), "2_residual_plots.png", "ETAPE 7/2_residual_plots.png")
    p3 = find_file(os.path.join("ETAPE 7", "3_mlp_learning_curves.png"), "3_mlp_learning_curves.png", "ETAPE 7/3_mlp_learning_curves.png")
    p4 = find_file(os.path.join("ETAPE 7", "4_error_maps_2d.png"), "4_error_maps_2d.png", "ETAPE 7/4_error_maps_2d.png")

    g1, g2 = st.columns(2)

    with g1:
        if p1 and os.path.exists(p1):
            st.image(p1, caption="Figure 1 : Diagrammes de parité (Prédit vs. Nastran réel)")
        else:
            st.info("Image 1 (Parity plots) introuvable.")

        if p3 and os.path.exists(p3):
            st.image(p3, caption="Figure 3 : Courbes d'apprentissage du réseau MLP")
        else:
            st.info("Image 3 (Learning curves) introuvable.")

    with g2:
        if p2 and os.path.exists(p2):
            st.image(p2, caption="Figure 2 : Analyse des résidus de prédiction")
        else:
            st.info("Image 2 (Residual plots) introuvable.")

        if p4 and os.path.exists(p4):
            st.image(p4, caption="Figure 4 : Cartes d'erreur 2D dans l'espace paramétrique (L, F)")
        else:
            st.info("Image 4 (Error maps 2D) introuvable.")

# ==============================================================================
# PAGE 3 — ARCHITECTURE & PIPELINE
# ==============================================================================

elif page == "Architecture & pipeline":

    st.title("Architecture logicielle et pipeline de traitement")
    st.markdown(
        "Présentation synthétique du pipeline automatisé STHENOS, du plan d'expériences "
        "jusqu'à l'inférence temps réel."
    )

    steps = [
        ("1. Modélisation géométrique EF",
         "Maillage volumique HEXA20 de la poutre sous MSC Patran. Fichier de référence : `Hex20_12mm.bdf`."),
        ("2. Plan d'expériences (DoE)",
         "Latin Hypercube Sampling (LHS) 2D sur la longueur L ∈ [500, 2 500] mm et la force F ∈ [1 000, 16 000] N. 500 cas générés."),
        ("3. Simulation batch MSC Nastran",
         "Scripting Python d'altération dynamique des cartes BDF `GRID` et `FORCE`, suivi de l'exécution automatique en batch (SOL 101)."),
        ("4. Parsing Regex & Nettoyage",
         "Extraction par automate à états des déplacements T2 et contraintes de Von Mises dans les `.f06`. Nettoyage des divergences. Dataset final : 296 points."),
        ("5. Standardisation des données",
         "Normalisation Z-score des entrées (L, F) et des sorties (Δ, Σ) pour assurer la convergence de la descente de gradient."),
        ("6. Entraînement des métamodèles",
         "Modélisation comparative avec scikit-learn : Régression Polynomiale Ridge (Deg 3), Random Forest (150 arbres) et Multi-Layer Perceptron (64-64-32)."),
        ("7. Validation métrologique",
         "Évaluation aveugle sur 60 cas de test (20 % split) via R², MAE, RMSE et MAPE."),
        ("8. Déploiement opérationnel",
         "Sérialisation Joblib des artefacts et interface de démonstration temps réel sous Streamlit."),
    ]

    for title, desc in steps:
        st.markdown(f"**{title}**")
        st.markdown(f'<p style="color:#475569; font-size:0.9rem; margin-top:-6px; margin-bottom:12px;">{desc}</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Extrait du code d'inférence (Multi-Layer Perceptron)**")
    st.code(
        """# Prétraitement et inférence directe
x_scaled = scaler_X.transform([[longueur_mm, force_n]])
y_pred_scaled = mlp_model.predict(x_scaled)
delta_max, sigma_max = scaler_y.inverse_transform(y_pred_scaled)[0]""",
        language="python"
    )

# ==============================================================================
# PAGE 4 — RAPPORT TECHNIQUE & LECTEUR PDF
# ==============================================================================

elif page == "Rapport technique":

    st.title("Rapport technique & document de synthèse PDF")
    st.markdown(
        "Consultez ou téléchargez le rapport d'ingénierie complet du projet STHENOS au format PDF."
    )

    # Fonction d'affichage du lecteur PDF intégré
    def render_pdf_viewer(pdf_source):
        if isinstance(pdf_source, str):
            with open(pdf_source, "rb") as f:
                pdf_bytes = f.read()
        else:
            pdf_bytes = pdf_source

        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900" type="application/pdf" style="border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.markdown("")
        st.download_button(
            label="Télécharger le rapport PDF",
            data=pdf_bytes,
            file_name="rapport_sthenos.pdf",
            mime="application/pdf",
            type="primary"
        )

    # Recherche automatique d'un fichier PDF existant
    pdf_file_path = find_file(
        "rapport_sthenos.pdf",
        "../rapport_sthenos.pdf",
        "../../rapport_sthenos.pdf",
        "Projet_FEM___Modélisation (2).pdf",
        "../Projet_FEM___Modélisation (2).pdf"
    )

    uploaded_pdf = st.file_uploader("Charger un document PDF (optionnel)", type=["pdf"])

    if uploaded_pdf is not None:
        st.success("Document PDF chargé depuis l'interface.")
        render_pdf_viewer(uploaded_pdf.read())
    elif pdf_file_path and os.path.exists(pdf_file_path):
        st.info(f"Fichier PDF détecté : `{os.path.basename(pdf_file_path)}`")
        render_pdf_viewer(pdf_file_path)
    else:
        st.warning(
            "Aucun fichier PDF (`rapport_sthenos.pdf`) n'a été automatiquement détecté dans le répertoire racine du projet.\n\n"
            "Veuillez déposer votre fichier PDF compilé à l'aide du champ ci-dessus pour l'afficher directement dans le lecteur intégré."
        )
