import os
import re
import time
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==============================================================================
# CONFIGURATION GLOBALE
# ==============================================================================

# URL brute GitHub du rapport LaTeX (à adapter selon votre dépôt)
# Format : https://raw.githubusercontent.com/<user>/<repo>/<branch>/<fichier.tex>
GITHUB_RAW_TEX_URL = "https://github.com/Nyzer94/Sthenos/blob/main/rapport_sthenos.tex"

st.set_page_config(
    page_title="STHENOS — Surrogate Model · IPSA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection CSS pour un rendu sobre et professionnel
st.markdown("""
<style>
    /* Police et couleurs générales */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    /* Sidebar sobre */
    section[data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3347;
    }
    section[data-testid="stSidebar"] * {
        color: #c8cdd8 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem;
        letter-spacing: 0.02em;
    }
    /* En-tête de page */
    h1 {
        font-weight: 600;
        font-size: 1.6rem;
        letter-spacing: -0.01em;
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
    }
    h2, h3 {
        font-weight: 600;
        color: #1e293b;
    }
    /* Métriques */
    [data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    /* Séparateur */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    /* Bloc d'info sidebar */
    .sidebar-info {
        font-size: 0.78rem;
        color: #8892a4 !important;
        line-height: 1.6;
    }
    /* Titres de section pipeline */
    .step-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #3b82f6;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CHARGEMENT DES MODÈLES
# ==============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
artifacts_path = os.path.join(current_dir, "models_artifacts.joblib")

@st.cache_resource
def load_models():
    return joblib.load(artifacts_path)

try:
    artifacts = load_models()
except Exception as e:
    st.error(
        f"Impossible de charger les artefacts de modèles. "
        f"Vérifiez que `export_models.py` a été exécuté au préalable. "
        f"Détail : {e}"
    )
    st.stop()

scaler_X  = artifacts['scaler_X']
scaler_y  = artifacts['scaler_y']
mlp_model = artifacts['mlp']
poly_delta = artifacts['poly_delta']
poly_sigma = artifacts['poly_sigma']
rf_model  = artifacts['rf']

# ==============================================================================
# BARRE DE NAVIGATION (SIDEBAR)
# ==============================================================================

st.sidebar.markdown("## STHENOS")
st.sidebar.markdown(
    '<p class="sidebar-info">Métamodèle d\'intelligence artificielle<br>'
    'pour le calcul de structures<br><br>'
    'IPSA — Mécanique / Structures Aéronautiques<br>'
    'Promotion 2024–2025</p>',
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
    '<p class="sidebar-info">Domaine de validité :<br>'
    'L ∈ [500 – 2 500] mm<br>'
    'F ∈ [1 000 – 16 000] N</p>',
    unsafe_allow_html=True
)

# ==============================================================================
# PAGE 1 — DÉMONSTRATEUR TEMPS RÉEL
# ==============================================================================

if page == "Démonstrateur temps réel":

    st.title("Démonstrateur — Prédiction structurelle en temps réel")
    st.markdown(
        "Prédiction instantanée de la réponse d'une poutre encastrée-libre "
        "par substitution du solveur éléments finis MSC Nastran (SOL 101)."
    )

    col_in1, col_in2, col_in3 = st.columns([1.2, 1.2, 1])

    with col_in1:
        st.markdown("**Paramètres géométriques et chargement**")
        L_val = st.number_input(
            "Longueur L (mm)",
            min_value=500.0, max_value=2500.0, value=1500.0, step=10.0
        )
        L_slider = st.slider(
            "Ajustement fin — Longueur",
            500.0, 2500.0, float(L_val), key="l_slide"
        )
        if L_slider != L_val:
            L_val = L_slider

    with col_in2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        F_val = st.number_input(
            "Force appliquée F (N)",
            min_value=1000.0, max_value=10000.0, value=5000.0, step=50.0
        )
        F_slider = st.slider(
            "Ajustement fin — Force",
            1000.0, 10000.0, float(F_val), key="f_slide"
        )
        if F_slider != F_val:
            F_val = F_slider

    with col_in3:
        st.markdown("**Modèle prédictif**")
        chosen_model = st.selectbox(
            "Sélection du modèle",
            [
                "Réseau de neurones (MLP)",
                "Régression polynomiale (degré 3)",
                "Forêt aléatoire (Random Forest)",
            ]
        )

    # --- Inférence ---
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
    st.markdown("**Résultats de l'inférence**")

    res1, res2, res3, res4 = st.columns(4)
    res1.metric("Flèche maximale (Δ)", f"{d_pred:.2f} mm")
    res2.metric("Contrainte de Von Mises (Σ)", f"{s_pred:.2f} MPa")
    res3.metric("Temps de calcul IA", f"{t_inference_us:.1f} µs")
    res4.metric("Gain de vitesse vs. Nastran", f"× {speed_up:,.0f}")

    # --- Visualisation de la déformée ---
    st.markdown("**Déformée structurelle prédite**")

    x_beam = np.linspace(0, L_val, 100)
    y_beam = -d_pred * (3 * (x_beam / L_val)**2 - (x_beam / L_val)**3) / 2.0

    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
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
        label=f"Déformée prédite (Δmax = {d_pred:.1f} mm)"
    )
    ax.plot(0, 0, marker='s', color='#1e293b', markersize=10, label="Encastrement")
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
    ax.set_ylabel("Flèche transversale (mm)", fontsize=9, color='#475569')
    ax.set_title(
        f"Déformée structurelle — Contrainte de Von Mises : {s_pred:.1f} MPa",
        fontsize=10, color='#1e293b', fontweight='600', pad=10
    )
    ax.tick_params(labelsize=8, colors='#64748b')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.8, edgecolor='#e2e8f0')
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================================================
# PAGE 2 — MÉTRIQUES & VALIDATION
# ==============================================================================

elif page == "Métriques & validation":

    st.title("Métriques & validation des modèles")
    st.markdown(
        "Évaluation comparative des trois architectures sur les **60 cas de test indépendants** "
        "(partitionnement 80 % / 20 %, stratifié, graine fixée)."
    )

    metrics_csv = os.path.join(parent_dir, "ETAPE 7", "tableau_comparatif_modeles.csv")
    if os.path.exists(metrics_csv):
        df_met = pd.read_csv(metrics_csv)
        st.dataframe(
            df_met.style.highlight_max(
                subset=[c for c in df_met.columns if 'R²' in c],
                color='#dcfce7'
            ).format(precision=4),
            use_container_width=True
        )
    else:
        st.warning(
            "Tableau comparatif introuvable. "
            "Exécutez `validates_models.py` (Étape 7) pour générer ce fichier."
        )

    st.markdown("---")
    st.markdown("**Graphiques de validation**")

    img_dir = os.path.join(parent_dir, "ETAPE 7")
    g1, g2  = st.columns(2)

    with g1:
        p1 = os.path.join(img_dir, "1_parity_plots.png")
        if os.path.exists(p1):
            st.image(p1, caption="Diagrammes de parité — Prédit vs. Nastran (référence)")
        p3 = os.path.join(img_dir, "3_mlp_learning_curves.png")
        if os.path.exists(p3):
            st.image(p3, caption="Courbes d'apprentissage du réseau MLP")

    with g2:
        p2 = os.path.join(img_dir, "2_residual_plots.png")
        if os.path.exists(p2):
            st.image(p2, caption="Analyse des résidus — Biais systématique et dispersion")
        p4 = os.path.join(img_dir, "4_error_maps_2d.png")
        if os.path.exists(p4):
            st.image(p4, caption="Carte d'erreur relative dans l'espace paramétrique (L, F)")

# ==============================================================================
# PAGE 3 — ARCHITECTURE & PIPELINE
# ==============================================================================

elif page == "Architecture & pipeline":

    st.title("Architecture du projet et pipeline de données")

    st.markdown("""
Le pipeline **STHENOS** est organisé en huit étapes séquentielles, de la
génération des données numériques jusqu'au déploiement du démonstrateur.
""")

    steps = [
        ("Modélisation géométrique",
         "Maillage de la poutre de référence (L₀ = 2 000 mm, section 50 × 50 mm) "
         "en éléments volumiques HEXA20 sous MSC Patran. Fichier de sortie : `Hex20_12mm.bdf`."),
        ("Plan d'expériences (DoE)",
         "Latin Hypercube Sampling sur le domaine paramétrique L ∈ [500, 2 500] mm "
         "et F ∈ [1 000, 16 000] N. Génération de 500 configurations."),
        ("Automatisation des simulations",
         "Modification dynamique des cartes `GRID` (coordonnées nodales) et `FORCE` "
         "dans les fichiers `.bdf`, puis exécution de MSC Nastran (SOL 101) en mode batch."),
        ("Extraction et nettoyage des résultats",
         "Parsing des fichiers `.f06` par automate à états et expressions régulières. "
         "Extraction de Δmax et σVM. Filtrage des cas divergents. Jeu de données final : 296 points."),
        ("Normalisation et séparation",
         "Standardisation Z-score des variables d'entrée et de sortie. "
         "Partition entraînement / test : 80 % / 20 % (236 / 60 observations)."),
        ("Entraînement des modèles",
         "Régression polynomiale Ridge (degré 3), Forêt aléatoire (150 arbres, profondeur 12), "
         "Réseau MLP (64 – 64 – 32 neurones, activation ReLU, optimiseur Adam)."),
        ("Évaluation métrologique",
         "Comparaison des modèles sur le jeu de test via R², MAE, RMSE et MAPE "
         "pour les deux sorties (flèche et contrainte)."),
        ("Sérialisation et déploiement",
         "Export des artefacts (scalers + modèles) via Joblib. "
         "Interface web interactive déployée sous Streamlit."),
    ]

    for i, (title, desc) in enumerate(steps, start=1):
        col_num, col_content = st.columns([0.06, 0.94])
        with col_num:
            st.markdown(
                f'<div style="background:#1e40af;color:#fff;border-radius:50%;'
                f'width:32px;height:32px;display:flex;align-items:center;'
                f'justify-content:center;font-weight:700;font-size:0.85rem;">'
                f'{i}</div>',
                unsafe_allow_html=True
            )
        with col_content:
            st.markdown(f"**{title}**")
            st.markdown(f'<p style="color:#475569;font-size:0.88rem;margin-top:-8px;">{desc}</p>',
                        unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("**Extrait de code — Inférence MLP**")
    st.code(
        """\
# Standardisation des entrées
x_scaled = scaler_X.transform([[longueur_mm, force_n]])

# Prédiction dans l'espace normalisé
y_pred_scaled = mlp_model.predict(x_scaled)

# Retour aux unités physiques (mm, MPa)
delta_max, sigma_max = scaler_y.inverse_transform(y_pred_scaled)[0]
""",
        language="python"
    )

# ==============================================================================
# PAGE 4 — RAPPORT TECHNIQUE
# ==============================================================================

elif page == "Rapport technique":

    st.title("Rapport technique — Formulation et théorie")

    # --- Récupération du fichier .tex depuis GitHub ---
    @st.cache_data(ttl=600)  # Cache 10 min pour éviter les requêtes répétées
    def fetch_latex_from_github(url: str):
        """Télécharge le source LaTeX depuis GitHub Raw et retourne son contenu."""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.text, None
        except requests.exceptions.RequestException as exc:
            return None, str(exc)

    def extract_tex_sections(tex_content: str) -> list[tuple[str, str]]:
        """
        Extrait les sections et sous-sections d'un source LaTeX.
        Retourne une liste de (niveau, titre, contenu).
        """
        # On extrait le corps entre \begin{document} et \end{document}
        body_match = re.search(
            r'\\begin\{document\}(.*?)\\end\{document\}',
            tex_content, re.DOTALL
        )
        body = body_match.group(1) if body_match else tex_content

        # Découpages selon les \section et \subsection
        pattern = re.compile(
            r'\\(section|subsection|subsubsection)\{([^}]+)\}',
            re.DOTALL
        )
        splits = list(pattern.finditer(body))
        sections = []
        for i, match in enumerate(splits):
            level  = match.group(1)
            title  = match.group(2).strip()
            start  = match.end()
            end    = splits[i + 1].start() if i + 1 < len(splits) else len(body)
            content = body[start:end].strip()
            sections.append((level, title, content))
        return sections

    def clean_latex(text: str) -> str:
        """Nettoie le LaTeX pour affichage en Markdown / texte brut."""
        # Suppression des commandes de mise en forme les plus courantes
        removals = [
            r'\\textbf\{([^}]*)\}',   # gras -> texte
            r'\\textit\{([^}]*)\}',   # italique -> texte
            r'\\emph\{([^}]*)\}',
            r'\\texttt\{([^}]*)\}',
            r'\\label\{[^}]*\}',
            r'\\ref\{[^}]*\}',
            r'\\cite\{[^}]*\}',
            r'\\noindent\s*',
            r'\\medskip\s*',
            r'\\bigskip\s*',
            r'\\newline\s*',
            r'\\\\',
            r'\\item\s*',
        ]
        for pat in removals:
            text = re.sub(pat, lambda m: m.group(1) if m.lastindex else ' ', text)

        # Suppression des environnements (figure, table, lstlisting, enumerate, itemize…)
        envs_to_remove = ['figure', 'table', 'lstlisting', 'lstinputlisting',
                          'equation', 'align', 'equation*', 'align*']
        for env in envs_to_remove:
            text = re.sub(
                rf'\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}',
                '', text, flags=re.DOTALL
            )

        # Nettoyage résiduel
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?\s*', ' ', text)
        text = re.sub(r'\{|\}', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def extract_equations(section_content: str) -> list[str]:
        """Extrait les équations LaTeX d'un bloc de contenu."""
        equations = []
        # Environnements equation/align
        for pat in [
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
        ]:
            for m in re.finditer(pat, section_content, re.DOTALL):
                eq = m.group(1).strip()
                eq = re.sub(r'\\label\{[^}]*\}', '', eq).strip()
                if eq:
                    equations.append(eq)
        return equations

    # --- Interface principale de la page rapport ---

    tab_online, tab_local = st.tabs(["Rapport depuis GitHub", "Formules de référence"])

    with tab_online:
        st.markdown(
            "Le rapport est récupéré automatiquement depuis le dépôt GitHub. "
            "Configurez l'URL dans la variable `GITHUB_RAW_TEX_URL` du fichier `app.py`."
        )

        col_url, col_btn = st.columns([3, 1])
        with col_url:
            custom_url = st.text_input(
                "URL GitHub Raw du fichier .tex",
                value=GITHUB_RAW_TEX_URL,
                label_visibility="collapsed",
                placeholder="https://raw.githubusercontent.com/..."
            )
        with col_btn:
            fetch_btn = st.button("Charger le rapport", type="primary", use_container_width=True)

        if fetch_btn or custom_url != GITHUB_RAW_TEX_URL:
            tex_content, error = fetch_latex_from_github(custom_url)
        else:
            tex_content, error = fetch_latex_from_github(GITHUB_RAW_TEX_URL)

        if error:
            st.error(
                f"Impossible de récupérer le rapport depuis GitHub. "
                f"Vérifiez l'URL et votre connexion. Erreur : {error}"
            )
            st.info(
                "Conseil : assurez-vous d'utiliser l'URL **Raw** de GitHub "
                "(commençant par `https://raw.githubusercontent.com/`)."
            )
        elif tex_content:
            # Extraction de la date (si présente) et du titre
            title_match = re.search(r'\\title\{([^}]+)\}', tex_content)
            doc_title = title_match.group(1) if title_match else "Rapport STHENOS"
            doc_title = re.sub(r'\\[a-zA-Z]+\{?([^}]*)\}?', r'\1', doc_title).strip()

            st.success(f"Rapport chargé avec succès — {len(tex_content):,} caractères.")
            st.markdown(f"### {doc_title}")

            sections = extract_tex_sections(tex_content)

            if not sections:
                st.warning("Aucune section détectée dans le fichier LaTeX.")
                with st.expander("Source brut"):
                    st.text(tex_content[:5000])
            else:
                # Affichage section par section
                for level, title, content in sections:
                    if level == 'section':
                        st.markdown(f"## {title}")
                    elif level == 'subsection':
                        st.markdown(f"### {title}")
                    else:
                        st.markdown(f"#### {title}")

                    # Texte nettoyé
                    clean_text = clean_latex(content)
                    if clean_text:
                        st.markdown(
                            f'<p style="color:#334155;font-size:0.9rem;'
                            f'line-height:1.7;">{clean_text[:800]}</p>',
                            unsafe_allow_html=True
                        )

                    # Équations extraites
                    eqs = extract_equations(content)
                    for eq in eqs[:3]:  # Max 3 équations par section
                        try:
                            st.latex(eq)
                        except Exception:
                            pass

            # Téléchargement du source brut
            st.markdown("---")
            st.download_button(
                label="Télécharger le source LaTeX (.tex)",
                data=tex_content.encode('utf-8'),
                file_name="rapport_sthenos.tex",
                mime="text/plain",
            )

    with tab_local:
        st.markdown("**Mécanique de la poutre encastrée-libre — Rappels théoriques**")

        st.markdown("Flèche maximale à l'extrémité libre :")
        st.latex(r"\delta_{\max} = \frac{F \cdot L^3}{3\,E\,I_z}")

        st.markdown("Contrainte normale maximale (fibre extrême, section d'encastrement) :")
        st.latex(r"\sigma_{\max} = \frac{F \cdot L \cdot (h/2)}{I_z}")

        st.markdown("Contrainte équivalente de Von Mises :")
        st.latex(
            r"\sigma_{VM} = \sqrt{\frac{(\sigma_1 - \sigma_2)^2 "
            r"+ (\sigma_2 - \sigma_3)^2 + (\sigma_3 - \sigma_1)^2}{2}}"
        )

        st.markdown("Gain de vitesse numérique :")
        st.latex(
            r"\text{Speed-Up} = \frac{T_{\text{Nastran}}}{T_{\text{IA}}} "
            r"\approx \frac{2.5 \text{ s}}{50\;\mu\text{s}} \approx 50\,000"
        )

        st.markdown("---")
        st.markdown("**Algorithme Adam — Mise à jour des paramètres**")
        st.latex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,g_t")
        st.latex(r"v_t = \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2")
        st.latex(r"\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad "
                 r"\hat{v}_t = \frac{v_t}{1 - \beta_2^t}")
        st.latex(r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\,\hat{m}_t")

        st.markdown("---")
        st.markdown("**Régression Ridge (pénalité L₂)**")
        st.latex(
            r"\hat{\beta}_{\text{Ridge}} = \underset{\beta}{\arg\min}"
            r"\left[\|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2\right] "
            r"= (X^\top X + \alpha I)^{-1} X^\top y"
        )

        st.markdown("**Métriques d'évaluation**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("Coefficient de détermination :")
            st.latex(r"R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}")
            st.markdown("Erreur absolue moyenne (MAE) :")
            st.latex(r"\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|")
        with col_m2:
            st.markdown("Racine de l'erreur quadratique (RMSE) :")
            st.latex(r"\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}")
            st.markdown("Erreur relative moyenne (MAPE) :")
            st.latex(r"\text{MAPE} = \frac{100}{N}\sum_{i=1}^N\left|\frac{y_i - \hat{y}_i}{y_i}\right|")
