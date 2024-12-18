import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit.components.v1 as components
from xgboost import XGBRegressor
# Configuration de la page Streamlit
st.set_page_config(page_title="Prédiction des prix immobiliers", layout="wide")

# Fonction pour charger le modèle Ridge (mise en cache)
@st.cache_resource
def load_lgb_model():
    model_path = 'ressource/modele_final/lgb_model.pkl'
    return joblib.load(model_path)

lgb_model = load_lgb_model()

# Chargement des autres Ressourcesss
pipeline = joblib.load('ressource/pipeline/pipeline.pkl')
data_performances = {
    "Linear Regression": joblib.load('ressource/performance/GS_lr_perform.pkl'),
    "ElasticNet": joblib.load('ressource/performance/ElasticNet_perform.pkl'),
    "Random Forest Regressor": joblib.load('ressource/performance/rfr_perform.pkl'),
    "XGBoost": joblib.load('ressource/performance/xgb_perform.pkl'),
    "LightGBM": joblib.load('ressource/performance/lgb_perform.pkl')
}

# Fonction pour charger les données (mise en cache)
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Le fichier {file_path} est introuvable.")
        return pd.DataFrame()

# Chargement des données
train_df = load_data("data/data_apuree.csv")
train_df_labelled = load_data("data/data_apuree.csv")

# Initialisation de l'état de la page (si ce n'est pas déjà fait)
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

# Titre de l'application
st.title("🏡 **Application de Prédiction des Prix Immobiliers**")

# Fonction pour changer la page active dans st.session_state
def set_page(page_name):
    st.session_state.page = page_name

# Barre de navigation horizontale avec des boutons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Accueil"):
        set_page("Accueil")
with col2:
    if st.button("📊 Analyse"):
        set_page("Analyse")
with col3:
    if st.button("🔍 Prédiction"):
        set_page("Prédiction")
with col4:
    if st.button("📈 Performance"):
        set_page("Performance")

# Section Accueil
if st.session_state.page == "Accueil":
    st.write("---")
    st.header("Bienvenue 👋")
    st.write("""
        Cette application vous offre des outils intuitifs pour :
        - 🟡 Prédire les **prix des maisons** à partir de caractéristiques clés.
        - 📊 Analyser les **tendances des prix immobiliers**.
        - 🛠️ Évaluer les **performances des modèles** utilisés.
    """)

    st.header("🗂 Description des Données")
    file_path = "Ressources/data_description.txt"


    try:
        with open(file_path, "r") as file:
            description = file.read()
    except FileNotFoundError:
        st.error(f"Le fichier '{file_path}' est introuvable.")
        st.stop()

    st.text_area("Aperçu de la description des données :", description, height=300)
    st.download_button("Télécharger la description des données", data=description, file_name="description.txt")

    st.info("Utilisez la barre de navigation pour explorer les différentes fonctionnalités.")

# Section Analyse
elif st.session_state.page == "Analyse":
    st.subheader("📊 Analyse des Données")
    if st.checkbox("Afficher les données brutes"):
        st.dataframe(train_df_labelled)

    st.write("### Statistiques descriptives")
    st.write(train_df_labelled.describe())

    st.write("### Visualisation de deux variables")
    variable_x = st.selectbox("Variable X", train_df_labelled.columns)
    variable_y = st.selectbox("Variable Y", train_df_labelled.columns)

    # Visualisation des relations entre les variables
    fig, ax = plt.subplots(figsize=(10, 8))
    if train_df_labelled[variable_x].dtype in ['int64', 'float64'] and train_df_labelled[variable_y].dtype in ['int64', 'float64']:
        sns.scatterplot(data=train_df_labelled, x=variable_x, y=variable_y, ax=ax, color="teal", s=100, edgecolor='black')
        ax.set_title(f"Nuage de points entre {variable_x} et {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12,rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
    elif train_df_labelled[variable_x].dtype == 'object' and train_df_labelled[variable_y].dtype == 'object':
        grouped_train_df_labelled = train_df_labelled.groupby([variable_x, variable_y]).size().unstack()
        grouped_train_df_labelled.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm')
        ax.set_title(f"Graphique en barres empilées de {variable_x} par {variable_y}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel("Effectifs", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12,rotation=45)
        ax.legend(title=variable_y, fontsize=12)
    else:
        sns.boxplot(data=train_df_labelled, x=variable_x, y=variable_y, ax=ax, palette="Set2")
        ax.set_title(f"Graphique de boîte de {variable_y} par {variable_x}", fontsize=16, fontweight='bold')
        ax.set_xlabel(variable_x, fontsize=14)
        ax.set_ylabel(variable_y, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12,rotation=45)

    st.pyplot(fig)
    st.write("---")

    st.write("### Matrice de Corrélation")
    correlation_matrix = train_df_labelled.select_dtypes(include=['int64', 'float64']).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", mask=mask, fmt=".2f")
    st.pyplot(fig_corr)
    st.write("---")

        # Titre de la page
    st.subheader("📄 Rapport en HTML")

    # Chemin du fichier HTML
    html_file_path = "Ressources/data_report.html"

    try:
        # Lire le contenu du fichier HTML
        with open(html_file_path, "r", encoding="utf-8") as html_file:
            html_content = html_file.read()

        # Afficher le contenu HTML dans Streamlit
        components.html(html_content, height=1200, scrolling=True)

    except FileNotFoundError:
        st.error(f"Le fichier '{html_file_path}' est introuvable.")

# Section Prédiction
elif st.session_state.page == "Prédiction":
    st.subheader("🔍 Prédiction des Prix")
    form_data = {}
    for col_label in train_df_labelled.columns:
        if train_df_labelled[col_label].dtype == 'object':
            form_data[col_label] = st.selectbox(f"{col_label}", train_df_labelled[col_label].unique())
        else:
            form_data[col_label] = st.number_input(f"{col_label}", float(train_df_labelled[col_label].min()), float(train_df_labelled[col_label].max()))

    input_data = pd.DataFrame([form_data])
    if st.checkbox("Afficher les données saisies:"):
        st.dataframe(input_data)
        st.write("---")
    if st.button("Prédire"):
        st.write("---")
        try:
            transformed_data = pipeline.transform(input_data)
            predicted_price = np.expm1(ridge_model.predict(transformed_data))
            st.success(f"Prix prédit : {predicted_price[0]:,.2f} unités monétaires")
        except Exception as e:
            st.error(f"Erreur : {e}")

# Section Performance
elif st.session_state.page == "Performance":
    st.subheader("📈 Performance des Modèles")
    cols = st.columns(4)
    for i, (model_name, performance_df) in enumerate(data_performances.items()):
        col = cols[i % 4]
        with col:
            st.write(f"### {model_name}")
            st.dataframe(performance_df)
            fig, ax = plt.subplots()
            ax.bar(performance_df["metric"], performance_df["train"], label="Train", alpha=0.7)
            ax.bar(performance_df["metric"], performance_df["test"], label="Test", alpha=0.7)
            ax.legend()
            ax.set_title(f"Performance : {model_name}")
            ax.tick_params(axis='both', which='major', labelsize=12,rotation=45)
            st.pyplot(fig)
            st.write("---")
