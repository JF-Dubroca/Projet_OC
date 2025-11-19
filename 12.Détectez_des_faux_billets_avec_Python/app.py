
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Détection de faux billets", layout="wide")
st.title("💶 Détection automatique de faux billets")

# --- Chargement du pipeline ---
pipeline = joblib.load('models/pipeline.joblib')
reg_robustscaled = pipeline['nan_predict']
robust_scaler = pipeline['nan_scaler']
clf_standard = pipeline['model']
log_standard_scaler = pipeline['log_scaler']
best_threshold = pipeline['best_threshold']
features = pipeline.get('features', ['diagonal','length','height_left','height_right','margin_low','margin_up'])
impute_features = pipeline.get('impute_features', ['length','margin_up','height_right','height_left'])

# --- Upload CSV ---
uploaded_file = st.file_uploader("Importez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Détection automatique du séparateur
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # --- Vérification des colonnes ---
    missing_cols = set(features + impute_features) - set(df.columns)
    if missing_cols:
        st.error(f"❌ Colonnes manquantes dans le fichier : {missing_cols}")
    else:
        # --- Imputation pour margin_low ---
        if df['margin_low'].isna().any():
            df_nan = df.loc[df['margin_low'].isna()]
            predict_var = df_nan[['length', 'margin_up', 'height_right', 'height_left']]

            # Remplir temporairement les NaN restantes avec 0 pour éviter ValueError
            predict_var_filled = predict_var.fillna(0)

            predict_var_scaled = pd.DataFrame(
                robust_scaler.transform(predict_var_filled),
                columns=predict_var.columns,
                index=predict_var.index
            )
            predict_nan = reg_robustscaled.predict(predict_var_scaled)
            df.loc[df['margin_low'].isna(), 'margin_low'] = predict_nan

        # --- Vérification d'autres valeurs manquantes ---
        nan_cols = df[clf_standard.feature_names_in_].columns[df[clf_standard.feature_names_in_].isna().any()].tolist()
        if nan_cols:
            st.warning(f"⚠️ Attention, présence de valeurs manquantes dans : {nan_cols}")
        else:
            st.success("✅ Aucune valeur manquante détectée")

        # --- Suppression des doublons ---
        df = df.drop_duplicates()

        # --- Préparation des données pour la prédiction ---
        X = df[clf_standard.feature_names_in_].copy()

        # Identifier les lignes complètes pour la prédiction
        complete_rows = X.dropna().index
        X_complete = X.loc[complete_rows]

        # Scaler et prédiction uniquement sur lignes complètes
        X_scaled = pd.DataFrame(
            log_standard_scaler.transform(X_complete),
            columns=clf_standard.feature_names_in_,
            index=X_complete.index
        )
        pred = clf_standard.predict(X_scaled)
        proba = clf_standard.predict_proba(X_scaled)[:, 1]
        pred_opt = (proba >= best_threshold).astype(int)

        # Ajouter les résultats dans df, NaN pour les lignes incomplètes
        df['prediction'] = pd.NA
        df['probabilities'] = pd.NA
        df.loc[complete_rows, 'prediction'] = [True if p == 1 else False for p in pred_opt]
        df.loc[complete_rows, 'probabilities'] = proba

        # --- Comptage résultats ---
        n_true = df['prediction'].sum(skipna=True)
        n_false = df['prediction'].count() - n_true
        n_true_pct = (n_true / df['prediction'].count() * 100).round(1)
        n_false_pct = (n_false / df['prediction'].count() * 100).round(1)
        # Liste des vrais et des faux billets
        true_bill = df.loc[df['prediction'] == True, 'id'].tolist()
        false_bill = df.loc[df['prediction'] == False, 'id'].tolist()

        # --- Billets incertains (±10% du seuil) ---
        threshold_pct = 10
        low_threshold = best_threshold * (1 - threshold_pct / 100)
        high_threshold = best_threshold * (1 + threshold_pct / 100)
        uncertain = df.loc[
            df['probabilities'].between(low_threshold, high_threshold),
            'id'
        ].dropna().tolist()
        n_uncertain = len(uncertain)
        pct_uncertain = round(n_uncertain / df['prediction'].count() * 100, 1)

        # --- Affichage des résultats ---
        st.subheader("📊 Résultats globaux")
        st.write(f"✅ Billets vrais : {n_true} ({n_true_pct}%)")
        st.write(f"{true_bill}")
        st.write(f"❌ Billets faux : {n_false} ({n_false_pct}%)")
        st.write(f"{false_bill}")
        st.write(f"👀 Billets à vérifier : {n_uncertain} ({pct_uncertain}%)")
        if n_uncertain > 0:
            st.dataframe(pd.DataFrame({'id_suspects': uncertain}))

            # --- Optionnel : télécharger les billets suspects ---
            csv = pd.DataFrame({'id_suspects': uncertain}).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les billets suspects",
                data=csv,
                file_name='billets_suspects.csv',
                mime='text/csv'
            )
