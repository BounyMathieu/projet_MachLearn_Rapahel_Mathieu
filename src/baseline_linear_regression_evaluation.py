"""
baseline_linear_regression.py
============================
Baseline de régression linéaire pour le projet CGMacros.
Ce script entraîne et évalue un modèle de régression linéaire sur le dataset
meal_windows_dataset.csv produit par meal_window_builder.py.

Deux configurations sont comparées comme suggéré par Mr Redjdal :
- Config A : features agrégées uniquement (7 statistiques CGM pré-repas)
- Config B : séquence CGM brute complète (60 colonnes cgm_t-X)

Donc ici, la tâche première est de prédire cgm_target_t60 et ensuite, cgm_target_t30 et cgm_target_t90. 

La validation se fait par validation croisée à 5 folds par patient (donc pas de data leakage). 

Auteurs : Palliere Raphael   
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


warnings.filterwarnings("ignore")


# Configuration
CONFIG = {
    "dataset_path": "data/processed/meal_windows_dataset.csv",
    "output_dir":    "data/results/baseline_linear",
    "random_state":  42,
    "n_folds":       5,

    #Prédiction de cgm_target_t60, cgm_target_t30 et cgm_target_t90
    "targets": {
        "t60": "cgm_target_t60",#cible secondaire
        "t30": "cgm_target_t30",#cible secondaire
        "t90": "cgm_target_t90", #cible principale
    },

    # Seuil clinique de référence (nous utilisons la norme ISO 15197de 15 mg/dL pour la RMSE, soit environ 0.83 mmol/L) pour évaluer la pertinence clinique des prédictions: 
    "clinical_rmse_threshold": 15.0,
}


#Définiton des features sur les 2 configurations

# Config A : features agrégées uniquement (7 statistiques CGM pré-repas)
FEATURES_AGG = [
    "cgm_at_meal",
    "cgm_pre_mean",
    "cgm_pre_std",
    "cgm_pre_min",
    "cgm_pre_max",
    "cgm_slope_15",
    "cgm_slope_30",

    "carbs",
    "protein",
    "fat",
    "fiber",

    "hour_sin",
    "hour_cos",

    "bio_A1c PDL (Lab)",
    "bio_Fasting GLU - PDL (Lab)",
    "bio_Insulin",
    "bio_BMI",
    "bio_Age",
    "bio_group_encoded",
    "bio_gender_encoded",
]

# Variables catégorielles 
CATEGORICAL_FEATURES = ["meal_type"]


def build_feature_matrix(df: pd.DataFrame, use_sequence: bool = False) -> pd.DataFrame:
    """
    Construit la matrice de features X à partir du dataset : 
    avec comme paramètres : 
    df : DataFramede meal_windows_dataset.csv
    use_sequence : False = Config A (agrégats), True = Config B (séquence brute)
 
    et retourne :
    X : le DataFrame de features prêt pour l'entraînement
    """
    # One-hot encoding du type de repas
    meal_dummies = pd.get_dummies(df["meal_type"], prefix="meal", drop_first=False)
 
    #Features numériques de base
    available_agg = [c for c in FEATURES_AGG if c in df.columns]
    X = df[available_agg].copy()
 
    # Ajouter le one-hot meal_type
    X = pd.concat([X, meal_dummies], axis=1)
 
    if use_sequence:
        #Config B : remplacer les agrégats CGM par la séquence brute
        seq_cols = sorted(
            [c for c in df.columns if c.startswith("cgm_t-")],
            key=lambda x: int(x.split("-")[1]),
            reverse=True,   # cgm_t-60 en premier (le plus ancien)
        )
        # Retirer les agrégats CGM et ajouter la séquence brute
        cgm_agg_cols = ["cgm_at_meal", "cgm_pre_mean", "cgm_pre_std",
                        "cgm_pre_min", "cgm_pre_max", "cgm_slope_15", "cgm_slope_30"]
        X = X.drop(columns=[c for c in cgm_agg_cols if c in X.columns])
        X = pd.concat([X, df[seq_cols]], axis=1)
 
    return X
 


# Evaluation par validation croisée à 5 folds par patient
def evaluate_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_name: str, n_folds: int = 5,) -> dict:
    """
    Évalue le modèle par validation croisée à 5 folds par patient.
     Split strictement par patient pour éviter le data leakage.

    Retourne un dictionnaire de métriques (RMSE, R2, MAE) avec moyennes et écarts-types.
    """
    # Pipeline : imputation → normalisation → régression Ridge
    # Ridge plutôt que LinearRegression pure car la régularisation L2 stabilise les coefficients quand les features sont nombreuses (Config B)
    pipeline = Pipeline([
        ("imputer",  SimpleImputer(strategy="median")),
        ("scaler",   StandardScaler()),
        ("model",    Ridge(alpha=1.0, random_state=CONFIG["random_state"])),
    ])
 
    gkf = GroupKFold(n_splits=n_folds)
 
    rmse_scores, mae_scores, r2_scores = [], [], []
    fold_details = []
 
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
        # Entraînement
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
 
        # Métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
 
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
 
        # Patients en test dans ce fold
        test_patients = groups.iloc[test_idx].unique().tolist()
        fold_details.append({
            "fold":     fold_idx + 1,
            "n_train":  len(train_idx),
            "n_test":   len(test_idx),
            "patients_test": test_patients,
            "rmse":     round(rmse, 2),
            "mae":      round(mae, 2),
            "r2":       round(r2, 3),
        })
 
    results = {
        "model_name":    model_name,
        "n_features":    X.shape[1],
        "rmse_mean":     round(np.mean(rmse_scores), 2),
        "rmse_std":      round(np.std(rmse_scores), 2),
        "mae_mean":      round(np.mean(mae_scores), 2),
        "mae_std":       round(np.std(mae_scores), 2),
        "r2_mean":       round(np.mean(r2_scores), 3),
        "r2_std":        round(np.std(r2_scores), 3),
        "fold_details":  fold_details,
        "pipeline":      pipeline,   # Pipeline entraîné sur le dernier fold (pour analyse)
        "X_columns":     X.columns.tolist(),
    }
 
    return results




#Analyse des coefficients du modèle linéaire entraîné sur la Config B (séquence brute) pour identifier les time points les plus prédictifs
def get_feature_importance(pipeline: Pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extrait les coefficients du modèle Ridge et les retourne triés par importance absolue.
    Les coefficients sont ceux après normalisation — comparables entre eux.
    """
    coefs = pipeline.named_steps["model"].coef_
    importance = pd.DataFrame({
        "feature":     feature_names,
        "coefficient": coefs,
        "abs_coef":    np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
 
    return importance


#Visualisation : 
def plot_predictions_vs_actual(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    title: str,
    output_path: str,
):
    """
    Scatter plot prédit vs réel sur le dernier fold de test.
    Inclut la droite parfaite y=x et les seuils cliniques.
    """
    gkf = GroupKFold(n_splits=CONFIG["n_folds"])
    splits = list(gkf.split(X, y, groups=groups))
    _, test_idx = splits[-1]   # Dernier fold pour la visualisation
 
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
 
    #Réentraîner le pipeline proprement sur le fold d'entraînement
    train_idx = splits[-1][0]
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipeline.predict(X_test)
 
    fig, ax = plt.subplots(figsize=(7, 6))
 
    ax.scatter(y_test, y_pred, alpha=0.6, color="#1D9E75", s=40, label="Fenêtres repas")
 
    #Droite de prédiction parfaite 
    lims = [min(y_test.min(), y_pred.min()) - 5, max(y_test.max(), y_pred.max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Prédiction parfaite")
 
    # Bandes d'erreur clinique (±15 mg/dL) selon la norme ISO 15197
    ax.fill_between(lims,
                    [l - 15 for l in lims],
                    [l + 15 for l in lims],
                    alpha=0.1, color="#1D9E75", label="±15 mg/dL (seuil ISO 15197)")
 
    #Seuils cliniques horizontaux
    for seuil, label, color in [
        (70,  "Hypo < 70",    "#E24B4A"),
        (140, "Normal < 140", "#EF9F27"),
        (180, "Hyper > 180",  "#E24B4A"),
    ]:
        ax.axhline(seuil, color=color, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(lims[0] + 1, seuil + 1, label, fontsize=8, color=color)
 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    ax.set_xlabel("Glycémie réelle à t+60 min (mg/dL)", fontsize=11)
    ax.set_ylabel("Glycémie prédite (mg/dL)", fontsize=11)
    ax.set_title(f"{title}\nRMSE = {rmse:.1f} mg/dL | MAE = {mae:.1f} mg/dL", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
 
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique sauvegardé : {output_path}")
 
 
def plot_feature_importance(importance_df: pd.DataFrame, title: str, output_path: str, top_n: int = 20):
    """
    Barplot horizontal des coefficients Ridge (top N features).
    """
    top = importance_df.head(top_n).copy()
 
    colors = ["#1D9E75" if c >= 0 else "#E24B4A" for c in top["coefficient"]]
 
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    bars = ax.barh(top["feature"][::-1], top["coefficient"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (après normalisation)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique sauvegardé : {output_path}")
def plot_rmse_comparison(results_a: dict, results_b: dict, output_path: str):
    """
    Barplot comparant RMSE Config A vs Config B sur les 3 horizons.
    """
    horizons = ["t+30", "t+60", "t+90"]
    rmse_a = [results_a[h]["rmse_mean"] for h in ["t30", "t60", "t90"]]
    rmse_b = [results_b[h]["rmse_mean"] for h in ["t30", "t60", "t90"]]
    std_a  = [results_a[h]["rmse_std"]  for h in ["t30", "t60", "t90"]]
    std_b  = [results_b[h]["rmse_std"]  for h in ["t30", "t60", "t90"]]
 
    x = np.arange(len(horizons))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, rmse_a, width, yerr=std_a, label="Config A — agrégats",
           color="#1D9E75", capsize=4, alpha=0.85)
    ax.bar(x + width/2, rmse_b, width, yerr=std_b, label="Config B — séquence brute",
           color="#3B8BD4", capsize=4, alpha=0.85)
 
    ax.axhline(CONFIG["clinical_rmse_threshold"], color="#E24B4A",
               linestyle="--", linewidth=1, label=f"Seuil clinique {CONFIG['clinical_rmse_threshold']} mg/dL")
 
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("RMSE (mg/dL)", fontsize=10)
    ax.set_title("Comparaison Config A vs Config B\nRégression linéaire (Ridge) — GroupKFold k=5", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(rmse_a), max(rmse_b)) * 1.4)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Graphique sauvegardé : {output_path}")



#Pipeline Principal d'exécution

def run_baseline(dataset_path: str, output_dir: str):
    """
    Lance la baseline complète : chargement, évaluation Config A et B,
    génération des graphiques et sauvegarde des résultats.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    print("=" * 60)
    print("  BASELINE — Régression Linéaire (Ridge)")
    print("  Projet CGMacros — Palliere / Bouny")
    print("=" * 60)
 
    # --- Chargement ---
    print(f"\n[1/4] Chargement du dataset : {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path}\n"
            f"Lancer d'abord meal_window_builder.py"
        )
 
    df = pd.read_csv(dataset_path)
    print(f"  → {len(df)} fenêtres repas | {df['patient_id'].nunique()} patients")
    print(f"  → Colonnes disponibles : {len(df.columns)}")
 
    # Vérifier que les cibles sont présentes
    for tgt in CONFIG["targets"].values():
        if tgt not in df.columns:
            raise ValueError(f"Colonne cible manquante : {tgt}")
 
    groups = df["patient_id"]
 
    # --- Construction des matrices de features ---
    print("\n[2/4] Construction des matrices de features...")
    X_agg = build_feature_matrix(df, use_sequence=False)
    X_seq = build_feature_matrix(df, use_sequence=True)
    print(f"  Config A (agrégats)      : {X_agg.shape[1]} features")
    print(f"  Config B (séquence brute): {X_seq.shape[1]} features")
 
    #Évaluation par validation croisée à 5 folds par patient
    print("\n[3/4] Évaluation par GroupKFold (k=5, split par patient)...")
 
    results_a, results_b = {}, {}
 
    for horizon_key, target_col in CONFIG["targets"].items():
        y = df[target_col].copy()
 
        # Supprimer les lignes où la cible est manquante
        valid_mask = y.notna()
        y_clean      = y[valid_mask]
        X_agg_clean  = X_agg[valid_mask]
        X_seq_clean  = X_seq[valid_mask]
        groups_clean = groups[valid_mask]
 
        print(f"\n  Horizon {horizon_key} ({target_col}) — {valid_mask.sum()} fenêtres valides")
 
        #Config A
        res_a = evaluate_model(
            X_agg_clean, y_clean, groups_clean,
            model_name=f"Ridge_ConfigA_{horizon_key}",
        )
        results_a[horizon_key] = res_a
        print(f"    Config A → RMSE {res_a['rmse_mean']} ± {res_a['rmse_std']} mg/dL | "
              f"MAE {res_a['mae_mean']} ± {res_a['mae_std']} mg/dL | "
              f"R² {res_a['r2_mean']} ± {res_a['r2_std']}")
 
        #config B
        res_b = evaluate_model(
            X_seq_clean, y_clean, groups_clean,
            model_name=f"Ridge_ConfigB_{horizon_key}",
        )
        results_b[horizon_key] = res_b
        print(f"    Config B → RMSE {res_b['rmse_mean']} ± {res_b['rmse_std']} mg/dL | "
              f"MAE {res_b['mae_mean']} ± {res_b['mae_std']} mg/dL | "
              f"R² {res_b['r2_mean']} ± {res_b['r2_std']}")
        




#Plots
    print("\n[4/4] Génération des graphiques...")
 
    #Graphique principal : RMSE Config A vs B sur les 3 horizons
    plot_rmse_comparison(
        results_a, results_b,
        output_path=os.path.join(output_dir, "rmse_comparison_A_vs_B.png"),
    )
 
    #Scatter prédit vs réel — Config A, horizon t+60
    y_t60 = df[CONFIG["targets"]["t60"]]
    valid_t60 = y_t60.notna()
    plot_predictions_vs_actual(
        pipeline=results_a["t60"]["pipeline"],
        X=X_agg[valid_t60],
        y=y_t60[valid_t60],
        groups=groups[valid_t60],
        title="Config A (agrégats) — Glycémie prédite vs réelle à t+60 min",
        output_path=os.path.join(output_dir, "scatter_configA_t60.png"),
    )
 
    #Importance des coefficients — Config A, horizon t+60
    importance_a = get_feature_importance(
        results_a["t60"]["pipeline"],
        results_a["t60"]["X_columns"],
    )
    importance_a.to_csv(os.path.join(output_dir, "coefficients_configA_t60.csv"), index=False)
    plot_feature_importance(
        importance_a,
        title="Coefficients Ridge — Config A, horizon t+60 min",
        output_path=os.path.join(output_dir, "coefficients_configA_t60.png"),
    )
 
    #Tableau de résultats
    rows = []
    for h in ["t30", "t60", "t90"]:
        for config_name, results in [("A - agrégats", results_a), ("B - séquence", results_b)]:
            r = results[h]
            rows.append({
                "Config":        config_name,
                "Horizon":       f"t+{h[1:]} min",
                "N features":    r["n_features"],
                "RMSE (mg/dL)":  f"{r['rmse_mean']} ± {r['rmse_std']}",
                "MAE (mg/dL)":   f"{r['mae_mean']} ± {r['mae_std']}",
                "R²":            f"{r['r2_mean']} ± {r['r2_std']}",
            })
 
    results_df = pd.DataFrame(rows)
    results_path = os.path.join(output_dir, "baseline_results.csv")
    results_df.to_csv(results_path, index=False)
 
    #Résumé final
    print("\n" + "=" * 60)
    print("  RÉSULTATS BASELINE")
    print("=" * 60)
    print(results_df.to_string(index=False))
 
    #Interprétation clinique automatique
    rmse_main = results_a["t60"]["rmse_mean"]
    threshold = CONFIG["clinical_rmse_threshold"]
    print(f"\n  Horizon principal t+60 min (Config A) :")
    print(f"  RMSE = {rmse_main} mg/dL (seuil clinique ISO 15197 = {threshold} mg/dL)")
    if rmse_main <= threshold:
        print(f"  ✅ Performance dans le seuil clinique acceptable")
    else:
        print(f"  ⚠️  Performance au-dessus du seuil clinique — à améliorer avec les modèles suivants")
 
    print(f"\n  Fichiers sauvegardés dans : {output_dir}/")
    return results_a, results_b, importance_a
 
 


# POINT D'ENTRÉE
 
if __name__ == "__main__":
    results_a, results_b, importance = run_baseline(
        dataset_path=CONFIG["dataset_path"],
        output_dir=CONFIG["output_dir"],
    )        
