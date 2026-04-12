"""
task2_logistic_regression.py

Classification : prédire l'état glycémique postprandial

Modèle    : Régression Logistique multiclasse (baseline classification)

La régression logistique est la baseline naturelle pour la classification, comme la régression linéaire l'était pour la régression continue.
Elle suppose une séparation linéaire des classes dans l'espace des features, hypothèse forte mais qui fixe un plancher de performance interprétable.

Nous utilisons class_weight='balanced' pour corriger le biais du déséquilibre des classes à l'entraînement. (cas normaux plus fréquents que les hypoglycémies sévères)
Validation par GroupKFold k=5 strictement par patient (pas de data leakage)

Auteurs : Palliere Raphael & Bouny Mathieu — E4 Bio
"""



import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
 
from config import (
    DATASET_PATH, RESULTS_DIR, RANDOM_STATE, N_FOLDS,
    CLASSIFICATION_TARGET, CLASS_ORDER,
    load_dataset, build_X, get_preprocessing, get_cv_splits, save_results,
)
 
warnings.filterwarnings("ignore")
 
OUTPUT_DIR = os.path.join(RESULTS_DIR, "task2_logistic_regression")
MODEL_NAME = "LogisticRegression"


#Hyperparamètres de la régression logistique;
LR_PARAMS = {
    "C":           1.0, #Inverse de la force de régularisation (1.0 = standard)
    "max_iter": 1000, #Nombre maximal d'itérations pour la convergence
    "class_weight": "balanced", #Correction du déséquilibre des classes
    "solver": "lbfgs", #Algorithme d'optimisation efficace pour les petits datasets
    "random_state": RANDOM_STATE,
}

# Horizon d'évaluation : t+60 min (horizon principal pour la classification)
HORIZON_TARGETS = {
    "t30": "cgm_target_t30",
    "t60": "cgm_target_t60",
    "t90": "cgm_target_t90",
}



# COnstruction des labels par horizon 
def build_labels(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Reconstruit les labels de classification depuis la valeur continue.
    Utilise les seuils cliniques définis dans config.py.
    """
    from config import label_from_value
    return df[target_col].apply(label_from_value)
 


#Evaluation par FOld 
def evaluate_fold(pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
 
    # Filtrer les labels 'unknown' si présents
    mask = y_test != "unknown"
    y_test_clean = y_test[mask]
    y_pred_clean = y_pred[mask]
 
    classes_present = [c for c in CLASS_ORDER if c in y_test_clean.values]
 
    return {
        "accuracy": float(accuracy_score(y_test_clean, y_pred_clean)),
        "recall":   float(recall_score(y_test_clean, y_pred_clean, average="macro", zero_division=0, labels=classes_present)),
        "f1":       float(f1_score(y_test_clean, y_pred_clean, average="macro", zero_division=0, labels=classes_present)),
        "y_test":   y_test_clean.values,
        "y_pred":   y_pred_clean,
    }
 



#Vsiualisation de la matrice de confusion
def plot_confusion_matrix(y_test, y_pred, horizon: str, output_path: str):
    """Matrice de confusion normalisée par ligne (rappel par classe)."""
    labels = [c for c in CLASS_ORDER if c in np.unique(np.concatenate([y_test, y_pred]))]
 
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
 
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(f"Matrice de confusion normalisée\n{MODEL_NAME} — {horizon}", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Matrice de confusion : {output_path}")

def plot_metrics_by_class(report_df: pd.DataFrame, horizon: str, output_path: str):
    """Barplot des métriques (precision/recall/f1) par classe."""
    classes = [c for c in CLASS_ORDER if c in report_df.index]
    metrics = ["precision", "recall", "f1-score"]
 
    x = np.arange(len(classes))
    width = 0.25
    colors = ["#1D9E75", "#3B8BD4", "#EF9F27"]
 
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in report_df.columns:
            values = [report_df.loc[c, metric] if c in report_df.index else 0 for c in classes]
            ax.bar(x + i * width, values, width, label=metric, color=color, alpha=0.85)
 
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{MODEL_NAME} — Métriques par classe | {horizon}", fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
 


#Pipeli,e pri,ncipal 
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  TÂCHE 2 — Classification | Modèle : {MODEL_NAME}")
    df     = load_dataset(DATASET_PATH)
    X      = build_X(df)
    groups = df["patient_id"]
 
    print(f"\n  Dataset : {len(df)} fenêtres | {df['patient_id'].nunique()} patients")
    print(f"  Features : {X.shape[1]} colonnes (Config A)")
 
    all_results = []
 
    for horizon_key, target_col in HORIZON_TARGETS.items():
        # Construire les labels depuis la valeur continue à cet horizon
        y     = build_labels(df, target_col)
        valid = (y != "unknown") & df[target_col].notna()
        X_v   = X[valid].reset_index(drop=True)
        y_v   = y[valid].reset_index(drop=True)
        g_v   = groups[valid].reset_index(drop=True)
 
        print(f"\n  Horizon {horizon_key} — {valid.sum()} fenêtres valides")
        print(f"  Distribution labels : {y_v.value_counts().to_dict()}")
 
        pipeline = Pipeline(
            get_preprocessing() + [("model", LogisticRegression(**LR_PARAMS))]
        )
 
        splits = get_cv_splits(X_v, y_v, g_v)
        accs, recalls, f1s = [], [], []
        all_y_test, all_y_pred = [], []
 
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_result = evaluate_fold(
                pipeline,
                X_v.iloc[train_idx], y_v.iloc[train_idx],
                X_v.iloc[test_idx],  y_v.iloc[test_idx],
            )
            accs.append(fold_result["accuracy"])
            recalls.append(fold_result["recall"])
            f1s.append(fold_result["f1"])
            all_y_test.extend(fold_result["y_test"])
            all_y_pred.extend(fold_result["y_pred"])
 
            print(f"    Fold {fold_idx+1} | Acc={fold_result['accuracy']:.3f} | "
                  f"Recall={fold_result['recall']:.3f} | F1={fold_result['f1']:.3f}"
                )
 
        acc_m,    acc_s    = np.mean(accs),    np.std(accs)
        recall_m, recall_s = np.mean(recalls), np.std(recalls)
        f1_m,     f1_s     = np.mean(f1s),     np.std(f1s)
 
        print(f"\n  → Moyenne | Acc={acc_m:.3f}±{acc_s:.3f} | "
              f"Recall={recall_m:.3f}±{recall_s:.3f} | F1={f1_m:.3f}±{f1_s:.3f}")
 
        # Rapport détaillé agrégé sur tous les folds
        y_test_all = np.array(all_y_test)
        y_pred_all = np.array(all_y_pred)
        classes_all = [c for c in CLASS_ORDER if c in np.unique(y_test_all)]
 
        report = classification_report(
            y_test_all, y_pred_all,
            labels=classes_all,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df.to_csv(
            os.path.join(OUTPUT_DIR, f"classification_report_{horizon_key}.csv")
        )
 
        # Graphiques
        plot_confusion_matrix(
            y_test_all, y_pred_all,
            f"t+{horizon_key[1:]} min",
            output_path=os.path.join(OUTPUT_DIR, f"confusion_matrix_{horizon_key}.png"),
        )
        plot_metrics_by_class(
            report_df, f"t+{horizon_key[1:]} min",
            output_path=os.path.join(OUTPUT_DIR, f"metrics_by_class_{horizon_key}.png"),
        )
 
        all_results.append({
            "model":      MODEL_NAME,
            "task":       "classification",
            "horizon":    f"t+{horizon_key[1:]} min",
            "accuracy_mean":  round(acc_m, 3),
            "accuracy_std":   round(acc_s, 3),
            "recall_mean":    round(recall_m, 3),
            "recall_std":     round(recall_s, 3),
            "f1_mean":        round(f1_m, 3),
            "f1_std":         round(f1_s, 3),
            "n_features":     X_v.shape[1],
            "class_weight":   "balanced",
        })
 
    save_results(all_results, os.path.join(OUTPUT_DIR, "results_logistic_regression_classification.csv"))
    
 

if __name__ == "__main__":
    run()
