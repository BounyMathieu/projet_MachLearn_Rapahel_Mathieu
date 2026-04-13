# Ordre d'éxécution des codes src : 

## Étape 1 — Préparation des données patients
### python src/build_patient_table.py
Ce script lit data/raw/cgmacros/bio.csv et ajoute une colonne group classifiant chaque patient selon son taux d'HbA1c :

healthy si HbA1c < 5.7 %
prediabetes si 5.7 % ≤ HbA1c ≤ 6.4 %
t2d si HbA1c > 6.4 %

### Produit : data/processed/bio_with_group.csv


## Étape 2 — Construction du dataset de fenêtres repas
### python src/meal_window_builder.py
C'est le cœur du pipeline de feature engineering. Pour chaque repas horodaté de chaque patient, ce script extrait :

la séquence CGM pré-repas : 60 minutes minute par minute avant le repas (signal Abbott FreeStyle Libre)
les agrégats statistiques de cette séquence (moyenne, écart-type, pentes, etc.)
les macronutriments du repas (glucides, protéines, lipides, fibres, calories)
l'encodage temporel cyclique de l'heure du repas (sin/cos sur 24h)
les biomarqueurs cliniques du patient répétés à chaque fenêtre
les valeurs cibles : glycémie à t+30, t+60 et t+90 min après le repas

On passe ainsi de 45 patients à ~1 300–1 500 fenêtres repas, chaque ligne représentant un événement alimentaire complet.

Note importante : le patient_id est conservé comme clé de groupement pour le split train/test. Les fenêtres d'un même patient ne sont jamais mélangées entre train et test.

### Produit : data/processed/meal_windows_dataset.csv
Pour générer le dictionnaire des colonnes :
### bashpython src/column_description_meal_window_builder.py

## Étape 3 — Modélisation : Tâche 1 — Régression continue
L'objectif est de prédire la valeur glycémique exacte en mg/dL à t+30, t+60 et t+90 minutes.
Métriques d'évaluation : RMSE (mg/dL), MAE (mg/dL), R²
Référence clinique : RMSE < 15 mg/dL (norme ISO 15197 pour les systèmes CGM)
Validation : GroupKFold k=5, split strictement par patient

Modèles linéaires : OLS (baseline), Ridge, Lasso, pipeline Lasso→Random Forest
# src/task1_linear_models.py

Baseline Ridge seule (pour compatibilité avec le fichier de comparaison)
# python src/baseline_linear_regression.py

Arbre de décision
# python src/task1_decision_tree.py

Random Forest
# python src/task1_random_forest.py

Comparaison de tous les modèles (graphiques + tableau synthèse)
# python src/compare_task1_regression.py


## Étape 4 — Modélisation : Tâche 2 — Classification glycémique

L'objectif est de prédire l'état glycémique postprandial en 4 classes :
Classe   Seuil (mg/dL)   Signification clinique
hypo       < 70          Hypoglycémie
normal     70 – 140      Normoglycémie postprandiale
hyper_mild   140 – 180   Hyperglycémie légère
hyper_severe  > 180      Hyperglycémie sévère

Métriques d'évaluation : Accuracy, Recall macro, F1-score macro
Note clinique : le Recall est la métrique prioritaire — un faux négatif sur hypo (hypoglycémie manquée) est cliniquement plus grave qu'un faux positif.

Régression logistique (baseline classification)
# python src/task2_logistic_regression.py

Arbre de décision + Random Forest
# python src/task2_trees_classification.py

Comparaison de tous les modèles (graphiques + heatmap F1)
# python src/compare_task2_classification.py

### Choix méthodologiques clés
## Capteur CGM retenu : Abbott FreeStyle Libre
Le dataset contient deux capteurs CGM (Abbott FreeStyle Libre et Dexcom G6). Seul l'Abbott est présent chez tous les patients — c'est donc lui qui sert de signal principal. Des écarts systématiques entre les deux capteurs ont été observés selon le statut métabolique, ce qui rendait leur combinaison risquée sans calibration inter-capteurs.
## Split train/test par patient — anti-data leakage
Le split est réalisé avec GroupKFold(n_splits=5, groups=patient_id). Toutes les fenêtres d'un même patient restent dans le même fold. Un split aléatoire sur les fenêtres aurait permis au modèle de mémoriser les patterns glycémiques basaux de chaque patient — ce n'est pas de la généralisation, c'est de la mémorisation.
✅ CORRECT : Fold 1 → train sur patients 10–45 / test sur patients 01–09
❌ INTERDIT : train sur fenêtres paires / test sur fenêtres impaires de TOUS les patients

## Configuration des features : Config A
Suite à l'analyse de l'importance des variables (indice de Gini) et à la sélection automatique par le Lasso, nous utilisons la Config A — agrégats CGM uniquement — plutôt que la séquence brute de 60 points minute par minute. Les deux approches convergent sur les mêmes variables pertinentes :

# CGM pré-repas:
cgm_at_meal, cgm_pre_mean, cgm_pre_std, cgm_pre_min, cgm_pre_max, cgm_slope_15, cgm_slope_30

# Nutrition
carbs, protein, fat, fiber

# Temporel
hour_sin, hour_cos (encodage cyclique)

# Biomarqueurs
bio_A1c PDL (Lab), bio_Fasting GLU, bio_Insulin, bio_BMI, bio_Age

# Type repas
meal_type (one-hot : breakfast, lunch, dinner, snacks)
