# Projet Machine Learning - Palliere Raphael & Bouny Mathieu

---

## Objectif

Prédire la réponse glycémique postprandiale (t+60 min) via CGM, macronutriments, activité physique et profil clinique.
Deux approches : **prédiction** (valeur exacte mg/dL) + **classification** (hypo / normo / hyperglycémie).

### Contexte et motivation

La prédiction de la réponse glycémique postprandiale représente un enjeu clinique majeur dans la prévention et la gestion des troubles métaboliques. Face à la prévalence croissante du diabète et du prédiabète, nous nous sommes fixé pour objectif de développer un modèle de Machine Learning capable de prédire avec fiabilité la glycémie 60 minutes après le début d'un repas. Ce projet, réalisé en collaboration avec nos collègues de l'école, s'inscrit dans une démarche d'innovation clinique où les données glycémiques continues rencontrent l'intelligence artificielle.

Le défi réside dans la complexité des interactions biologiques : la réponse glycémique ne dépend pas seulement de la composition du repas, mais aussi de l'historique métabolique du patient, de son activité physique, de son profil clinique global et potentiellement de sa microbiote. Nous avons choisi d'explorer simultanément une population diversifiée — sujets sains, prédiabétiques et diabétiques de type 2 — afin de capturer cette variabilité et de construire un modèle robuste et généralisable.

### Présentation du dataset CGMacros

Nous disposons d'un dataset original comportant 45 participants suivis en conditions de vie réelle sur 10 jours consécutifs. Cette population se répartit équitablement entre trois profils : 15 sujets sains, 16 prédiabétiques et 14 diabétiques de type 2. La caractérisation physiologique repose sur un ensemble riche de mesures : variables biologiques cliniques (HbA1c, glycémie à jeun, insuline, lipides), données de capteurs de glycémie continue (Dexcom G6 Pro et Abbott FreeStyle Libre), suivi d'activité physique par Fitbit, et données nutritionnelles détaillées des repas.

En particulier, nous avons accumulé environ 129 600 points glycémiques au total, soit une résolution temporelle très fine grâce aux deux capteurs CGM interpolés à la minute. Les données nutritionnelles incluent les macronutriments essentiels (glucides, protéines, lipides, fibres) et l'apport calorique total. L'ajout de données microbiote — 1 979 bactéries identifiées — ouvre des perspectives intéressantes, bien que leur utilisation dans notre modèle reste encore exploratoire. L'ensemble du dataset est accessible via PhysioNet après certification CITI Program, garantissant le respect des normes éthiques.

### Difficultés rencontrées et choix méthodologiques

Nous avons rapidement identifié plusieurs défis pratiques lors du nettoyage des données. Le capteur Dexcom GL a présenté des lacunes importantes et des divergences significatives avec les mesures de l'Abbott FreeStyle Libre. Après une évaluation comparative approfondie, nous avons pris la décision d'utiliser exclusivement les données du capteur Abbott, ce qui nous garantit une cohérence et une fiabilité accrues.

Un autre point notable concerne l'exclusion du patient n°12, dont les valeurs de triglycérides et LDL étaient anormalement élevées. Bien que ces valeurs ne soient pas techniquement aberrantes, elles reflètent un problème cardiaque indépendant de la pathologie glycémique principale, justifiant ainsi son exclusion de l'analyse. Nous avons également observé un déséquilibre dans la distribution du genre (environ deux fois plus de femmes) et de l'ethnie (majorité hispanique et latino-américaine), un biais que nous avons documenté et qui doit être pris en compte lors de l'interprétation des résultats.

L'extraction des fenêtres repas s'est avérée délicate : nous avons identifié environ 1 700 épisodes repas à partir des 45 patients (en moyenne trois repas et collations par jour). Une corrélation intra-patient forte nous a poussés à adopter une validation croisée GroupKFold stratifiée par patient, ce qui prévient le data leakage et garantit une évaluation réaliste de la généralisation du modèle.

### Protocole de validation et approche méthodologique

Nous avons opté pour une double approche : d'une part une tâche de régression visant à prédire la valeur glycémique exacte aux horizons t+30, t+60 et t+90 minutes, et d'autre part une tâche de classification multiclasse permettant de prédire le risque d'hypoglycémie, d'euglycémie ou d'hyperglycémie. Cette dualité nous offre à la fois une granularité numérique et une pertinence clinique.

Pour la régression, nous utilisons les métriques classiques : RMSE, MAE et R². Pour la classification, nous nous appuyons sur la précision, le rappel et le score F1 afin de capturer les performances de façon équilibrée. Les modèles testés couvrent un spectre allant de la régression linéaire (baseline simple) aux arbres de décision, Random Forest et réseaux de neurones, ce qui nous permet d'identifier le meilleur compromis entre complexité et performance.


---

## Dataset - CGMacros

**Source** : [PhysioNet CGMacros v1.0.0](https://physionet.org/content/cgmacros/1.0.0/)


| Paramètre | Valeur |
| :--- | :--- |
| Participants | 44 (45 - patient n°12 exclu) |
| Groupes | 15 sains / 16 prédiabétiques / 14 Diabétiques (Type2) |
| Durée suivi | 10 jours en conditions réelles |
| Capteur retenu | Abbott FreeStyle Libre Pro (15 min, interpolé 1 min) |
| Capteur exclu | Dexcom G6 Pro (données manquantes, incohérences) |
| Points glycémie bruts | ~129 600 |
| Fenêtres repas extraites | 1 700 |

### Variables `bio.csv`

| Nom de la variable | Type | Description | Signification |
| :--- | :--- | :--- | :--- |
| **subject** | Numérique (entier) | Identifiant unique du participant | Indispensable pour GroupKFold - évite data leakage |
| **group** | Catégoriel (nominal) | Statut métabolique (healthy / prediabetes / t2d) | Variable cible clinique - ajoutée manuellement via A1c |
| **Age** | Numérique (entier) | Âge en années | Facteur de risque - sensibilité insuline |
| **Gender** | Catégoriel (binaire) | Sexe (M/F) | Impact cinétique métabolique - déséquilibre 2F:1H |
| **BMI** | Numérique (continu) | Indice de Masse Corporelle | Corrélé insulinorésistance |
| **Body weight** | Numérique (continu) | Poids en livres (lbs) | Conversion kg nécessaire |
| **Height** | Numérique (continu) | Taille en pouces (inches) | Conversion cm nécessaire |
| **Self-identify** | Catégoriel (nominal) | Ethnie auto-déclarée | Pertinente cliniquement - exclue phase 1, intégrable phase 2 |
| **A1c PDL (Lab)** | Numérique (continu) | HbA1c en % | Référence classification groupe - moyenne 2-3 mois |
| **Fasting GLU - PDL (Lab)** | Numérique (continu) | Glycémie à jeun (mg/dL) | Capacité basale régulation glycémique |
| **Insulin** | Numérique (continu) | Insuline à jeun | Marqueur précoce insulinorésistance |
| **Triglycerides** | Numérique (continu) | Triglycérides (mg/dL) | Marqueur syndrome métabolique |
| **Cholesterol** | Numérique (continu) | Cholestérol total (mg/dL) | Profil cardiovasculaire de base |
| **HDL** | Numérique (continu) | Lipoprotéines haute densité (mg/dL) | "Bon cholestérol" |
| **Non HDL** | Numérique (continu) | Cholestérol total - HDL (mg/dL) | Lipoprotéines athérogènes |
| **LDL (Cal)** | Numérique (continu) | Lipoprotéines basse densité calculées (mg/dL) | "Mauvais cholestérol" - risque cardiovasculaire |
| **VLDL (Cal)** | Numérique (continu) | Lipoprotéines très basse densité calculées (mg/dL) | Transport triglycérides foie → tissus |
| **Cho/HDL Ratio** | Numérique (continu) | Cholestérol total / HDL | Indice risque cardiovasculaire composite |
| **Collection time PDL (Lab)** | Temporel (Datetime) | Heure prélèvement sanguin | Contextualisation bilan à jeun |
| **#1 Contour Fingerstick GLU** | Numérique (continu) | 1ère glycémie capillaire (mg/dL) | Contrôle manuel - référence CGM |
| **Time (t)** | Temporel (Time) | Heure 1ère mesure capillaire | Synchronisation avec signal CGM |
| **#2 Contour Fingerstick GLU** | Numérique (continu) | 2ème glycémie capillaire (mg/dL) | Suivi ponctuel glycémie |
| **Time (t).1** | Temporel (Time) | Heure 2ème mesure capillaire | Horodatage mesure 2 |
| **#3 Contour Fingerstick GLU** | Numérique (continu) | 3ème glycémie capillaire (mg/dL) | Dernier contrôle manuel série |
| **Time (t).2** | Temporel (Time) | Heure 3ème mesure capillaire | Horodatage mesure 3 |

---

## Structure du dépôt

```

├── README.md
│
├── requirements.txt
│
├── .gitignore
│
├── data/
│ ├── processed
│ │ └── bio_with_group.csv
│ │ └── column_description_meal_window.csv
│ │ └── meal_windows_dataset.csv
│ │ └── test.py
│ │
│ ├── raw
│ │ └── README.md        # Instructions accès PhysioNet
│ │
│ ├── results
│ │ └── baseline_linear
│ │ │ └── baseline_results.csv
│ │ │ └── coefficients_configA_t60.csv
│ │ │ └── coefficients_configA_t60.png
│ │ │ └── rmse_comparaison_A_vs_B.png
│ │ │ └── scatter_configA_t60.png
│
├── notebooks/
│ ├── 01_EDA.ipynb        # Analyse exploratoire
│ ├── 02_preprocessing.ipynb
│ └── 03_modelling.ipynb
│
├── src/
│ ├── Analyse exploratoire
│ │ └── graphes_repartitions.ipynb
│ │ 
│ ├── build_patient_table.py
│ ├── column_description_meal_window.py
│ └── meal_windows_builder.py
│
├── dashboard/
│ └── app.py        # Streamlit
│ └── report/
│
├── rapport.pdf
│ └── figures/

```

---

## Installation

```bash
git clone https://github.com/BounyMathieu/projet_MachLearn_Rapahel_Mathieu.git
cd projet_MachLearn_Rapahel_Mathieu
pip install -r requirements.txt
```

---

## Pipeline

```
bio.csv + CGMacros-0XX.xlsx
        ↓
Ajout colonne group (via A1c)
Suppression patient n°12 (comorbidité cardiaque)
Suppression signal Dexcom G6
        ↓
meal_window_builder.py → ~1 700 fenêtres repas
Filtre : min 30 points valides sur 60 min pré-repas
        ↓
GroupKFold (groups = patient_id)  ← anti data leakage
        ↓
Régression / Classification
```


---

## Auteurs

| Nom | Mail |
| :--- | :--- |
| Palliere Raphael | palliere.raphael@icloud.com |
| Bouny Mathieu | bouny.mathieu@gmail.com |


Accréditation CITI Program - recherche sur sujets humains






