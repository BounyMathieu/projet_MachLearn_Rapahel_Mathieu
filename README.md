# Projet Machine Learning - Palliere Raphael & Bouny Mathieu

## Objectif

Prédire la réponse glycémique postprandiale (t+60 min) via CGM, macronutriments, activité physique et profil clinique.
Deux approches : **prédiction** (valeur exacte mg/dL) + **classification** (hypo / normo / hyperglycémie).

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






