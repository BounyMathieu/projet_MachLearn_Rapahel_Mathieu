Ordre d'éxécution des codes src : 
Étape 1 — Préparation des données patients
python src/build_patient_table.py
Ce script lit data/raw/cgmacros/bio.csv et ajoute une colonne group classifiant chaque patient selon son taux d'HbA1c :

healthy si HbA1c < 5.7 %
prediabetes si 5.7 % ≤ HbA1c ≤ 6.4 %
t2d si HbA1c > 6.4 %

Produit : data/processed/bio_with_group.csv


