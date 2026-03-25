import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement des données
# Remplacez 'bio.csv' par le chemin exact de votre fichier si nécessaire
df = pd.read_csv('bio.csv')

# 2. Configuration du style visuel pour un rendu professionnel
sns.set_theme(style="whitegrid")

# 3. Séparation des variables numériques et catégorielles
# Attention à respecter scrupuleusement les majuscules et espaces de vos colonnes
colonnes_numeriques = [
    "Age", "BMI", "Fasting GLU - PDL (Lab)", 
    "Triglycerides", "Cholesterol", "HDL", "LDL (Cal)"
]
colonne_categorielle = "Self-identity"

# 4. Création d'une grande figure (canevas) contenant 8 sous-graphiques (4 lignes x 2 colonnes)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.3) # Ajustement des espaces entre les graphes
axes = axes.flatten() # Aplatissement de la matrice pour itérer facilement dessus

# 5. Boucle pour tracer les histogrammes des variables numériques
for i, col in enumerate(colonnes_numeriques):
    # Tracé de l'histogramme avec la courbe de tendance (kde=True)
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color="steelblue", bins=15)
    axes[i].set_title(f'Répartition de : {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Nombre de patients')

# 6. Tracé du graphique en barres pour la variable catégorielle (Self-identity) sur le dernier axe
sns.countplot(data=df, y=colonne_categorielle, ax=axes[7], palette="viridis", order=df[colonne_categorielle].value_counts().index)
axes[7].set_title(f'Répartition de : {colonne_categorielle}', fontsize=12, fontweight='bold')
axes[7].set_xlabel('Nombre de patients')
axes[7].set_ylabel('')

# 7. Sauvegarde de l'image globale et affichage
plt.savefig('distribution_variables.png', bbox_inches='tight', dpi=300)
plt.show()