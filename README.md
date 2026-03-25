# Projet de Machine Learning - Palliere Raphael &amp; Bouny Mathieu



| Nom de la variable | Type de donnée | Description | Explication de la signification |
| :--- | :--- | :--- | :--- |
| **subject** | Numérique (entier) | Identifiant unique attribué à chaque participant. | Élément technique indispensable pour grouper les données temporelles par individu et appliquer notre stratégie de validation croisée sans fuite d'information (data leakage). |
| **group** | Catégoriel (nominal) | Statut métabolique du patient (healthy, prediabetes, t2d). | Variable cible clinique de base permettant de stratifier la cohorte et d'analyser les différences de réponses physiologiques face à un même apport nutritionnel. |
| **Age** | Numérique (entier) | Âge du participant exprimé en années. | Facteur de risque majeur dans l'apparition des troubles métaboliques, la sensibilité à l'insuline et la régulation glycémique évoluant avec le vieillissement cellulaire. |
| **Gender** | Catégoriel (binaire) | Sexe biologique du participant (M/F). | Paramètre influençant le métabolisme de base, la répartition des masses tissulaires (graisse viscérale vs musculaire) et, par conséquent, la cinétique d'assimilation des glucides. |
| **Body weight / Height** | Numérique (continu) | Poids (en livres) et taille (en pouces) du patient. | Données brutes anthropométriques (nécessitant une conversion dans le système métrique) indispensables pour évaluer la corpulence globale de l'individu. |
| **BMI (IMC)** | Numérique (continu) | Indice de Masse Corporelle. | Indicateur clinique standardisé du surpoids et de l'obésité, conditions physiologiques fortement corrélées à l'insulinorésistance et au diabète de type 2. |
| **Self-identify** | Catégoriel (nominal) | Ethnie auto-déclarée par le participant. | Variable démographique permettant de capturer des prédispositions génétiques ou des habitudes socio-environnementales (fréquentes dans les études de cohortes américaines). |
| **A1c PDL (Lab)** | Numérique (continu) | Taux d'hémoglobine glyquée en pourcentage. | Marqueur biologique de référence reflétant l'équilibre glycémique moyen sur les 2 à 3 mois précédant la prise de sang. C'est un critère diagnostique fondamental du diabète. |
| **Fasting GLU** | Numérique (continu) | Glycémie à jeun en mg/dL. | Mesure de la concentration de glucose sanguin après une nuit de jeûne, évaluant la capacité basale de l'organisme à réguler sa glycémie sans contrainte alimentaire. |
| **Insulin** | Numérique (continu) | Insulinémie à jeun. | L'insuline étant l'hormone hypoglycémiante clé, un taux basal anormalement élevé face à une glycémie normale ou haute est le principal marqueur de l'insulinorésistance périphérique. |
| **Lipid Panel** | Numérique (continu) | Bilan lipidique sanguin complet en mg/dL (cholestérol total, bon/mauvais cholestérol, triglycérides). | L'évaluation des lipides est indissociable du risque métabolique global. Une dyslipidémie accompagne très fréquemment le diabète de type 2 (syndrome métabolique). |
| **Fingerstick GLU** | Numérique (continu) | Mesure de la glycémie capillaire ponctuelle en mg/dL. | Relevé sanguin direct servant de contrôle qualité (ou de calibration) pour vérifier la fiabilité et la précision des capteurs interstitiels continus (CGM). |
| **CGM Glucose** | Numérique (continu) | Mesure continue de la glycémie interstitielle (toutes les 5 ou 15 min). | Le cœur de notre variable prédictive temporelle. Permet de capter la cinétique (pente, vitesse d'augmentation) de la glycémie avant et après le repas. |
| **Macronutrients** | Numérique (continu) | Composition du repas (Glucides, Protéines, Lipides, Fibres en grammes). | La source primaire de la perturbation glycémique. Les glucides dictent le pic d'hyperglycémie, tandis que les lipides et protéines modifient la vitesse d'absorption. |
| **Physical Activity** | Numérique (continu) | Fréquence cardiaque et équivalents métaboliques (METs). | L'activité physique modifie drastiquement la sensibilité musculaire à l'insuline et la consommation périphérique du glucose, impactant directement la courbe postprandiale. |
