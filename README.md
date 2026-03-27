# Projet de Machine Learning - Palliere Raphael &amp; Bouny Mathieu


Lien JDD sur PhysioNet : https://physionet.org/content/cgmacros/1.0.0/



| Nom de la variable | Type de donnée | Description | Explication de la signification |
| :--- | :--- | :--- | :--- |
| **subject** | Numérique (entier) | Identifiant unique du participant. | Élément technique indispensable pour grouper les données par individu et éviter la fuite de données (data leakage) lors de la validation croisée. |
| **group** | Catégoriel (nominal) | Statut métabolique du patient (healthy, prediabetes, t2d). | Variable cible clinique de base permettant de stratifier la cohorte et d'analyser les différences de réponses physiologiques. |
| **Age** | Numérique (entier) | Âge du participant exprimé en années. | Facteur de risque majeur influençant la sensibilité à l'insuline et la régulation métabolique globale. |
| **Gender** | Catégoriel (binaire) | Sexe biologique du participant (M/F). | Paramètre influençant la répartition des masses tissulaires et la cinétique métabolique basale. |
| **BMI** | Numérique (continu) | Indice de Masse Corporelle (IMC). | Indicateur clinique standardisé de la corpulence, fortement corrélé aux risques d'insulinorésistance. |
| **Body weight** | Numérique (continu) | Poids corporel mesuré en livres (lbs). | Mesure anthropométrique brute nécessitant une conversion en kilogrammes pour les standards internationaux. |
| **Height** | Numérique (continu) | Taille corporelle mesurée en pouces (inches). | Mesure anthropométrique brute nécessitant une conversion en centimètres. |
| **Self-identify** | Catégoriel (nominal) | Ethnie auto-déclarée par le participant. | Variable démographique capturant de potentielles prédispositions génétiques ou habitudes socio-environnementales. |
| **A1c PDL (Lab)** | Numérique (continu) | Taux d'hémoglobine glyquée en pourcentage. | Marqueur clinique de référence évaluant l'équilibre glycémique moyen sur les 2 à 3 derniers mois. |
| **Fasting GLU - PDL (Lab)** | Numérique (continu) | Glycémie à jeun mesurée en laboratoire (mg/dL). | Évalue la capacité basale de l'organisme à réguler sa glycémie après une nuit complète sans apport calorique. |
| **Insulin** | Numérique (continu) | Taux d'insuline sanguin à jeun. | Crucial pour identifier une hyperinsulinémie compensatoire, marqueur précoce de la résistance à l'insuline. |
| **Triglycerides** | Numérique (continu) | Concentration de triglycérides sanguins (mg/dL). | Lipides dont l'excès est un marqueur très fort du syndrome métabolique et du risque cardiovasculaire. |
| **Cholesterol** | Numérique (continu) | Cholestérol total sanguin (mg/dL). | Mesure globale des lipides circulants utilisée pour évaluer le profil cardiovasculaire de base. |
| **HDL** | Numérique (continu) | Lipoprotéines de haute densité (mg/dL). | Souvent appelé "bon cholestérol", il participe à l'élimination du cholestérol en excès dans le sang. |
| **Non HDL** | Numérique (continu) | Cholestérol total moins le HDL (mg/dL). | Représente l'ensemble des lipoprotéines athérogènes (qui favorisent les plaques sur les artères). |
| **LDL (Cal)** | Numérique (continu) | Lipoprotéines de basse densité calculées (mg/dL). | Le "mauvais cholestérol", principal responsable du risque de maladies cardiovasculaires associé au diabète. |
| **VLDL (Cal)** | Numérique (continu) | Lipoprotéines de très basse densité calculées (mg/dL). | Particules transportant principalement les triglycérides synthétisés par le foie vers les tissus. |
| **Cho/HDL Ratio** | Numérique (continu) | Rapport entre le cholestérol total et le HDL. | Indice clinique composite utilisé par les médecins pour affiner l'évaluation du risque cardiovasculaire. |
| **Collection time PDL (Lab)** | Temporel (Datetime) | Heure du prélèvement sanguin en laboratoire. | Permet de contextualiser avec précision le moment où le bilan biologique à jeun a été réalisé. |
| **#1 Contour Fingerstick GLU** | Numérique (continu) | 1ère mesure de glycémie capillaire (mg/dL). | Relevé sanguin par piqûre au bout du doigt, servant de contrôle manuel fiable pour la glycémie. |
| **Time (t)** | Temporel (Time) | Heure exacte de la 1ère mesure capillaire. | Horodatage indispensable pour synchroniser cette mesure avec les données continues du capteur CGM. |
| **#2 Contour Fingerstick GLU** | Numérique (continu) | 2ème mesure de glycémie capillaire (mg/dL). | Deuxième point de contrôle manuel pour suivre l'évolution ponctuelle de la glycémie. |
| **Time (t).1** | Temporel (Time) | Heure exacte de la 2ème mesure capillaire. | Horodatage associé à la deuxième mesure de contrôle. |
| **#3 Contour Fingerstick GLU** | Numérique (continu) | 3ème mesure de glycémie capillaire (mg/dL). | Dernier point de contrôle de cette série de mesures manuelles. |
| **Time (t).2** | Temporel (Time) | Heure exacte de la 3ème mesure capillaire. | Horodatage associé à la troisième mesure de contrôle. |
