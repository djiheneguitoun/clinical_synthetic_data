"""
clinical_synthetic_data : génération de données cliniques synthétiques.

Pipeline complet conformément au rapport (sections 1 à 8) :
    - Définition des classes et variables
    - Modélisation des plages et critères diagnostiques
    - Génération (Méthode 1 : copule gaussienne, Méthode 2 : CTGAN)
    - Validation clinique (bornes, R1-R4, cohérence valeurs/classe)
    - Analyses statistiques, visualisation, évaluation ML
"""

__version__ = "0.1.0"
