"""
Zentrale Konstanten für die Rettungsdienst-Standortplanung-Demo
(Sebastian Hanisch - Operations Research und Machine Learning).
"""

MAP_SIZE = 10.0  # Kantenlänge des quadratischen Einsatzgebiets

DEFAULT_N_SERVERS = 4       # tatsächlich platzierte Fahrzeuge
DEFAULT_N_CANDIDATES = 10   # zur Auswahl stehende Standorte
DEFAULT_N_DEMAND = 12
DEFAULT_MU = 1.0            # Bedienrate (1 / mittlere Einsatzdauer)
DEFAULT_UTILIZATION = 0.5   # Ziel-Systemauslastung (lambda_total / (N * mu))
DEFAULT_TIME_THRESHOLD = 4.0  # Entfernungs-/Zeit-Schwelle für "Abdeckung"

# Obergrenze für die Anzahl platzierter Fahrzeuge: bei N Fahrzeugen hat die
# Markov-Kette 2^N Zustände. Bei N=8 (256 Zustände) dauert eine Lösung rund
# 0,5ms (siehe OPENBLAS_NUM_THREADS=1 in app.py/conftest.py - ohne diese
# Begrenzung braucht OpenBLAS' Multi-Thread-Koordination bei so kleinen
# Matrizen selbst ~15-20x so lange wie die eigentliche Rechnung) - bei der
# lokalen Suche werden pro Konfiguration mehrere Dutzend Nachbarn ausgewertet,
# N=8 bleibt damit auf kostenlosem Hosting reaktionsschnell.
MAX_SERVERS = 8

LOCAL_SEARCH_MAX_ITER = 30

# GA/ACO-Parameter: an Blank (Dissertation, Kapitel 6.3) angelehnt - dort
# identifiziert als die im eigenen Metaheuristik-Vergleich am besten
# funktionierende ACO-Parameterkombination (Supd=6, Verdunstung 0,1,
# Populationsgroesse 20). Fuer den GA verwendet diese Demo dieselbe
# Populationsgroesse, um beide Verfahren mit vergleichbarem "Budget" an
# HQM-Auswertungen je Generation/Iteration gegenueberzustellen.
GA_POPULATION_SIZE = 20
GA_MAX_GENERATIONS = 30
GA_CROSSOVER_PROB = 0.8
GA_MUTATION_PROB = 0.15
GA_NO_IMPROVE_PATIENCE = 8

ACO_N_ANTS = 20
ACO_S_UPD = 6
ACO_EVAPORATION = 0.1
ACO_MAX_ITERATIONS = 30
ACO_NO_IMPROVE_PATIENCE = 8

# Memetische Hybridisierung (GA/ACO + kurze lokale Suche auf dem jeweils
# besten Individuum/der besten Ameise je Generation/Iteration): mehrere
# Studien (u.a. eine Metaheuristik-Vergleichsstudie fuer Ambulanz-Allokation,
# GECCO 2023) finden, dass ein memetischer Algorithmus reinen GA/ACO-Varianten
# ueberlegen ist. max_iter bewusst klein gehalten, da dieser Schritt in jeder
# Generation/Iteration erneut ausgefuehrt wird.
GA_MEMETIC_POLISH_STEPS = 3
ACO_MEMETIC_POLISH_STEPS = 3

METAHEURISTIC_SEED_OFFSET = 51971  # abweichender Seed-Offset fuer GA/ACO-Zufallszahlen
