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
# Markov-Kette 2^N Zustände. Bei N=8 (256 Zustände) dauert eine Lösung noch
# rund 15ms - bei der lokalen Suche werden pro Konfiguration mehrere Dutzend
# Nachbarn ausgewertet, N=8 bleibt damit auf kostenlosem Hosting reaktionsschnell.
# Siehe Prototyp-Benchmark: N=10 (1024 Zustände) liegt schon bei ~65ms/Lösung.
MAX_SERVERS = 8

LOCAL_SEARCH_MAX_ITER = 30
