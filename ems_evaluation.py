"""Kennzahlen-Aufbereitung für Anzeige/Export - reine Funktionen ohne
Streamlit-Abhängigkeit, damit sie unabhängig testbar sind."""

import numpy as np


def naive_self_assessed_art(server_indices, candidate_sites, demand_positions, demand_weights):
    """Wie ein kongestionsblindes Coverage-Modell die erwartete Reaktionszeit
    einschätzen würde: Entfernung zum jeweils nächstgelegenen platzierten
    Fahrzeug, gewichtet nach Nachfrage - ohne jede Berücksichtigung von
    Verfügbarkeit/Auslastung."""
    server_pos = candidate_sites[server_indices]
    dist = np.linalg.norm(demand_positions[:, None, :] - server_pos[None, :, :], axis=2)
    nearest = dist.min(axis=1)
    return float((nearest * demand_weights).sum() / demand_weights.sum())


def hqm_summary(hqm_result, demand_weights, time_threshold):
    """Fasst ein solve_hqm()-Ergebnis in Kennzahlen zusammen:
    - art_served: erwartete Reaktionszeit, bedingt darauf, dass der Anruf
      tatsächlich bedient wird (nicht verloren geht)
    - coverage_pct: demand-gewichteter Anteil, bei dem der TATSÄCHLICH
      zuständige Server (nicht der nächstgelegene!) die Zeitschwelle einhält
    - p_loss: Wahrscheinlichkeit, dass ein Anruf verloren geht (alle Server belegt)
    - workload: Auslastung je Server
    """
    dispatch_freq = hqm_result["dispatch_freq"]  # (N, J)
    dists = hqm_result["dists"]  # (J, N)
    total_weight = demand_weights.sum()

    served_mass_per_demand = dispatch_freq.sum(axis=0)  # (J,)
    served_distance = (dispatch_freq * dists.T).sum()
    served_mass_total = served_mass_per_demand.sum()
    art_served = served_distance / served_mass_total if served_mass_total > 1e-9 else float("nan")

    within_threshold = dists.T <= time_threshold  # (N, J)
    covered_mass_per_demand = (dispatch_freq * within_threshold).sum(axis=0)  # (J,)
    coverage_pct = float((covered_mass_per_demand * demand_weights).sum() / total_weight * 100.0)

    return {
        "art_served": float(art_served),
        "coverage_pct": coverage_pct,
        "p_loss": hqm_result["p_loss"],
        "workload": hqm_result["workload"],
    }
