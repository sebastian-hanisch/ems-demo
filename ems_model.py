"""
Erzeugung von Nachfragepunkten und Standort-Kandidaten für die Rettungs-
dienst-Standortplanung-Demo.
"""

import numpy as np

from ems_constants import MAP_SIZE


def generate_demand_points(n_demand, n_peaks, peak_conc, seed):
    """Erzeugt Nachfragepunkte mit Positionen und relativen Gewichten
    (Bevölkerungsdichte-artig, mit ein paar Nachfrage-Schwerpunkten) -
    analog zum Peak-Muster in den anderen Demos."""
    rng = np.random.default_rng(int(seed))
    centers = rng.uniform(0, MAP_SIZE, size=(int(n_peaks), 2))
    spread = 1.0 + (1.0 - float(peak_conc)) * 3.0

    positions = np.zeros((n_demand, 2))
    weights = np.zeros(n_demand)
    for i in range(n_demand):
        if rng.random() < 0.85 and n_peaks > 0:
            center = centers[rng.integers(0, n_peaks)]
            pos = rng.normal(center, spread)
            pos = np.clip(pos, 0, MAP_SIZE)
        else:
            pos = rng.uniform(0, MAP_SIZE, size=2)
        positions[i] = pos
        weights[i] = rng.uniform(0.5, 1.5)
    weights = weights / weights.sum()
    return positions, weights


def generate_candidate_sites(n_candidates, seed, demand_positions=None):
    """Erzeugt Kandidatenstandorte - teils zufällig über das Gebiet verteilt,
    teils in der Nähe der Nachfrage (realistischer: Standorte werden selten
    fernab jeder Nachfrage vorgeschlagen)."""
    rng = np.random.default_rng(int(seed) + 9973)  # abweichender Seed-Offset, damit Standorte nicht 1:1 den Nachfragepunkten folgen
    sites = np.zeros((n_candidates, 2))
    for i in range(n_candidates):
        if demand_positions is not None and len(demand_positions) > 0 and rng.random() < 0.6:
            base = demand_positions[rng.integers(0, len(demand_positions))]
            pos = rng.normal(base, MAP_SIZE * 0.12)
            pos = np.clip(pos, 0, MAP_SIZE)
        else:
            pos = rng.uniform(0, MAP_SIZE, size=2)
        sites[i] = pos
    return sites


def scale_demand_to_utilization(weights, target_utilization, n_servers, mu):
    """Skaliert relative Nachfragegewichte so, dass lambda_total / (N*mu)
    genau der gewünschten Ziel-Systemauslastung entspricht."""
    lambda_total = float(target_utilization) * n_servers * mu
    return weights * lambda_total
