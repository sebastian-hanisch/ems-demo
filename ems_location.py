"""
Zwei Standortstrategien im Vergleich:

1. Naive Coverage (Greedy-MCLP): klassischer Greedy-Algorithmus für das
   Maximal Covering Location Problem - wählt wiederholt den Kandidaten, der
   die meiste noch nicht abgedeckte (gewichtete) Nachfrage innerhalb einer
   Zeitschwelle abdeckt. Kennt keine Warteschlangentheorie: geht implizit
   davon aus, dass ein Fahrzeug immer verfügbar ist, wenn es gebraucht wird.

2. HQM-bewusste lokale Suche: startet bei einer Konfiguration und tauscht
   wiederholt einen platzierten gegen einen nicht platzierten Standort, wenn
   das die über das Hypercube Queueing Model TATSÄCHLICH berechnete
   Zielgröße verbessert - Verlust-Wahrscheinlichkeit (kein Fahrzeug frei)
   eingeschlossen.
"""

import numpy as np

from ems_constants import LOCAL_SEARCH_MAX_ITER, MAP_SIZE
from ems_hqm import solve_hqm

LOST_CALL_PENALTY_DISTANCE = MAP_SIZE * 2.0  # siehe hqm_objective()


def greedy_mclp(candidate_sites, demand_positions, demand_weights, n_servers, time_threshold):
    """Klassischer Greedy für das Maximal Covering Location Problem -
    kongestionsblind, dient als "naive" Baseline."""
    n_candidates = len(candidate_sites)
    dist = np.linalg.norm(
        demand_positions[:, None, :] - candidate_sites[None, :, :], axis=2
    )  # (J, n_candidates)
    covers = dist <= time_threshold

    chosen = []
    covered = np.zeros(len(demand_positions), dtype=bool)
    remaining = list(range(n_candidates))
    for _ in range(n_servers):
        best_site, best_gain = None, -1.0
        for c in remaining:
            newly_covered = covers[:, c] & ~covered
            gain = demand_weights[newly_covered].sum()
            if gain > best_gain:
                best_gain, best_site = gain, c
        if best_site is None:
            break
        chosen.append(best_site)
        covered |= covers[:, best_site]
        remaining.remove(best_site)
    return chosen


def hqm_objective(server_indices, candidate_sites, demand_positions, demand_weights, mu):
    """Demand-gewichtete erwartete Distanz (als Näherung für die Reaktions-
    zeit) über das Hypercube Queueing Model - inklusive eines Strafterms für
    verlorene Anrufe (alle Fahrzeuge belegt). Ohne diesen Strafterm könnte
    die lokale Suche eine hohe Verlustwahrscheinlichkeit in Kauf nehmen,
    solange die noch bedienten Anrufe im Schnitt kurze Wege haben - das wäre
    für ein Rettungsdienstsystem ein unsinniges Optimum."""
    server_pos = candidate_sites[server_indices]
    result = solve_hqm(server_pos, demand_positions, demand_weights, mu)
    served_distance = float((result["dispatch_freq"] * result["dists"].T).sum())
    return served_distance + result["p_loss"] * demand_weights.sum() * LOST_CALL_PENALTY_DISTANCE


def local_search(initial_indices, candidate_sites, demand_positions, demand_weights, mu, max_iter=LOCAL_SEARCH_MAX_ITER, cache=None):
    """Best-Improvement-Lokalsuche: tauscht in jeder Iteration den Zug, der
    die HQM-Zielgröße am meisten verbessert, bis keine Verbesserung mehr
    gefunden wird oder max_iter erreicht ist. Gibt zusätzlich die Historie
    der Zielwerte zurück (für eine "Konvergenz"-Anzeige im UI).

    (Ein First-Improvement-Variante - der erste statt der beste verbessernde
    Zug je Iteration - wurde getestet, war aber ueber 10 Testszenarien im
    Schnitt 48% LANGSAMER: sie braucht 2-5x mehr Iterationen bis zur
    Konvergenz, was den Einspareffekt pro Iteration wieder auffrisst. Deshalb
    bleibt es bei Best-Improvement.)

    cache: optionales dict {Standort-Tupel: Zielgroesse} zur Memoisierung -
    wird von den Metaheuristiken in ems_metaheuristics.py hereingereicht, die
    local_search auch als Politur-Schritt aufrufen, damit dort schon
    berechnete Konfigurationen nicht erneut das (teure) HQM-Gleichungssystem
    loesen muessen. Ohne uebergebenes dict wird lokal ein neues angelegt."""
    if cache is None:
        cache = {}

    def objective(indices):
        key = tuple(sorted(indices))
        if key not in cache:
            cache[key] = hqm_objective(indices, candidate_sites, demand_positions, demand_weights, mu)
        return cache[key]

    n_candidates = len(candidate_sites)
    current = list(initial_indices)
    current_obj = objective(current)
    history = [current_obj]

    for _ in range(max_iter):
        best_swap, best_obj = None, current_obj
        chosen_set = set(current)
        for out_pos, out_idx in enumerate(current):
            for in_idx in range(n_candidates):
                if in_idx in chosen_set:
                    continue
                candidate = current.copy()
                candidate[out_pos] = in_idx
                obj = objective(candidate)
                if obj < best_obj - 1e-9:
                    best_obj, best_swap = obj, (out_pos, in_idx)
        if best_swap is None:
            break
        out_pos, in_idx = best_swap
        current[out_pos] = in_idx
        current_obj = best_obj
        history.append(current_obj)

    return current, history
