"""
Zwei weitere Standortstrategien, die Blank (Dissertation, Kapitel 6) explizit
miteinander vergleicht: Genetischer Algorithmus (GA) und Ant-Colony-
Optimization (ACO). Beide werden hier - wie dort - direkt auf dieselbe
HQM-Zielgroesse angewandt wie die lokale Suche in ems_location.py
(demand-gewichtete erwartete Distanz auf Basis der tatsaechlichen
Zuteilungswahrscheinlichkeiten, siehe hqm_objective).

Blank kommt in eigenen Experimenten zu einem klaren Ergebnis (S. 168f. der
Dissertation): "the ACO performs better than the GA in the proposed
experimental setting" - die ACO reagiert deutlich robuster auf die
Parameterwahl und wird deshalb fuer den Rest der Arbeit als alleinige
Optimierungstechnik verwendet. Diese Demo bildet beide Verfahren nach, damit
sich dieser Befund direkt am eigenen Szenario nachvollziehen laesst (siehe
Expander "Metaheuristik-Vergleich" in app.py).

GA-Chromosom bzw. ACO-"Ameise": eine Liste von n_servers verschiedenen
Standort-Indizes (ein Standort je platziertem Fahrzeug) - dieselbe
Repraesentation, die auch local_search verwendet.

Beide Verfahren sind memetisch hybridisiert: das jeweils beste Individuum
bzw. die beste Ameise wird je Generation/Iteration mit ein paar Schritten
lokaler Suche (Wiederverwendung von local_search aus ems_location.py)
nachpoliert, bevor Selektion bzw. Pheromon-Update darauf aufbauen. Mehrere
Studien - u.a. eine Metaheuristik-Vergleichsstudie fuer Ambulanz-Allokation
(GECCO 2023) sowie der allgemeine ACO-Literaturstrang "ACO + lokale Suche" -
finden, dass diese Hybridisierung reine populationsbasierte Verfahren
uebertrifft.

Beide Verfahren memoisieren HQM-Auswertungen ueber ein lokales cache-dict
(Schluessel: sortiertes Tupel der Standort-Indizes), das auch an die
local_search-Politur weitergereicht wird - bei hohen Fahrzeugzahlen (grosser
HQM-Zustandsraum) werden ansonsten dieselben Konfigurationen wiederholt neu
geloest, gerade waehrend der spaeten, kaum noch verbessernden Generationen.
"""

import time

import numpy as np

from ems_constants import (
    ACO_EVAPORATION,
    ACO_MAX_ITERATIONS,
    ACO_MEMETIC_POLISH_STEPS,
    ACO_N_ANTS,
    ACO_NO_IMPROVE_PATIENCE,
    ACO_S_UPD,
    GA_CROSSOVER_PROB,
    GA_MAX_GENERATIONS,
    GA_MEMETIC_POLISH_STEPS,
    GA_MUTATION_PROB,
    GA_NO_IMPROVE_PATIENCE,
    GA_POPULATION_SIZE,
)
from ems_location import hqm_objective, local_search


def _random_individual(rng, n_candidates, n_servers):
    return list(rng.choice(n_candidates, size=n_servers, replace=False))


def _repair_duplicates(rng, individual, n_candidates):
    """Nach Crossover kann derselbe Standort-Index doppelt im Kind auftauchen
    (zwei Fahrzeuge am selben Standort) - ersetzt Duplikate durch zufaellige,
    im Individuum noch nicht verwendete Indizes, um die Loesung gueltig zu
    halten (ein Standort pro platziertem Fahrzeug)."""
    seen = set()
    free = None
    for pos, gene in enumerate(individual):
        if gene in seen:
            if free is None:
                free = [i for i in range(n_candidates) if i not in individual]
                rng.shuffle(free)
            individual[pos] = free.pop()
        seen.add(individual[pos])
    return individual


def _mutate(rng, individual, n_candidates, mutation_prob):
    for pos in range(len(individual)):
        if rng.random() < mutation_prob:
            choices = [i for i in range(n_candidates) if i not in individual]
            if choices:
                individual[pos] = int(rng.choice(choices))
    return individual


def genetic_algorithm(
    candidate_sites, demand_positions, demand_weights, mu, n_servers, seed,
    population_size=GA_POPULATION_SIZE, max_generations=GA_MAX_GENERATIONS,
    crossover_prob=GA_CROSSOVER_PROB, mutation_prob=GA_MUTATION_PROB,
    no_improve_patience=GA_NO_IMPROVE_PATIENCE, polish_steps=GA_MEMETIC_POLISH_STEPS,
):
    """Genetischer Algorithmus nach Blank, Kapitel 6.1: fitness-proportionale
    Selektion (Eq. 6.1.2 der Dissertation, da die HQM-Zielgroesse minimiert
    wird - kleinere Fitnesswerte gelten als ueberlegen), Ein-Punkt-Crossover,
    Mutation, elitistische Nachfolge (die besten population_size Individuen
    aus Eltern- und Kindgeneration ueberleben). Memetische Hybridisierung:
    das beste Individuum jeder Generation wird zusaetzlich mit polish_steps
    Schritten lokaler Suche nachpoliert (siehe Moduldoc). Bricht ab, wenn
    sich das beste Ergebnis ueber no_improve_patience Generationen nicht mehr
    verbessert.

    Rueckgabe: (beste Standort-Indizes, history der je Generation bislang
    besten Zielgroesse - garantiert monoton fallend, siehe Tests -, Laufzeit
    in Sekunden)."""
    start = time.perf_counter()
    n_candidates = len(candidate_sites)
    rng = np.random.default_rng(seed)
    cache = {}

    def objective(individual):
        key = tuple(sorted(individual))
        if key not in cache:
            cache[key] = hqm_objective(individual, candidate_sites, demand_positions, demand_weights, mu)
        return cache[key]

    population = [_random_individual(rng, n_candidates, n_servers) for _ in range(population_size)]
    fitness = [objective(ind) for ind in population]

    best_idx = int(np.argmin(fitness))
    best, best_obj = list(population[best_idx]), fitness[best_idx]
    history = [best_obj]
    stale = 0

    for _ in range(max_generations):
        inv_fit = 1.0 / np.array(fitness)
        probs = inv_fit / inv_fit.sum()

        offspring = []
        while len(offspring) < population_size:
            p1, p2 = rng.choice(population_size, size=2, replace=False, p=probs)
            parent1, parent2 = population[p1], population[p2]
            if n_servers > 1 and rng.random() < crossover_prob:
                cut = int(rng.integers(1, n_servers))
                child1 = parent1[:cut] + parent2[cut:]
                child2 = parent2[:cut] + parent1[cut:]
            else:
                child1, child2 = list(parent1), list(parent2)
            child1 = _mutate(rng, _repair_duplicates(rng, child1, n_candidates), n_candidates, mutation_prob)
            child2 = _mutate(rng, _repair_duplicates(rng, child2, n_candidates), n_candidates, mutation_prob)
            offspring.extend([child1, child2])
        offspring = offspring[:population_size]
        offspring_fitness = [objective(ind) for ind in offspring]

        combined = sorted(zip(population + offspring, fitness + offspring_fitness), key=lambda pair: pair[1])
        population = [ind for ind, _ in combined[:population_size]]
        fitness = [f for _, f in combined[:population_size]]

        polished, polish_history = local_search(
            population[0], candidate_sites, demand_positions, demand_weights, mu, max_iter=polish_steps, cache=cache,
        )
        if polish_history[-1] < fitness[0] - 1e-9:
            population[0], fitness[0] = polished, polish_history[-1]

        if fitness[0] < best_obj - 1e-9:
            best, best_obj, stale = list(population[0]), fitness[0], 0
        else:
            stale += 1
        history.append(best_obj)
        if stale >= no_improve_patience:
            break

    return best, history, time.perf_counter() - start


def ant_colony_optimization(
    candidate_sites, demand_positions, demand_weights, mu, n_servers, seed,
    n_ants=ACO_N_ANTS, s_upd=ACO_S_UPD, evaporation=ACO_EVAPORATION,
    max_iterations=ACO_MAX_ITERATIONS, no_improve_patience=ACO_NO_IMPROVE_PATIENCE,
    polish_steps=ACO_MEMETIC_POLISH_STEPS,
):
    """Ant-Colony-Optimization nach Blank, Kapitel 6.2: jede Ameise waehlt
    n_servers Standorte nacheinander ohne Zuruecklegen, mit Wahrscheinlichkeit
    proportional zum aktuellen Pheromonwert tau_i jedes noch verfuegbaren
    Kandidaten. Je Iteration aktualisieren die s_upd besten Ameisen die
    Pheromone (Eq. 6.2.1 der Dissertation): Verdunstung um den Faktor
    (1 - evaporation), anschliessend Zuwachs proportional zur relativen
    Loesungsqualitaet (bessere Ameisen tragen mehr beim Update bei). Memetische
    Hybridisierung ("ACO + lokale Suche", siehe Moduldoc): die beste Ameise
    jeder Iteration wird vor der Pheromon-Aktualisierung mit polish_steps
    Schritten lokaler Suche nachpoliert - die verbesserte Loesung fliesst dann
    auch in die Pheromon-Verstaerkung ein. Bricht ab, wenn sich das beste
    Ergebnis ueber no_improve_patience Iterationen nicht mehr verbessert.

    Rueckgabe: (beste Standort-Indizes, history der je Iteration bislang
    besten Zielgroesse - garantiert monoton fallend, siehe Tests -, Laufzeit
    in Sekunden)."""
    start = time.perf_counter()
    n_candidates = len(candidate_sites)
    rng = np.random.default_rng(seed)
    cache = {}

    def objective(individual):
        key = tuple(sorted(individual))
        if key not in cache:
            cache[key] = hqm_objective(individual, candidate_sites, demand_positions, demand_weights, mu)
        return cache[key]

    tau = np.full(n_candidates, 1.0 / n_candidates)

    best, best_obj = None, float("inf")
    history = []
    stale = 0

    for _ in range(max_iterations):
        ants = []
        for _ in range(n_ants):
            remaining = list(range(n_candidates))
            chosen = []
            for _ in range(n_servers):
                weights = tau[remaining]
                probs = weights / weights.sum()
                pick = rng.choice(len(remaining), p=probs)
                chosen.append(remaining.pop(pick))
            ants.append(chosen)
        ant_fitness = [objective(ind) for ind in ants]

        best_ant_idx = int(np.argmin(ant_fitness))
        polished, polish_history = local_search(
            ants[best_ant_idx], candidate_sites, demand_positions, demand_weights, mu, max_iter=polish_steps, cache=cache,
        )
        if polish_history[-1] < ant_fitness[best_ant_idx] - 1e-9:
            ants[best_ant_idx], ant_fitness[best_ant_idx] = polished, polish_history[-1]

        order = list(np.argsort(ant_fitness)[:s_upd])
        quality = np.array([1.0 / ant_fitness[i] for i in order])
        elite_weights = quality / quality.sum()

        tau *= (1.0 - evaporation)
        for rank, ant_idx in enumerate(order):
            for site in ants[ant_idx]:
                tau[site] += evaporation * elite_weights[rank]

        iter_best_idx = int(np.argmin(ant_fitness))
        if ant_fitness[iter_best_idx] < best_obj - 1e-9:
            best, best_obj, stale = list(ants[iter_best_idx]), ant_fitness[iter_best_idx], 0
        else:
            stale += 1
        history.append(best_obj)
        if stale >= no_improve_patience:
            break

    return best, history, time.perf_counter() - start
