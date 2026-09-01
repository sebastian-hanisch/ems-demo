"""
Unit-Tests der reinen Modell-/HQM-/Standortlogik (kein Streamlit-UI-Code).

Der wichtigste Testblock (TestErlangBIdentity) ist kein gewöhnlicher
Regressionstest, sondern eine von der Implementierung unabhängige,
mathematisch scharfe Korrektheitsprüfung: siehe Docstring von
ems_hqm.solve_hqm für die Herleitung, warum die Verlustwahrscheinlichkeit
bei homogener Bedienrate exakt der klassischen Erlang-B-Formel entsprechen
MUSS, unabhängig von räumlicher Anordnung oder Präferenzlisten.

Ausführen mit: pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

APP_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(APP_DIR))

from ems_evaluation import hqm_summary, naive_self_assessed_art
from ems_hqm import build_preference_lists, erlang_b, solve_hqm
from ems_location import greedy_mclp, hqm_objective, local_search
from ems_metaheuristics import ant_colony_optimization, genetic_algorithm
from ems_model import generate_candidate_sites, generate_demand_points, scale_demand_to_utilization
from ems_pdf_export import generate_location_plan_pdf


# ==========================================================================
# Erlang-B-Identität - der zentrale, implementierungsunabhängige Korrektheitstest
# ==========================================================================

class TestErlangBIdentity:
    def test_homogeneous_single_demand_node(self):
        for n in [2, 3, 5, 7]:
            for lam in [1.0, 3.0, 8.0]:
                server_pos = [(i, 0) for i in range(n)]
                demand_pos = [(0.5, 0.5)]
                result = solve_hqm(server_pos, demand_pos, [lam], 1.0)
                expected = erlang_b(n, lam)
                assert result["p_loss"] == pytest.approx(expected, abs=1e-8), f"n={n}, lam={lam}"

    def test_heterogeneous_demand_random_trials(self):
        rng = np.random.default_rng(0)
        for _ in range(8):
            n = int(rng.integers(2, 7))
            j = int(rng.integers(2, 9))
            server_pos = rng.uniform(0, 10, size=(n, 2))
            demand_pos = rng.uniform(0, 10, size=(j, 2))
            demand_rates = rng.uniform(0.2, 2.0, size=j)
            result = solve_hqm(server_pos, demand_pos, demand_rates, 1.0)
            expected = erlang_b(n, demand_rates.sum())
            assert result["p_loss"] == pytest.approx(expected, abs=1e-6)

    def test_different_mu(self):
        rng = np.random.default_rng(1)
        n, j, mu = 4, 5, 2.3
        server_pos = rng.uniform(0, 10, size=(n, 2))
        demand_pos = rng.uniform(0, 10, size=(j, 2))
        demand_rates = rng.uniform(0.2, 2.0, size=j)
        result = solve_hqm(server_pos, demand_pos, demand_rates, mu)
        expected = erlang_b(n, demand_rates.sum() / mu)
        assert result["p_loss"] == pytest.approx(expected, abs=1e-6)


# ==========================================================================
# Weitere Erhaltungssätze (Flow-Conservation, Dispatch-Conservation)
# ==========================================================================

class TestConservationLaws:
    def test_workload_matches_offered_load(self):
        rng = np.random.default_rng(2)
        for _ in range(6):
            n = int(rng.integers(2, 7))
            j = int(rng.integers(2, 9))
            mu = 1.7
            server_pos = rng.uniform(0, 10, size=(n, 2))
            demand_pos = rng.uniform(0, 10, size=(j, 2))
            demand_rates = rng.uniform(0.2, 2.0, size=j)
            result = solve_hqm(server_pos, demand_pos, demand_rates, mu)
            lhs = result["workload"].sum()
            rhs = (demand_rates.sum() / mu) * (1 - result["p_loss"])
            assert lhs == pytest.approx(rhs, abs=1e-6)

    def test_dispatch_frequency_sums_to_one_minus_p_loss(self):
        rng = np.random.default_rng(3)
        for _ in range(6):
            n = int(rng.integers(2, 7))
            j = int(rng.integers(2, 9))
            server_pos = rng.uniform(0, 10, size=(n, 2))
            demand_pos = rng.uniform(0, 10, size=(j, 2))
            demand_rates = rng.uniform(0.2, 2.0, size=j)
            result = solve_hqm(server_pos, demand_pos, demand_rates, 1.0)
            col_sums = result["dispatch_freq"].sum(axis=0)
            assert np.allclose(col_sums, 1 - result["p_loss"], atol=1e-6)

    def test_pi_is_a_valid_probability_distribution(self):
        rng = np.random.default_rng(4)
        for _ in range(10):
            n = int(rng.integers(2, 8))
            j = int(rng.integers(2, 10))
            server_pos = rng.uniform(0, 10, size=(n, 2))
            demand_pos = rng.uniform(0, 10, size=(j, 2))
            demand_rates = rng.uniform(0.2, 2.0, size=j)
            result = solve_hqm(server_pos, demand_pos, demand_rates, 1.0)
            assert result["pi"].sum() == pytest.approx(1.0, abs=1e-9)
            assert (result["pi"] >= -1e-12).all()


# ==========================================================================
# Präferenzlisten
# ==========================================================================

def test_preference_list_is_sorted_by_distance():
    server_pos = [(0, 0), (10, 0), (5, 0)]
    demand_pos = [(4, 0)]
    prefs, dists = build_preference_lists(server_pos, demand_pos)
    ordered_dists = dists[0][prefs[0]]
    assert list(ordered_dists) == sorted(ordered_dists)


# ==========================================================================
# Standortstrategien
# ==========================================================================

class TestLocationStrategies:
    def test_greedy_mclp_returns_requested_count(self):
        rng = np.random.default_rng(5)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        chosen = greedy_mclp(candidates, demand_pos, weights, n_servers=4, time_threshold=3.0)
        assert len(chosen) == 4
        assert len(set(chosen)) == 4  # keine Duplikate

    def test_greedy_mclp_prefers_high_demand_coverage(self):
        """Ein einzelner Kandidat, der viel Nachfrage abdeckt, muss vor einem
        gewählt werden, der wenig abdeckt."""
        candidates = np.array([[0.0, 0.0], [10.0, 10.0]])
        demand_pos = np.array([[0.1, 0.1], [0.2, 0.2], [0.1, 0.2], [9.9, 9.9]])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        chosen = greedy_mclp(candidates, demand_pos, weights, n_servers=1, time_threshold=1.0)
        assert chosen == [0]  # deckt 3 von 4 Nachfragepunkten ab, Kandidat 1 nur 1

    def test_local_search_never_worsens_the_objective(self):
        rng = np.random.default_rng(6)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        initial = list(range(4))
        _, history = local_search(initial, candidates, demand_pos, weights, mu=1.0, max_iter=15)
        assert all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1)), \
            "Zielgröße darf während der lokalen Suche nie steigen"

    def test_local_search_finds_the_true_optimum_on_a_small_enumerable_case(self):
        """Stärkerer Test als ein willkürlicher Verbesserungs-Schwellenwert:
        Bei nur 6 Kandidaten und 2 zu wählenden Standorten lassen sich alle
        C(6,2)=15 Kombinationen per Brute Force durchprobieren - die lokale
        Suche muss von einer offensichtlich schlechten Startkonfiguration
        (beide Fahrzeuge weit weg von aller Nachfrage) trotzdem beim
        nachweislich global besten Ergebnis landen."""
        import itertools

        candidates = np.array([[0, 0], [0.5, 0], [1, 0], [5, 5], [9, 9], [9, 0]], dtype=float)
        demand_pos = np.array([[5, 5], [5.2, 5.1], [4.8, 4.9], [9, 9]], dtype=float)
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        bad_start = [0, 1]  # beide weit weg von aller Nachfrage

        brute_force_best = min(
            (hqm_objective(list(combo), candidates, demand_pos, weights, 1.0), combo)
            for combo in itertools.combinations(range(len(candidates)), 2)
        )

        final, history = local_search(bad_start, candidates, demand_pos, weights, mu=1.0)
        assert sorted(final) == sorted(brute_force_best[1])
        assert history[-1] == pytest.approx(brute_force_best[0])
        assert history[-1] < history[0]  # tatsächlich eine Verbesserung ggü. dem schlechten Start


# ==========================================================================
# Metaheuristiken (Genetischer Algorithmus, Ant-Colony-Optimization)
# ==========================================================================

class TestMetaheuristics:
    def test_ga_returns_requested_count_and_no_duplicates(self):
        rng = np.random.default_rng(10)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        chosen, _, _ = genetic_algorithm(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=1)
        assert len(chosen) == 4
        assert len(set(chosen)) == 4

    def test_aco_returns_requested_count_and_no_duplicates(self):
        rng = np.random.default_rng(11)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        chosen, _, _ = ant_colony_optimization(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=1)
        assert len(chosen) == 4
        assert len(set(chosen)) == 4

    def test_ga_history_never_worsens(self):
        rng = np.random.default_rng(12)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        _, history, _ = genetic_algorithm(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=2)
        assert all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1)), \
            "Elitistische Nachfolge: das bislang beste Ergebnis darf sich nie verschlechtern"

    def test_aco_history_never_worsens(self):
        rng = np.random.default_rng(13)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        _, history, _ = ant_colony_optimization(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=2)
        assert all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1)), \
            "Das bislang beste Ergebnis darf sich nie verschlechtern"

    def test_ga_finds_the_true_optimum_on_a_small_enumerable_case(self):
        """Dieselbe scharfe Brute-Force-Referenz wie fuer die lokale Suche
        (siehe test_local_search_finds_the_true_optimum_on_a_small_enumerable_case) -
        bei nur C(6,2)=15 Kombinationen sollte ein GA mit Populationsgroesse 20
        das global beste Ergebnis zuverlaessig finden."""
        import itertools

        candidates = np.array([[0, 0], [0.5, 0], [1, 0], [5, 5], [9, 9], [9, 0]], dtype=float)
        demand_pos = np.array([[5, 5], [5.2, 5.1], [4.8, 4.9], [9, 9]], dtype=float)
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        brute_force_best = min(
            hqm_objective(list(combo), candidates, demand_pos, weights, 1.0)
            for combo in itertools.combinations(range(len(candidates)), 2)
        )

        _, history, _ = genetic_algorithm(candidates, demand_pos, weights, mu=1.0, n_servers=2, seed=3)
        assert history[-1] == pytest.approx(brute_force_best)

    def test_aco_finds_the_true_optimum_on_a_small_enumerable_case(self):
        import itertools

        candidates = np.array([[0, 0], [0.5, 0], [1, 0], [5, 5], [9, 9], [9, 0]], dtype=float)
        demand_pos = np.array([[5, 5], [5.2, 5.1], [4.8, 4.9], [9, 9]], dtype=float)
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        brute_force_best = min(
            hqm_objective(list(combo), candidates, demand_pos, weights, 1.0)
            for combo in itertools.combinations(range(len(candidates)), 2)
        )

        _, history, _ = ant_colony_optimization(candidates, demand_pos, weights, mu=1.0, n_servers=2, seed=3)
        assert history[-1] == pytest.approx(brute_force_best)

    def test_ga_deterministic_with_fixed_seed(self):
        rng = np.random.default_rng(14)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        chosen1, hist1, _ = genetic_algorithm(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=99)
        chosen2, hist2, _ = genetic_algorithm(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=99)
        assert sorted(chosen1) == sorted(chosen2)
        assert hist1 == hist2

    def test_aco_deterministic_with_fixed_seed(self):
        rng = np.random.default_rng(15)
        candidates = rng.uniform(0, 10, size=(10, 2))
        demand_pos = rng.uniform(0, 10, size=(8, 2))
        weights = rng.uniform(0.5, 1.5, size=8)
        chosen1, hist1, _ = ant_colony_optimization(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=99)
        chosen2, hist2, _ = ant_colony_optimization(candidates, demand_pos, weights, mu=1.0, n_servers=4, seed=99)
        assert sorted(chosen1) == sorted(chosen2)
        assert hist1 == hist2


# ==========================================================================
# Modellgenerierung
# ==========================================================================

def test_generate_demand_points_shapes_and_weights_normalized():
    positions, weights = generate_demand_points(n_demand=15, n_peaks=2, peak_conc=0.5, seed=42)
    assert positions.shape == (15, 2)
    assert weights.shape == (15,)
    assert weights.sum() == pytest.approx(1.0)
    assert (positions >= 0).all()


def test_generate_candidate_sites_shape():
    demand_pos, _ = generate_demand_points(10, 2, 0.5, 1)
    sites = generate_candidate_sites(n_candidates=12, seed=1, demand_positions=demand_pos)
    assert sites.shape == (12, 2)


def test_scale_demand_to_utilization_matches_target():
    weights = np.array([0.2, 0.3, 0.5])
    n_servers, mu, target = 4, 1.5, 0.6
    scaled = scale_demand_to_utilization(weights, target, n_servers, mu)
    assert scaled.sum() / (n_servers * mu) == pytest.approx(target)


# ==========================================================================
# Kennzahlen
# ==========================================================================

def test_naive_self_assessed_art_uses_nearest_server_only():
    candidates = np.array([[0, 0], [10, 0]], dtype=float)
    demand_pos = np.array([[1, 0]], dtype=float)
    weights = np.array([1.0])
    art = naive_self_assessed_art([0, 1], candidates, demand_pos, weights)
    assert art == pytest.approx(1.0)  # naeher am Standort (0,0)


def test_hqm_summary_coverage_and_loss_in_valid_ranges():
    rng = np.random.default_rng(7)
    server_pos = rng.uniform(0, 10, size=(4, 2))
    demand_pos = rng.uniform(0, 10, size=(8, 2))
    weights = rng.uniform(0.3, 1.0, size=8)
    result = solve_hqm(server_pos, demand_pos, weights, mu=1.0)
    summary = hqm_summary(result, weights, time_threshold=4.0)
    assert 0.0 <= summary["coverage_pct"] <= 100.0
    assert 0.0 <= summary["p_loss"] <= 1.0
    assert summary["art_served"] >= 0.0


# ==========================================================================
# PDF-Export
# ==========================================================================

def test_pdf_export_returns_valid_bytes():
    rng = np.random.default_rng(8)
    candidates = rng.uniform(0, 10, size=(8, 2))
    demand_pos = rng.uniform(0, 10, size=(6, 2))
    weights = rng.uniform(0.3, 1.0, size=6)
    chosen = greedy_mclp(candidates, demand_pos, weights, n_servers=3, time_threshold=3.0)
    result = solve_hqm(candidates[chosen], demand_pos, weights, mu=1.0)
    summary = hqm_summary(result, weights, time_threshold=3.0)
    pdf_bytes = generate_location_plan_pdf("Test", chosen, candidates, summary, summary, 3, 0.5)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"
