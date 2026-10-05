"""Unabhängige Orakel für das Hypercube Queueing Model und die Standortsuche.

Die vorhandene Erlang-B-Identität prüft nur P(alle belegt). Hier zusätzlich, auf anderem Rechenweg:
- explizite Zustandsaufzählung mit Schleifen und Gleichgewicht über den Nullraum (SVD) statt `np.linalg.solve`:
  stationäre Verteilung, Auslastung je Fahrzeug, Zuteilungswahrscheinlichkeiten, hqm_summary (Reaktionszeit, Abdeckung);
- Anzahl belegter Fahrzeuge = abgeschnittene Poisson-Verteilung (Geburts-Todes-Kette);
- Handbeispiel N=2 (Auslastungen 0,5 und 0,3, Verlust 0,2);
- Ereignissimulation (fester Zufallsstrom) mit Vier-Sigma-Band;
- Greedy-MCLP-Neufassung, Zielgröße und lokale Optimalität der Standortsuche, GA/ACO-Historie gegen Neuberechnung.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from ems_evaluation import hqm_summary, naive_self_assessed_art
from ems_hqm import erlang_b, solve_hqm
from ems_location import greedy_mclp, hqm_objective, local_search
from ems_metaheuristics import ant_colony_optimization, genetic_algorithm


def chain_oracle(server_pos, demand_pos, rates, mu):
    n_srv, n_dem = len(server_pos), len(demand_pos)
    dist = [[math.dist(demand_pos[j], server_pos[n]) for n in range(n_srv)] for j in range(n_dem)]
    pref = [sorted(range(n_srv), key=lambda n: (dist[j][n], n)) for j in range(n_dem)]
    states = list(itertools.product([0, 1], repeat=n_srv))  # 1 = belegt
    index = {s: i for i, s in enumerate(states)}

    def responder(s, j):
        return next((n for n in pref[j] if s[n] == 0), None)

    q = np.zeros((len(states), len(states)))
    for s in states:
        for n in range(n_srv):
            if s[n]:
                t = list(s)
                t[n] = 0
                q[index[s], index[tuple(t)]] += mu
        for j in range(n_dem):
            n = responder(s, j)
            if n is not None:
                t = list(s)
                t[n] = 1
                q[index[s], index[tuple(t)]] += rates[j]
    for i in range(len(states)):
        q[i, i] = -q[i].sum()
    pi = np.linalg.svd(q.T)[2][-1]
    pi = pi / pi.sum()
    workload = np.array([sum(pi[index[s]] for s in states if s[n]) for n in range(n_srv)])
    dispatch = np.zeros((n_srv, n_dem))
    for s in states:
        for j in range(n_dem):
            n = responder(s, j)
            if n is not None:
                dispatch[n, j] += pi[index[s]]
    return dict(states=states, pi=pi, workload=workload, dispatch=dispatch, dist=np.array(dist), p_loss=pi[index[(1,) * n_srv]])


def _objective_oracle(idx, sites, demand, rates, mu):
    o = chain_oracle(sites[list(idx)], demand, rates, mu)
    total = 0.0
    for j in range(len(demand)):
        served = o["dispatch"][:, j].sum()
        total += rates[j] * ((o["dispatch"][:, j] * o["dist"][j]).sum() / served if served > 1e-9 else 0.0)
    return total


def test_hand_example_two_servers_one_demand_node():
    r = solve_hqm([(0, 0), (5, 0)], [(1, 0)], [1.0], 1.0)
    assert r["workload"] == pytest.approx([0.5, 0.3])
    assert r["p_loss"] == pytest.approx(0.2)


def test_stationary_quantities_match_explicit_chain():
    rng = np.random.default_rng(2026)
    for it in range(40):
        n = int(rng.integers(1, 6))
        j = int(rng.integers(1, 8))
        mu = float(rng.choice([0.5, 1.0, 1.7, 3.0]))
        server_pos = rng.uniform(0, 10, size=(n, 2))
        demand_pos = rng.uniform(0, 10, size=(j, 2))
        rates = rng.uniform(0.1, 2.5, size=j) * (3.0 if it % 7 == 0 else 1.0)
        r = solve_hqm(server_pos, demand_pos, rates, mu)
        o = chain_oracle(server_pos, demand_pos, rates, mu)
        pi_demo = np.zeros(1 << n)
        for k, s in enumerate(o["states"]):
            pi_demo[sum(b << m for m, b in enumerate(s))] = o["pi"][k]
        assert np.abs(r["pi"] - pi_demo).max() < 1e-9
        assert np.abs(r["workload"] - o["workload"]).max() < 1e-9
        assert np.abs(r["dispatch_freq"] - o["dispatch"]).max() < 1e-9
        a = rates.sum() / mu
        busy = np.array([sum(s) for s in o["states"]])
        norm = sum(a**m / math.factorial(m) for m in range(n + 1))
        for k in range(n + 1):
            assert o["pi"][busy == k].sum() == pytest.approx(a**k / math.factorial(k) / norm, abs=1e-9)
        assert r["p_loss"] == pytest.approx(erlang_b(n, a), abs=1e-9)
        w = rates / rates.sum()
        summary = hqm_summary(r, w, 4.0)
        art = cov = 0.0
        for jj in range(j):
            served = o["dispatch"][:, jj].sum()
            art += w[jj] * ((o["dispatch"][:, jj] * o["dist"][jj]).sum() / served if served > 1e-9 else 0.0)
            cov += w[jj] * sum(o["dispatch"][m, jj] for m in range(n) if o["dist"][jj][m] <= 4.0)
        assert summary["art_served"] == pytest.approx(art, abs=1e-9)
        assert summary["coverage_pct"] == pytest.approx(cov * 100, abs=1e-7)


def test_event_simulation_agrees_within_four_sigma():
    rg = np.random.default_rng(100)
    n, j, mu, calls = 3, 4, 1.0, 60_000
    rr = np.random.default_rng(10)
    server_pos = rr.uniform(0, 10, size=(n, 2))
    demand_pos = rr.uniform(0, 10, size=(j, 2))
    rates = rr.uniform(0.3, 1.5, size=j) * (n * mu * 0.6 / (j * 0.9))
    r = solve_hqm(server_pos, demand_pos, rates, mu)
    dist = np.linalg.norm(demand_pos[:, None, :] - server_pos[None, :, :], axis=2)
    pref = [sorted(range(n), key=lambda m: (dist[jj][m], m)) for jj in range(j)]
    free_at = np.zeros(n)
    t = 0.0
    n_call = np.zeros(j)
    n_served = np.zeros((n, j))
    lost = 0
    gaps = rg.exponential(1 / rates.sum(), size=calls)
    nodes = rg.choice(j, p=rates / rates.sum(), size=calls)
    durations = rg.exponential(1 / mu, size=calls)
    for gap, node, dur in zip(gaps, nodes, durations):
        t += gap
        n_call[node] += 1
        for m in pref[node]:
            if free_at[m] <= t:
                free_at[m] = t + dur
                n_served[m, node] += 1
                break
        else:
            lost += 1
    emp = n_served / n_call[None, :]
    assert np.abs(emp - r["dispatch_freq"]).max() < 4 * math.sqrt(0.25 / n_call.min())
    assert lost / calls == pytest.approx(r["p_loss"], abs=4 * math.sqrt(0.25 / calls))


def _greedy_oracle(sites, demand, w, k, threshold):
    cover = [[math.dist(demand[j], sites[c]) <= threshold for c in range(len(sites))] for j in range(len(demand))]
    chosen, covered = [], [False] * len(demand)
    for _ in range(k):
        best, best_gain = None, -1.0
        for c in range(len(sites)):
            if c in chosen:
                continue
            gain = sum(w[j] for j in range(len(demand)) if cover[j][c] and not covered[j])
            if gain > best_gain:
                best_gain, best = gain, c
        chosen.append(best)
        covered = [covered[j] or cover[j][best] for j in range(len(demand))]
    return chosen


def test_location_strategies_against_independent_evaluation():
    rng = np.random.default_rng(55)
    for _ in range(12):
        n = int(rng.integers(2, 5))
        n_cand = int(rng.integers(n + 1, 8))
        j = int(rng.integers(3, 10))
        sites = rng.uniform(0, 10, size=(n_cand, 2))
        demand = rng.uniform(0, 10, size=(j, 2))
        mu = float(rng.choice([0.5, 1.0, 2.0]))
        rates = rng.uniform(0.2, 1.5, size=j) * (n * mu * float(rng.choice([0.2, 0.5, 0.9])) / j)
        thr = float(rng.choice([2.0, 3.5, 6.0]))
        start = greedy_mclp(sites, demand, rates, n, thr)
        assert start == _greedy_oracle(sites, demand, rates, n, thr)
        assert hqm_objective(start, sites, demand, rates, mu) == pytest.approx(_objective_oracle(start, sites, demand, rates, mu), abs=1e-9)
        assert naive_self_assessed_art(start, sites, demand, rates) == pytest.approx(
            sum(rates[jj] * min(math.dist(demand[jj], sites[c]) for c in start) for jj in range(j)) / rates.sum(), abs=1e-9)
        final, history = local_search(start, sites, demand, rates, mu)
        value = _objective_oracle(final, sites, demand, rates, mu)
        assert history[-1] == pytest.approx(value, abs=1e-9)
        for pos in range(n):  # lokales Optimum unter Einzeltausch
            for c in set(range(n_cand)) - set(final):
                cand = list(final)
                cand[pos] = c
                assert _objective_oracle(cand, sites, demand, rates, mu) >= value - 1e-8
        best = min(_objective_oracle(combo, sites, demand, rates, mu) for combo in itertools.combinations(range(n_cand), n))
        for fn in (genetic_algorithm, ant_colony_optimization):
            chosen, hist, _ = fn(sites, demand, rates, mu, n, 7)
            assert len(set(chosen)) == n
            assert hist[-1] == pytest.approx(_objective_oracle(chosen, sites, demand, rates, mu), abs=1e-9)
            assert hist[-1] >= best - 1e-8
