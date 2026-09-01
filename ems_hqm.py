"""
Hypercube Queueing Model (LARSON 1974) - exakte Berechnung der stationären
Verteilung eines Rettungsdienstsystems mit N Fahrzeugen.

Jeder Fahrzeugstatus (frei/beschäftigt) ergibt zusammen 2^N Systemzustände
(daher "Hypercube"). Ein Notruf von Nachfragepunkt j wird an den nächst-
gelegenen FREIEN Wagen vermittelt (Präferenzliste nach Entfernung sortiert);
sind alle Wagen beschäftigt, geht der Anruf verloren (Verlustsystem, wie ein
klassisches Erlang-B-System). Die Zustandsübergänge bilden eine zeitkonti-
nuierliche Markov-Kette, deren stationäre Verteilung pi exakt über ein
lineares Gleichungssystem berechnet wird (kein Simulationsrauschen).

Validiert (siehe tests/test_model.py): Für ein System mit homogener
Bedienrate mu ist die Wahrscheinlichkeit "alle Fahrzeuge beschäftigt" IMMER
gleich der klassischen Erlang-B-Formel B(N, lambda/mu) - komplett unabhängig
von der räumlichen Anordnung oder den Präferenzlisten. Das folgt daraus, dass
die Anzahl beschäftigter Server für sich genommen eine Geburts-Todes-Kette
mit Rate lambda (Geburt, solange N nicht erreicht) bzw. k*mu (Tod bei k
beschäftigten Servern) bildet - unabhängig davon, WELCHE Server beschäftigt
sind. Diese Eigenschaft dient als scharfer, von der eigentlichen Implemen-
tierung unabhängiger Korrektheitstest (siehe tests/test_model.py).
"""

import numpy as np

MAX_STATES_HARD_LIMIT = 4096  # entspricht 12 Fahrzeugen - Notbremse gegen versehentlich riesige Zustandsräume


def build_preference_lists(server_pos, demand_pos):
    """Für jeden Nachfragepunkt: Server-Indizes nach Entfernung sortiert (nächster zuerst)."""
    server_pos = np.asarray(server_pos, dtype=float)
    demand_pos = np.asarray(demand_pos, dtype=float)
    dists = np.linalg.norm(demand_pos[:, None, :] - server_pos[None, :, :], axis=2)  # (J, N)
    prefs = np.argsort(dists, axis=1)
    return prefs, dists


def solve_hqm(server_pos, demand_pos, demand_rates, mu):
    """Berechnet die stationäre Verteilung sowie die daraus abgeleiteten
    Kennzahlen (Auslastung je Server, Zuteilungswahrscheinlichkeiten,
    System-Verlustwahrscheinlichkeit) für eine gegebene Server-Konfiguration.

    Rückgabe ist ein dict mit:
      pi            - stationäre Verteilung über alle 2^N Zustände
      workload      - Auslastung je Server (N,)
      dispatch_freq - dispatch_freq[n, j] = P(Server n bedient Nachfrage j)  (N, J)
      p_loss        - Wahrscheinlichkeit, dass ein Anruf verloren geht (alle Server belegt)
      dists         - Entfernungsmatrix Server<->Nachfrage (N, J), für Kennzahlen weiterverwendet
    """
    N = len(server_pos)
    J = len(demand_pos)
    n_states = 1 << N
    if n_states > MAX_STATES_HARD_LIMIT:
        raise ValueError(f"Zu viele Fahrzeuge für eine exakte Lösung ({N}, {n_states} Zustände).")

    prefs, dists = build_preference_lists(server_pos, demand_pos)
    demand_rates = np.asarray(demand_rates, dtype=float)

    # bit_matrix[s, n] = 1, wenn Server n im Zustand s beschäftigt ist.
    bit_matrix = ((np.arange(n_states)[:, None] >> np.arange(N)[None, :]) & 1).astype(bool)

    # responder[s, j] = Index des Servers, der im Zustand s Nachfrage j bedient (-1 = kein freier Server).
    responder = np.full((n_states, J), -1, dtype=int)
    free_in_pref_order = ~bit_matrix[:, prefs]  # (n_states, J, N): frei? in Präferenzreihenfolge
    has_free = free_in_pref_order.any(axis=2)
    first_free_rank = np.argmax(free_in_pref_order, axis=2)  # erster True-Index je (s, j)
    responder = np.where(has_free, prefs[np.arange(J)[None, :], first_free_rank], -1)

    Q = np.zeros((n_states, n_states))
    state_idx = np.arange(n_states)

    # Bedienende Abschlüsse: aus jedem Zustand mit beschäftigtem Server n -> Zustand ohne n, Rate mu.
    for n in range(N):
        busy_mask = bit_matrix[:, n]
        s_from = state_idx[busy_mask]
        s_to = s_from - (1 << n)
        Q[s_from, s_to] += mu

    # Ankünfte: aus jedem Zustand, in dem Nachfrage j von Server n bedient würde (n frei) -> Zustand mit n belegt.
    for j in range(J):
        resp = responder[:, j]
        valid = resp >= 0
        s_from = state_idx[valid]
        n_resp = resp[valid]
        s_to = s_from + (1 << n_resp)
        np.add.at(Q, (s_from, s_to), demand_rates[j])

    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    # pi Q = 0, sum(pi) = 1: eine Zeile der Normierungsbedingung ersetzt eine
    # (redundante) Gleichgewichtsgleichung - das Gleichungssystem pi*Q=0 hat
    # sonst Rang n_states-1 (Generator-Matrizen sind singulär).
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n_states)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()

    workload = (pi[:, None] * bit_matrix).sum(axis=0)

    dispatch_freq = np.zeros((N, J))
    for j in range(J):
        resp = responder[:, j]
        valid = resp >= 0
        np.add.at(dispatch_freq, (resp[valid], j), pi[valid])

    p_loss = float(pi[-1])  # letzter Zustand (alle Bits gesetzt) = alle Server belegt

    return {
        "pi": pi,
        "workload": workload,
        "dispatch_freq": dispatch_freq,
        "p_loss": p_loss,
        "dists": dists,
    }


def per_demand_expected_distance(dispatch_freq, dists):
    """Erwartete Distanz je Nachfragepunkt, bedingt darauf, dass ein Anruf
    von dort tatsaechlich bedient wird (0, falls ein Punkt nie bedient wird -
    Randfall bei extremer Auslastung). Wird von ems_evaluation.hqm_summary
    (fuer die angezeigte Reaktionszeit) und ems_location.hqm_objective (fuer
    die Zielgroesse der Standortsuche) verwendet - beide muessen das
    Ergebnis noch mit den Nachfragegewichten (Anrufraten) gewichten, sonst
    zaehlt jeder Nachfragepunkt gleich viel, unabhaengig von seinem
    tatsaechlichen Anrufvolumen - genau das war bei beiden Funktionen zuvor
    der Fall und fuehrte bei ungleich verteilter Nachfrage zu teils
    drastisch verzerrten Ergebnissen (siehe tests/test_model.py,
    test_*_weighted_by_demand_volume)."""
    served_mass_per_demand = dispatch_freq.sum(axis=0)  # (J,)
    distance_per_demand = np.zeros_like(served_mass_per_demand)
    nonzero = served_mass_per_demand > 1e-9
    distance_per_demand[nonzero] = (dispatch_freq * dists.T).sum(axis=0)[nonzero] / served_mass_per_demand[nonzero]
    return distance_per_demand, served_mass_per_demand


def erlang_b(n, offered_load):
    """Klassische Erlang-B-Verlustformel, stabile Rekursion.
    Dient hier v. a. als unabhängiger Korrektheitstest für solve_hqm (siehe
    Moduldoc oben) - wird aber auch für die Kalibrierung des Auslastungs-
    Reglers in ems_model.py verwendet."""
    b = 1.0
    for k in range(1, n + 1):
        b = (offered_load * b) / (k + offered_load * b)
    return b
