"""
Rettungsdienst-Standortplanung (Hypercube Queueing Model) - interaktive Demo
Sebastian Hanisch - Operations Research und Machine Learning

Angeregt durch die Dissertation von BLANK, F. (Julius-Maximilians-Universitaet
Wuerzburg, 2021): "The use of the Hypercube Queueing Model for the location
optimization decision of Emergency Medical Service systems". Diese Demo
implementiert den Kern des dort behandelten Modells (LARSON 1974) sowie einen
vereinfachten Ausschnitt der dort entwickelten Standortoptimierung - inklusive
der beiden dort verglichenen Metaheuristiken (Genetischer Algorithmus und
Ant-Colony-Optimization, siehe ems_metaheuristics.py) - bewusst ohne die volle
Modelltiefe der Dissertation (Goal Programming, robuste Optimierung ueber
mehrere Szenarien), um das Kernprinzip greifbar zu machen. Fuer die
Rechenzeit bei hohen Fahrzeugzahlen sorgt eine einfache Memoisierung der
HQM-Auswertungen (siehe local_search/cache in ems_location.py) - eine
schlankere Variante von Blanks "Dynamic Caching Strategy", nicht deren
volle Umsetzung.

Kernidee: Ein Fahrzeug (Server) ist nicht immer verfuegbar, wenn ein Notruf
eingeht - es koennte gerade im Einsatz sein. Das Hypercube Queueing Model
berechnet ueber eine zeitkontinuierliche Markov-Kette (2^N Systemzustaende
bei N Fahrzeugen) exakt, wie oft welches Fahrzeug tatsaechlich antwortet,
und liefert daraus realistische Kennzahlen (Reaktionszeit, Abdeckung,
Verlustwahrscheinlichkeit) - im Gegensatz zu klassischen Coverage-Modellen,
die Verfuegbarkeit ignorieren. Verglichen werden eine kongestionsblinde
Greedy-Standortwahl (klassisches Maximal Covering Location Problem) und eine
HQM-bewusste lokale Suche, die genau diesen Unterschied einpreist.

Mathematisch validiert (siehe tests/test_model.py): fuer ein System mit
homogener Bedienrate entspricht die berechnete Verlustwahrscheinlichkeit
IMMER exakt der klassischen Erlang-B-Formel, unabhaengig von der raeumlichen
Anordnung - ein von der eigentlichen Implementierung unabhaengiger, scharfer
Korrektheitstest.

Lauffaehig mit: streamlit run app.py
"""

import os

# MUSS vor dem ersten numpy-Import passieren: OpenBLAS liest die Thread-Anzahl
# beim Laden der Bibliothek (also beim ersten numpy-Import ueberhaupt), nicht
# lazy bei jedem Aufruf - nachtraeglich gesetzt wirkt die Variable nicht mehr.
# Grund: solve_hqm() loest viele KLEINE Gleichungssysteme (bis 2^8=256x256,
# siehe ems_hqm.py) - fuer solche Groessen frisst der Synchronisationsoverhead
# von OpenBLAS' Multi-Thread-Koordination mehr Zeit als er einspart. Per
# cProfile gemessen: ~20x schneller pro Loesung mit einem Thread statt der
# OpenBLAS-Standardeinstellung (bis zu 24 Threads).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import streamlit as st

from ems_constants import DEFAULT_TIME_THRESHOLD, METAHEURISTIC_SEED_OFFSET
from ems_evaluation import hqm_summary, naive_self_assessed_art
from ems_hqm import solve_hqm
from ems_location import greedy_mclp, local_search
from ems_metaheuristics import ant_colony_optimization, genetic_algorithm
from ems_model import generate_candidate_sites, generate_demand_points, scale_demand_to_utilization
from ems_pdf_export import generate_location_plan_pdf
from ems_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ems_visualization import convergence_figure, map_figure, multi_convergence_figure, workload_figure


@st.cache_data(show_spinner="Standorte werden optimiert …")
def _compute(n_servers, n_candidates, n_demand, n_peaks, peak_conc, seed, service_time, utilization, time_threshold, cache_key):
    mu = 1.0 / service_time
    demand_positions, demand_weights_rel = generate_demand_points(n_demand, n_peaks, peak_conc, seed)
    candidate_sites = generate_candidate_sites(n_candidates, seed, demand_positions)
    demand_weights = scale_demand_to_utilization(demand_weights_rel, utilization, n_servers, mu)

    naive_indices = greedy_mclp(candidate_sites, demand_positions, demand_weights, n_servers, time_threshold)
    hqm_indices, history = local_search(naive_indices, candidate_sites, demand_positions, demand_weights, mu)

    hqm_naive = solve_hqm(candidate_sites[naive_indices], demand_positions, demand_weights, mu)
    hqm_optimized = solve_hqm(candidate_sites[hqm_indices], demand_positions, demand_weights, mu)

    return {
        "demand_positions": demand_positions,
        "demand_weights": demand_weights,
        "candidate_sites": candidate_sites,
        "naive_indices": naive_indices,
        "hqm_indices": hqm_indices,
        "history": history,
        "hqm_naive": hqm_naive,
        "hqm_optimized": hqm_optimized,
        "mu": mu,
    }


@st.cache_data(show_spinner="Genetischer Algorithmus und Ant-Colony-Optimization werden verglichen …")
def _compute_metaheuristics(n_servers, n_candidates, n_demand, n_peaks, peak_conc, seed, service_time, utilization, cache_key):
    mu = 1.0 / service_time
    demand_positions, demand_weights_rel = generate_demand_points(n_demand, n_peaks, peak_conc, seed)
    candidate_sites = generate_candidate_sites(n_candidates, seed, demand_positions)
    demand_weights = scale_demand_to_utilization(demand_weights_rel, utilization, n_servers, mu)

    meta_seed = int(seed) + METAHEURISTIC_SEED_OFFSET
    ga_indices, ga_history, ga_time = genetic_algorithm(candidate_sites, demand_positions, demand_weights, mu, n_servers, meta_seed)
    aco_indices, aco_history, aco_time = ant_colony_optimization(candidate_sites, demand_positions, demand_weights, mu, n_servers, meta_seed)

    return {
        "ga_indices": ga_indices, "ga_history": ga_history, "ga_time": ga_time,
        "aco_indices": aco_indices, "aco_history": aco_history, "aco_time": aco_time,
    }


st.set_page_config(page_title="Rettungsdienst-Standortplanung – Sebastian Hanisch", layout="wide")

st.title("🚑 Rettungsdienst-Standortplanung (Hypercube Queueing Model)")
st.markdown(
    """
Interaktive Demo zur Standortplanung von Rettungsfahrzeugen. Kernfrage: Wie stark
ändert sich die beste Standortwahl, wenn man berücksichtigt, dass ein Fahrzeug oft
gerade **im Einsatz und nicht verfügbar** ist? Verglichen werden eine klassische,
**kongestionsblinde Coverage-Strategie** und eine **HQM-bewusste** Strategie, die
über das **Hypercube Queueing Model** (LARSON, 1974 - eine zeitkontinuierliche
Markov-Kette über alle Fahrzeug-Verfügbarkeitszustände) die tatsächliche
Erreichbarkeit exakt berechnet. Angeregt durch die Dissertation von Felix Blank
(Uni Würzburg, 2021) zu genau diesem Thema - Details im Expander
"Wie funktioniert diese Demo?" unten.
"""
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_col1, preset_col2, preset_col3 = st.columns(3)
with preset_col1:
    st.button(
        "🌙 Ruhige Nachtschicht", use_container_width=True,
        on_click=apply_preset, args=(4, 10, 12, 2, 0.5, 42, 0.5, 0.25, 4.0),
        help="Niedrige Auslastung - naive und HQM-bewusste Standortwahl unterscheiden sich kaum.",
    )
with preset_col2:
    st.button(
        "🔥 Stoßzeit", use_container_width=True,
        on_click=apply_preset, args=(4, 10, 12, 2, 0.5, 42, 0.5, 0.75, 4.0),
        help="Hohe Auslastung - hier zeigt sich der Unterschied zwischen beiden Strategien deutlich.",
    )
with preset_col3:
    st.button(
        "🏙️ Größeres System", use_container_width=True,
        on_click=apply_preset, args=(6, 14, 18, 3, 0.4, 7, 0.4, 0.7, 3.5),
        help="Mehr Fahrzeuge und Nachfragepunkte - mehr Systemzustände (2^N), die exakte Lösung dauert dadurch etwas länger.",
    )

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_servers = st.slider("Anzahl Fahrzeuge", *bounds("n_servers_slider"), key="n_servers_slider",
                           help="Mehr Fahrzeuge bedeuten mehr Systemzustände (2^N) - die exakte Lösung wird langsamer.")
    n_candidates = st.slider("Anzahl Standort-Kandidaten", *bounds("n_candidates_slider"), key="n_candidates_slider")
    n_demand = st.slider("Anzahl Nachfragepunkte", *bounds("n_demand_slider"), key="n_demand_slider")
    n_peaks = st.slider("Anzahl Nachfrage-Schwerpunkte", *bounds("n_peaks_slider"), key="n_peaks_slider")
    peak_conc = st.slider("Konzentration der Schwerpunkte", *bounds("peak_conc_slider"), step=0.05, key="peak_conc_slider")
    seed_lo, seed_hi = bounds("seed_input")
    seed = st.number_input("Zufalls-Seed", min_value=seed_lo, max_value=seed_hi, step=1, key="seed_input")

    st.markdown("**Betrieb**")
    service_time = st.slider(
        "Mittlere Einsatzdauer (Stunden)", *bounds("service_time_slider"), step=0.1, key="service_time_slider",
        help="Zeit, die ein Fahrzeug je Einsatz gebunden ist (Anfahrt, Vor-Ort-Versorgung, Transport, Rüstzeit).",
    )
    utilization = st.slider(
        "Ziel-Systemauslastung", *bounds("utilization_slider"), step=0.05, key="utilization_slider",
        help="Anrufrate relativ zur Gesamtkapazität (Anzahl Fahrzeuge / mittlere Einsatzdauer). "
             "Steuert, wie oft Fahrzeuge tatsächlich beschäftigt sind, wenn ein neuer Notruf eingeht.",
    )
    time_threshold = st.slider(
        "Zeitschwelle für 'Abdeckung'", *bounds("time_threshold_slider"), step=0.5, key="time_threshold_slider",
        help="Vereinfachend: 1 Karteneinheit ≈ 1 Zeiteinheit Fahrzeit (wie bei den anderen synthetischen Karten dieses Portfolios).",
    )

    st.button("🎲 Neues Szenario generieren", use_container_width=True, on_click=randomize_seed)

if n_candidates <= n_servers:
    st.warning("Die Anzahl Standort-Kandidaten muss größer als die Anzahl Fahrzeuge sein. Bitte in der Seitenleiste anpassen.")
    st.stop()

sync_query_params(n_servers, n_candidates, n_demand, n_peaks, peak_conc, seed, service_time, utilization, time_threshold)

cache_key = (n_servers, n_candidates, n_demand, n_peaks, peak_conc, int(seed), service_time, utilization, time_threshold)
data = _compute(n_servers, n_candidates, n_demand, n_peaks, peak_conc, int(seed), service_time, utilization, time_threshold, cache_key)

demand_positions = data["demand_positions"]
demand_weights = data["demand_weights"]
candidate_sites = data["candidate_sites"]
naive_indices = data["naive_indices"]
hqm_indices = data["hqm_indices"]
mu = data["mu"]

naive_self_assessed = naive_self_assessed_art(naive_indices, candidate_sites, demand_positions, demand_weights)
metrics_naive = hqm_summary(data["hqm_naive"], demand_weights, time_threshold)
metrics_hqm = hqm_summary(data["hqm_optimized"], demand_weights, time_threshold)

st.markdown("## 🎯 Naiv vs. HQM-bewusst")

optimism_gap_pct = (metrics_naive["art_served"] - naive_self_assessed) / naive_self_assessed * 100 if naive_self_assessed > 0 else 0.0
if optimism_gap_pct > 3:
    st.warning(
        f"⚠️ Die naive Standortwahl schätzt ihre eigene Reaktionszeit auf **{naive_self_assessed:.2f}** "
        f"(reine Entfernung zum nächsten Fahrzeug, Verfügbarkeit ignoriert) - tatsächlich liegt sie, korrekt "
        f"über das HQM berechnet, bei **{metrics_naive['art_served']:.2f}** ({optimism_gap_pct:+.0f}%). "
        "Ein kongestionsblindes Modell ist an dieser Stelle systematisch zu optimistisch."
    )

m1, m2, m3 = st.columns(3)
m1.metric(
    "Reaktionszeit (real, HQM-bewusst)", f"{metrics_hqm['art_served']:.2f}",
    delta=f"{metrics_hqm['art_served'] - metrics_naive['art_served']:+.2f} ggü. naiver Standortwahl (real)",
    delta_color="inverse",
)
m2.metric(
    "Abdeckung", f"{metrics_hqm['coverage_pct']:.0f}%",
    delta=f"{metrics_hqm['coverage_pct'] - metrics_naive['coverage_pct']:+.0f} Prozentpunkte ggü. naiv",
)
m3.metric("Verlustwahrscheinlichkeit", f"{metrics_hqm['p_loss']*100:.1f}%")
m3.caption(
    "Bleibt bei beiden Strategien identisch - sie hängt bei diesem Modell nachweislich nur von "
    "Fahrzeuganzahl und Systemauslastung ab, nicht von der Standortwahl (Erlang-B-Identität, "
    "siehe „📐 Mathematische Formulierung“ unten)."
)

map_col1, map_col2 = st.columns(2)
with map_col1:
    st.plotly_chart(
        map_figure(demand_positions, demand_weights, candidate_sites, naive_indices, "Naive Coverage-Wahl (Greedy-MCLP)", time_threshold),
        use_container_width=True,
    )
    st.plotly_chart(workload_figure(metrics_naive["workload"], "Auslastung je Fahrzeug (naiv)"), use_container_width=True)
with map_col2:
    st.plotly_chart(
        map_figure(demand_positions, demand_weights, candidate_sites, hqm_indices, "HQM-bewusste Standortwahl", time_threshold),
        use_container_width=True,
    )
    st.plotly_chart(workload_figure(metrics_hqm["workload"], "Auslastung je Fahrzeug (HQM-bewusst)"), use_container_width=True)

pdf_bytes = generate_location_plan_pdf("Rettungsdienst-Standortplanung", hqm_indices, candidate_sites, metrics_naive, metrics_hqm, n_servers, utilization)
st.download_button(
    "📄 Standortplan als PDF herunterladen", data=pdf_bytes,
    file_name="standortplan.pdf", mime="application/pdf",
)

st.markdown("---")

with st.expander("🔧 Lokale Suche: wie kam die HQM-bewusste Lösung zustande?"):
    st.plotly_chart(convergence_figure(data["history"], "Zielgröße über die Tausch-Iterationen"), use_container_width=True)
    st.caption(
        "Startpunkt ist die naive Greedy-Lösung. In jeder Iteration wird der Standorttausch gewählt, der die "
        "über das HQM berechnete Zielgröße (gewichtete Reaktionszeit + Strafterm für verlorene Anrufe) am "
        "stärksten verbessert - bis kein verbessernder Tausch mehr gefunden wird."
    )

with st.expander("🧬 Metaheuristik-Vergleich: Genetischer Algorithmus vs. Ant-Colony-Optimization"):
    st.markdown(
        """
Neben der lokalen Suche oben implementiert diese Demo zwei weitere Verfahren, die Blank
(Dissertation, Kapitel 6) für dieselbe Aufgabenstellung explizit miteinander vergleicht: einen
**genetischen Algorithmus (GA)** und **Ant-Colony-Optimization (ACO)** - beide angewandt auf
dieselbe HQM-Zielgröße wie die lokale Suche, aber ohne deren Startpunkt (naive Lösung) zu kennen.
Blank kommt in eigenen Experimenten zu einem klaren Ergebnis (S. 168f. der Dissertation):

> *"the ACO performs better than the GA in the proposed experimental setting"* - die ACO reagiert
> robuster auf die Parameterwahl und wird deshalb für den Rest der Arbeit als alleinige
> Optimierungstechnik verwendet.

Blank betont dabei selbst, dass ein GA-vs-ACO-Vergleich stark vom konkreten Anwendungsfall abhängt
- und die breitere Literatur bestätigt das: IANNONI, MORABITO UND SAYDAM (2008) kombinieren das
Hypercube-Modell umgekehrt mit einem GA (eine von Blanks eigenen Inspirationsquellen), und für
andere EMS-Standort-/Einsatzmodelle (z.B. BENABDOUALLAH UND BOJJI zum "Dynamic Double Standard
Model") schneidet in direkten Vergleichen gerade der **GA besser** ab als die ACO. Es gibt also
keinen universellen Sieger - mit dem Button unten lässt sich das eigene Szenario direkt selbst
nachvollziehen, inklusive Rechenzeit, denn genau die war neben der Lösungsgüte ausschlaggebend für
Blanks Entscheidung.

Beide Verfahren sind hier zusätzlich **memetisch hybridisiert**: das jeweils beste Individuum
bzw. die beste Ameise wird je Generation/Iteration mit ein paar Schritten lokaler Suche
nachpoliert - eine Erweiterung, die weder Blank noch die oben genannten Vergleichsstudien
verwenden, aber laut einer separaten Metaheuristik-Vergleichsstudie für Ambulanz-Allokation
(GECCO 2023) reine populationsbasierte Verfahren übertrifft.
"""
    )
    run_comparison = st.button("🧬 Vergleich berechnen (bei vielen Fahrzeugen ca. 1-2 Sekunden)")
    if run_comparison:
        st.session_state["metaheuristic_cache_key"] = cache_key

    if st.session_state.get("metaheuristic_cache_key") == cache_key:
        meta = _compute_metaheuristics(n_servers, n_candidates, n_demand, n_peaks, peak_conc, int(seed), service_time, utilization, cache_key)

        st.plotly_chart(
            multi_convergence_figure(
                {
                    "Lokale Suche": data["history"],
                    "Genetischer Algorithmus": meta["ga_history"],
                    "Ant-Colony-Optimization": meta["aco_history"],
                },
                "Zielgröße über Iterationen/Generationen je Verfahren",
            ),
            use_container_width=True,
        )

        ga_obj, aco_obj, ls_obj = meta["ga_history"][-1], meta["aco_history"][-1], data["history"][-1]
        t1, t2, t3 = st.columns(3)
        t1.metric("Lokale Suche", f"{ls_obj:.3f}", help="Zielgröße (gewichtete Distanz + Verlust-Strafe)")
        t2.metric("Genetischer Algorithmus", f"{ga_obj:.3f}", delta=f"{ga_obj - ls_obj:+.3f} ggü. lokaler Suche", delta_color="inverse")
        t2.caption(f"Rechenzeit: {meta['ga_time']*1000:.0f} ms")
        t3.metric("Ant-Colony-Optimization", f"{aco_obj:.3f}", delta=f"{aco_obj - ls_obj:+.3f} ggü. lokaler Suche", delta_color="inverse")
        t3.caption(f"Rechenzeit: {meta['aco_time']*1000:.0f} ms")

        st.caption(
            "Die x-Achse ist zwischen den Verfahren nicht direkt vergleichbar (ein GA-Generationsschritt oder eine "
            "ACO-Iteration wertet deutlich mehr Konfigurationen aus als ein einzelner Tausch-Schritt der lokalen "
            "Suche) - Zielgröße und Rechenzeit oben sind der faire Vergleich. Alle drei Verfahren nutzen dieselbe "
            "HQM-Zielgröße, GA und ACO starten aber - anders als die lokale Suche - bewusst nicht von der naiven "
            "Lösung, sondern von zufälligen Konfigurationen."
        )
    else:
        st.caption("Noch nicht berechnet - Button oben klicken.")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        """
**Die Problemstellung:** Rettungsfahrzeuge werden an einer begrenzten Anzahl Standorten
stationiert. Bei einem Notruf wird das nächstgelegene **freie** Fahrzeug entsandt - ist
es beschäftigt, übernimmt das nächste freie auf der Präferenzliste, oder der Anruf geht
verloren, falls alle Fahrzeuge belegt sind.

**Naive Coverage-Wahl (Greedy-MCLP):** Klassischer Greedy-Algorithmus für das Maximal
Covering Location Problem - wählt wiederholt den Standort, der die meiste noch nicht
abgedeckte Nachfrage innerhalb einer Zeitschwelle abdeckt. Kennt keine Warteschlangentheorie:
geht implizit von 100% Verfügbarkeit aus.

**Hypercube Queueing Model (HQM):** Bei N Fahrzeugen gibt es 2^N mögliche
Verfügbarkeitszustände (jedes Fahrzeug frei oder beschäftigt) - daher "Hypercube". Diese
Zustände bilden eine zeitkontinuierliche Markov-Kette: ein Zustand wechselt, wenn ein
Fahrzeug einen Einsatz abschließt (Rate μ) oder ein neuer Notruf einem freien Fahrzeug
zugewiesen wird. Die stationäre Verteilung dieser Kette wird **exakt** über ein lineares
Gleichungssystem berechnet (kein Simulationsrauschen) und liefert daraus: wie oft welches
Fahrzeug tatsächlich antwortet, die reale Reaktionszeit, die Abdeckung und die
Verlustwahrscheinlichkeit.

**HQM-bewusste lokale Suche:** Startet bei der naiven Lösung und tauscht wiederholt einen
Standort gegen einen besseren, sofern das die über das HQM berechnete Zielgröße verbessert -
bis keine Verbesserung mehr gefunden wird. Ohne Strafterm für verlorene Anrufe könnte die
Suche eine hohe Verlustwahrscheinlichkeit in Kauf nehmen, solange die noch bedienten Anrufe
kurze Wege haben - deshalb fließt die Verlustwahrscheinlichkeit direkt in die Zielgröße ein.

**Genetischer Algorithmus und Ant-Colony-Optimization:** Zwei weitere, populationsbasierte
Metaheuristiken für dieselbe Zielgröße (siehe Expander "Metaheuristik-Vergleich" oben). Der
GA lässt eine Population von Standort-Konfigurationen über Generationen "evolvieren"
(fitness-proportionale Selektion, Crossover, Mutation). Die ACO lässt Standort-Konfigurationen
schrittweise durch Pheromon-Werte je Kandidat entstehen, die nach jeder Iteration entsprechend
der Lösungsgüte verstärkt bzw. verdunstet werden. Beide sind unabhängig von der lokalen Suche
und kennen deren Startpunkt nicht - beide sind aber **memetisch hybridisiert**: das jeweils
beste Individuum bzw. die beste Ameise wird je Generation/Iteration mit ein paar Schritten
lokaler Suche nachpoliert, bevor Selektion bzw. Pheromon-Update darauf aufbauen. Mehrere Studien
(u.a. eine Metaheuristik-Vergleichsstudie für Ambulanz-Allokation, GECCO 2023) finden, dass
diese Hybridisierung reine populationsbasierte Verfahren übertrifft.

**Warum der Unterschied mit der Auslastung wächst:** Bei niedriger Auslastung ist fast immer
ein Fahrzeug frei - naive und HQM-bewusste Standortwahl fallen kaum unterschiedlich aus. Bei
hoher Auslastung ist das nächstgelegene Fahrzeug häufiger beschäftigt, entferntere Fahrzeuge
übernehmen spürbar mehr Einsätze als ein kongestionsblindes Modell annimmt - das lässt sich
mit dem Auslastungs-Regler direkt nachvollziehen.

**In echten Projekten** kämen meist weitere Aspekte dazu (mehrtägige Nachfrageszenarien,
mehrere gewichtete Zielgrößen, Backup-Fahrzeugtypen, echte Straßennetze statt Luftlinie) -
genau das behandelt die Dissertation, auf der diese Demo aufbaut, in voller Tiefe
(Goal Programming, robuste Optimierung über mehrere Szenarien) - das Grundprinzip aus
Markov-Kette und Standortsuche bleibt aber dasselbe.
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Systemzustand.** Bei $N$ Fahrzeugen ist der Systemzustand ein Binärvektor
$b = (b_1, \ldots, b_N) \in \{0,1\}^N$, $b_n=1$ genau dann, wenn Fahrzeug $n$ beschäftigt
ist - insgesamt $2^N$ mögliche Zustände.

**Übergänge.** Aus Zustand $b$ mit beschäftigtem Fahrzeug $n$ führt eine Bedienbeendigung
(Rate $\mu$) zu $b - e_n$. Ein Notruf von Nachfragepunkt $j$ (Rate $\lambda_j$) wird an das
erste freie Fahrzeug auf $j$s Präferenzliste (nach Entfernung sortiert) vermittelt und führt
zu $b + e_n$; sind alle Fahrzeuge auf der Liste belegt, bleibt der Zustand unverändert
(Anruf verloren).

**Stationäre Verteilung.** Diese Übergänge bilden eine zeitkontinuierliche Markov-Kette mit
Generatormatrix $Q$. Die stationäre Verteilung $\pi$ löst:
"""
    )
    st.latex(r"\pi Q = 0, \qquad \sum_{b} \pi(b) = 1")
    st.markdown(
        r"""
Daraus: Auslastung $\rho_n = \sum_{b: b_n=1} \pi(b)$, Zuteilungswahrscheinlichkeit
$f_{n,j} = \sum_{b: n \text{ bedient } j \text{ in } b} \pi(b)$ und Verlustwahrscheinlichkeit
$P_{loss} = \pi(\mathbf{1})$ (alle Fahrzeuge belegt).

**Validierung (Erlang-B-Identität).** Bei homogener Bedienrate $\mu$ bildet die Anzahl
beschäftigter Fahrzeuge für sich genommen eine Geburts-Todes-Kette mit Rate $\lambda =
\sum_j \lambda_j$ (Geburt, solange nicht alle $N$ belegt) bzw. $k\mu$ (Tod bei $k$ belegten
Fahrzeugen) - **unabhängig davon, welche Fahrzeuge konkret belegt sind**. Damit muss gelten:
"""
    )
    st.latex(r"P_{loss} = B(N, \lambda/\mu) \quad \text{(klassische Erlang-B-Formel)}")
    st.markdown(
        r"""
Diese Identität hält für JEDE räumliche Anordnung und Präferenzliste - sie dient als von der
eigentlichen Implementierung unabhängiger, scharfer Korrektheitstest (siehe
`tests/test_model.py`, `test_erlang_b_identity_*`).

**Zielgröße der lokalen Suche.** Für eine Standortauswahl $S \subseteq$ Kandidaten,
$|S|=N$:
"""
    )
    st.latex(r"\min_S \; \sum_{n \in S}\sum_j f_{n,j}(S) \cdot d(n,j) \;+\; P_{loss}(S) \cdot \Lambda \cdot \text{Strafdistanz}")
    st.markdown(
        r"""
mit $\Lambda = \sum_j \lambda_j$. Der erste Term ist die erwartete, demand-gewichtete
gefahrene Distanz der tatsächlich bedienten Anrufe; der zweite Term bestraft eine hohe
Verlustwahrscheinlichkeit - ohne ihn könnte die Suche viele Anrufe "verlieren", solange die
verbleibenden im Schnitt kurze Wege haben.

**Bezug zum Code:** `ems_hqm.py` (`solve_hqm`) baut $Q$ auf und löst das Gleichungssystem.
`ems_location.py` implementiert die naive Greedy-Wahl (`greedy_mclp`) sowie die lokale
Suche (`local_search`, `hqm_objective`). `ems_metaheuristics.py` implementiert Genetischen
Algorithmus und Ant-Colony-Optimization auf derselben Zielgröße. `ems_evaluation.py` leitet
die angezeigten Kennzahlen aus dem HQM-Ergebnis ab.
"""
    )

st.markdown("---")
st.caption(
    "Diese Demo ist bewusst vereinfacht (synthetisches Gebiet, ein Fahrzeugtyp, statische Nachfrage) "
    "und implementiert nur einen Ausschnitt der zugrundeliegenden Dissertation. Reale Systeme haben deutlich "
    "mehr Nebenbedingungen und Datenvolumen - die Methodik dahinter ist aber dieselbe."
)
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
