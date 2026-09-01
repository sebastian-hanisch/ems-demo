# 🚑 Rettungsdienst-Standortplanung (Hypercube Queueing Model)

Interaktive Demo zur Standortplanung von Rettungsfahrzeugen. Kernfrage: Wie stark ändert sich die beste Standortwahl, wenn man berücksichtigt, dass ein Fahrzeug oft gerade im Einsatz und nicht verfügbar ist?

**[→ Demo live ausprobieren](https://sebastianhanisch-ems-demo.streamlit.app/)**

Angeregt durch die Dissertation von Felix Blank (Julius-Maximilians-Universität Würzburg, 2021): *„The use of the Hypercube Queueing Model for the location optimization decision of Emergency Medical Service systems“*.

## Worum geht's?

Klassische Coverage-Modelle für Standortplanung gehen implizit von 100 % Verfügbarkeit aus — in Wirklichkeit ist ein Fahrzeug aber mit einer gewissen Wahrscheinlichkeit gerade im Einsatz, wenn der nächste Notruf eingeht. Das **Hypercube Queueing Model** (Larson, 1974) berechnet über eine zeitkontinuierliche Markov-Kette über alle 2ᴺ Fahrzeug-Verfügbarkeitszustände **exakt**, wie oft welches Fahrzeug tatsächlich antwortet — und liefert daraus realistische Reaktionszeiten, Abdeckung und Verlustwahrscheinlichkeit.

## Methodik

- Vergleich einer **kongestionsblinden Greedy-Standortwahl** (klassisches Maximal Covering Location Problem) mit einer **HQM-bewussten lokalen Suche**, die die tatsächliche Erreichbarkeit einpreist
- Zusätzlich: **Genetischer Algorithmus** und **Ant-Colony-Optimization** als Metaheuristik-Vergleich, angelehnt an Blanks Kapitel 6
- **Mathematisch validiert**: Für ein System mit homogener Bedienrate entspricht die berechnete Verlustwahrscheinlichkeit nachweislich immer exakt der klassischen Erlang-B-Formel, unabhängig von der räumlichen Anordnung — ein von der Implementierung unabhängiger, scharfer Korrektheitstest (siehe `tests/test_model.py`)
- PDF-Export, Permalink

Bewusst ohne die volle Modelltiefe der Dissertation (Goal Programming, robuste Optimierung über mehrere Szenarien), um das Kernprinzip greifbar zu machen.

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `pytest tests/ -v`

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von [Sebastian Hanisch](https://sebastianhanisch.net) — Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
