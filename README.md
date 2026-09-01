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

## Wie ist das entstanden?

Ausgangspunkt war die Frage, ob sich aus Blanks Dissertation eine gute Ergänzung zum Demo-Portfolio bauen lässt — die Antwort brauchte erst ein echtes Verständnis der Arbeit (PDF gelesen, Kernmodell und Forschungsfragen extrahiert), bevor überhaupt eine Zeile Code geschrieben wurde.

**Erst validiert, dann gebaut.** Die Markov-Ketten-Mathematik des Hypercube Queueing Models ist deutlich anspruchsvoller als bei den übrigen Demos dieses Portfolios — ein Implementierungsfehler wäre leicht möglich und schwer zu bemerken gewesen. Deshalb stand vor der eigentlichen App ein Validierungsschritt gegen ein von der eigenen Implementierung komplett unabhängiges Referenzresultat: Bei homogener Bedienrate muss die Verlustwahrscheinlichkeit exakt der klassischen Erlang-B-Formel entsprechen, unabhängig von Fahrzeuganzahl, Standorten oder Präferenzlisten — das folgt daraus, dass die Anzahl beschäftigter Fahrzeuge für sich genommen eine gewöhnliche Geburts-Todes-Kette bildet. Über viele Zufallsinstanzen bestätigt (`tests/test_model.py`, `TestErlangBIdentity`), bevor überhaupt eine Zeile UI-Code entstand. Zusätzlich wurden Lösungszeiten für 5 bis 10 Fahrzeuge gemessen (256 Zustände ≈ 15 ms, 1024 Zustände ≈ 65 ms), um die Fahrzeug-Obergrenze im Regler (8) so zu setzen, dass die lokale Suche mit vielen Auswertungen pro Klick reaktionsschnell bleibt.

**Ein Kennzahl-Verwirrung gefunden und in eine Erklärung verwandelt:** Die Verlustwahrscheinlichkeit zeigte in den Vergleichs-Kacheln naiv vs. HQM-bewusst immer exakt 0,0 Prozentpunkte Unterschied — sah zunächst wie ein toter oder kaputter Wert aus. Tatsächlich ist das ein beweisbares Resultat (siehe Erlang-B-Identität oben): Die Standortwahl kann die Verlustwahrscheinlichkeit bei diesem Modell gar nicht beeinflussen, nur *wer* antwortet, nicht *ob überhaupt* jemand verfügbar ist. Aus der potenziell verwirrenden Kennzahl wurde ein expliziter Hinweistext mit Verweis auf die formale Herleitung.

**Später erweitert** um Genetischen Algorithmus und Ant-Colony-Optimization (`ems_metaheuristics.py`) — direkt nach Blanks Kapitel 6, inklusive seines eigenen, wörtlich zitierten Befunds *„the ACO performs better than the GA in the proposed experimental setting"*, den man über den Vergleichs-Button am eigenen Szenario nachvollziehen kann. Beide Verfahren zusätzlich memetisch hybridisiert (bestes Individuum je Generation/Iteration wird mit der bestehenden lokalen Suche nachpoliert) und über ein gemeinsames Cache-Dict memoisiert, damit wiederholt betrachtete Standort-Konfigurationen nicht erneut das Gleichungssystem lösen müssen. Dabei auch eine Vereinfachung gefunden: Die Zielgröße der lokalen Suche enthielt ursprünglich einen Strafterm für die Verlustwahrscheinlichkeit — der ist wegen der Erlang-B-Identität für jede Konfiguration exakt identisch (eine additive Konstante ändert kein Minimierungsproblem) und wurde ersatzlos entfernt. Für die Rechenzeit bei vielen kleinen Gleichungssystemen sorgt außerdem eine feste Ein-Thread-Vorgabe für OpenBLAS/OMP/MKL (per Messung ~20x schneller pro Lösung als die Standard-Multithreading-Einstellung, siehe `app.py`).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `pytest tests/ -v`

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von [Sebastian Hanisch](https://sebastianhanisch.net) — Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
