"""
UI-Tests über streamlit.testing.v1.AppTest - prüft, dass die App mit allen
Presets und Slider-Extremen lädt, ohne abzustürzen.

Ausführen mit: pytest tests/ -v
"""

import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = os.path.join(os.path.dirname(__file__), "..")
APP_PATH = os.path.join(APP_DIR, "app.py")
TIMEOUT = 90

sys.path.insert(0, os.path.abspath(APP_DIR))


def fresh_app():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    return at


def assert_ok(at):
    assert not at.exception, f"Unerwartete Exception(s): {[e.message for e in at.exception]}"


def test_default_load():
    at = fresh_app()
    assert_ok(at)
    assert len(at.metric) >= 3


@pytest.mark.parametrize("label", ["Ruhige Nachtschicht", "Stoßzeit", "Größeres System"])
def test_presets_apply_without_crash(label):
    at = fresh_app()
    btn = [b for b in at.button if label in b.label][0]
    btn.click().run(timeout=TIMEOUT)
    assert_ok(at)


def test_low_vs_high_utilization_preset_metrics_differ():
    """Kernbehauptung der Demo: bei hoher Auslastung weicht die reale
    (HQM-berechnete) Reaktionszeit stärker von der naiven Selbsteinschätzung
    ab als bei niedriger Auslastung."""
    at_low = fresh_app()
    btn_low = [b for b in at_low.button if "Ruhige Nachtschicht" in b.label][0]
    btn_low.click().run(timeout=TIMEOUT)
    assert_ok(at_low)

    at_high = fresh_app()
    btn_high = [b for b in at_high.button if "Stoßzeit" in b.label][0]
    btn_high.click().run(timeout=TIMEOUT)
    assert_ok(at_high)

    # Verlustwahrscheinlichkeit sollte bei Stoßzeit klar höher liegen.
    loss_low = [m for m in at_low.metric if "Verlustwahrscheinlichkeit" in m.label][0]
    loss_high = [m for m in at_high.metric if "Verlustwahrscheinlichkeit" in m.label][0]

    def _parse_pct(value):
        return float(value.replace("%", "").replace(",", "."))

    assert _parse_pct(loss_high.value) > _parse_pct(loss_low.value)


def test_regenerate_seed_button_changes_seed():
    at = fresh_app()
    seed_before = at.sidebar.number_input(key="seed_input").value
    seed_btn = [b for b in at.sidebar.button if "Neues Szenario" in b.label][0]
    seed_btn.click().run(timeout=TIMEOUT)
    assert_ok(at)
    seed_after = at.sidebar.number_input(key="seed_input").value
    assert seed_after != seed_before


@pytest.mark.parametrize("n_servers", [2, 8])
def test_server_count_extremes(n_servers):
    at = fresh_app()
    slider = [s for s in at.sidebar.slider if "Anzahl Fahrzeuge" in s.label][0]
    slider.set_value(n_servers).run(timeout=TIMEOUT)
    assert_ok(at)


def test_too_few_candidates_shows_warning_not_crash():
    at = fresh_app()
    servers_slider = [s for s in at.sidebar.slider if "Anzahl Fahrzeuge" in s.label][0]
    candidates_slider = [s for s in at.sidebar.slider if "Standort-Kandidaten" in s.label][0]
    servers_slider.set_value(6).run(timeout=TIMEOUT)
    candidates_slider.set_value(4).run(timeout=TIMEOUT)
    assert_ok(at)
    assert any("größer als die Anzahl Fahrzeuge" in str(w.value) for w in at.warning)


def test_pdf_download_button_present():
    at = fresh_app()
    assert_ok(at)
    assert any("PDF" in b.label for b in at.download_button)


def test_no_feedback_ui_present():
    """Regressionsschutz: diese Demo hat bewusst KEINEN Feedback-Mechanismus
    (siehe Entscheidung für das gesamte Portfolio - Streamlit Community
    Cloud hat kein persistentes Dateisystem)."""
    at = fresh_app()
    assert not any("hilfreich" in b.label for b in at.button)
