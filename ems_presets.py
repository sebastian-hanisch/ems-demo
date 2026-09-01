"""
Ein-Klick-Beispielszenarien und Permalink-Logik - dasselbe SETTING_SPECS-
Muster wie in den anderen Demos dieses Workspace.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

SETTING_SPECS = {}


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_servers_slider": SettingSpec("servers", int, 4, 2, 8),
    "n_candidates_slider": SettingSpec("cand", int, 10, 4, 16),
    "n_demand_slider": SettingSpec("demand", int, 12, 5, 25),
    "n_peaks_slider": SettingSpec("peaks", int, 2, 1, 3),
    "peak_conc_slider": SettingSpec("conc", float, 0.5, 0.0, 1.0),
    "seed_input": SettingSpec("seed", int, 42, 0, 2_000_000_000),
    "service_time_slider": SettingSpec("svc", float, 0.5, 0.1, 2.0),
    "utilization_slider": SettingSpec("util", float, 0.5, 0.1, 0.95),
    "time_threshold_slider": SettingSpec("thr", float, 4.0, 1.0, 8.0),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def apply_preset(n_servers, n_candidates, n_demand, n_peaks, peak_conc, seed, service_time, utilization, time_threshold):
    st.session_state["n_servers_slider"] = n_servers
    st.session_state["n_candidates_slider"] = n_candidates
    st.session_state["n_demand_slider"] = n_demand
    st.session_state["n_peaks_slider"] = n_peaks
    st.session_state["peak_conc_slider"] = peak_conc
    st.session_state["seed_input"] = seed
    st.session_state["service_time_slider"] = service_time
    st.session_state["utilization_slider"] = utilization
    st.session_state["time_threshold_slider"] = time_threshold


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def sync_query_params(n_servers, n_candidates, n_demand, n_peaks, peak_conc, seed, service_time, utilization, time_threshold):
    try:
        st.query_params["servers"] = str(n_servers)
        st.query_params["cand"] = str(n_candidates)
        st.query_params["demand"] = str(n_demand)
        st.query_params["peaks"] = str(n_peaks)
        st.query_params["conc"] = str(peak_conc)
        st.query_params["seed"] = str(int(seed))
        st.query_params["svc"] = str(service_time)
        st.query_params["util"] = str(utilization)
        st.query_params["thr"] = str(time_threshold)
    except Exception:
        pass
