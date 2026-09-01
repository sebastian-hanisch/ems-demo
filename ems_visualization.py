"""Plotly-Visualisierungen für die Rettungsdienst-Standortplanung-Demo."""

import plotly.graph_objects as go

from ems_constants import MAP_SIZE


def map_figure(demand_positions, demand_weights, candidate_sites, chosen_indices, title, time_threshold=None):
    fig = go.Figure()

    unchosen = [i for i in range(len(candidate_sites)) if i not in chosen_indices]
    if unchosen:
        fig.add_trace(go.Scatter(
            x=candidate_sites[unchosen, 0], y=candidate_sites[unchosen, 1],
            mode="markers", name="Nicht gewählte Standorte",
            marker=dict(symbol="circle-open", size=10, color="#94a3b8"),
        ))

    fig.add_trace(go.Scatter(
        x=demand_positions[:, 0], y=demand_positions[:, 1],
        mode="markers", name="Nachfragepunkte",
        marker=dict(size=8 + 22 * (demand_weights / demand_weights.max()), color="#D68A2E", opacity=0.55),
    ))

    chosen_pos = candidate_sites[chosen_indices]
    if time_threshold is not None:
        for x, y in chosen_pos:
            fig.add_shape(
                type="circle", x0=x - time_threshold, x1=x + time_threshold,
                y0=y - time_threshold, y1=y + time_threshold,
                line=dict(color="#3E8E86", width=1, dash="dot"), fillcolor="rgba(62,142,134,0.05)",
            )
    fig.add_trace(go.Scatter(
        x=chosen_pos[:, 0], y=chosen_pos[:, 1],
        mode="markers", name="Fahrzeugstandorte",
        marker=dict(symbol="star", size=16, color="#14233B", line=dict(color="white", width=1)),
    ))

    fig.update_layout(
        title=title, height=420, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=[0, MAP_SIZE], title=None, showgrid=False),
        yaxis=dict(range=[0, MAP_SIZE], title=None, showgrid=False, scaleanchor="x"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def workload_figure(workload, title):
    labels = [f"Fahrzeug {i+1}" for i in range(len(workload))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=workload * 100, marker_color="#3E8E86"))
    fig.update_layout(
        title=title, yaxis_title="Auslastung (%)", yaxis_range=[0, 100],
        height=280, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def convergence_figure(history, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(history))), y=history, mode="lines+markers",
        line=dict(color="#14233B", width=2),
    ))
    fig.update_layout(
        title=title, xaxis_title="Tausch-Iteration", yaxis_title="Zielgröße (gewichtete Distanz + Verlust-Strafe)",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
