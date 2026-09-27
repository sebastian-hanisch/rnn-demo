"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe."""
import numpy as np
import plotly.graph_objects as go

import rnn_constants as C

COLOR_POS = "#1f77b4"
COLOR_NEG = "#d62728"
COLOR_FIRST = "#d62728"
COLOR_LAST = "#1f77b4"


def build_sequence_figure(seq: np.ndarray, label: float, title: str = ""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=seq, mode="lines+markers", name="Sequenz",
                             line=dict(color="#7f7f7f"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[C.SIGNAL_POS], y=[seq[C.SIGNAL_POS]], mode="markers",
                             name="Signal", marker=dict(size=14, symbol="star",
                             color=COLOR_POS if label > 0 else COLOR_NEG)))
    fig.update_layout(
        title=title or f"Sequenz (Signal = {'+1' if label > 0 else '-1'} an Position {C.SIGNAL_POS})",
        xaxis=dict(title="Zeitschritt t", fixedrange=True),
        yaxis=dict(title="Wert", fixedrange=True), height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_t_sweep_figure(rows, title="Erfolgsquote vs. Sequenzlänge T"):
    Ts = [r["T"] for r in rows]
    rates = [r["rate"] * 100 for r in rows]
    fig = go.Figure(go.Bar(x=[str(t) for t in Ts], y=rates, marker_color=COLOR_POS))
    fig.update_layout(
        title=title, xaxis=dict(title="Sequenzlänge T", fixedrange=True),
        yaxis=dict(title="Erfolgsquote (%)", fixedrange=True, range=[0, 105]),
        height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_gradient_norm_figure(rows, title="Gradientennorm: erster vs. letzter Zeitschritt"):
    Ts = [r["T"] for r in rows]
    first = [r["norm_first"] for r in rows]
    last = [r["norm_last"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ts, y=first, mode="lines+markers", name="||δ₁|| (erster Schritt)",
                             line=dict(color=COLOR_FIRST)))
    fig.add_trace(go.Scatter(x=Ts, y=last, mode="lines+markers", name="||δ_T|| (letzter Schritt)",
                             line=dict(color=COLOR_LAST)))
    fig.update_layout(
        title=title, xaxis=dict(title="Sequenzlänge T", fixedrange=True, type="log"),
        yaxis=dict(title="Gradientennorm", fixedrange=True, type="log"),
        height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_error_curve_figure(errors_per_epoch, title="Fehlerzahl je Epoche"):
    epochs = list(range(1, len(errors_per_epoch) + 1))
    fig = go.Figure(go.Scatter(x=epochs, y=errors_per_epoch, mode="lines", line=dict(color=COLOR_NEG)))
    fig.update_layout(
        title=title, xaxis=dict(title="Epoche", fixedrange=True),
        yaxis=dict(title="Anzahl Fehler", fixedrange=True, rangemode="tozero"),
        height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
