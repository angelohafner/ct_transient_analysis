# Comments in English only
import numpy as np
import plotly.graph_objects as go


def make_currents_figure(t: np.ndarray,
                         is_arr: np.ndarray,
                         i2_arr: np.ndarray,
                         is_rms: np.ndarray,
                         i2_rms: np.ndarray) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=is_arr, mode="lines", name="is(t) ideal"))
    fig.add_trace(go.Scatter(x=t, y=i2_arr, mode="lines", name="i2(t) actual"))
    fig.add_trace(go.Scatter(x=t, y=is_rms, mode="lines", name="is_rms(t) 1-cycle"))
    fig.add_trace(go.Scatter(x=t, y=i2_rms, mode="lines", name="i2_rms(t) 1-cycle"))

    fig.update_layout(
        title="Currents (instantaneous and 1-cycle RMS)",
        xaxis_title="Time (s)",
        yaxis_title="Current (A)",
        hovermode="x unified",
        legend=dict(orientation="h")
    )
    return fig


def make_flux_excitation_figure(t: np.ndarray,
                                lam: np.ndarray,
                                ie_arr: np.ndarray) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=lam, mode="lines", name="lambda(t)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=t, y=ie_arr, mode="lines", name="ie(t) excitation", yaxis="y2"))

    fig.update_layout(
        title="Flux-linkages and excitation current",
        xaxis_title="Time (s)",
        yaxis=dict(title="Lambda (Wb-turns)", side="left"),
        yaxis2=dict(title="Excitation current ie (A)", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h")
    )
    return fig
