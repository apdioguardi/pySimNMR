# -- coding: utf-8 --
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:
    go = None; pio = None

def save_lines_html(x: np.ndarray, Y: np.ndarray, *, xlab: str, ylab: str, title: str, out_html: Path) -> None:
    if go is None or pio is None:
        raise RuntimeError("plotly is not installed; install pySimNMR[vis]")
    fig = go.Figure()
    if Y.ndim == 1:
        fig.add_trace(go.Scatter(x=x, y=Y, mode='lines', name='line'))
    else:
        for j in range(Y.shape[1]):
            fig.add_trace(go.Scatter(x=x, y=Y[:,j], mode='lines', name=f'L{j}'))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, template='plotly_white')
    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), include_plotlyjs='cdn', auto_open=False)


def save_internal_field_distribution_html(vectors: np.ndarray, *, weights: Optional[np.ndarray]=None, title: str="Internal Field Distribution", out_html: Path) -> None:
    """Persist a 3D scatter plot of internal field vectors to an HTML file."""
    if go is None or pio is None:
        raise RuntimeError("plotly is not installed; install pySimNMR[vis]")
    vectors = np.asarray(vectors, float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (n_samples, 3)")
    n_samples = vectors.shape[0]
    if n_samples == 0:
        raise ValueError("vectors must contain at least one sample")
    if weights is None:
        weights_arr = np.full(n_samples, 1.0 / n_samples, float)
    else:
        weights_arr = np.asarray(weights, float)
        if weights_arr.shape != (n_samples,):
            raise ValueError("weights must have shape (n_samples,)")
        weight_sum = float(np.sum(weights_arr))
        if not np.isfinite(weight_sum) or weight_sum == 0.0:
            raise ValueError("weights must sum to a finite, non-zero value")
        weights_arr = weights_arr / weight_sum
    norm = weights_arr.max() or 1.0
    marker_colors = weights_arr / norm
    marker_sizes = 4 + 8 * marker_colors
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vectors[:, 0],
        y=vectors[:, 1],
        z=vectors[:, 2],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            colorscale='Viridis',
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(title='Weight')
        ),
        name='Internal fields'
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='Hx (T)'),
            yaxis=dict(title='Hy (T)'),
            zaxis=dict(title='Hz (T)'),
            aspectmode='data'
        ),
        template='plotly_white',
        margin=dict(l=10, r=10, b=10, t=40)
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), include_plotlyjs='cdn', auto_open=False)
