from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Any, Union
import base64
import sys
import html

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except Exception:  # pragma: no cover - plotly is an optional heavy dependency
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    pio = None  # type: ignore


_PLOTLY_IMPORT_ERROR = (
    "Plotly is required for this command. Install pySimNMR[vis] or add "
    "'plotly>=5.0' and 'kaleido>=0.2.1' to your environment."
)


def _table_columns(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[List[str]]:
    """Convert row-oriented data to Plotly's column-wise table input."""
    n_cols = len(header)
    if n_cols == 0:
        return []
    columns = [[] for _ in range(n_cols)]
    for row in rows:
        padded = list(row)[:n_cols]
        if len(padded) < n_cols:
            padded.extend([""] * (n_cols - len(padded)))
        for idx in range(n_cols):
            value = padded[idx]
            columns[idx].append("" if value is None else str(value))
    return columns


def _metadata_cell_value(value: str) -> str:
    """Insert zero-width break markers so long paths wrap inside Plotly tables."""
    text = str(value)
    text = text.replace("\\", "\\\u200b")
    text = text.replace("/", "/\u200b")
    return text


@dataclass
class SpectrumPlotData:
    x: np.ndarray
    total_with_background: np.ndarray
    per_site_normalized: List[np.ndarray]
    site_labels: List[str]
    plot_individual: bool
    plot_sum: bool
    title: str
    x_label: str
    y_label: str
    legend_lines: List[Union[str, Tuple[str, str]]] = field(default_factory=list)
    legend_width_ratio: Sequence[float] = field(default_factory=lambda: [3.25, 1.0])
    x_limits: Optional[Tuple[float, float]] = None
    y_limits: Optional[Tuple[float, float]] = None
    exp_x: Optional[np.ndarray] = None
    exp_y: Optional[np.ndarray] = None
    exp_label: str = "experiment"
    overlays: List["OverlayTrace"] = field(default_factory=list)
    site_table_header: Optional[List[str]] = None
    site_table_rows: Optional[List[List[str]]] = None
    global_table_header: Optional[List[str]] = None
    global_table_rows: Optional[List[List[str]]] = None
    metadata_entries: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class RenderedPlot:
    figure: Any


@dataclass
class OverlayTrace:
    x: np.ndarray
    y: np.ndarray
    label: str
    mpl_style: str = 'k-'
    line_width: float = 2.0
    marker_size: float = 6.0
    opacity: float = 1.0
    show_legend: bool = True


def _require_plotly() -> None:
    if go is None or make_subplots is None or pio is None:
        raise RuntimeError(_PLOTLY_IMPORT_ERROR)


_COLOR_MAP = {
    'k': '#000000',
    'K': '#000000',
    'r': '#d62728',
    'R': '#d62728',
    'g': '#2ca02c',
    'G': '#2ca02c',
    'b': '#1f77b4',
    'B': '#1f77b4',
    'c': '#17becf',
    'C': '#17becf',
    'm': '#e377c2',
    'M': '#e377c2',
    'y': '#bcbd22',
    'Y': '#bcbd22',
    'w': '#ffffff',
    'W': '#ffffff',
}

_MARKER_MAP = {
    'o': 'circle',
    's': 'square',
    '^': 'triangle-up',
    'v': 'triangle-down',
    '<': 'triangle-left',
    '>': 'triangle-right',
    'x': 'x',
    '+': 'cross',
    'd': 'diamond',
    'D': 'diamond',
    '*': 'star',
}

_LINE_MAP = {
    '--': 'dash',
    '-.': 'dashdot',
    ':': 'dot',
    '-': 'solid',
}


def _plotly_style_from_mpl(style: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    style = style or ''
    dash = None
    line_present = False
    remaining = style
    for token in ('-.', '--', ':', '-'):
        if token in remaining:
            dash = _LINE_MAP[token]
            line_present = True
            remaining = remaining.replace(token, '', 1)
            break
    color = None
    marker = None
    for ch in remaining:
        if ch in _COLOR_MAP:
            color = _COLOR_MAP[ch]
        elif ch in _MARKER_MAP:
            marker = _MARKER_MAP[ch]
    if marker and line_present:
        mode = 'lines+markers'
    elif marker:
        mode = 'markers'
    else:
        mode = 'lines'
    return color, dash, marker, mode


def create_plotly_spectrum_figure(data: SpectrumPlotData) -> "go.Figure":
    """Render a Plotly figure with parameter tables on the right and metadata on the bottom."""
    _require_plotly()
    right_tables: list[
        tuple[List[str], List[List[str]], dict[str, Any] | None, dict[str, Any] | None]
    ] = []
    bottom_tables: list[
        tuple[List[str], List[List[str]], dict[str, Any] | None, dict[str, Any] | None]
    ] = []
    if data.site_table_header and data.site_table_rows:
        right_tables.append((data.site_table_header, data.site_table_rows, None, None))
    if data.global_table_header and data.global_table_rows:
        right_tables.append((data.global_table_header, data.global_table_rows, None, None))
    if data.metadata_entries:
        metadata_rows = [[label, _metadata_cell_value(path)] for label, path in data.metadata_entries]
        bottom_tables.append((
            ["File", "Path"],
            metadata_rows,
            {"font": {"family": "Courier New, monospace", "size": 11}},
            {"columnwidth": [0.8, 3.2]},
        ))

    right_rows = max(1, len(right_tables)) if right_tables else 1
    bottom_rows = len(bottom_tables)
    total_rows = right_rows + bottom_rows
    column_widths = [0.7, 0.3] if right_tables else [1.0, 0.0]

    specs: list[list[dict[str, Any] | None]] = []
    row_heights: list[float] = []

    if right_tables:
        specs.append([{"type": "xy", "rowspan": right_rows}, {"type": "table"}])
        row_heights.append(0.7 if total_rows > 1 else 1.0)
        for _ in range(1, right_rows):
            specs.append([None, {"type": "table"}])
        if right_rows > 1:
            remaining = 0.3
            per = remaining / (right_rows - 1)
            row_heights.extend([per] * (right_rows - 1))
    else:
        specs.append([{"type": "xy", "colspan": 2}, None])
        row_heights.append(0.75 if bottom_rows else 1.0)

    if bottom_rows:
        for _ in range(bottom_rows):
            specs.append([{"type": "table", "colspan": 2}, None])
        remaining_height = max(0.25, 1.0 - sum(row_heights))
        per_bottom = remaining_height / bottom_rows
        row_heights.extend([per_bottom] * bottom_rows)

    fig = make_subplots(
        rows=total_rows,
        cols=2,
        column_widths=column_widths,
        row_heights=row_heights,
        specs=specs,
        vertical_spacing=0.07,
        horizontal_spacing=0.05,
    )
    x = np.asarray(data.x, dtype=float)

    for overlay in data.overlays:
        x_overlay = np.asarray(overlay.x, dtype=float)
        y_overlay = np.asarray(overlay.y, dtype=float)
        color, dash, marker_symbol, mode = _plotly_style_from_mpl(overlay.mpl_style)
        trace = go.Scatter(
            x=x_overlay,
            y=y_overlay,
            mode=mode,
            name=overlay.label,
            opacity=overlay.opacity,
            showlegend=overlay.show_legend,
        )
        if 'lines' in mode:
            line_dict = {'width': overlay.line_width}
            if color:
                line_dict['color'] = color
            if dash:
                line_dict['dash'] = dash
            trace.line = line_dict
        if 'markers' in mode or mode == 'markers':
            marker_dict = {'size': overlay.marker_size}
            if marker_symbol:
                marker_dict['symbol'] = marker_symbol
            if color:
                marker_dict['color'] = color
            trace.marker = marker_dict
        fig.add_trace(trace, row=1, col=1)

    if data.plot_individual and len(data.per_site_normalized) > 1:
        for label, spec in zip(data.site_labels, data.per_site_normalized):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.asarray(spec, dtype=float),
                    mode="lines",
                    name=label,
                    fill="tozeroy",
                    line=dict(width=1.5),
                    opacity=0.5,
                ),
                row=1,
                col=1
            )

    if data.plot_sum:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(data.total_with_background, dtype=float),
                mode="lines",
                name="total",
                line=dict(color="#d62728", width=2.5),
            ),
            row=1,
            col=1
        )

    if data.exp_x is not None and data.exp_y is not None:
        fig.add_trace(
            go.Scatter(
                x=np.asarray(data.exp_x, dtype=float),
                y=np.asarray(data.exp_y, dtype=float),
                mode="lines",
                name=data.exp_label,
                line=dict(color="#1f77b4", width=1.5, dash="dot"),
            ),
            row=1,
            col=1
        )

    for idx, (header, rows, cell_opts, table_kwargs) in enumerate(right_tables, start=1):
        table_kwargs = table_kwargs or {}
        fig.add_trace(
            go.Table(
                header=dict(values=header, align="left", fill_color="#f2f2f2"),
                cells=dict(values=_table_columns(header, rows), align="left", **(cell_opts or {})),
                **table_kwargs,
            ),
            row=idx,
            col=2,
        )

    for bottom_idx, (header, rows, cell_opts, table_kwargs) in enumerate(bottom_tables, start=1):
        table_kwargs = table_kwargs or {}
        fig.add_trace(
            go.Table(
                header=dict(values=header, align="left", fill_color="#f2f2f2"),
                cells=dict(values=_table_columns(header, rows), align="left", **(cell_opts or {})),
                **table_kwargs,
            ),
            row=right_rows + bottom_idx,
            col=1,
        )

    fig.update_xaxes(title=data.x_label, range=list(data.x_limits) if data.x_limits else None)
    fig.update_yaxes(title=data.y_label, range=list(data.y_limits) if data.y_limits else None)

    fig.update_layout(
        title=data.title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=90, b=80),
        autosize=True,
    )
    return fig


def write_plotly_html(fig: "go.Figure", out_html: Path, *, extra_sections: Optional[str] = None) -> None:
    _require_plotly()
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html_str = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
        config={"responsive": True},
    )
    if extra_sections:
        html_str = html_str.replace("</body>", f"{extra_sections}</body>", 1)
    out_html.write_text(html_str, encoding="utf-8")


def save_plotly_image(fig: "go.Figure", out_path: Path, *, scale: float = 1.0) -> None:
    _require_plotly()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_path.suffix.lstrip(".").lower()
    if fmt not in {"png", "svg", "pdf"}:
        raise ValueError(f"Unsupported figure format: {fmt}")
    scale = max(scale, 0.5)
    try:
        pio.write_image(fig, str(out_path), format=fmt, scale=scale)
    except ValueError as exc:  # kaleido not installed
        raise RuntimeError(
            "Plotly static image export requires the 'kaleido' package. "
            "Install pySimNMR[vis] or pip install kaleido."
        ) from exc
    except TimeoutError as exc:
        if fmt == "png":
            print(
                "[pySimNMR] Plotly image export timed out. Writing placeholder PNG. "
                "Re-run with kaleido available to capture the figure.",
                file=sys.stderr,
            )
            _write_placeholder_png(out_path)
        else:
            raise RuntimeError(
                "Plotly static image export timed out; try rerunning the command or exporting to PNG."
            ) from exc


def _create_plotly_line_figure(x: np.ndarray,
                               series: List[np.ndarray],
                               labels: List[str],
                               title: str,
                               x_label: str,
                               y_label: str) -> "go.Figure":
    _require_plotly()
    fig = go.Figure()
    for label, arr in zip(labels, series):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=arr,
                mode="lines",
                name=label,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _write_placeholder_png(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PLACEHOLDER_PNG)


def save_line_plot(x: np.ndarray,
                   series: List[np.ndarray],
                   *,
                   labels: Optional[List[str]] = None,
                   title: str = "",
                   x_label: str = "",
                   y_label: str = "",
                   save_path: Path,
                   html_path: Optional[Path] = None,
                   dpi: int = 150) -> RenderedPlot:
    x = np.asarray(x, dtype=float)
    arrays = [np.asarray(arr, dtype=float) for arr in series]
    n_series = len(arrays)
    if not labels or len(labels) != n_series:
        labels = [f"Series {idx+1}" for idx in range(n_series)]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = Path(html_path) if html_path else None

    fig = _create_plotly_line_figure(x, arrays, labels, title, x_label, y_label)
    if html_path is not None:
        write_plotly_html(fig, html_path)
    scale = max(dpi / 96.0, 1.0)
    try:
        save_plotly_image(fig, save_path, scale=scale)
    except RuntimeError as exc:
        print(f"[pySimNMR] {exc} Writing placeholder PNG. Install kaleido>=0.2.1 for image exports.", file=sys.stderr)
        _write_placeholder_png(save_path)
    return RenderedPlot(figure=fig)


def save_spectrum_plot(plot_data: SpectrumPlotData,
                       *,
                       save_path: Path,
                       html_path: Optional[Path] = None,
                       dpi: int = 150) -> RenderedPlot:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = create_plotly_spectrum_figure(plot_data)
    extra_sections = _build_extra_sections(plot_data) if html_path is not None else None
    if html_path is not None:
        write_plotly_html(fig, html_path, extra_sections=extra_sections)
    scale = max(dpi / 96.0, 1.0)
    try:
        save_plotly_image(fig, save_path, scale=scale)
    except RuntimeError as exc:
        print(f"[pySimNMR] {exc} Writing placeholder PNG. Install kaleido>=0.2.1 for image exports.", file=sys.stderr)
        _write_placeholder_png(save_path)
    return RenderedPlot(figure=fig)


def _table_html(title: str, header: List[str], rows: List[List[str]]) -> str:
    header_cells = "".join(f"<th align='left'>{html.escape(str(h))}</th>" for h in header)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table style='width:100%; border-collapse:collapse; margin-bottom:16px;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody></table>"
    )


def _metadata_html(entries: List[tuple[str, str]]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td><pre>{html.escape(path)}</pre></td></tr>"
        for label, path in entries
    )
    return (
        "<h3>Files</h3>"
        "<table style='width:100%; border-collapse:collapse;'>"
        "<thead><tr><th align='left'>File</th><th align='left'>Absolute path</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_extra_sections(data: SpectrumPlotData) -> Optional[str]:
    sections: list[str] = []
    if data.site_table_header and data.site_table_rows:
        sections.append(_table_html("Per-site parameters", data.site_table_header, data.site_table_rows))
    if data.global_table_header and data.global_table_rows:
        sections.append(_table_html("Global parameters", data.global_table_header, data.global_table_rows))
    if data.metadata_entries:
        sections.append(_metadata_html(data.metadata_entries))
    if not sections:
        return None
    return "<div style='margin-top:24px;font-family:Arial, sans-serif;'>" + "".join(sections) + "</div>"


def finalize_rendered_figure(rendered: RenderedPlot, show_plot: bool) -> None:
    if show_plot:
        try:
            rendered.figure.show()
        except Exception:
            pass
