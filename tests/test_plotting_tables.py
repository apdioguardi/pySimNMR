import numpy as np
import pytest

from pysimnmr import plotting
from pysimnmr.plotting import SpectrumPlotData, create_plotly_spectrum_figure


pytestmark = pytest.mark.skipif(plotting.go is None, reason="plotly is not available")


def test_plotly_spectrum_includes_parameter_tables() -> None:
    x = np.linspace(0.0, 1.0, 5)
    spectrum = np.linspace(0.0, 1.0, 5)
    site_header = ["Parameter", "<sup>75</sup>As"]
    site_rows = [
        [r"$m_i$", "0.5"],
        [r"$K_a$", "1.0"],
    ]
    global_header = ["Global parameter", "Value"]
    global_rows = [["Carrier freq.", "14.5"]]
    metadata_entries = [
        ("Config file", "/tmp/config.py"),
        ("Output spectrum", "/tmp/out.txt"),
    ]
    data = SpectrumPlotData(
        x=x,
        total_with_background=spectrum,
        per_site_normalized=[spectrum],
        site_labels=["<sup>75</sup>As"],
        plot_individual=False,
        plot_sum=True,
        title="Test spectrum",
        x_label="Field (T)",
        y_label="Intensity",
        legend_lines=[],
        site_table_header=site_header,
        site_table_rows=site_rows,
        global_table_header=global_header,
        global_table_rows=global_rows,
        metadata_entries=metadata_entries,
    )

    fig = create_plotly_spectrum_figure(data)
    table_traces = [trace for trace in fig.data if getattr(trace, "type", "") == "table"]
    assert len(table_traces) == 3

    site_table, global_table, metadata_table = table_traces
    assert tuple(site_table.header.values) == tuple(site_header)
    assert site_table.cells.values[0][0] == site_rows[0][0]
    assert tuple(global_table.header.values) == tuple(global_header)
    assert global_table.cells.values[1][0] == global_rows[0][1]
    assert tuple(metadata_table.header.values) == ("File", "Path")
    assert metadata_table.cells.values[0][0] == metadata_entries[0][0]
    path_cell = metadata_table.cells.values[1][0]
    assert "config.py" in path_cell
    assert "\u200b" in path_cell
    assert "<span" not in path_cell
    assert metadata_table.columnwidth == (0.8, 3.2)
