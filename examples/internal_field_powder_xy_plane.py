"""Convenience wrapper to run the hyperfine-distribution example config."""
from __future__ import annotations
import argparse
from pathlib import Path

from pysimnmr.hyperfine_cli import load_hyperfine_config, run_hyperfine_simulation


DEFAULT_CONFIG = Path(__file__).with_name('hyperfine_distribution_xy_plane.py')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Run the 75As hyperfine-distribution example configuration.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG,
        help='Configuration file (Python CONFIG module or YAML) describing the hyperfine distribution.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('output/75As_xy_internal_fields'),
        help='Output directory for the generated spectrum/plots.',
    )
    parser.add_argument(
        '--no-plotly',
        action='store_true',
        help='Skip Plotly HTML export even if enabled in the config.',
    )
    args = parser.parse_args(argv)
    cfg = load_hyperfine_config(args.config)
    run_hyperfine_simulation(cfg, args.out, enable_plotly=not args.no_plotly)


if __name__ == '__main__':
    main()
