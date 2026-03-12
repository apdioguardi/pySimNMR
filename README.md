# pySimNMR

Simulation software for solid-state nuclear magnetic resonance (NMR) spectra. Frequency/field swept spectra of both single crystals and powders.

## Getting Started

You will need Python 3.9+ to run this code. A 64-bit Python is recommended for large arrays, especially for powder spectra.

### Zero-to-running walkthrough

1. **Install Python 3.10+**  
   - Download from [python.org](https://www.python.org/downloads/) and enable the "Add Python to PATH" checkbox on Windows.
2. **Download pySimNMR**  
   - Either clone with Git (`git clone https://github.com/.../pySimNMR.git`) or download the ZIP from your repository host and extract it.
3. **Create a virtual environment** (from the project root):  
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`  
   - Windows PowerShell: `py -m venv .venv; .\.venv\Scripts\Activate.ps1`  
   - Windows Command Prompt: `py -m venv .venv && .\.venv\Scripts\activate.bat`
4. **Install pySimNMR and dependencies**  
   - `pip install --upgrade pip` (recommended)  
   - `pip install -e .` (installs the package in editable mode)
5. **Run a sanity check**  
   - `nmr-freq-powder --config examples/freq_powder_modern.py --no-show --save-fig output/test_powder.png`
   - The CLI prints progress and writes a PNG into `output/`.
6. **Explore other examples**  
   - Swap the `--config` path with any Python or YAML/JSON config under `examples/` or `examples/reference/`.

### Prerequisites

Dependencies include:
- numpy>=1.23
- scipy>=1.9
- lmfit>=1.3.4
- pydantic>=2.12 (config validation)
- pyyaml>=6.0
- h5py>=3.14
- joblib>=1.5
- plotly>=5.0
- kaleido>=0.2.1 (for static PNG/SVG/PDF exports)
- tqdm>=4.66 (for CLI progress bars)

### Quick Start (venv)

- Create and activate a virtual environment:
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows PowerShell: `py -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - Windows Command Prompt: `py -m venv .venv && .\.venv\Scripts\activate.bat`
- Install in editable mode: `pip install -e .`

### Running CLI tools

The CLIs now follow the `field`/`freq`/`vary` naming scheme so you can see at a glance whether you are simulating spectra, fitting them, or exploring parameter trends. The compatibility wrapper `nmr-sweep` still exists, but each workflow has a dedicated command:

| Workflow | Single crystal | Powder |
| --- | --- | --- |
| Field-spectrum simulation | `nmr-field-spec --config examples/reference/multisite/field_spectrum.py --out output/field_spectrum_multisite` | `nmr-field-powder --config examples/reference/powder/field_spectrum_borides.py --out output/field_powder_spectrum` |
| Field-spectrum fitting | `nmr-field-fit --config examples/field_fit_multisite.py --out output/field_fit_multisite` | `nmr-field-powder-fit --config examples/field_fit_powder_modern.py --out output/field_powder_fit` |
| Frequency-spectrum simulation | `nmr-freq --config examples/reference/multisite/freq_spectrum.py --out output/freq_spectrum_multisite` | `nmr-freq-powder --config examples/freq_powder_modern.py --out output/freq_powder_spectrum` |
| Frequency-spectrum fitting | `nmr-freq-fit --config examples/freq_fit_multisite.py --out output/freq_fit_multisite` | `nmr-freq-powder-fit --config examples/freq_fit_powder_modern.py --out output/freq_powder_fit` |
| Parameter trends | `nmr-vary-eta --config examples/freq_vs_eta.py --out output/freq_vs_eta`  \| `nmr-vary-angle --config examples/freq_vs_angle.py --out output/freq_vs_angle` | (powder variants tracked in TODO) |

- Each CLI now ships with at least one heavily commented CONFIG file under
`examples/reference/`. Those files preserve the exact parameter sets that used
to live in the legacy scripts while documenting every field inline. Powder
plotting/fitting configs still expose a `sim_type` switch so you can choose the
full exact-diagonalisation path (`exact diag`) or the faster perturbative helper
(`2nd order`). Use `n_samples` to control the random orientation pool: ED clamps
the cache to 2,000 orientations for speed, while the 2nd-order helper directly
uses `nrands = n_samples`.

### Configuration formats (Python, YAML, or JSON)

All CLI entry points now accept Python modules that expose a top-level `CONFIG` dictionary. New examples (e.g., `examples/freq_vs_field.py` and everything under `examples/reference/`) use this format so parameters live in regular Python rather than YAML. YAML/JSON remain fully supported for older configs; several variations are still provided as `.yaml` files while their conversions happen.

A JSON variant (`examples/reference/multisite/freq_spectrum.json`) is included for users who prefer JSON or integrate with tooling that emits JSON. Usage is identical:

```bash
nmr-freq \
  --config examples/reference/multisite/freq_spectrum.json \
  --out output/freq_spectrum_multisite_json --no-show
```

For a ready-to-run fit, the commented CONFIG at `examples/freq_fit_multisite.py` mirrors the legacy parameters and can be invoked via:

```bash
nmr-freq-fit --config examples/freq_fit_multisite.py --out output/freq_fit_multisite --max-nfev 200 --no-show
```

Field-swept multisite spectra follow the same pattern via:

```bash
nmr-field-spec \
  --config examples/reference/multisite/field_spectrum.py \
  --out output/field_spectrum_multisite --no-show
```

And the matching fitter uses:

```bash
nmr-field-fit --config examples/field_fit_multisite.py --out output/field_fit_multisite --max-nfev 200 --no-show
```

Powder spectra (frequency axis) now have a consolidated CLI as well:

```bash
nmr-freq-powder \
  --config examples/freq_powder_modern.py \
  --out output/freq_powder_modern --no-show
```

You can fit powder spectra with:

```bash
nmr-freq-powder-fit \
  --config examples/freq_fit_powder_modern.py \
  --out output/freq_powder_fit --max-nfev 200 --no-show
```

Field-swept powder spectra mirror this flow:

```bash
nmr-field-powder \
  --config examples/field_powder_modern.py \
  --out output/field_powder_modern --no-show
```

```bash
nmr-field-powder-fit \
  --config examples/field_fit_powder_modern.py \
  --out output/field_powder_fit --max-nfev 200 --no-show
```

### CLI parameter names

`nmr-field`, `nmr-freq`, `nmr-elevels`, `nmr-vary-eta`, `nmr-vary-angle`, and `nmr-hyperfine` share canonical YAML keys so configs can be reused between tools:

- External fields use `B_min_T`, `B_max_T`, `B_points`, and `B0_T` (fixed field).
- Shift/quad parameters use `Ka/Kb/Kc` (percent) and `vQ_MHz` (a.k.a. `vc`).
- Euler angles are `phi`, `theta`, `psi` in radians; legacy degree names like
  `phi_z_deg`, `theta_x_prime_deg`, and `psi_z_prime_deg` are accepted and
  converted automatically.
- Powder-angle variations use `angle_name`, `angle_min_deg`, `angle_max_deg`,
  `angle_points`, and fixed orientations `fixed_phi_deg`, `fixed_theta_deg`,
  `fixed_psi_deg`. Legacy names (`angle_to_vary`, `angle_start`, `angle_stop`,
  `n_angles`, `*_deg_init_list`) map to the canonical keys.
- Hint vectors prefer the PAS tuple `Hint_pas: {x: ..., y: ..., z: ...}` but
  the historic `Hinta/Hintb/Hintc` fields are still honored.

These aliases mean configs written for the legacy scripts (e.g., using
`min_field`, `n_fields`, `vc`, `phi_z_deg`) can be fed straight into
`nmr-field`/`nmr-vary-*`/`nmr-hyperfine` without manual renaming. The compatibility command `nmr-sweep` still parses the old subcommands, but you are encouraged to move to the explicit CLIs listed above.

### Headless usage

All plotting entry points support headless runs (no GUI). Pass `--no-show` to suppress interactive windows and `--save-fig <path>` to write a static image. Use `--fig-format png|svg|pdf` to control the output format and `--dpi <n>` to set the resolution (default 150 dpi). Example:

```bash
nmr-freq --config examples/reference/multisite/freq_spectrum.py --no-show --save-fig output/multisite_freq.png
```

`--no-show` automatically suppresses the progress display in non-TTY environments (e.g., CI). To disable progress bars explicitly use `--no-progress`.

## Powder spectra from internal field distributions

The helper `pysimnmr.hyperfine_distribution.freq_spectrum_from_internal_field_distribution` 
combines individual exact-diagonalisation spectra using a probability distribution 
over internal hyperfine fields. Provide samples in the PAS along with weights and the
function handles normalization as well as optional PAS rotations per sample.
Under the hood it relies on the exact-diagonalisation helper (`freq_spec_ed_mix`), avoiding the legacy `freq_prob_trans_ed` path.

A YAML-driven workflow hangs these parameters off a `hyperfine_distribution` block, for example:

```yaml
hyperfine_distribution:
  samples: 1000
  seed: 20251007
  plane: xy
  angle_distribution:
    type: uniform        # uniform angle in the chosen plane
  magnitude_distribution:
    type: gaussian
    mean_T: 0.20
    sigma_T: 0.04
  weights:
    type: equal           # or provide explicit values

# Alternative (uniform directions with fixed magnitude in the xy plane):
# hyperfine_distribution:
#   samples: 1000
#   seed: 20251007
#   plane: xy
#   angle_distribution:
#     type: uniform
#   magnitude_distribution:
#     type: delta
#     value_T: 0.20
#   weights:
#     type: equal

# Explicit vectors with custom weights:
# hyperfine_distribution:
#   vectors:
#     - [0.0, 0.0, 0.00]
#     - [0.0, 0.0, 0.05]
#   weights:
#     values: [0.7, 0.3]

# Custom generator (requires allow_custom_generators: true):
# allow_custom_generators: true
# hyperfine_distribution:
#   custom_generator:
#     module: my_package.hyperfine_generators
#     function: build_vectors
#     params:
#       samples: 1000
#       seed: 20251007
#   weights:
#     type: equal
```

The helper remains available for direct use from Python:

```python
import numpy as np
from pysimnmr.core import SimNMR
from pysimnmr.hyperfine_distribution import freq_spectrum_from_internal_field_distribution

sim = SimNMR('1H')
freq_axis = np.linspace(18.0, 24.0, 512)
hint_samples = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]])
weights = np.array([0.7, 0.3])

spectrum = freq_spectrum_from_internal_field_distribution(
    sim,
    freq_axis,
    hint_pas_samples_T=hint_samples,
    weights=weights,
    Ka=0.0,
    Kb=0.0,
    Kc=0.0,
    vQ_MHz=0.1,
    eta=0.0,
    H0_T=0.5,
    FWHM_MHz=0.05,
)
```

For a ready-to-run demo with 75As fields confined to the xy-plane, use the provided YAML config:

- `nmr-hyperfine --config examples/hyperfine_distribution_xy_plane.py --out output/75As_xy_internal_fields`
- or equivalently `python examples/internal_field_powder_xy_plane.py --config examples/hyperfine_distribution_xy_plane.py --out output/75As_xy_internal_fields`

Inspect the exported spectrum/HTML artefacts in the chosen directory once the run completes.

When you want an interactive look at the sampled hyperfine fields,
`pysimnmr.plotly_utils.save_internal_field_distribution_html` writes a Plotly-based
3D scatter plot:

```python
from pathlib import Path
from pysimnmr.plotly_utils import save_internal_field_distribution_html

save_internal_field_distribution_html(
    hint_samples,
    weights=weights,
    title='Internal field distribution for 1H',
    out_html=Path('output/internal_fields.html'),
)
```

- Add `--no-show` to suppress windows and `--save-fig <path>` to write images.
- Optional `--fig-format png|svg|pdf` and `--dpi 150` control output.

Example: `nmr-freq --config examples/reference/multisite/freq_spectrum.py --no-show --save-fig output/multisite_freq.png`

## Running the tests

Run the automated suite with `pytest`. The new powder helpers are covered by unit tests:

- `pytest tests/test_hyperfine_distribution.py` checks the weighted spectrum helper against an explicit summation.
- The same module exercises the Plotly exporter (skipped if Plotly is missing).

## Versioning

We follow a simple, human-friendly SemVer policy to keep things predictable:

- MAJOR (X.0.0): breaking API changes or sweeping refactors that require user action.
- MINOR (X.Y.0): backward-compatible features, improvements, or additions.
- PATCH (X.Y.Z): backward-compatible bug fixes, performance/UX improvements, and test/packaging changes.

Release checklist (match the project's style):

- Bump the version in `pyproject.toml` under `[project].version`.
- Add a new entry at the top of `pySimNMR_change_log.txt` using the format:
  - `vX.Y.Z - YYYY-MM-DD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
  - Brief, bullet-style notes of changes and rationale.
- Tag the commit (optional): `git tag -a vX.Y.Z -m "Release vX.Y.Z"` and push tags.

Programmatic version:

- `pysimnmr.__version__` is available at runtime (pulled from package metadata).

## Authors

* **Adam Paul Dioguardi** - *Initial work* - [apdioguardi](https://github.com/apdioguardi)
* **Piotr Lepucki** - *addition of offset, linear, and gaussian backgrounds*


## Acknowledgments

* We thank H.-J. Grafe for help with testing, basic physics, and mentorship. We thank N. J. Curro for mentorship and the pre-python version on which our the exact diagonalization eigen-decomposition methodology is based, and also for discussion of properly encoding the character of the parent eigenstates.

## Documentation Roadmap

- A LaTeX-based user manual is planned to cover:
  - The vectorized exact-diagonalization method and rotation operators used under the hood.
  - Practical usage, config parameters, and examples for each CLI tool.
  - Environment setup and reproducible workflows.
- Sources live under `doc/` (see `doc/pySimNMR_manual.tex` and `doc/references.bib`).
- Build locally with LaTeX:
  - `cd doc`
  - `pdflatex pySimNMR_manual.tex && bibtex pySimNMR_manual && pdflatex pySimNMR_manual.tex && pdflatex pySimNMR_manual.tex`
  - Or: `make -C doc pdf` (uses `latexmk`)
  - Or: `doc/build_manual.sh`

## Project TODOs

See `TODO.md` for a living checklist: CLI config support for legacy scripts, ED consolidation, headless plotting options, config validation, and a path toward a GUI.
