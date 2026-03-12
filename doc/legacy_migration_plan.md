# Modernizing pySimNMR Legacy Scripts

This note records the detailed plan for removing the "legacy" label from the
plotting/fitting stack. It captures the Phase 0 inventory and the concrete
Phase 1 tasks so we can resume the migration at any point.

## Goals

- Keep every existing CLI workflow (powder patterns, multisite spectra, fitting)
  available under first-class entry points instead of `legacy_scripts.*`.
- Share one implementation of the ED/2nd-order kernels so physics logic lives in
  `pysimnmr`, not inside scripts.
- Preserve the example parameter sets that currently live inside the scripts'
  `CONFIG` dicts; move them into dedicated, commented config files so newcomers
  understand every knob the first time they open a file.
- Provide schema validation + alias handling for the migrated CLIs to guarantee
  friendly error messages and compatibility with previous config names.
- Establish smoke tests for each CLI path (ED + 2nd-order, tiny grids) so CI
  guards the refactor.

### Command naming grid

| Workflow | Single crystal CLI/module | Powder CLI/module |
| --- | --- | --- |
| Field-spectrum simulation | `nmr-field-spec`, `nmr-field-powder` | powder CLI built on `pysimnmr.powder_field_cli` |
| Field-spectrum fitting | `nmr-field-fit`, `nmr-field-powder-fit` | powder fitter CLI replacing the legacy script |
| Frequency-spectrum simulation | `nmr-freq` | `nmr-freq-powder` |
| Frequency-spectrum fitting | `nmr-freq-fit`, `nmr-freq-powder-fit` | `nmr-pp-freq-fit` (existing legacy script for powder) |
| Parameter trends (eta/angle) | `nmr-vary-eta`, `nmr-vary-angle` | _planned powder variants_ |

The "vary" commands replaced the legacy "nmr-sweep" multi-tool. The old command remains as a thin shim that forwards to the new modules while emitting a `DeprecationWarning`.

## Phase 0 Inventory (completed)

| Script | Category | Notes |
| --- | --- | --- |
| `plot_freq_vs_field.py`, `plot_elevels_vs_field.py`, `plot_freq_vs_eta.py`, `plot_freq_vs_angle.py` | Single-crystal variations | Already refactored once to load CONFIG dicts, but still embed plotting/export code and rely on `pySimNMR.SimNMR`. `plot_freq_vs_angle` supports multiple sites and weighted scatter sizes, something the new `nmr-field`/`nmr-vary-*` CLIs currently lack. |
| `plot_freq_powder_spectrum.py`, `plot_field_powder_spectrum.py` | Powder plotting | Multi-site support, ED vs 2nd-order switch, orientation caching, background corrections, optional experimental overlays, ability to dump individual site spectra. Field version already plugs into `pysimnmr.progress`. |
| `plot_freq_spectrum_multisite.py`, `plot_field_spectrum_multisite.py` | Multisite single-crystal spectra | Similar ED/2nd-order split, background control, experimental overlay scaling, file exports. |
| `fit_freq_powder_spectrum_devel.py`, `fit_freq_spectrum_multisite.py`, `fit_freq_vs_angle.py` | LMFit-based fitting flows | Accept lists of parameter definitions (value, vary flag, bounds, expressions). They re-use the powder/multisite simulation kernels but wire them to lmfit minimizers and optional constraints. |

Existing helpers that we can reuse:

- `pysimnmr.vary_helpers` - example of moving variation logic out of scripts.
- `pysimnmr.config_aliases` / `config_schema` - canonical keys + Pydantic models.
- `pysimnmr.progress.ProgressManager` - progress bars already used in field powder CLI.
- `pysimnmr.plotting_utils` - axis normalization + limit helpers.
- `pysimnmr.hyperfine_cli` - reference for a modern CLI that reads configs, normalizes keys, and orchestrates plotting/export helpers.

## Phase 1 Design (in progress)

### Module layout

1. **`pysimnmr/powder_cli_helpers.py`**
   - Contains pure functions for frequency- and field-swept powder spectra.
   - Exposes `PowderSpectrumConfig` dataclasses/Pydantic models for ED and 2nd-order.
   - Handles orientation sampling caches, ED vs 2nd-order branching, background correction, export of per-site spectra, and optional experimental overlays (returning structured data so CLI layers can plot/save consistently).
   - Plumbs `ProgressManager` so GUI/CLI behavior stays consistent.

2. **`pysimnmr/multisite_cli_helpers.py`**
   - Shared logic for `freq_spectrum_multisite` and `field_spectrum_multisite`.
   - Accepts site lists, hint vectors, convolution settings, etc., and returns stacked spectra ready for plotting/export.

3. **`pysimnmr/fitting_cli.py` (or `fitting_helpers.py`)**
   - Wraps the powder/multisite kernels with LMFit parameter parsing (value/vary/bounds expressions).
   - Keeps the list-driven config style but converts inputs into strongly-typed objects so we can validate before starting minimization.
   - Provides helper functions to build lmfit `Parameters`, run the configured minimizer, and emit best-fit spectra.

4. **`pysimnmr/config_schema.py` extensions**
   - Add Pydantic models for each new CLI (powder freq/field, multisite freq/field, powder/multisite fitting).
   - Provide strict typing for site lists, convolution settings, sample counts, etc.
   - Bake in safe defaults (e.g., clamp `n_samples` for ED) and per-field descriptions for eventual docs/autocomplete.

5. **`pysimnmr/config_aliases.py` extensions**
   - Map historic keys (e.g., `min_freq`, `max_field`, `vc_list`, `Hinta_list`, background aliases) into canonical names consumed by the new schemas.
   - Ensure both `vc_list` and `vQ_list` inputs work; convert single-value fields (e.g., `H0`) into the new names (`B0_T` or `freq_axis_MHz`).

### Single-crystal frequency simulation plan

Next up is migrating `legacy_scripts/plot_freq_spectrum_multisite.py` (and the matching fitter) into first-class helpers/CLIs:

1. **Helper/module structure**
   - Add `pysimnmr/freq_simulation.py` with a `SingleCrystalFreqConfig` Pydantic model plus a `simulate_single_crystal_freq(config)` function returning:
     - The frequency axis (MHz).
     - Per-site spectra (already scaled by multiplicity).
     - The summed/normalized spectrum, background corrections applied separately so callers can choose how to render.
     - Optional experimental data traces (loaded by the CLI, not the helper).
   - Distinguish between ED vs 2nd-order via an enum on the config (`sim_mode: Literal["ed","perturbative"]`). The helper orchestrates:
     - Shared orientation handling (ED uses lists of ZXZ Euler angles; perturbative uses lab-frame `theta/phi`). Convert legacy lists into canonical arrays before calling `SimNMR`.
     - Internal-field vectors per site, with aliasing for `Hint_list` vs `Hinta_list/Hinta` etc.
   - Provide convenience dataclasses for per-site inputs to keep the config tidy (`SingleSiteParams`).

2. **CLI (`pysimnmr/freq_cli.py`)**
   - Entry point `nmr-freq` (simulation only) that:
     - Loads Python/YAML/JSON configs.
     - Normalizes aliases (reuse `normalize_common_aliases`, new `normalize_freq_config` branch).
     - Validates via `SingleCrystalFreqConfig`.
     - Calls `simulate_single_crystal_freq` and handles plotting/export (Plotly lines/fills with optional experimental overlay plus `.txt/.png/.html` output, same behavior as the legacy script).
   - Keep `legacy_scripts.plot_freq_spectrum_multisite` available during the transition; eventually point a console shim at `nmr-freq`.

3. **Configuration schema highlights**
   - Canonical names:
     - `freq_min_MHz`, `freq_max_MHz`, `freq_points`.
     - `broadening.type` (gauss/lor), `broadening_FWHM_MHz`, `broadening_dvQ_FWHM_MHz`.
     - `site_multiplicity`, `phi_lab_deg`, `theta_lab_deg` for perturbative mode, `phi_z_deg`, `theta_x_prime_deg`, `psi_z_prime_deg` for ED mode. Alias mapper copies whichever set is provided into the canonical fields (ED lists win if both are given).
     - `Hint_pas_list` to house internal fields per site; legacy configs using `Hint_list` or `Hinta_list` convert automatically.
     - Background parameters adopt a structured block (`background: {mode: "offset"/"line"/"gaussian", params: [...]}`) while still understanding the old list-of-lists form.
   - Provide defaults for optional fields (e.g., `site_multiplicity=1`, `plot_individual=True`).

4. **Testing**
   - New pytest module `tests/test_cli_freq.py` covering ED + perturbative paths with tiny configs (1-2 sites, small grids) and verifying plot/data outputs.
   - Unit tests for `simulate_single_crystal_freq` to ensure both modes return consistent shapes and handle alias conversions.

5. **Docs**
   - README grid: add `nmr-freq` row once the CLI lands; highlight how it differs from `nmr-vary-*`.
   - Manual: new subsection for "Single-crystal frequency spectra (`nmr-freq`)" referencing the config schema and CLI usage.
   - Update `examples/` with a commented config file (converted from the legacy `examples/reference/freq_spectrum_multisite_legacy.py`), plus YAML/JSON variants if useful.

6. **Fitting path preview**
   - After the simulation helper is stable, reuse its internals to build `pysimnmr/freq_fit.py` and a `nmr-freq-fit` CLI that wraps LMFit with the same parameter list schema the legacy script uses. This will share the same config model, with additional sections describing variable definitions/bounds.

### CLI & entry point strategy

- Introduce a consolidated CLI (working name `nmr-powder`) with subcommands:
  1. `nmr-powder freq-spectrum` (powder, frequency axis).
  2. `nmr-powder field-spectrum`.
  3. `nmr-powder fit-freq`.
  4. `nmr-powder fit-field` (if needed).
- Extend the new `nmr-field`/`nmr-vary-*` family with multi-site variants (or add a separate CLI module mirroring the hyperfine CLI style).
- Update `pyproject.toml` so the current "legacy" console scripts (`nmr-plot-freq-powder-spectrum`, etc.) import the new CLI entry points and emit `DeprecationWarning`s. This keeps existing users unblocked while we finalize docs.
- Eventually adjust the README to drop the "legacy" term and reference the new CLI commands.

### Config + example files

- Move every `CONFIG` dict currently embedded in `legacy_scripts/*.py` into `examples/powder/`, `examples/multisite/`, and `examples/fitting/`.
- Each config file should:
  - Keep the original parameter values verbatim.
  - Convert to the new canonical schema names.
  - Include inline comments describing each parameter (unit, reasoning, when to tweak). E.g.:
    ```python
    CONFIG = {
        "isotope_list": ["75As"],  # Target nucleus for each crystallographic site
        "Ka_list": [0.25],         # Knight-shift tensor components (%)
        ...
    }
    ```
- Provide YAML equivalents once the Python versions are vetted, so users can choose their preferred format.
- Update `README.md` and `doc/pySimNMR_manual.tex` with a short "configuration anatomy" section pointing to these examples.

### Testing

- Add smoke tests under `tests/cli/` that invoke each new CLI via `subprocess.run`/`pytest` with tiny configs:
  - Powder freq/field (ED with ~32 orientations, 2nd-order with ~500 random draws).
  - Multisite freq/field (two sites, quick grid).
  - Fitting flows using mocked experimental data (already in `tests/artifacts`).
- Ensure the tests assert on generated output files (txt/png) so we know the CLI produced results even when plots are headless.
- Cover both CLI paths and helper functions directly (e.g., verifying alias normalization, background handling, orientation cache reuse).

### Compatibility considerations

- Orientation caches: `plot_freq_powder_spectrum` currently calls `SimNMR.random_rotation_matrices` once per run and reuses the stack across isotopes. We should surface this as a helper inside `powder_cli_helpers` so we can (a) reuse caches between CLI invocations via `.npz` snapshots, (b) allow deterministic seeds for tests, and (c) expose a `cache_path` option in configs.
- `fit_freq_powder_spectrum_devel` warns before running huge ED sims. The new schema should include `confirm_large_jobs` and we can replicate the guard via CLI prompts or a `--yes` flag.
- Plotting: single-crystal scripts currently produce scatter plots colored by transition probability. The modern variation helpers only emit line plots. We should either:
  1. enhance `nmr-field`/`nmr-vary-*` to optionally render the scatter view (via Plotly); or
  2. ship a lightweight plotting utility that consumes the saved txt data and replicates the style.
  Decision: start with a Plotly scatter support in the CLI so parity is immediate.
- Multi-site angle variation: `plot_freq_vs_angle` accepts multiple sites with different initial Euler offsets and hint vectors. The new CLI should allow `isotope_list` plus parallel lists for parameters, matching the powder/multisite style.

## Next steps

1. **Single-crystal frequency simulations** - migrate the plot/fitting scripts that sweep frequency at fixed field (multisite + fitting) into modern helpers/CLIs. [x] Simulation (`nmr-freq`) and fitting (`nmr-freq-fit`) are now in place; the next phases extend the same approach to field/powder workflows.
2. **Single-crystal field simulations** - repeat for the field-swept multisite plot/fitting flows.
3. **Powder frequency simulations** - move `plot_freq_powder_spectrum`/`fit_freq_powder_spectrum_devel` into the new helpers once the single-crystal infrastructure is stable.
4. **Powder field simulations** - finally port the field-swept powder pipeline (including progress bars and joblib integration).

Implementation checklist for each bullet above:

1. Implement the shared helpers/config models for the targeted workflow (single-crystal frequency first), including ED/2nd-order branching, background correction, and export logic.
2. Wire a new CLI module (`pysimnmr/powder_cli.py` or similar) that:
   - Parses configs (Python/YAML/JSON).
   - Validates with the new schemas.
   - Calls the helpers and drives plotting/export (Plotly with HTML + static image outputs).
3. Update `pyproject.toml` entry points and `pysimnmr/legacy_entrypoints.py` to delegate to the new CLI (with warnings).
4. Extract each script's `CONFIG` block into dedicated, commented example files under `examples/`.
5. Add smoke tests covering the new commands.
6. Refresh README/manual sections and remove references to "legacy scripts".

-> See `TODO.md` ("Remove the legacy label from the plotting/fitting stack") for a condensed checklist tied to this document.
