from pysimnmr.core import SimNMR


def test_rotation_matrix_cache_lives_in_output(tmp_path, monkeypatch):
    """rotation_matrices.h5 should be created under the configured output directory."""
    out_dir = tmp_path / "output-cache"
    monkeypatch.setenv("PYSIMNMR_OUTPUT_DIR", str(out_dir))
    cache_path = out_dir / "rotation_matrices.h5"
    if cache_path.exists():
        cache_path.unlink()

    sim = SimNMR("1H")
    sim.random_rotation_matrices(['1H'], recalc_random_samples=True, n_samples=32)
    assert cache_path.exists()
    before_mtime = cache_path.stat().st_mtime

    # Second call should reuse the cached file instead of rewriting it.
    sim.random_rotation_matrices(['1H'], recalc_random_samples=False, n_samples=32)
    assert cache_path.stat().st_mtime == before_mtime
