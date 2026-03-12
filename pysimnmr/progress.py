from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from typing import Iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    tqdm = None


class _BaseBar:
    def update(self, n: int = 1) -> None:
        ...

    def complete(self) -> None:
        ...

    def close(self) -> None:
        ...


class _TqdmBar(_BaseBar):
    def __init__(self, total: int, desc: str, disable: bool) -> None:
        self._bar = None
        if not disable and tqdm is not None:
            self._bar = tqdm(total=total, desc=desc, leave=False)

    def update(self, n: int = 1) -> None:
        if self._bar:
            self._bar.update(n)

    def complete(self) -> None:
        if self._bar and self._bar.total is not None:
            remaining = self._bar.total - self._bar.n
            if remaining > 0:
                self._bar.update(remaining)

    def close(self) -> None:
        if self._bar:
            self._bar.close()


class _NullBar(_BaseBar):
    def update(self, n: int = 1) -> None:  # pragma: no cover - trivial
        return

    def complete(self) -> None:  # pragma: no cover - trivial
        return

    def close(self) -> None:  # pragma: no cover - trivial
        return


class _SimpleBar(_BaseBar):
    """Fallback textual progress indicator when tqdm is unavailable."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = max(total, 1)
        self.desc = desc
        self.current = 0
        self._print()

    def _print(self) -> None:
        percent = (self.current / self.total) * 100.0
        msg = f"{self.desc}: {self.current}/{self.total} ({percent:5.1f}%)"
        print(msg, end='\r', file=sys.stderr, flush=True)

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        self._print()

    def complete(self) -> None:
        self.current = self.total
        self._print()

    def close(self) -> None:
        print(file=sys.stderr)


class ProgressManager:
    """Factory for consistent progress bars across CLIs."""

    def __init__(self, enabled: bool = True) -> None:
        if enabled and tqdm is None:
            warnings.warn(
                "Progress bars require the 'tqdm' package; install it or pass --no-progress to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._enabled = enabled
        self._disable_render = not sys.stderr.isatty()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bar(self, total: int, desc: str) -> _BaseBar:
        if not self._enabled:
            return _NullBar()
        if tqdm is not None:
            disable = self._disable_render
            return _TqdmBar(total=total, desc=desc, disable=disable)
        return _SimpleBar(total=total, desc=desc)

    @contextmanager
    def iter(self, iterable, desc: str) -> Iterator:
        if not self._enabled:
            yield iterable
            return
        if tqdm is None or self._disable_render:
            yield iterable
            return
        with tqdm(iterable, desc=desc, leave=False) as bar:
            yield bar


def is_tqdm_available() -> bool:
    return tqdm is not None


@contextmanager
def joblib_progress(bar: _BaseBar) -> Iterator[None]:
    """Patch joblib so each completed batch updates the provided bar."""
    if isinstance(bar, _NullBar):
        yield
        return
    try:
        import joblib.parallel
    except Exception:  # pragma: no cover - joblib always present in package deps
        yield
        return

    original_callback = joblib.parallel.BatchCompletionCallBack

    class _ProgressCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            bar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _ProgressCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
