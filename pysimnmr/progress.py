from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    tqdm = None


def _needs_ascii() -> bool:
    """Return True when the terminal cannot render Unicode block characters.

    tqdm defaults to Unicode block-fill characters which display as '?' in
    Windows cmd.exe and other terminals that lack full UTF-8 support.  Using
    ascii=True switches to plain '#' fill characters that work everywhere.
    """
    if sys.platform == 'win32':
        return True
    encoding = getattr(sys.stderr, 'encoding', None) or ''
    return encoding.lower().replace('-', '') not in {'utf8', 'utf-8', 'utf_8'}


class _BaseBar:
    def update(self, n: int = 1) -> None:
        ...

    def complete(self) -> None:
        ...

    def close(self) -> None:
        ...

    def set_description(self, desc: str) -> None:
        ...


class _TqdmBar(_BaseBar):
    def __init__(self,
                 total: Optional[int],
                 desc: str,
                 disable: bool,
                 leave: bool = True) -> None:
        self._bar = None
        if not disable and tqdm is not None:
            self._bar = tqdm(
                total=total,
                desc=desc,
                leave=leave,
                ascii=_needs_ascii(),
                dynamic_ncols=True,
            )

    def update(self, n: int = 1) -> None:
        if self._bar:
            self._bar.update(n)

    def complete(self) -> None:
        # Only force-advance to total when the total is known; indeterminate
        # bars (total=None) are left at their current count.
        if self._bar and self._bar.total is not None:
            remaining = self._bar.total - self._bar.n
            if remaining > 0:
                self._bar.update(remaining)

    def close(self) -> None:
        if self._bar:
            self._bar.close()

    def set_description(self, desc: str) -> None:
        if self._bar:
            self._bar.set_description(desc)


class _NullBar(_BaseBar):
    def update(self, n: int = 1) -> None:  # pragma: no cover - trivial
        return

    def complete(self) -> None:  # pragma: no cover - trivial
        return

    def close(self) -> None:  # pragma: no cover - trivial
        return

    def set_description(self, desc: str) -> None:  # pragma: no cover - trivial
        return


class _SimpleBar(_BaseBar):
    """Fallback textual progress indicator when tqdm is unavailable."""

    def __init__(self,
                 total: Optional[int],
                 desc: str,
                 leave: bool = True) -> None:
        self.total = max(total, 1) if total is not None else None
        self.desc = desc
        self.current = 0
        self.leave = leave
        self._print()

    def _print(self) -> None:
        if self.total is not None:
            percent = (self.current / self.total) * 100.0
            msg = f"{self.desc}: {self.current}/{self.total} ({percent:5.1f}%)"
        else:
            msg = f"{self.desc}: {self.current} iterations"
        print(msg, end='\r', file=sys.stderr, flush=True)

    def update(self, n: int = 1) -> None:
        if self.total is not None:
            self.current = min(self.total, self.current + n)
        else:
            self.current += n
        self._print()

    def complete(self) -> None:
        if self.total is not None:
            self.current = self.total
        self._print()

    def set_description(self, desc: str) -> None:
        self.desc = desc
        self._print()

    def close(self) -> None:
        if self.leave:
            # Print a final newline so the completed bar stays visible.
            print(file=sys.stderr)
        else:
            # Erase the transient line.
            print('\r' + ' ' * 80 + '\r', end='', file=sys.stderr, flush=True)


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

    def bar(self,
            total: Optional[int],
            desc: str,
            leave: bool = True) -> _BaseBar:
        """Create a progress bar.

        Parameters
        ----------
        total:
            Expected number of steps.  Pass ``None`` for an indeterminate bar
            that shows a running count without a percentage (useful for fitting
            loops where the total number of evaluations is not known ahead of
            time).
        desc:
            Short label shown to the left of the bar.
        leave:
            If True (default), the bar remains visible after it completes.
            Pass False for transient inner bars (e.g., per-site solver bars)
            that should disappear once a site finishes.
        """
        if not self._enabled:
            return _NullBar()
        if tqdm is not None:
            disable = self._disable_render
            return _TqdmBar(total=total, desc=desc, disable=disable, leave=leave)
        return _SimpleBar(total=total, desc=desc, leave=leave)

    @contextmanager
    def iter(self, iterable, desc: str) -> Iterator:
        if not self._enabled:
            yield iterable
            return
        if tqdm is None or self._disable_render:
            yield iterable
            return
        with tqdm(iterable, desc=desc, leave=False,
                  ascii=_needs_ascii(), dynamic_ncols=True) as bar:
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
