"""Minimal numpy stub for tests that don't require real numpy."""

from __future__ import annotations

import sys

__stub__ = True


# ``numpy.float32`` is commonly used as a callable for type casting.  The
# previous stub exposed ``float32`` as the integer ``0`` which meant calls like
# ``numpy.float32(x)`` resulted in ``TypeError: 'int' object is not
# callable``.  Use the built-in :class:`float` type instead so it can be called
# to perform basic casting.
class _float32(float):  # pragma: no cover - simple type marker
    pass


class _float64(float):  # pragma: no cover - simple type marker
    pass


float32 = _float32
float64 = _float64
class _int64(int):  # pragma: no cover - simple type marker
    pass

int64 = _int64
bool_ = bool


class ndarray(list):
    """Very small stand-in for :class:`numpy.ndarray`."""

    def __init__(self, seq=(), dtype=float32):  # pragma: no cover - simple init
        super().__init__(seq)
        self.dtype = dtype

    def tolist(self) -> list:  # pragma: no cover - trivial helper
        """Return a plain ``list`` copy of the array."""
        return list(self)

    def astype(self, _dtype) -> ndarray:  # pragma: no cover - trivial helper
        """Return ``self`` cast to ``_dtype``."""
        return ndarray(self, dtype=_dtype)

    # basic ``shape`` attribute mimicking numpy's tuple return
    @property
    def shape(self):  # pragma: no cover - trivial helper
        if self and isinstance(self[0], (list, ndarray)):
            return (len(self), len(self[0]))
        return (len(self),)

    @property
    def ndim(self):  # pragma: no cover - simple dimension helper
        return 2 if self and isinstance(self[0], (list, ndarray)) else 1

    def reshape(self, rows: int, cols: int):  # pragma: no cover - simple reshape
        """Very small subset of ``ndarray.reshape`` supporting ``(1, -1)``."""
        if rows != 1:
            raise NotImplementedError("stub only supports reshape to (1, -1)")
        if cols not in (-1, len(self)):
            raise ValueError("invalid reshape dimensions")
        return ndarray([ndarray(self)])

    # --- Arithmetic and comparison helpers ---------------------------------
    def __truediv__(self, other):  # pragma: no cover - simple elementwise op
        if isinstance(other, (int, float)):
            return ndarray([x / other for x in self])
        if isinstance(other, ndarray):
            if len(other) == 1:
                div = other[0]
                if isinstance(div, (ndarray, list)):
                    div = div[0] if div else 1
                if div == 0:
                    div = 1
                return ndarray([x / div for x in self])
            result = []
            for x, y in zip(self, other, strict=False):
                if isinstance(y, (list, ndarray)):
                    y_val = y[0] if y else 1
                else:
                    y_val = y
                if isinstance(x, (list, ndarray)):
                    result.append([val / y_val for val in x])
                else:
                    result.append(x / y_val)
            return ndarray(result)
        return NotImplemented

    def __eq__(self, other):  # pragma: no cover - simple elementwise op
        if isinstance(other, ndarray):
            return ndarray([x == y for x, y in zip(self, other, strict=False)])
        if self and isinstance(self[0], (list, ndarray)):
            return ndarray([ndarray([val == other for val in row]) for row in self])
        return ndarray([x == other for x in self])

    def __setitem__(self, key, value):  # pragma: no cover - mask assignment
        if isinstance(key, ndarray):
            for i, (mask, row) in enumerate(zip(key, self, strict=False)):
                # ``mask`` may itself be an ``ndarray`` when the boolean index
                # has an extra dimension (e.g. the result of ``arr == 0`` on a
                # 2-D array).  Checking its truthiness directly would always be
                # ``True`` because non-empty lists are truthy, so we instead
                # inspect the contained boolean value.
                if isinstance(mask, (list, ndarray)):
                    mask_val = bool(mask[0]) if mask else False
                else:
                    mask_val = bool(mask)
                if mask_val:
                    if isinstance(row, list):
                        super().__setitem__(i, [value])
                    else:
                        super().__setitem__(i, value)
            return None
        if isinstance(key, list):
            # Basic fancy indexing support used in the tests.  When ``value`` is
            # a 2-D structure we assign row by row; otherwise we broadcast the
            # value to each index.
            if isinstance(value, (list, ndarray)) and key:
                for idx, val in zip(key, value, strict=False):
                    super().__setitem__(idx, val)
            else:
                for idx in key:
                    super().__setitem__(idx, value)
            return None
        return super().__setitem__(key, value)


def _zeros(shape) -> ndarray:
    if isinstance(shape, int):
        return ndarray([0] * shape)
    if isinstance(shape, (tuple, list)):
        if len(shape) == 2:
            return ndarray([ndarray([0] * shape[1]) for _ in range(shape[0])])
        if len(shape) == 1:
            return ndarray([0] * shape[0])
    return ndarray([])


def empty(shape, dtype=None):
    return _zeros(shape)


def zeros(shape, dtype=None):
    return _zeros(shape)


def array(seq, dtype=float32):
    """Return an :class:`ndarray` built from ``seq``."""

    if isinstance(seq, ndarray):
        seq = list(seq)
    if seq and isinstance(seq[0], (list, tuple, ndarray)):
        return ndarray([array(s, dtype=dtype) for s in seq], dtype=dtype)
    cast = dtype if callable(dtype) else float
    return ndarray([cast(x) for x in seq], dtype=dtype)


def stack(list_of_arrays):
    """Stack ``list_of_arrays`` into a 2-D :class:`ndarray`."""

    return array(list_of_arrays)


def vstack(arrays):
    """Compatibility wrapper calling :func:`stack`."""

    return stack(arrays)


def asarray(arr, dtype=None):
    return arr if isinstance(arr, ndarray) and dtype is None else array(arr, dtype)


def dot(a, b):
    """Compute the inner product of two iterables."""

    return sum(float(x) * float(y) for x, y in zip(a, b, strict=False))


def argsort(arr):
    return sorted(range(len(arr)), key=lambda i: arr[i])


def array_equal(a, b):  # pragma: no cover - simple helper
    return array(a).tolist() == array(b).tolist()


def isclose(a, b, *, rtol: float = 1e-05, atol: float = 1e-08):
    """Return ``True`` if ``a`` and ``b`` are approximately equal.

    This lightweight implementation mirrors :func:`numpy.isclose` for the
    simple scalar values used throughout the test-suite.  Inputs may be plain
    numbers or :class:`ndarray` instances (in which case the first element is
    compared).  The behaviour is intentionally minimal but sufficient for tests
    that only require a truthy/falsey answer without needing the full numpy
    broadcasting rules.
    """

    def _to_float(x):
        if isinstance(x, ndarray):
            # ``ndarray`` instances in the stub behave like lists; when a 1-D
            # array is provided we simply compare its first element which
            # mirrors the usage in the tests.
            x = x[0] if x else 0.0
        return float(x)

    a_f = _to_float(a)
    b_f = _to_float(b)
    return abs(a_f - b_f) <= (atol + rtol * abs(b_f))


def allclose(a, b, *, rtol: float = 1e-05, atol: float = 1e-08):
    """Return ``True`` if all elements of ``a`` and ``b`` are approximately equal.

    This is a very small subset of :func:`numpy.allclose` implemented purely for
    the test-suite.  Inputs may be scalars, lists or :class:`ndarray` instances
    (including nested structures).  The function recursively flattens both
    inputs and performs a pairwise :func:`isclose` comparison.
    """

    def _flatten(x):
        if isinstance(x, ndarray):
            x = x.tolist()
        if isinstance(x, list):
            for item in x:
                yield from _flatten(item)
        else:
            yield x

    a_flat = list(_flatten(a))
    b_flat = list(_flatten(b))
    if len(a_flat) != len(b_flat):
        return False
    return all(isclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a_flat, b_flat, strict=False))


class random:
    _counter = 0

    @staticmethod
    def rand(*shape):
        """Return a deterministic array with the requested shape.

        Values increase linearly which is sufficient for tests that only need
        non-zero, repeatable data without relying on external randomness.
        """

        if not shape:
            return ndarray([0.0])
        random._counter += 1
        base = random._counter
        if len(shape) == 1:
            n = int(shape[0])
            return ndarray([(base + i) / max(n, 1) for i in range(n)])
        first, *rest = shape
        return ndarray([random.rand(*rest) for _ in range(int(first))])

    @staticmethod
    def default_rng(_seed):
        class _RNG:
            def normal(self, size):
                return ndarray([0] * size)

        return _RNG()


# Expose ``numpy.random`` as an actual module so ``import numpy.random`` works
# with the stub.  The class defined above provides the required interface which
# is sufficient for the tests.
sys.modules[__name__ + ".random"] = random  # type: ignore[assignment]


def isscalar(x) -> bool:
    """Return ``True`` if ``x`` is a scalar value.

    The real :func:`numpy.isscalar` checks whether *x* is a Python or numpy
    scalar.  For the purposes of these tests we simply treat lists, tuples and
    ``ndarray`` instances as non-scalars and everything else as a scalar.
    """

    return not isinstance(x, (list, tuple, ndarray))


class linalg:
    @staticmethod
    def norm(arr, axis=None, keepdims=False):
        """Compute a basic Euclidean norm.

        For 1-D arrays the standard vector norm is returned.  For 2-D arrays
        ``axis=1`` computes row-wise norms.  ``keepdims`` preserves the
        dimensions of the result similarly to :func:`numpy.linalg.norm`.
        """

        arr = array(arr, dtype=float)
        if axis is None:
            # Flatten any nested arrays and compute the norm.
            if arr and isinstance(arr[0], (list, ndarray)):
                values = [x for row in arr for x in row]
            else:
                values = arr
            result = (sum(x * x for x in values)) ** 0.5
            if keepdims:
                return ndarray([result])
            return result

        if axis in (1, -1):
            if arr and not isinstance(arr[0], (list, ndarray)):
                arr = [arr]
            norms = [(sum(x * x for x in row)) ** 0.5 for row in arr]
            if keepdims:
                return ndarray([[n] for n in norms])
            return ndarray(norms)

        raise NotImplementedError("Only axis=None or axis=1 supported")


class _testing:
    @staticmethod
    def assert_array_equal(a, b):  # pragma: no cover - simple helper
        if array(a).tolist() != array(b).tolist():
            raise AssertionError("arrays are not equal")


testing = _testing()

__all__ = [
    "argsort",
    "array",
    "asarray",
    "dot",
    "empty",
    "float32",
    "linalg",
    "ndarray",
    "random",
    "stack",
    "testing",
    "vstack",
    "zeros",
]

# Provide ``numpy.ndarray`` alias used throughout the codebase.
ndarray = ndarray
