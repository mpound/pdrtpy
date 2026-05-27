# Bug: `ValueError: inconsistent shapes` when using `jac_sparsity` with `method='least_squares'`

**lmfit version:** 1.3.4
**scipy version:** ≥ 1.9 (tested with 1.17.1)
**Python version:** 3.12

**Version information:**
Python: 3.12.13 (main, Mar  3 2026, 14:59:34) [Clang 21.1.4 ]
lmfit: 1.3.4, scipy: 1.17.1, numpy: 1.26.4,asteval: 1.0.8, uncertainties: 3.2.3

Yes, I read the instructions and I am sure this is a GitHub Issue.

Full disclosure: This issue was uncovered by Claude while performance optimizing my code which uses lmfit.

---

## Description

Calling `Minimizer.minimize(method='least_squares', jac_sparsity=...)` raises
`ValueError: inconsistent shapes` during covariance estimation when the number of
residuals exceeds the number of parameters (the usual case).

The crash is in `minimizer.py:1596`, which computes the approximate Hessian using
element-wise multiplication on a non-square sparse matrix:

```python
# minimizer.py line 1595–1596
if issparse(ret.jac):
    hess = (ret.jac.T * ret.jac).toarray()   # BUG: * is element-wise in scipy >= 1.9
```

`ret.jac` has shape `(m, n)` where `m` = number of residuals and `n` = number of
parameters. `ret.jac.T` has shape `(n, m)`. When `m ≠ n`, the element-wise `*`
requires matching shapes and raises `ValueError: inconsistent shapes (n, m) and (m, n)`.

In scipy < 1.9, `*` on sparse matrices meant matrix multiplication (equivalent to `@`),
so the code worked. scipy 1.9 aligned sparse matrix arithmetic with NumPy dense array
semantics, making `*` element-wise. The lmfit code was never updated.

---

## Minimal Failing Example

```python
import numpy as np
from scipy.sparse import lil_matrix
from lmfit import Minimizer, Parameters

# Simple linear model: y = a*x + b, 4 observations, 2 parameters
x_data = np.array([1.0, 2.0, 3.0, 4.0])
y_data = np.array([2.1, 4.0, 5.9, 8.1])

def residual(params):
    a = params['a'].value
    b = params['b'].value
    return a * x_data + b - y_data   # 4 residuals

params = Parameters()
params.add('a', value=1.0)
params.add('b', value=0.0)

# Jacobian sparsity: 4 residuals × 2 parameters, all entries non-zero
sparsity = lil_matrix((4, 2), dtype=np.int8)
sparsity[:, :] = 1
sparsity = sparsity.tocsr()

mini = Minimizer(residual, params)
result = mini.minimize(method='least_squares', jac_sparsity=sparsity)
# ValueError: inconsistent shapes (2, 4) and (4, 2)
```

**Traceback:**
```
Traceback (most recent call last):
  File "...", line N, in <module>
    result = mini.minimize(method='least_squares', jac_sparsity=sparsity)
  ...
  File ".../lmfit/minimizer.py", line 1596, in least_squares
    hess = (ret.jac.T * ret.jac).toarray()
  ...
ValueError: inconsistent shapes (2, 4) and (4, 2)
```

---

## Root Cause

`scipy.sparse` matrix `*` operator semantics changed in **scipy 1.9**:

| scipy version | `A * B` for sparse matrices |
|---|---|
| < 1.9 | Matrix multiplication (same as `A @ B`) |
| ≥ 1.9 | Element-wise multiply (same as `np.multiply(A, B)`) |

`ret.jac` (shape `m × n`) and `ret.jac.T` (shape `n × m`) have different shapes when
`m ≠ n`, so element-wise `*` raises `ValueError`.

The same pattern appears on line 1599 for `LinearOperator`, but `LinearOperator.__mul__`
still performs matrix multiplication, so that branch is unaffected.

---

## Suggested Fix

Replace element-wise `*` with the matrix-multiply operator `@` on line 1596:

```python
# minimizer.py line 1595–1596  (current, broken on scipy >= 1.9)
if issparse(ret.jac):
    hess = (ret.jac.T * ret.jac).toarray()

# Fix:
if issparse(ret.jac):
    hess = (ret.jac.T @ ret.jac).toarray()
```

`@` calls `__matmul__` which performs matrix multiplication for both dense and sparse
arrays in all scipy versions.

---

## Impact

Any user who passes `jac_sparsity` to `Minimizer.minimize(method='least_squares')` with
more residuals than parameters (i.e., virtually all real fitting problems) will hit this
crash on scipy ≥ 1.9.

The workaround is to call `scipy.optimize.least_squares` directly and extract per-parameter
standard errors manually from the Jacobian:

```python
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np

result = least_squares(fun, x0, bounds=bounds, jac_sparsity=sparsity, method='trf')

# Manual covariance from Jacobian (works for both sparse and dense)
J = result.jac
if hasattr(J, 'toarray'):
    J = J.toarray()
try:
    cov = np.linalg.inv(J.T @ J)
    stderr = np.sqrt(np.diag(cov))
except np.linalg.LinAlgError:
    stderr = np.full(len(x0), None)
```
