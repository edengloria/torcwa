# TORCWA v2 Fourier Convolution Operator Review

Date: 2026-04-24

## Summary

The validation prototype computes the same truncated Fourier convolution as the
legacy dense `_material_conv` matrix, but applies it directly to vectors without
storing the dense matrix.  It is **not** connected to the solver path.

Conclusion: keep the operator as a validation/research helper for now.  It is
accurate and differentiable in smoke checks, but the direct implementation is
slower than dense assembly/matmul at the measured sizes, and the current dense
eigensolver still requires dense `P @ Q` matrices.

## Validation Results

Commands:

```bash
python3 benchmarks/fourier_operator_review.py --quick --devices cpu
python3 benchmarks/fourier_operator_review.py --quick --devices cuda
```

Representative results:

```text
| device | dtype | case | size | dense ms | operator ms | max abs | rel l2 | grad |
|---|---|---|---|---:|---:|---:|---:|---|
| cpu | complex64 | rectangle | order=[3,3] grid=32x32 rhs=3 | 0.243 | 0.806 | 9.611e-07 | 1.423e-07 | True |
| cpu | complex128 | rectangle | order=[3,3] grid=32x32 rhs=3 | 0.103 | 0.832 | 1.351e-15 | 2.318e-16 | True |
| cuda | complex64 | rectangle | order=[3,3] grid=32x32 rhs=3 | 0.320 | 5.414 | 9.555e-07 | 1.591e-07 | True |
| cuda | complex128 | rectangle | order=[3,3] grid=32x32 rhs=3 | 0.408 | 4.813 | 2.747e-15 | 3.555e-16 | True |
```

The same pattern holds for random and circular material grids: the operator
matches the dense convention within dtype tolerance, but it is slower because it
uses a Python loop over retained Fourier modes.

## Adoption Decision

- **Do not use as the default solver path yet.**  Dense eigensolves dominate the
  non-homogeneous layer path, so replacing only `conv @ vector` does not remove
  the need to materialize dense matrices.
- **Safe to keep as a correctness oracle.**  The helper is useful for checking
  Fourier indexing and future operator/FFT implementations against the legacy
  dense convention.
- **Revisit when the eigensolver becomes operator-aware.**  A practical speed or
  memory win likely requires an iterative/operator eigensolver or an FFT-backed
  convolution matvec with exact padding/truncation tests.

## Requirements Before Solver Integration

- Dense-vs-operator validation for `order=[10,10]`, `[15,15]`, and `[20,20]`
  with multiple RHS counts.
- `complex128` gradcheck for material tensors and geometry-derived tensors.
- External RCWA fixture comparison after operator coupling into any eigensolver
  or field reconstruction path.
- Clear fallback to the dense path for small orders where dense assembly remains
  faster.
