# psfi

**Point spread function interpolation.** Evaluate an integral kernel
Φ(y, x) at arbitrary points by interpolating *transported* impulse responses
sampled at scattered locations — the kernel-approximation core of the
PSF method of [Alger, Hartland, Petra, Ghattas, SISC 2024](https://doi.org/10.1137/23M1584745).
Header-only C++17 with Python bindings; depends on Eigen and
[etree](https://github.com/NickAlger/ellipsoid_tree).

> **Status: work in progress.** The full evaluation pipeline is implemented
> and tested: the impulse-response container, the per-neighbor prediction
> machinery (all frame maps and scalings, with data-requirement validation),
> the RBF layer (six kernels, polynomial tails, ridge smoothing;
> cross-checked against scipy's `RBFInterpolator`), and the threaded
> `KernelEvaluator` (cols-only and symmetric). Examples, docs, and the PyPI
> release are still to come.

## The idea

An operator A with a locally supported integral kernel is probed by applying
it to Dirac combs, giving *batches* of impulse responses φ_i — functions on a
simplicial mesh, each localized inside a support ellipsoid (μ_i, Σ_i). To
evaluate the kernel at an arbitrary pair (y, x), each nearby sample point x_i
predicts

```
f_i = s_i · φ_i(T_i(y)),
```

where the **frame map** T_i identifies the neighborhood of x with the
neighborhood of x_i and the **scaling** s_i corrects the amplitude; the
predictions {(x_i, f_i)} are then combined by radial basis function
interpolation at x. The two axes are independent:

| frame map `T_i(y)` | paper eq. |
| --- | --- |
| `identity`: y | (4.7) |
| `translation`: y − x + x_i | (4.8) |
| `mean_translation`: y − μ(x) + μ_i | (4.9) |
| `whitened_affine`: μ_i + Σ_i^{1/2} Σ(x)^{−1/2} (y − μ(x)) | new |

| scaling `s_i` | meaning |
| --- | --- |
| `none`: 1 | raw values (4.9) |
| `volume`: V(x)/V_i | preserves peak values (4.10) |
| `volume_det`: (V(x)/V_i)·√(det Σ_i / det Σ(x)) | preserves mass under `whitened_affine` |

Here V, μ, Σ are the moment fields of the impulse responses (mass, mean,
covariance), evaluated at sample points and — when a configuration needs
them — at target points via CG1 interpolation of vertex fields. A **support
gate** (`Support.ellipsoid`) zeroes predictions whose transported point falls
outside the sample's ellipsoid at scale τ, which is what isolates individual
impulses within a multi-impulse batch; transported points outside the mesh
exclude the sample entirely (the method never evaluates impulse responses
outside their domain of definition).

**Data requirements are minimal per configuration** and validated up front:
`identity`/`translation` with no gate and no scaling needs nothing but the
batches and sample points (classical scattered-PSF interpolation of
single-impulse batches), while `whitened_affine` + `volume_det` needs all
per-sample moments and all vertex moment fields.
`ImpulseResponseField.validate(config)` lists exactly what is missing.

## Quick look (Python)

```python
import numpy as np, psfi

F = psfi.ImpulseResponseField(vertices, cells)   # (nv, d), (nc, d+1); points are rows
F.set_moment_fields(V, mu, Sigma)                # (nv,), (nv, d), (nv, d, d)
F.add_batch(points, psi, V_i, mu_i, Sigma_i)     # one Dirac-comb response per batch

cfg = psfi.EvalConfig(frame=psfi.Frame.whitened_affine,
                      scaling=psfi.Scaling.volume_det,
                      tau=3.0, num_neighbors=10)
indices, points, values = F.predictions(y, x, cfg)   # per-neighbor predictions at (y, x)

K = psfi.KernelEvaluator(F, config=cfg,
                         rbf=psfi.RBFScheme(kernel=psfi.RBFKernel.gaussian, shape=3.0))
value = K(y, x)         # one kernel entry
B = K.block(yy, xx)     # (num_y, num_x) block, threaded
```

## Building and testing

```sh
cmake -S . -B build && cmake --build build -j $(nproc) && ctest --test-dir build
```

etree is found via `find_package(etree)`, with a pinned FetchContent download
as fallback (for local development against a checkout:
`-DFETCHCONTENT_SOURCE_DIR_ETREE=/path/to/ellipsoid_tree`). Python module:
`pip install .`, or `cmake -B build -DPSFI_BUILD_PYTHON=ON` and use
`build/bindings`. Binding tests (`bindings/tests`) check the C++ against a
pure-numpy reference implementation over every configuration axis.

## Locality (a note for distributed use)

The mesh is *a* mesh, not necessarily the whole domain: in a domain-decomposed
setting each rank holds a submesh plus halo and the samples whose ellipsoids
reach it. All queries are answered from local data; "outside the mesh" is
treated as "outside the domain". Callers must provide enough halo that this
identification is correct for the queries they issue — the membership test is
isolated in one place so a global-domain indicator can plug in later.

## References

- N. Alger, T. Hartland, N. Petra, O. Ghattas, *Point spread function
  approximation of high-rank Hessians with locally supported non-negative
  integral kernels*, SIAM Journal on Scientific Computing 46(3), 2024,
  A1658–A1689 — the method this library extracts and extends.
- [etree](https://github.com/NickAlger/ellipsoid_tree) — geometry layer
  (simplicial meshes, kd-trees, ellipsoid intersections, batch picking).

MIT license.
