# IEEEBound — Certified Interval-Arithmetic & Taylor-Model Global Optimizer

A self-contained, dependency-free Rust workspace for **certified global
optimization, root-finding, and range enclosure** using **interval arithmetic**
and **Taylor models**. Every enclosure is mathematically sound: arithmetic is
outward-rounded against IEEE-754, and out-of-domain inputs are handled by
domain clamping / empty intersection rather than by leaking `NaN`.

## Workspace layout

| Crate | Purpose |
|-------|---------|
| `veribound-interval` | Outward-rounded `Interval` (`[lo, hi]`, `∅`, `(−∞,+∞)`), set ops, elementary functions, `extended_div` (0/1/2 pieces), integer powers, `BoxDomain`. |
| `veribound-expr`     | `RefCell`-arena `Expr` AST, `f64`/interval evaluation, reverse-mode **interval AD** `(value, gradient)`, benchmark builders. |
| `veribound-taylor`   | Multivariate `TaylorModel` (interval coefficients), polynomial add/mul/truncation, Taylor series for `exp/sin/cos/ln/sqrt/recip`, `eval_at` and `bound_range`. |
| `veribound-solver`   | Priority-queue branch-and-bound `minimize_global` (lower/upper cutoff + monotonicity pruning), 1-D `interval_newton_1d`. |
| `veribound-cli`      | `veribound minimize` / `veribound roots` with JSON or human output. |

## Build & test

```sh
cargo build --release   # build everything
cargo test              # 21 unit + integration tests
```

## CLI usage (`veribound`)

```sh
# Certified global minimization (interval or Taylor-model mode)
cargo run --release -p veribound-cli -- minimize --bench himmelblau --mode taylor --degree 4 --eps 5e-3
cargo run --release -p veribound-cli -- minimize --bench rosenbrock --mode interval --domain -2,2,-2,2 --json

# Certified 1-D root isolation via Interval Newton
cargo run --release -p veribound-cli -- roots --eq cubic --eps 1e-9 --json
cargo run --release -p veribound-cli -- roots --eq cosx --eps 1e-12
```

Supported flags:
- `minimize`: `--bench <himmelblau|rosenbrock|rastrigin|ackley>`, `--mode <interval|taylor>`, `--degree <1..=4>`, `--eps <f64>`, `--domain lo0,hi0,lo1,hi1`, `--json`
- `roots`: `--eq <cubic|cosx|sqrt2>`, `--eps <f64>`, `--json`

## Library usage

```rust
use veribound_expr::benchmarks;
use veribound_interval::{BoxDomain, Interval};
use veribound_solver::{minimize_global, EvalMode, SolverOptions};

let (expr, root) = benchmarks::himmelblau();
let domain = BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]);
let res = minimize_global(&expr, root, &domain, SolverOptions::default());
assert!(res.best_lower <= 0.0 && 0.0 <= res.best_upper);
```

## Soundness guarantees

* **Outward rounding** — lower bounds use `f64::next_down`, upper bounds use
  `f64::next_up`; `−0.0`/`+0.0` and `±∞` are handled explicitly; `NaN` is
  *contained* (treated as `∅`).
* **Domain discipline** — `sqrt(x<0) → ∅`, `ln(x≤0) → ∅` (or `−∞` lower bound
  when the domain is `(0, hi]`), `extended_div` never panics on `0 ∈ divisor`.
* **Taylor models** — `T(x) = P_d(x−x₀) + I_r` with interval coefficients and a
  sound remainder bounding truncation **and** rounding. Remainders are
  `O(width^{d+1})` for smooth functions.
* **Branch-and-bound** — boxes are discarded only when *provably* unable to
  contain the global minimum (lower-bound cutoff, upper-bound cutoff, or
  strict-sign monotonicity of a partial derivative).

## Verified benchmarks

Global minima (all `f* = 0`) are bracketed by `[best_lower, best_upper]` with
`best_upper ≈ 0` for **Himmelblau** (`[-5,5]²`, four minima isolated),
**Rosenbrock** (`[-2,2]²`), **Rastrigin** and **Ackley** (small domains).
Interval Newton isolates `x³−2x−5 = 0` and `cos x − x = 0` to ~1 ULP.

## Notes on the interval-vs-Taylor comparison

The integration test `certification.rs` reports node counts and bracket widths
for both modes. For the *polynomial* benchmarks the plain-interval path is
frequently competitive because the Taylor polynomial's range is bounded by the
natural (monomial-sum) interval extension, which re-introduces the dependency
effect inside the polynomial itself; Taylor models shine on *smooth* functions
on small boxes, where the remainder is `O(width^{d+1})`. See the "Extension
notes" in the project write-up for tighter polynomial bounds (Horner/centered
forms, Bernstein coefficients) and argument reduction for `sin`/`cos`.
