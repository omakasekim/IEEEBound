//! Certified branch-and-bound global minimization and interval Newton root
//! isolation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use veribound_expr::{Expr, NodeId};
use veribound_interval::{BoxDomain, Interval};
use veribound_taylor::TaylorModel;

/// Solver configuration.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    /// Stop when a box's diameter or its enclosure width is ≤ `eps`.
    pub eps: f64,
    /// Safety cap on the number of processed nodes.
    pub max_nodes: usize,
    /// Use Taylor-model enclosures (`true`) or plain interval (`false`).
    pub use_taylor: bool,
    /// Taylor-model order (1–4).
    pub taylor_degree: usize,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            eps: 1e-3,
            max_nodes: 1_000_000,
            use_taylor: false,
            taylor_degree: 2,
        }
    }
}

/// Result of a global minimization run.
#[derive(Clone, Debug)]
pub struct MinimizationResult {
    /// Verified lower bound: the global minimum is ≥ `best_lower`.
    pub best_lower: f64,
    /// Verified upper bound: the global minimum is ≤ `best_upper`.
    pub best_upper: f64,
    /// Boxes that survived pruning and may contain a global minimum.
    pub leaves: Vec<BoxDomain>,
    /// Number of nodes processed by the branch-and-bound loop.
    pub node_count: usize,
    /// Enclosure mode used ("interval" or "taylor").
    pub mode: &'static str,
}

impl MinimizationResult {
    /// Width of the verified bracket `[best_lower, best_upper]`.
    pub fn bracket_width(&self) -> f64 {
        self.best_upper - self.best_lower
    }
}

/// A priority-queue entry ordered by *ascending* lower bound.
struct QueueItem {
    lb: f64,
    bx: BoxDomain,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.lb == other.lb
    }
}
impl Eq for QueueItem {}
impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed so that `BinaryHeap` (a max-heap) pops the smallest `lb`.
        other.lb.partial_cmp(&self.lb).unwrap_or(Ordering::Equal)
    }
}

/// Enclosure of `expr` over `bx` in the selected mode.
fn enclosure(expr: &Expr, root: NodeId, bx: &BoxDomain, opts: &SolverOptions) -> Interval {
    if opts.use_taylor {
        TaylorModel::from_expr(expr, root, bx, opts.taylor_degree).bound_range()
    } else {
        expr.eval_interval(root, bx)
    }
}

/// Monotonicity pruning: if `∂f/∂xᵢ` is strictly positive (negative) over the
/// box, `f` is increasing (decreasing) in `xᵢ` and the minimum lies on the
/// lower (upper) face, so the box is contracted to that face.
fn monotonic_prune(expr: &Expr, root: NodeId, bx: &BoxDomain) -> BoxDomain {
    let (_v, grads) = expr.interval_gradient(root, bx);
    let mut out = bx.clone();
    for i in 0..out.dim() {
        let g = grads[i];
        if g.lo > 0.0 {
            let lo = out.get(i).lo;
            out.set(i, Interval::new(lo, lo));
        } else if g.hi < 0.0 {
            let hi = out.get(i).hi;
            out.set(i, Interval::new(hi, hi));
        }
    }
    out
}

/// Certified global minimization of `expr` (root `root`) over `domain`.
///
/// The returned `[best_lower, best_upper]` bracket is guaranteed to contain the
/// true global minimum (subject to the enclosure soundness of the underlying
/// interval/Taylor arithmetic). Boxes are pruned only when they are *provably*
/// unable to contain a global minimum: lower-bound cutoff, upper-bound cutoff,
/// or monotonicity.
pub fn minimize_global(
    expr: &Expr,
    root: NodeId,
    domain: &BoxDomain,
    opts: &SolverOptions,
) -> MinimizationResult {
    let mut heap: BinaryHeap<QueueItem> = BinaryHeap::new();
    let mut best_upper = f64::INFINITY;
    let mut leaves: Vec<(f64, BoxDomain)> = Vec::new();
    let mut node_count = 0usize;

    // Seed the verified upper bound with the midpoint sample and the enclosure
    // upper bound (both are valid upper bounds on the minimum).
    let init_enc = enclosure(expr, root, domain, opts);
    let mp = domain.mid();
    best_upper = best_upper.min(init_enc.hi).min(expr.eval_f64(root, &mp));

    heap.push(QueueItem {
        lb: init_enc.lo,
        bx: domain.clone(),
    });

    while let Some(item) = heap.pop() {
        node_count += 1;
        let bx = item.bx;

        // Upper-bound cutoff: this box's best possible value exceeds the best
        // value already found elsewhere, so it cannot contain the minimum.
        if item.lb > best_upper {
            continue;
        }

        let enc = enclosure(expr, root, &bx, opts);
        if enc.is_empty() {
            continue; // no feasible point in this box
        }
        if enc.lo > best_upper {
            continue; // lower-bound prune
        }

        // Monotonicity contraction (sound: only removes provably non-minimal
        // faces).
        let bx = monotonic_prune(expr, root, &bx);

        let enc = enclosure(expr, root, &bx, opts);
        if enc.is_empty() || enc.lo > best_upper {
            continue;
        }

        // Update the verified upper bound from a sample and the enclosure.
        let mp = bx.mid();
        let fmid = expr.eval_f64(root, &mp);
        best_upper = best_upper.min(fmid).min(enc.hi);

        let lb = enc.lo;
        if lb > best_upper {
            continue;
        }

        // Termination: box or its enclosure is small enough.
        if bx.diameter() <= opts.eps || enc.width_up() <= opts.eps {
            leaves.push((lb, bx));
            continue;
        }

        // Split along the widest dimension and enqueue sound children.
        let (l, r) = bx.split();
        let el = enclosure(expr, root, &l, opts);
        let er = enclosure(expr, root, &r, opts);
        if !el.is_empty() && el.lo <= best_upper {
            heap.push(QueueItem { lb: el.lo, bx: l });
        }
        if !er.is_empty() && er.lo <= best_upper {
            heap.push(QueueItem { lb: er.lo, bx: r });
        }

        if node_count >= opts.max_nodes {
            break;
        }
    }

    // Global lower bound = min lower bound over all still-alive boxes (leaves
    // plus any un-processed queue entries).
    let mut best_lower = f64::INFINITY;
    for (lb, _) in &leaves {
        best_lower = best_lower.min(*lb);
    }
    while let Some(it) = heap.pop() {
        best_lower = best_lower.min(it.lb);
    }
    if leaves.is_empty() {
        best_lower = best_upper;
    }

    MinimizationResult {
        best_lower,
        best_upper,
        leaves: leaves.into_iter().map(|(_, b)| b).collect(),
        node_count,
        mode: if opts.use_taylor { "taylor" } else { "interval" },
    }
}

/// Interval Newton root isolation in one dimension.
///
/// Contracts boxes around roots of `g(x) = 0` using
/// `N(X) = m − g(m) / G′(X)` (with `m = mid(X)` and `G′` an interval enclosure
/// of the derivative obtained by reverse-mode AD). When the derivative
/// enclosure contains `0` the Newton operator is not applicable and the box is
/// split cleanly.
pub fn interval_newton_1d(
    g: &Expr,
    g_root: NodeId,
    domain: Interval,
    eps: f64,
) -> Vec<Interval> {
    let mut worklist = vec![domain];
    let mut roots: Vec<Interval> = Vec::new();
    let mut iters = 0usize;
    const MAX_ITERS: usize = 100_000;

    while let Some(x) = worklist.pop() {
        iters += 1;
        if iters > MAX_ITERS {
            break;
        }
        if x.is_empty() || x.width_up() <= eps {
            roots.push(x);
            continue;
        }

        let bx = BoxDomain::new(vec![x]);
        let enc_g = g.eval_interval(g_root, &bx);
        if !enc_g.contains(0.0) {
            continue; // no root in this box (g does not change sign through 0)
        }

        let (_v, grads) = g.interval_gradient(g_root, &bx);
        let gprime = grads[0];
        if gprime.lo <= 0.0 && gprime.hi >= 0.0 {
            // derivative straddles 0 → split instead of applying Newton
            let (l, r) = split_interval(x);
            worklist.push(l);
            worklist.push(r);
            continue;
        }

        let m = x.mid();
        let gm = g.eval_f64(g_root, &[m]);
        let n = Interval::point(m).sub(Interval::point(gm).div(gprime));
        let contracted = x.intersect(n);
        if contracted.is_empty() {
            continue; // Newton proved there is no root in X
        }
        if contracted.width_up() >= x.width_up() {
            // No progress: split.
            let (l, r) = split_interval(x);
            worklist.push(l);
            worklist.push(r);
            continue;
        }
        worklist.push(contracted);
    }

    roots
}

fn split_interval(x: Interval) -> (Interval, Interval) {
    let m = x.mid();
    let m = if m <= x.lo { x.hi } else if m >= x.hi { x.lo } else { m };
    (Interval::new(x.lo, m), Interval::new(m, x.hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use veribound_expr::benchmarks;

    #[test]
    fn himmelblau_finds_zero_and_four_minima() {
        let (e, root) = benchmarks::himmelblau();
        let domain = BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]);
        let opts = SolverOptions {
            eps: 5e-3,
            ..Default::default()
        };
        let r = minimize_global(&e, root, &domain, &opts);
        assert!(r.best_lower <= 0.0, "lower bound must not exceed 0");
        assert!(r.best_upper >= 0.0, "upper bound must not be below 0");
        assert!(r.best_upper < 0.2, "upper bound should be near 0, got {}", r.best_upper);

        let minima = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)];
        for &(mx, my) in &minima {
            let found = r.leaves.iter().any(|b| {
                (b.get(0).lo - mx).abs() < 0.5 && (b.get(1).lo - my).abs() < 0.5
            });
            assert!(found, "minimum ({mx}, {my}) not isolated");
        }
    }

    #[test]
    fn newton_isolates_cubic_root() {
        // x^3 - 2x - 5 = 0 has a root ≈ 2.09455
        let e = Expr::new(1);
        let x = e.var(0);
        let x3 = e.powi(x, 3);
        let twox = e.mul(e.constant(2.0), x);
        let inner = e.sub(x3, twox);
        let g = e.sub(inner, e.constant(5.0));
        let roots = interval_newton_1d(&e, g, Interval::new(0.0, 4.0), 1e-9);
        assert!(!roots.is_empty());
        let target = 2.0945514815423265;
        let hit = roots.iter().any(|r| r.contains(target));
        assert!(hit, "root {} not enclosed: {:?}", target, roots);
    }

    #[test]
    fn newton_isolates_cosx_minus_x() {
        // cos(x) - x = 0 has a root ≈ 0.739085
        let e = Expr::new(1);
        let x = e.var(0);
        let cx = e.cos(x);
        let g = e.sub(cx, x);
        let roots = interval_newton_1d(&e, g, Interval::new(-1.0, 1.0), 1e-9);
        assert!(!roots.is_empty());
        let target = 0.7390851332151607;
        let hit = roots.iter().any(|r| r.contains(target));
        assert!(hit, "root {} not enclosed: {:?}", target, roots);
    }
}
