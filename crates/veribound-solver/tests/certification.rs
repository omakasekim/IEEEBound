//! Certification tests: empirical enclosure soundness via dense sampling, and
//! classic global-optimization benchmarks comparing plain-interval vs
//! Taylor-model enclosures and branch-and-bound node counts.

use veribound_expr::{benchmarks, Expr};
use veribound_interval::{BoxDomain, Interval};
use veribound_solver::{minimize_global, SolverOptions};
use veribound_taylor::TaylorModel;

// -------------------------------------------------------------------- PRNG --
/// Deterministic splitmix64 generator (no external dependencies).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn in_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn random_point(rng: &mut Rng, b: &BoxDomain) -> Vec<f64> {
    (0..b.dim())
        .map(|i| rng.in_range(b.get(i).lo, b.get(i).hi))
        .collect()
}

// --------------------------------------------- enclosure soundness sampling --

#[test]
fn enclosure_soundness_on_benchmarks() {
    let benches: Vec<(&str, BoxDomain)> = vec![
        ("himmelblau", BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)])),
        ("rosenbrock", BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)])),
        ("rastrigin", BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)])),
        ("ackley", BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)])),
    ];

    let mut rng = Rng::new(0x1234_5678_9abc_def0);
    const SAMPLES: usize = 4000;

    for (name, domain) in benches {
        let (expr, root) = match name {
            "himmelblau" => benchmarks::himmelblau(),
            "rosenbrock" => benchmarks::rosenbrock(),
            "rastrigin" => benchmarks::rastrigin(),
            _ => benchmarks::ackley(),
        };

        let enc_interval = expr.eval_interval(root, &domain);
        for degree in [2, 3] {
            let tm = TaylorModel::from_expr(&expr, root, &domain, degree);
            let enc_taylor = tm.bound_range();

            for _ in 0..SAMPLES {
                let p = random_point(&mut rng, &domain);
                let fv = expr.eval_f64(root, &p);
                assert!(
                    enc_interval.contains(fv),
                    "{name}: interval enclosure {} missed f({p:?}) = {fv}",
                    enc_interval
                );
                assert!(
                    enc_taylor.contains(fv),
                    "{name}: taylor(deg {degree}) enclosure {} missed f({p:?}) = {fv}",
                    enc_taylor
                );
                // Per-point Taylor evaluation must also be a sound enclosure.
                let at = tm.eval_at(&p);
                assert!(
                    at.contains(fv),
                    "{name}: taylor eval_at {} missed f({p:?}) = {fv}",
                    at
                );
            }
        }
    }
}

#[test]
fn smooth_function_taylor_soundness() {
    // Exercises exp, sin, cos, sqrt, ln in the Taylor path:
    //   f(x,y) = x·exp(y) + sin(x)·cos(y) + sqrt(x²+y²+1) + ln(x²+y²+1) − x·y
    let e = Expr::new(2);
    let x = e.var(0);
    let y = e.var(1);
    let t1 = e.mul(x, e.exp(y));
    let t2 = e.mul(e.sin(x), e.cos(y));
    let r2 = e.add(e.sqr(x), e.sqr(y));
    let s = e.add(r2, e.constant(1.0));
    let t3 = e.sqrt(s);
    let t4 = e.ln(s);
    let t5 = e.mul(x, y);
    let root = e.sub(e.add(e.add(t1, t2), e.add(t3, t4)), t5);

    let domain = BoxDomain::new(vec![Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0)]);
    let enc_interval = e.eval_interval(root, &domain);
    let mut rng = Rng::new(0xdead_beef_cafe_f00d);

    for degree in [2, 3, 4] {
        let tm = TaylorModel::from_expr(&e, root, &domain, degree);
        let enc_taylor = tm.bound_range();
        for _ in 0..4000 {
            let p = random_point(&mut rng, &domain);
            let fv = e.eval_f64(root, &p);
            assert!(enc_interval.contains(fv), "interval missed f({p:?}) = {fv}");
            assert!(
                enc_taylor.contains(fv),
                "taylor(deg {degree}) {} missed f({p:?}) = {fv}",
                enc_taylor
            );
        }
    }
}

// ---------------------------------------------- global optimization benchmarks --

#[test]
fn global_optimization_benchmarks_compare_modes() {
    struct Row {
        name: &'static str,
        domain: BoxDomain,
        eps: f64,
    }
    let rows = vec![
        Row {
            name: "himmelblau",
            domain: BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]),
            eps: 5e-3,
        },
        Row {
            name: "rosenbrock",
            domain: BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)]),
            eps: 2e-2,
        },
        Row {
            name: "rastrigin",
            domain: BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)]),
            eps: 5e-2,
        },
        Row {
            name: "ackley",
            domain: BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)]),
            eps: 5e-2,
        },
    ];

    for row in rows {
        let (expr, root) = match row.name {
            "himmelblau" => benchmarks::himmelblau(),
            "rosenbrock" => benchmarks::rosenbrock(),
            "rastrigin" => benchmarks::rastrigin(),
            _ => benchmarks::ackley(),
        };

        // Interval mode.
        let opts_i = SolverOptions {
            eps: row.eps,
            use_taylor: false,
            ..Default::default()
        };
        let ri = minimize_global(&expr, root, &row.domain, &opts_i);

        // Taylor mode (order 3 for smooth functions, 4 for the quartics).
        let degree = if row.name == "himmelblau" || row.name == "rosenbrock" { 4 } else { 3 };
        let opts_t = SolverOptions {
            eps: row.eps,
            use_taylor: true,
            taylor_degree: degree,
            ..Default::default()
        };
        let rt = minimize_global(&expr, root, &row.domain, &opts_t);

        // Soundness: the true global minimum (0 for all four) must be bracketed.
        assert!(ri.best_lower <= 0.0, "{}: interval lower {} > 0", row.name, ri.best_lower);
        assert!(ri.best_upper >= 0.0, "{}: interval upper {} < 0", row.name, ri.best_upper);
        assert!(rt.best_lower <= 0.0, "{}: taylor lower {} > 0", row.name, rt.best_lower);
        assert!(rt.best_upper >= 0.0, "{}: taylor upper {} < 0", row.name, rt.best_upper);

        // The verified upper bound must be close to the known minimum.
        assert!(
            ri.best_upper < 0.5,
            "{}: interval best_upper {} not near 0",
            row.name,
            ri.best_upper
        );
        assert!(
            rt.best_upper < 0.5,
            "{}: taylor best_upper {} not near 0",
            row.name,
            rt.best_upper
        );

        println!(
            "{:<10} interval: nodes={:>6}  upper={:>8.3e}  bracket={:>8.3e} | \
             taylor(d{}): nodes={:>6}  upper={:>8.3e}  bracket={:>8.3e}",
            row.name,
            ri.node_count,
            ri.best_upper,
            ri.bracket_width(),
            degree,
            rt.node_count,
            rt.best_upper,
            rt.bracket_width(),
        );
    }
}

#[test]
fn himmelblau_isolates_four_minima_both_modes() {
    let (e, root) = benchmarks::himmelblau();
    let domain = BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]);
    let minima = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ];
    for mode in [false, true] {
        let opts = SolverOptions {
            eps: 5e-3,
            use_taylor: mode,
            taylor_degree: 4,
            ..Default::default()
        };
        let r = minimize_global(&e, root, &domain, &opts);
        for &(mx, my) in &minima {
            let found = r.leaves.iter().any(|b| {
                (b.get(0).lo - mx).abs() < 0.5 && (b.get(1).lo - my).abs() < 0.5
            });
            assert!(found, "mode={mode}: minimum ({mx}, {my}) not isolated");
        }
    }
}
