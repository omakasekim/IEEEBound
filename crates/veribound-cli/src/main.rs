//! `veribound` — command-line interface for IEEEBound.
//!
//! Subcommands:
//!   `minimize --bench <name> [--mode interval|taylor] [--degree D] [--eps E]
//!             [--domain lo0,hi0,lo1,hi1,...] [--json]`
//!   `roots   --eq <name> [--eps E] [--json]`

use std::env;
use std::process::ExitCode;

use veribound_expr::{benchmarks, Expr, NodeId};
use veribound_interval::{BoxDomain, Interval};
use veribound_solver::{interval_newton_1d, minimize_global, SolverOptions};

const USAGE: &str = "\
usage:
  veribound minimize --bench <himmelblau|rosenbrock|rastrigin|ackley>
                      [--mode <interval|taylor>] [--degree <1..=4>]
                      [--eps <f64>] [--domain lo0,hi0,lo1,hi1] [--json]
  veribound roots   --eq <cubic|cosx|sqrt2> [--eps <f64>] [--json]
";

#[derive(Clone, Copy, PartialEq)]
enum Bench {
    Himmelblau,
    Rosenbrock,
    Rastrigin,
    Ackley,
}

impl Bench {
    fn parse(s: &str) -> Option<Bench> {
        match s {
            "himmelblau" => Some(Bench::Himmelblau),
            "rosenbrock" => Some(Bench::Rosenbrock),
            "rastrigin" => Some(Bench::Rastrigin),
            "ackley" => Some(Bench::Ackley),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Bench::Himmelblau => "himmelblau",
            Bench::Rosenbrock => "rosenbrock",
            Bench::Rastrigin => "rastrigin",
            Bench::Ackley => "ackley",
        }
    }
    fn build(self) -> (Expr, NodeId) {
        match self {
            Bench::Himmelblau => benchmarks::himmelblau(),
            Bench::Rosenbrock => benchmarks::rosenbrock(),
            Bench::Rastrigin => benchmarks::rastrigin(),
            Bench::Ackley => benchmarks::ackley(),
        }
    }
    fn default_domain(self) -> BoxDomain {
        match self {
            Bench::Himmelblau => BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]),
            Bench::Rosenbrock => BoxDomain::new(vec![Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)]),
            Bench::Rastrigin => BoxDomain::new(vec![Interval::new(-5.12, 5.12), Interval::new(-5.12, 5.12)]),
            Bench::Ackley => BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]),
        }
    }
}

fn build_equation(name: &str) -> (Expr, NodeId) {
    let e = Expr::new(1);
    let x = e.var(0);
    match name {
        // x^3 - 2x - 5 = 0
        "cubic" => {
            let x3 = e.powi(x, 3);
            let twox = e.mul(e.constant(2.0), x);
            let inner = e.sub(x3, twox);
            let g = e.sub(inner, e.constant(5.0));
            (e, g)
        }
        // cos(x) - x = 0
        "cosx" => {
            let cx = e.cos(x);
            let g = e.sub(cx, x);
            (e, g)
        }
        // x^2 - 2 = 0
        "sqrt2" => {
            let x2 = e.sqr(x);
            let g = e.sub(x2, e.constant(2.0));
            (e, g)
        }
        _ => (e, x),
    }
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_domain(s: &str) -> Option<BoxDomain> {
    let vals: Vec<f64> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    if vals.len() % 2 != 0 || vals.is_empty() {
        return None;
    }
    let dims: Vec<Interval> = vals
        .chunks(2)
        .map(|c| Interval::new(c[0], c[1]))
        .collect();
    Some(BoxDomain::new(dims))
}

fn cmd_minimize(args: &[String]) -> Result<(), String> {
    let bench = get_arg(args, "--bench")
        .and_then(|s| Bench::parse(&s))
        .ok_or("missing/invalid --bench")?;
    let json = args.iter().any(|a| a == "--json");
    let eps: f64 = get_arg(args, "--eps").map(|s| s.parse().unwrap_or(1e-3)).unwrap_or(1e-3);
    let mode = get_arg(args, "--mode").unwrap_or_else(|| "interval".to_string());
    let use_taylor = match mode.as_str() {
        "taylor" => true,
        "interval" => false,
        _ => return Err("--mode must be interval or taylor".into()),
    };
    let degree: usize = get_arg(args, "--degree")
        .map(|s| s.parse().unwrap_or(2).clamp(1, 4))
        .unwrap_or(2);
    let domain = get_arg(args, "--domain")
        .map(|s| parse_domain(&s).ok_or("bad --domain"))
        .transpose()?
        .unwrap_or_else(|| bench.default_domain());

    let (expr, root) = bench.build();
    let opts = SolverOptions {
        eps,
        use_taylor,
        taylor_degree: degree,
        ..Default::default()
    };
    let r = minimize_global(&expr, root, &domain, &opts);

    if json {
        println!(
            "{{\"bench\":\"{}\",\"mode\":\"{}\",\"degree\":{},\"eps\":{},\"best_lower\":{:.17e},\"best_upper\":{:.17e},\"node_count\":{},\"leaves\":{}}}",
            bench.name(),
            r.mode,
            degree,
            eps,
            r.best_lower,
            r.best_upper,
            r.node_count,
            r.leaves.len()
        );
    } else {
        println!("bench      = {}", bench.name());
        println!("mode       = {} (order {})", r.mode, degree);
        println!("eps        = {eps}");
        println!("domain     = {domain}");
        println!("node count = {}", r.node_count);
        println!("leaves     = {}", r.leaves.len());
        println!("minimum ∈  [{:.12e}, {:.12e}]", r.best_lower, r.best_upper);
        println!("bracket width = {:.3e}", r.bracket_width());
        for (i, b) in r.leaves.iter().take(8).enumerate() {
            println!("  leaf {i}: {b}");
        }
    }
    Ok(())
}

fn cmd_roots(args: &[String]) -> Result<(), String> {
    let eq = get_arg(args, "--eq").ok_or("missing --eq")?;
    if !["cubic", "cosx", "sqrt2"].contains(&eq.as_str()) {
        return Err(format!("unknown equation '{eq}'"));
    }
    let json = args.iter().any(|a| a == "--json");
    let eps: f64 = get_arg(args, "--eps").map(|s| s.parse().unwrap_or(1e-9)).unwrap_or(1e-9);

    let (expr, root) = build_equation(&eq);
    let domain = match eq.as_str() {
        "cubic" => Interval::new(0.0, 4.0),
        "cosx" => Interval::new(-2.0, 2.0),
        _ => Interval::new(0.0, 2.0),
    };
    let roots = interval_newton_1d(&expr, root, domain, eps);

    if json {
        print!("{{\"eq\":\"{eq}\",\"eps\":{eps},\"roots\":[");
        for (i, r) in roots.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("[{:.17e},{:.17e}]", r.lo, r.hi);
        }
        println!("]}}");
    } else {
        println!("equation   = {eq}");
        println!("eps        = {eps}");
        println!("domain     = {domain}");
        println!("roots      = {} found", roots.len());
        for (i, r) in roots.iter().enumerate() {
            println!("  root {i}: {r}");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let result = if args.len() < 2 {
        eprint!("{USAGE}");
        return ExitCode::FAILURE;
    } else if args[1] == "minimize" {
        cmd_minimize(&args[2..])
    } else if args[1] == "roots" {
        cmd_roots(&args[2..])
    } else {
        eprint!("{USAGE}");
        return ExitCode::FAILURE;
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}
