//! Arena-based expression AST with interval evaluation and reverse-mode
//! interval automatic differentiation.
//!
//! Expressions are stored in a flat arena (`Vec<Node>`); every non-leaf node
//! references its children by index, and the builder guarantees a child's
//! index is always lower than its parent's, so trees are naturally in
//! topological order. `NodeId` is simply the arena slot of a sub-expression.
//!
//! The arena is wrapped in a [`RefCell`] so builder methods take `&self` and can
//! therefore be nested freely (e.g. `e.add(e.sqr(x), e.constant(1.0))`).

use std::cell::RefCell;

use veribound_interval::{BoxDomain, Interval};

/// Index of a node inside an [`Expr`] arena.
pub type NodeId = usize;

/// An expression node.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// Constant leaf.
    Const(f64),
    /// Variable leaf `x_i`.
    Var(usize),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Neg(NodeId),
    Powi(NodeId, i32),
    Sqrt(NodeId),
    Exp(NodeId),
    Ln(NodeId),
    Sin(NodeId),
    Cos(NodeId),
    Abs(NodeId),
}

/// An arena of expression nodes.
#[derive(Clone, Debug)]
pub struct Expr {
    nodes: RefCell<Vec<Node>>,
    n_vars: usize,
}

/// Reciprocal that returns [`Interval::ENTIRE`] instead of ∅ when `0` is in the
/// denominator, so derivative enclosures stay sound (an unbounded derivative is
/// a valid enclosure, an empty one is not).
#[inline]
fn safe_recip(x: Interval) -> Interval {
    let r = x.recip();
    if r.is_empty() {
        Interval::ENTIRE
    } else {
        r
    }
}

impl Expr {
    /// Create an empty arena for an expression over `n_vars` variables.
    pub fn new(n_vars: usize) -> Self {
        Expr {
            nodes: RefCell::new(Vec::new()),
            n_vars,
        }
    }

    /// Number of free variables.
    #[inline]
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Number of nodes in the arena.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.borrow().is_empty()
    }

    /// Inspect a node (cloned).
    #[inline]
    pub fn node(&self, id: NodeId) -> Node {
        self.nodes.borrow()[id].clone()
    }

    #[inline]
    fn push(&self, n: Node) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(n);
        nodes.len() - 1
    }

    // ------------------------------------------------------------- builders

    pub fn constant(&self, c: f64) -> NodeId {
        self.push(Node::Const(c))
    }

    pub fn var(&self, i: usize) -> NodeId {
        assert!(i < self.n_vars, "variable index out of range");
        self.push(Node::Var(i))
    }

    pub fn add(&self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Add(a, b))
    }

    pub fn sub(&self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Sub(a, b))
    }

    pub fn mul(&self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Mul(a, b))
    }

    pub fn div(&self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Div(a, b))
    }

    pub fn neg(&self, a: NodeId) -> NodeId {
        self.push(Node::Neg(a))
    }

    pub fn powi(&self, a: NodeId, n: i32) -> NodeId {
        self.push(Node::Powi(a, n))
    }

    pub fn sqrt(&self, a: NodeId) -> NodeId {
        self.push(Node::Sqrt(a))
    }

    pub fn exp(&self, a: NodeId) -> NodeId {
        self.push(Node::Exp(a))
    }

    pub fn ln(&self, a: NodeId) -> NodeId {
        self.push(Node::Ln(a))
    }

    pub fn sin(&self, a: NodeId) -> NodeId {
        self.push(Node::Sin(a))
    }

    pub fn cos(&self, a: NodeId) -> NodeId {
        self.push(Node::Cos(a))
    }

    pub fn abs(&self, a: NodeId) -> NodeId {
        self.push(Node::Abs(a))
    }

    /// `x²`.
    pub fn sqr(&self, a: NodeId) -> NodeId {
        self.mul(a, a)
    }

    // ----------------------------------------------------------- evaluation

    /// Evaluate at a concrete point `x` using `f64` (for sampling, not for
    /// enclosure proofs).
    pub fn eval_f64(&self, node: NodeId, x: &[f64]) -> f64 {
        let nodes = self.nodes.borrow();
        eval_f64_inner(&nodes, node, x)
    }

    /// Sound interval enclosure of the expression over `b`.
    pub fn eval_interval(&self, node: NodeId, b: &BoxDomain) -> Interval {
        let nodes = self.nodes.borrow();
        let mut cache = vec![None; nodes.len()];
        eval_interval_inner(&nodes, node, b, &mut cache)
    }

    /// Forward interval evaluation together with a sound enclosure of the
    /// gradient (reverse-mode interval AD): returns
    /// `(value_interval, [∂f/∂x₀, …, ∂f/∂xₙ₋₁])`.
    pub fn interval_gradient(&self, node: NodeId, b: &BoxDomain) -> (Interval, Vec<Interval>) {
        let nodes = self.nodes.borrow();
        let n = nodes.len();
        // Forward pass: interval value at every node. The builder guarantees
        // children are stored before their parents, so a single linear scan
        // suffices.
        let mut vals = vec![Interval::EMPTY; n];
        for i in 0..n {
            vals[i] = match &nodes[i] {
                Node::Const(c) => Interval::point(*c),
                Node::Var(v) => b.get(*v),
                Node::Add(a, bb) => vals[*a].add(vals[*bb]),
                Node::Sub(a, bb) => vals[*a].sub(vals[*bb]),
                Node::Mul(a, bb) => vals[*a].mul(vals[*bb]),
                Node::Div(a, bb) => vals[*a].mul(safe_recip(vals[*bb])),
                Node::Neg(a) => vals[*a].neg(),
                Node::Powi(a, p) => vals[*a].powi(*p),
                Node::Sqrt(a) => vals[*a].sqrt(),
                Node::Exp(a) => vals[*a].exp(),
                Node::Ln(a) => vals[*a].ln(),
                Node::Sin(a) => vals[*a].sin(),
                Node::Cos(a) => vals[*a].cos(),
                Node::Abs(a) => vals[*a].abs(),
            };
        }
        // Reverse pass: adjoints.
        let mut adj = vec![Interval::ZERO; n];
        adj[node] = Interval::ONE;
        for i in (0..n).rev() {
            let a = adj[i];
            if a.is_empty() || (a.lo == 0.0 && a.hi == 0.0) {
                continue;
            }
            match &nodes[i] {
                Node::Const(_) | Node::Var(_) => {}
                Node::Add(l, r) => {
                    adj[*l] = adj[*l].add(a);
                    adj[*r] = adj[*r].add(a);
                }
                Node::Sub(l, r) => {
                    adj[*l] = adj[*l].add(a);
                    adj[*r] = adj[*r].add(a.neg());
                }
                Node::Mul(l, r) => {
                    adj[*l] = adj[*l].add(a.mul(vals[*r]));
                    adj[*r] = adj[*r].add(a.mul(vals[*l]));
                }
                Node::Div(l, r) => {
                    let dr = safe_recip(vals[*r]);
                    adj[*l] = adj[*l].add(a.mul(dr));
                    adj[*r] = adj[*r].add(a.mul(vals[*l].neg()).mul(dr).mul(dr));
                }
                Node::Neg(c) => {
                    adj[*c] = adj[*c].add(a.neg());
                }
                Node::Powi(c, p) => {
                    let d = vals[*c].powi(*p - 1).scale(*p as f64);
                    adj[*c] = adj[*c].add(a.mul(d));
                }
                Node::Sqrt(c) => {
                    let s = vals[*c].sqrt();
                    let d = if s.is_empty() {
                        Interval::ENTIRE
                    } else {
                        safe_recip(s.scale(2.0))
                    };
                    adj[*c] = adj[*c].add(a.mul(d));
                }
                Node::Exp(c) => {
                    adj[*c] = adj[*c].add(a.mul(vals[*c].exp()));
                }
                Node::Ln(c) => {
                    adj[*c] = adj[*c].add(a.mul(safe_recip(vals[*c])));
                }
                Node::Sin(c) => {
                    adj[*c] = adj[*c].add(a.mul(vals[*c].cos()));
                }
                Node::Cos(c) => {
                    adj[*c] = adj[*c].add(a.mul(vals[*c].sin().neg()));
                }
                Node::Abs(c) => {
                    let d = if vals[*c].lo > 0.0 {
                        Interval::ONE
                    } else if vals[*c].hi < 0.0 {
                        Interval::new(-1.0, -1.0)
                    } else {
                        Interval::new(-1.0, 1.0)
                    };
                    adj[*c] = adj[*c].add(a.mul(d));
                }
            }
        }
        // Gather gradient at variable leaves.
        let mut grads = vec![Interval::ZERO; self.n_vars];
        for (i, node) in nodes.iter().enumerate() {
            if let Node::Var(v) = node {
                grads[*v] = grads[*v].add(adj[i]);
            }
        }
        (vals[node], grads)
    }
}

fn eval_f64_inner(nodes: &[Node], node: NodeId, x: &[f64]) -> f64 {
    match &nodes[node] {
        Node::Const(c) => *c,
        Node::Var(i) => x[*i],
        Node::Add(a, b) => eval_f64_inner(nodes, *a, x) + eval_f64_inner(nodes, *b, x),
        Node::Sub(a, b) => eval_f64_inner(nodes, *a, x) - eval_f64_inner(nodes, *b, x),
        Node::Mul(a, b) => eval_f64_inner(nodes, *a, x) * eval_f64_inner(nodes, *b, x),
        Node::Div(a, b) => eval_f64_inner(nodes, *a, x) / eval_f64_inner(nodes, *b, x),
        Node::Neg(a) => -eval_f64_inner(nodes, *a, x),
        Node::Powi(a, n) => eval_f64_inner(nodes, *a, x).powi(*n),
        Node::Sqrt(a) => eval_f64_inner(nodes, *a, x).sqrt(),
        Node::Exp(a) => eval_f64_inner(nodes, *a, x).exp(),
        Node::Ln(a) => eval_f64_inner(nodes, *a, x).ln(),
        Node::Sin(a) => eval_f64_inner(nodes, *a, x).sin(),
        Node::Cos(a) => eval_f64_inner(nodes, *a, x).cos(),
        Node::Abs(a) => eval_f64_inner(nodes, *a, x).abs(),
    }
}

fn eval_interval_inner(
    nodes: &[Node],
    node: NodeId,
    b: &BoxDomain,
    cache: &mut [Option<Interval>],
) -> Interval {
    if let Some(v) = cache[node] {
        return v;
    }
    let v = match &nodes[node] {
        Node::Const(c) => Interval::point(*c),
        Node::Var(i) => b.get(*i),
        Node::Add(a, bb) => eval_interval_inner(nodes, *a, b, cache)
            .add(eval_interval_inner(nodes, *bb, b, cache)),
        Node::Sub(a, bb) => eval_interval_inner(nodes, *a, b, cache)
            .sub(eval_interval_inner(nodes, *bb, b, cache)),
        Node::Mul(a, bb) => eval_interval_inner(nodes, *a, b, cache)
            .mul(eval_interval_inner(nodes, *bb, b, cache)),
        Node::Div(a, bb) => {
            let den = eval_interval_inner(nodes, *bb, b, cache);
            eval_interval_inner(nodes, *a, b, cache).mul(safe_recip(den))
        }
        Node::Neg(a) => eval_interval_inner(nodes, *a, b, cache).neg(),
        Node::Powi(a, n) => eval_interval_inner(nodes, *a, b, cache).powi(*n),
        Node::Sqrt(a) => eval_interval_inner(nodes, *a, b, cache).sqrt(),
        Node::Exp(a) => eval_interval_inner(nodes, *a, b, cache).exp(),
        Node::Ln(a) => eval_interval_inner(nodes, *a, b, cache).ln(),
        Node::Sin(a) => eval_interval_inner(nodes, *a, b, cache).sin(),
        Node::Cos(a) => eval_interval_inner(nodes, *a, b, cache).cos(),
        Node::Abs(a) => eval_interval_inner(nodes, *a, b, cache).abs(),
    };
    cache[node] = Some(v);
    v
}

/// Ready-made benchmark / test expressions.
pub mod benchmarks {
    use super::*;

    /// Himmelblau's function `(x² + y − 11)² + (x + y² − 7)²`.
    /// Four global minima at value `0`.
    pub fn himmelblau() -> (Expr, NodeId) {
        let e = Expr::new(2);
        let x = e.var(0);
        let y = e.var(1);
        let x2 = e.sqr(x);
        let y2 = e.sqr(y);
        let t1 = e.sqr(e.sub(e.add(x2, y), e.constant(11.0)));
        let t2 = e.sqr(e.sub(e.add(x, y2), e.constant(7.0)));
        let root = e.add(t1, t2);
        (e, root)
    }

    /// Rosenbrock's function `(1 − x)² + 100(y − x²)²`, global minimum `0` at
    /// `(1, 1)`.
    pub fn rosenbrock() -> (Expr, NodeId) {
        let e = Expr::new(2);
        let x = e.var(0);
        let y = e.var(1);
        let a2 = e.sqr(e.sub(e.constant(1.0), x));
        let b2 = e.sqr(e.sub(y, e.sqr(x)));
        let root = e.add(a2, e.mul(e.constant(100.0), b2));
        (e, root)
    }

    /// 2-D Rastrigin `20 + Σ (xᵢ² − 10·cos(2πxᵢ))`, global minimum `0` at the
    /// origin.
    pub fn rastrigin() -> (Expr, NodeId) {
        let e = Expr::new(2);
        let mut acc = e.constant(20.0);
        for i in 0..2 {
            let xi = e.var(i);
            let term = e.sub(
                e.sqr(xi),
                e.mul(e.constant(10.0), e.cos(e.mul(e.constant(2.0 * std::f64::consts::PI), xi))),
            );
            acc = e.add(acc, term);
        }
        (e, acc)
    }

    /// 2-D Ackley, global minimum `0` at the origin.
    pub fn ackley() -> (Expr, NodeId) {
        let e = Expr::new(2);
        let x = e.var(0);
        let y = e.var(1);
        let r2 = e.mul(e.constant(0.5), e.add(e.sqr(x), e.sqr(y)));
        let t1 = e.mul(e.constant(-20.0), e.exp(e.mul(e.constant(-0.2), e.sqrt(r2))));
        let cx = e.cos(e.mul(e.constant(2.0 * std::f64::consts::PI), x));
        let cy = e.cos(e.mul(e.constant(2.0 * std::f64::consts::PI), y));
        let t2 = e.exp(e.mul(e.constant(0.5), e.add(cx, cy)));
        let root = e.add(e.sub(t1, t2), e.constant(20.0 + std::f64::consts::E));
        (e, root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_eval_contains_point() {
        let (e, root) = benchmarks::himmelblau();
        let b = BoxDomain::new(vec![Interval::new(-5.0, 5.0), Interval::new(-5.0, 5.0)]);
        let enc = e.eval_interval(root, &b);
        // f(0,0) = (−11)² + (−7)² = 170
        assert!(enc.contains(170.0));
        assert!(enc.contains(0.0) || enc.lo >= 0.0);
    }

    #[test]
    fn gradient_encloses_analytic_value() {
        // f(x, y) = x*y + x^2  →  ∂f/∂x = y + 2x, ∂f/∂y = x
        let e = Expr::new(2);
        let x = e.var(0);
        let y = e.var(1);
        let xy = e.mul(x, y);
        let x2 = e.sqr(x);
        let root = e.add(xy, x2);
        let b = BoxDomain::point(&[2.0, 3.0]);
        let (_v, g) = e.interval_gradient(root, &b);
        assert!(g[0].contains(3.0 + 2.0 * 2.0)); // 7
        assert!(g[1].contains(2.0));
    }

    #[test]
    fn f64_and_interval_agree_on_point() {
        let (e, root) = benchmarks::rosenbrock();
        let p = [0.5, 0.25];
        let b = BoxDomain::point(&p);
        let iv = e.eval_interval(root, &b);
        let fv = e.eval_f64(root, &p);
        assert!(iv.contains(fv));
    }
}
