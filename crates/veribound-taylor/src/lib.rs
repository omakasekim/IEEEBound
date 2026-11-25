//! Multivariate Taylor models with sound interval remainders.
//!
//! A Taylor model of order `d` over a box `X ⊆ ℝⁿ` centred at `x₀` represents a
//! function as
//!
//! ```text
//!   T_f(x) = P_d(u) + I_r ,   u = x − x₀ ,  |uᵢ| ≤ rᵢ
//! ```
//!
//! where `P_d` is a truncated multivariate polynomial (with *interval*
//! coefficients so that every rounding during coefficient arithmetic is itself
//! enclosed) and `I_r` is a sound interval remainder bounding both the
//! truncation error and the rounding error. `bound_range()` returns a rigorous
//! enclosure of the function's range over the box.
//!
//! # Rounding policy
//!
//! Interval arithmetic on coefficients is fully outward-rounded. The only
//! non-rigorous step is the *construction* of the Taylor coefficients
//! `f⁽ᵏ⁾(c)/k!` of the elementary functions, which are produced by correctly
//! rounded `f64` primitives and then inflated by a 16-ulp relative/absolute
//! safety margin (see [`iv`]); that margin conservatively dominates the ~1 ulp
//! error of each correctly-rounded primitive, so the resulting interval
//! coefficient is a sound enclosure of the true real coefficient.

use std::fmt;

use veribound_expr::{Expr, Node, NodeId};
use veribound_interval::{BoxDomain, Interval};

/// Round down for a lower bound (leave non-finite values untouched).
#[inline]
fn lo_round(x: f64) -> f64 {
    if x.is_finite() {
        x.next_down()
    } else {
        x
    }
}

/// Round up for an upper bound.
#[inline]
fn hi_round(x: f64) -> f64 {
    if x.is_finite() {
        x.next_up()
    } else {
        x
    }
}

/// Wrap a floating-point value in a sound interval coefficient: widen by a
/// 16-ulp relative + absolute margin, then round outward. This dominates the
/// accumulated error of the (correctly rounded) elementary-function call that
/// produced `v`.
#[inline]
fn iv(v: f64) -> Interval {
    let m = (v.abs() * 16.0 + 16.0) * f64::EPSILON;
    Interval::new(lo_round(v - m), hi_round(v + m))
}

/// `k!` for small `k`.
#[inline]
fn factorial(k: usize) -> f64 {
    (1..=k).fold(1.0_f64, |a, b| a * b as f64)
}

/// A single monomial `c · u₀^e₀ · … · u_{n−1}^e_{n−1}` with an interval
/// coefficient.
#[derive(Clone, Debug)]
pub struct Monomial {
    pub exps: Vec<u32>,
    pub coeff: Interval,
}

impl Monomial {
    fn total_degree(&self) -> u32 {
        self.exps.iter().sum()
    }
}

/// A multivariate Taylor model.
#[derive(Clone, Debug)]
pub struct TaylorModel {
    n_vars: usize,
    degree: usize,
    /// Expansion centre `x₀` (box midpoint).
    x0: Vec<f64>,
    /// Half-widths: `|uᵢ| = |xᵢ − x₀ᵢ| ≤ radius[i]` for every `x ∈ X`.
    radius: Vec<f64>,
    /// Polynomial part (interval coefficients).
    terms: Vec<Monomial>,
    /// Sound remainder interval.
    rem: Interval,
}

impl TaylorModel {
    /// An empty (zero) Taylor model with the given shape.
    pub fn new(n_vars: usize, degree: usize, x0: Vec<f64>, radius: Vec<f64>) -> Self {
        debug_assert_eq!(x0.len(), n_vars);
        debug_assert_eq!(radius.len(), n_vars);
        TaylorModel {
            n_vars,
            degree,
            x0,
            radius,
            terms: Vec::new(),
            rem: Interval::ZERO,
        }
    }

    /// Build the Taylor models `T_{x_i}(x) = x0ᵢ + uᵢ` for each variable.
    pub fn variables(b: &BoxDomain) -> Vec<TaylorModel> {
        let n = b.dim();
        let x0 = b.mid();
        let radius: Vec<f64> = (0..n)
            .map(|i| {
                let m = x0[i];
                let d = b.get(i);
                hi_round((d.lo - m).abs().max((d.hi - m).abs()))
            })
            .collect();
        let base = TaylorModel::new(n, 1, x0.clone(), radius.clone());
        (0..n).map(|i| base.variable(i)).collect()
    }

    /// Translate an expression (in topological arena order) into a Taylor model
    /// of the given degree over `b`.
    pub fn from_expr(expr: &Expr, root: NodeId, b: &BoxDomain, degree: usize) -> TaylorModel {
        let n = b.dim();
        let x0 = b.mid();
        let radius: Vec<f64> = (0..n)
            .map(|i| {
                let m = x0[i];
                let d = b.get(i);
                hi_round((d.lo - m).abs().max((d.hi - m).abs()))
            })
            .collect();
        let base = TaylorModel::new(n, degree, x0, radius);
        let mut tms: Vec<TaylorModel> = Vec::with_capacity(expr.len());
        for i in 0..expr.len() {
            let tm = match expr.node(i) {
                Node::Const(c) => base.constant(c),
                Node::Var(v) => base.variable(v),
                Node::Add(a, bb) => tms[a].add(&tms[bb]),
                Node::Sub(a, bb) => tms[a].sub(&tms[bb]),
                Node::Mul(a, bb) => tms[a].mul(&tms[bb]),
                Node::Div(a, bb) => tms[a].div(&tms[bb]),
                Node::Neg(a) => tms[a].neg(),
                Node::Powi(a, p) => tms[a].powi(p),
                Node::Sqrt(a) => tms[a].sqrt_tm(),
                Node::Exp(a) => tms[a].exp_tm(),
                Node::Ln(a) => tms[a].ln_tm(),
                Node::Sin(a) => tms[a].sin_tm(),
                Node::Cos(a) => tms[a].cos_tm(),
                Node::Abs(a) => tms[a].abs_tm(),
            };
            tms.push(tm);
        }
        tms[root].clone()
    }

    // ------------------------------------------------------------- constructors

    fn zero_like(&self) -> TaylorModel {
        TaylorModel::new(self.n_vars, self.degree, self.x0.clone(), self.radius.clone())
    }

    /// A Taylor model that is the constant `k`.
    pub fn constant(&self, k: f64) -> TaylorModel {
        let mut m = self.zero_like();
        m.terms.push(Monomial {
            exps: vec![0; self.n_vars],
            coeff: Interval::point(k),
        });
        m
    }

    /// `T_{x_i}(x) = x0ᵢ + uᵢ`.
    pub fn variable(&self, i: usize) -> TaylorModel {
        let mut m = self.zero_like();
        // constant term x0[i]
        m.terms.push(Monomial {
            exps: vec![0; self.n_vars],
            coeff: Interval::point(self.x0[i]),
        });
        // linear term u_i
        let mut exps = vec![0; self.n_vars];
        exps[i] = 1;
        m.terms.push(Monomial {
            exps,
            coeff: Interval::ONE,
        });
        m
    }

    /// A degenerate model whose range is exactly the interval `iv` (no
    /// polynomial information) — used as a sound fallback.
    fn fallback(&self, iv: Interval) -> TaylorModel {
        TaylorModel {
            n_vars: self.n_vars,
            degree: self.degree,
            x0: self.x0.clone(),
            radius: self.radius.clone(),
            terms: Vec::new(),
            rem: iv,
        }
    }

    // --------------------------------------------------------------- helpers

    /// Midpoint of the constant-term coefficient (≈ value at `x₀`).
    fn constant_mid(&self) -> f64 {
        for t in &self.terms {
            if t.exps.iter().all(|&e| e == 0) {
                return t.coeff.mid();
            }
        }
        0.0
    }

    /// Subtract a scalar from the constant term (shifting the expansion).
    fn shift(&self, delta: f64) -> TaylorModel {
        let mut m = self.clone();
        let mut found = false;
        for t in m.terms.iter_mut() {
            if t.exps.iter().all(|&e| e == 0) {
                t.coeff = t.coeff.sub(Interval::point(delta));
                found = true;
            }
        }
        if !found {
            m.terms.push(Monomial {
                exps: vec![0; self.n_vars],
                coeff: Interval::point(-delta),
            });
        }
        m
    }

    /// Multiply the whole model (polynomial + remainder) by the interval `k`.
    pub fn scale(&self, k: Interval) -> TaylorModel {
        let mut m = self.zero_like();
        m.terms = self
            .terms
            .iter()
            .map(|t| Monomial {
                exps: t.exps.clone(),
                coeff: t.coeff.mul(k),
            })
            .collect();
        m.rem = self.rem.mul(k);
        m
    }

    /// Merge monomials with equal exponent vectors and drop zero coefficients.
    fn normalize(&self) -> TaylorModel {
        let mut merged: Vec<Monomial> = Vec::new();
        'outer: for t in &self.terms {
            for existing in merged.iter_mut() {
                if existing.exps == t.exps {
                    existing.coeff = existing.coeff.add(t.coeff);
                    continue 'outer;
                }
            }
            merged.push(t.clone());
        }
        merged.retain(|t| !t.coeff.is_empty() && !(t.coeff.lo == 0.0 && t.coeff.hi == 0.0));
        TaylorModel {
            n_vars: self.n_vars,
            degree: self.degree,
            x0: self.x0.clone(),
            radius: self.radius.clone(),
            terms: merged,
            rem: self.rem,
        }
    }

    /// Range of a single monomial over the box.
    fn monomial_range(&self, t: &Monomial) -> Interval {
        let mut r = t.coeff;
        for i in 0..self.n_vars {
            let e = t.exps[i];
            if e == 0 {
                continue;
            }
            let rr = self.radius[i];
            let p = rr.powi(e as i32);
            let u = if e % 2 == 0 {
                Interval::new(0.0, hi_round(p))
            } else {
                Interval::new(lo_round(-p), hi_round(p))
            };
            r = r.mul(u);
        }
        r
    }

    /// Range of the polynomial part only (no remainder).
    pub fn poly_bound(&self) -> Interval {
        let mut acc = Interval::ZERO;
        for t in &self.terms {
            acc = acc.add(self.monomial_range(t));
        }
        acc
    }

    /// Rigorous enclosure of the function's range over the box.
    pub fn bound_range(&self) -> Interval {
        self.poly_bound().add(self.rem)
    }

    /// Enclosure of the model's value at a specific point `x` (the full box
    /// remainder is included, so this is sound but per-point conservative).
    pub fn eval_at(&self, x: &[f64]) -> Interval {
        let mut acc = Interval::ZERO;
        for t in &self.terms {
            let mut v = t.coeff;
            for i in 0..self.n_vars {
                let u = x[i] - self.x0[i];
                v = v.mul(Interval::point(u).powi(t.exps[i] as i32));
            }
            acc = acc.add(v);
        }
        acc.add(self.rem)
    }

    // ------------------------------------------------------------ arithmetic

    pub fn add(&self, other: &TaylorModel) -> TaylorModel {
        let mut m = self.zero_like();
        m.terms = self.terms.clone();
        m.terms.extend(other.terms.iter().cloned());
        m.rem = self.rem.add(other.rem);
        m.normalize()
    }

    pub fn sub(&self, other: &TaylorModel) -> TaylorModel {
        self.add(&other.neg())
    }

    pub fn neg(&self) -> TaylorModel {
        self.scale(Interval::new(-1.0, -1.0))
    }

    pub fn mul(&self, other: &TaylorModel) -> TaylorModel {
        let mut terms = Vec::with_capacity(self.terms.len() * other.terms.len());
        for a in &self.terms {
            for b in &other.terms {
                let mut exps = vec![0; self.n_vars];
                for i in 0..self.n_vars {
                    exps[i] = a.exps[i] + b.exps[i];
                }
                terms.push(Monomial {
                    exps,
                    coeff: a.coeff.mul(b.coeff),
                });
            }
        }
        // (P_a + r_a)(P_b + r_b) = P_a P_b + P_a r_b + r_a P_b + r_a r_b
        let bound_a = self.poly_bound();
        let bound_b = other.poly_bound();
        let rem = bound_a
            .mul(other.rem)
            .add(bound_b.mul(self.rem))
            .add(self.rem.mul(other.rem));
        TaylorModel {
            n_vars: self.n_vars,
            degree: self.degree,
            x0: self.x0.clone(),
            radius: self.radius.clone(),
            terms,
            rem,
        }
        .truncate()
        .normalize()
    }

    /// Integer power via exponentiation by squaring (positive) or via the
    /// reciprocal (negative).
    pub fn powi(&self, n: i32) -> TaylorModel {
        if n == 0 {
            return self.constant(1.0);
        }
        if n < 0 {
            return self.recip_tm().powi(-n);
        }
        let mut result = self.constant(1.0);
        let mut base = self.clone();
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// Truncate: drop monomials of degree `> degree`, folding their range into
    /// the remainder.
    fn truncate(&self) -> TaylorModel {
        let mut m = self.zero_like();
        m.rem = self.rem;
        for t in &self.terms {
            if t.total_degree() as usize > self.degree {
                m.rem = m.rem.add(self.monomial_range(t));
            } else {
                m.terms.push(t.clone());
            }
        }
        m
    }

    /// Division `self / other` via the reciprocal Taylor model.
    pub fn div(&self, other: &TaylorModel) -> TaylorModel {
        self.mul(&other.recip_tm())
    }

    // -------------------------------------------------- unary (Taylor series)

    /// Shared machinery for `f(a)` given the operand `a = self`:
    /// `coeffs[k] = f⁽ᵏ⁾(c)/k!`, `deriv` an interval enclosure of
    /// `f⁽ᵈ⁺¹⁾(ξ)` over `hull(c, x_int)`.
    fn finish_series(&self, coeffs: &[Interval], deriv: Interval) -> TaylorModel {
        let d = self.degree;
        let x_int = self.bound_range();
        let c = self.constant_mid();
        // u = a − c, keeping the operand remainder. `shift(delta)` subtracts
        // `delta` from the constant term, so pass `+c`.
        let u = self.shift(c);
        let mut acc = self.constant_tm(coeffs[0]);
        let mut power = u.clone();
        for k in 1..=d {
            acc = acc.add(&power.scale(coeffs[k]));
            if k < d {
                power = power.mul(&u);
            }
        }
        // Lagrange remainder: |f⁽ᵈ⁺¹⁾(ξ)/(d+1)! · t^(d+1)|, t = a − c ∈ x_int − c.
        let t = x_int.sub(Interval::point(c));
        let mag_t = t.mag();
        let fact = factorial(d + 1);
        let rd = hi_round(deriv.mag() / fact * mag_t.powi((d + 1) as i32));
        let rem = acc.rem.add(Interval::new(-rd, rd));
        TaylorModel {
            n_vars: self.n_vars,
            degree: d,
            x0: self.x0.clone(),
            radius: self.radius.clone(),
            terms: acc.terms,
            rem,
        }
        .normalize()
    }

    fn constant_tm(&self, k: Interval) -> TaylorModel {
        let mut m = self.zero_like();
        m.terms.push(Monomial {
            exps: vec![0; self.n_vars],
            coeff: k,
        });
        m
    }

    pub fn exp_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() {
            return self.fallback(Interval::EMPTY);
        }
        let c = self.constant_mid();
        let ec = c.exp();
        let coeffs: Vec<Interval> = (0..=self.degree)
            .map(|k| iv(ec / factorial(k)))
            .collect();
        let dom = x_int.hull(Interval::point(c));
        let deriv = dom.exp();
        self.finish_series(&coeffs, deriv)
    }

    pub fn sin_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() {
            return self.fallback(Interval::EMPTY);
        }
        let c = self.constant_mid();
        let sc = c.sin();
        let cc = c.cos();
        let coeffs: Vec<Interval> = (0..=self.degree)
            .map(|k| {
                let v = match k % 4 {
                    0 => sc,
                    1 => cc,
                    2 => -sc,
                    _ => -cc,
                };
                iv(v / factorial(k))
            })
            .collect();
        self.finish_series(&coeffs, Interval::new(-1.0, 1.0))
    }

    pub fn cos_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() {
            return self.fallback(Interval::EMPTY);
        }
        let c = self.constant_mid();
        let cc = c.cos();
        let sc = c.sin();
        let coeffs: Vec<Interval> = (0..=self.degree)
            .map(|k| {
                let v = match k % 4 {
                    0 => cc,
                    1 => -sc,
                    2 => -cc,
                    _ => sc,
                };
                iv(v / factorial(k))
            })
            .collect();
        self.finish_series(&coeffs, Interval::new(-1.0, 1.0))
    }

    pub fn ln_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() || x_int.hi <= 0.0 {
            return self.fallback(x_int.ln());
        }
        if x_int.lo <= 0.0 {
            // Not differentiable near 0: fall back to the interval enclosure.
            return self.fallback(x_int.ln());
        }
        let c = self.constant_mid();
        // f⁽ᵏ⁾(c)/k! = (−1)^(k−1)/(k c^k) for k ≥ 1; k = 0 → ln(c).
        let coeffs: Vec<Interval> = (0..=self.degree)
            .map(|k| {
                if k == 0 {
                    iv(c.ln())
                } else {
                    let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
                    iv(sign / (k as f64 * c.powi(k as i32)))
                }
            })
            .collect();
        let dom = x_int.hull(Interval::point(c));
        // |f⁽ᵈ⁺¹⁾(x)| = d! / x^(d+1), decreasing → sup at dom.lo.
        let deriv = Interval::point(factorial(self.degree))
            .mul(Interval::point(dom.lo).powi(-(self.degree as i32 + 1)));
        self.finish_series(&coeffs, deriv)
    }

    pub fn sqrt_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() || x_int.hi < 0.0 {
            return self.fallback(x_int.sqrt());
        }
        if x_int.lo <= 0.0 {
            // Not differentiable at 0: interval fallback (still sound).
            return self.fallback(x_int.sqrt());
        }
        let c = self.constant_mid();
        let d = self.degree;
        let coeffs: Vec<Interval> = (0..=d)
            .map(|k| {
                let v = falling_factorial_halves(k) * c.powf(0.5 - k as f64) / factorial(k);
                iv(v)
            })
            .collect();
        let dom = x_int.hull(Interval::point(c));
        // |f⁽ᵈ⁺¹⁾(x)| = |C_d| · x^(−d−1/2) with C_d = ∏_{j=0}^{d}(1/2 − j).
        let cd = falling_factorial_halves(d + 1);
        let denom = Interval::point(dom.lo)
            .powi(d as i32)
            .mul(Interval::point(dom.lo).sqrt());
        let deriv = Interval::point(cd.abs())
            .mul(recip_or_entire(denom))
            .mul(Interval::new(-1.0, 1.0));
        self.finish_series(&coeffs, deriv)
    }

    /// Reciprocal `1/self` (used for division and negative powers).
    pub fn recip_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() {
            return self.fallback(Interval::EMPTY);
        }
        if x_int.lo <= 0.0 && x_int.hi >= 0.0 {
            // 0 ∈ range: reciprocal unbounded (or empty for exact zero).
            return self.fallback(recip_or_entire(x_int));
        }
        let c = self.constant_mid();
        let d = self.degree;
        // f⁽ᵏ⁾(c)/k! = (−1)^k / c^(k+1).
        let coeffs: Vec<Interval> = (0..=d)
            .map(|k| {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                iv(sign / c.powi(k as i32 + 1))
            })
            .collect();
        let dom = x_int.hull(Interval::point(c));
        // |f⁽ᵈ⁺¹⁾(x)| = (d+1)! / x^(d+2), decreasing → sup at dom.lo.
        let deriv = Interval::point(factorial(d + 1))
            .mul(Interval::point(dom.lo).powi(-(d as i32 + 2)));
        self.finish_series(&coeffs, deriv)
    }

    pub fn abs_tm(&self) -> TaylorModel {
        let x_int = self.bound_range();
        if x_int.is_empty() {
            return self.fallback(Interval::EMPTY);
        }
        if x_int.lo >= 0.0 {
            self.clone()
        } else if x_int.hi <= 0.0 {
            self.neg()
        } else {
            self.fallback(x_int.abs())
        }
    }
}

/// `∏_{j=0}^{k−1} (1/2 − j)`, the derivative prefactor of `x^(1/2)`.
fn falling_factorial_halves(k: usize) -> f64 {
    (0..k).fold(1.0_f64, |a, j| a * (0.5 - j as f64))
}

/// Reciprocal returning `ENTIRE` instead of `∅` when `0` is in the interval.
fn recip_or_entire(x: Interval) -> Interval {
    let r = x.recip();
    if r.is_empty() {
        Interval::ENTIRE
    } else {
        r
    }
}

impl fmt::Display for TaylorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T(x) range = {} (order {}, {} terms)", self.bound_range(), self.degree, self.terms.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veribound_interval::Interval;

    fn box2() -> BoxDomain {
        BoxDomain::new(vec![Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0)])
    }

    #[test]
    fn variable_tm_encloses_box() {
        let vs = TaylorModel::variables(&box2());
        // x0 = 0, radius = 1 → range [-1, 1]
        assert!(vs[0].bound_range().contains_interval(Interval::new(-1.0, 1.0)));
        assert!(vs[1].bound_range().contains_interval(Interval::new(-1.0, 1.0)));
    }

    #[test]
    fn polynomial_tm_sound() {
        // f(x, y) = x*y + x^2 over [-1,1]^2
        let b = box2();
        let vs = TaylorModel::variables(&b);
        let f = vs[0].mul(&vs[1]).add(&vs[0].powi(2));
        let range = f.bound_range();
        // exact range of x*y + x^2 is [-0.25, 2]; check endpoints contained
        assert!(range.lo <= -0.25);
        assert!(range.hi >= 2.0);
    }

    #[test]
    fn exp_tm_contains_point_values() {
        let b = box2();
        let x = TaylorModel::variables(&b)[0].clone();
        let f = x.exp_tm();
        // at x = 0.5, exp(0.5) ≈ 1.6487
        let at = f.eval_at(&[0.5, 0.0]);
        assert!(at.contains(0.5_f64.exp()));
    }
}
