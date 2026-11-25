//! Certified outward-rounded interval arithmetic.
//!
//! # Representation
//!
//! An [`Interval`] is a closed set `[lo, hi] ⊆ ℝ ∪ {±∞}` stored as two `f64`
//! endpoints. The special values follow the classical convention used by
//! interval libraries (e.g. the "empty = reversed bound" trick):
//!
//! * [`Interval::EMPTY`] is encoded as `lo = +∞, hi = −∞` (i.e. `lo > hi`).
//! * [`Interval::ENTIRE`] is `lo = −∞, hi = +∞` (i.e. `(−∞, +∞)`).
//! * A `NaN` in either endpoint is *contained*: every query treats such a value
//!   as empty, so a `NaN` produced by an out-of-domain operation can never leak
//!   an unsound "result".
//!
//! # Enclosure discipline (outward rounding)
//!
//! Every arithmetic/transcendental operation returns an interval whose machine
//! endpoints are *outside* the true real result:
//!
//! * lower bounds are pushed down with [`f64::next_down`];
//! * upper bounds are pushed up with [`f64::next_up`];
//! * `−0.0`/`+0.0` compare equal, so a rounded `0.0` endpoint is widened to
//!   `±5e−324` as appropriate and still contains the exact value;
//! * infinite endpoints are never "rounded" further (they are already exact);
//! * out-of-domain inputs (`ln(x)` for `x ≤ 0`, `sqrt(x)` for `x < 0`) are
//!   handled by intersecting with the mathematical domain, returning [`EMPTY`]
//!   when no valid point remains.

use std::fmt;

mod boxdomain;
pub use boxdomain::BoxDomain;

/// A rigorous closed interval `[lo, hi]`.
#[derive(Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower endpoint. May be `−∞`.
    pub lo: f64,
    /// Upper endpoint. May be `+∞`.
    pub hi: f64,
}

/// The empty interval: `lo > hi` (encoded `[+∞, −∞]`).
pub const EMPTY: Interval = Interval {
    lo: f64::INFINITY,
    hi: f64::NEG_INFINITY,
};

/// The entire real line `(−∞, +∞)`.
pub const ENTIRE: Interval = Interval {
    lo: f64::NEG_INFINITY,
    hi: f64::INFINITY,
};

/// A degenerate zero interval `[0, 0]`.
pub const ZERO: Interval = Interval { lo: 0.0, hi: 0.0 };

/// A degenerate one interval `[1, 1]`.
pub const ONE: Interval = Interval { lo: 1.0, hi: 1.0 };

/// Round `x` down toward `−∞` for use as a lower bound. Non-finite values are
/// already exact endpoints (or NaN, which we leave alone so it is contained by
/// [`Interval::is_empty`]).
#[inline]
fn round_down(x: f64) -> f64 {
    if x.is_finite() {
        x.next_down()
    } else {
        x
    }
}

/// Round `x` up toward `+∞` for use as an upper bound.
#[inline]
fn round_up(x: f64) -> f64 {
    if x.is_finite() {
        x.next_up()
    } else {
        x
    }
}

/// Extended-real multiplication used by interval multiplication. The only NaN
/// hazard in the four-corner product is `0 * ±∞`; for a closed interval the
/// value `0` is itself an endpoint, so the true product set maps a zero corner
/// to `0` while neighbouring non-zero corners supply the ±∞ limits. The sign of
/// a zero corner is irrelevant (`−0.0 == 0.0`).
#[inline]
fn mul_ext(x: f64, y: f64) -> f64 {
    if x == 0.0 || y == 0.0 {
        0.0
    } else {
        x * y
    }
}

impl Interval {
    /// The empty interval `[+∞, −∞]`.
    pub const EMPTY: Interval = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    /// The entire real line `(−∞, +∞)`.
    pub const ENTIRE: Interval = Interval {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };
    /// The degenerate zero interval `[0, 0]`.
    pub const ZERO: Interval = Interval { lo: 0.0, hi: 0.0 };
    /// The degenerate one interval `[1, 1]`.
    pub const ONE: Interval = Interval { lo: 1.0, hi: 1.0 };

    /// Build an interval from already-rounded endpoints, normalising `NaN`
    /// (contained → empty) and reversed bounds (`lo > hi` → empty).
    #[inline]
    pub fn new(lo: f64, hi: f64) -> Self {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            EMPTY
        } else {
            Interval { lo, hi }
        }
    }

    /// Build an interval from *exact* (real) endpoints, applying outward
    /// rounding: the returned interval is guaranteed to contain `[lo, hi]`.
    #[inline]
    pub fn from_exact(lo: f64, hi: f64) -> Self {
        Self::new(round_down(lo), round_up(hi))
    }

    /// Degenerate interval `[x, x]` with outward rounding.
    #[inline]
    pub fn point(x: f64) -> Self {
        Self::from_exact(x, x)
    }

    /// `true` iff the interval contains no points.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.lo.is_nan() || self.hi.is_nan() || self.lo > self.hi
    }

    /// `true` iff the interval is the entire real line.
    #[inline]
    pub fn is_entire(self) -> bool {
        self.lo == f64::NEG_INFINITY && self.hi == f64::INFINITY
    }

    /// `true` iff the interval is a single point.
    #[inline]
    pub fn is_point(self) -> bool {
        !self.is_empty() && self.lo == self.hi
    }

    /// `true` iff `x ∈ [lo, hi]`.
    #[inline]
    pub fn contains(self, x: f64) -> bool {
        !self.is_empty() && self.lo <= x && x <= self.hi
    }

    /// `true` iff `other ⊆ self`.
    #[inline]
    pub fn contains_interval(self, other: Interval) -> bool {
        if other.is_empty() {
            return true; // the empty set is a subset of everything
        }
        !self.is_empty() && self.lo <= other.lo && other.hi <= self.hi
    }

    /// Width `hi − lo` (may be `+∞`); `0.0` for the empty interval.
    #[inline]
    pub fn width(self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// Width rounded up, for termination criteria that must stay conservative.
    #[inline]
    pub fn width_up(self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            round_up(self.hi - self.lo)
        }
    }

    /// Radius `width / 2`, rounded up.
    #[inline]
    pub fn radius(self) -> f64 {
        round_up(self.width() * 0.5)
    }

    /// Midpoint. For infinite endpoints a finite representative is returned.
    /// Returns `NaN` for the empty interval.
    #[inline]
    pub fn mid(self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        if self.lo == f64::NEG_INFINITY {
            return if self.hi == f64::INFINITY { 0.0 } else { self.hi };
        }
        if self.hi == f64::INFINITY {
            return self.lo;
        }
        // Overflow-safe midpoint, clamped back into the interval.
        (self.lo + (self.hi - self.lo) * 0.5).clamp(self.lo, self.hi)
    }

    /// Magnitude `max(|lo|, |hi|)` rounded up (0 for empty).
    #[inline]
    pub fn mag(self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            round_up(self.lo.abs().max(self.hi.abs()))
        }
    }

    // ---------------------------------------------------------------- set ops

    /// Convex hull of two intervals.
    #[inline]
    pub fn hull(self, other: Interval) -> Interval {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Interval::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    /// Intersection `self ∩ other` (empty when disjoint).
    #[inline]
    pub fn intersect(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return EMPTY;
        }
        Interval::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    /// Additive negation `−[lo, hi] = [−hi, −lo]`.
    #[inline]
    pub fn neg(self) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        Interval::new(round_down(-self.hi), round_up(-self.lo))
    }

    // ------------------------------------------------------------- arithmetic

    /// `self + other`.
    #[inline]
    pub fn add(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return EMPTY;
        }
        Interval::new(
            round_down(self.lo + other.lo),
            round_up(self.hi + other.hi),
        )
    }

    /// `self − other`.
    #[inline]
    pub fn sub(self, other: Interval) -> Interval {
        self.add(other.neg())
    }

    /// `self · other` via the four-corner product.
    #[inline]
    pub fn mul(self, other: Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return EMPTY;
        }
        let c1 = mul_ext(self.lo, other.lo);
        let c2 = mul_ext(self.lo, other.hi);
        let c3 = mul_ext(self.hi, other.lo);
        let c4 = mul_ext(self.hi, other.hi);
        let lo = c1.min(c2).min(c3).min(c4);
        let hi = c1.max(c2).max(c3).max(c4);
        Interval::new(round_down(lo), round_up(hi))
    }

    /// Reciprocal `1 / [lo, hi]` for an interval that excludes `0`. Returns
    /// [`EMPTY`] when `0 ∈ self` (use [`Interval::extended_div`] to split).
    #[inline]
    pub fn recip(self) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        if self.lo > 0.0 {
            // decreasing on the positive branch
            Interval::new(round_down(1.0 / self.hi), round_up(1.0 / self.lo))
        } else if self.hi < 0.0 {
            Interval::new(round_down(1.0 / self.hi), round_up(1.0 / self.lo))
        } else {
            EMPTY // 0 ∈ [lo, hi]
        }
    }

    /// `self / other` where `other` excludes `0`. Returns [`EMPTY`] when
    /// `0 ∈ other` (division is genuinely unbounded; use [`extended_div`]).
    #[inline]
    pub fn div(self, other: Interval) -> Interval {
        let r = other.recip();
        if r.is_empty() {
            EMPTY
        } else {
            self.mul(r)
        }
    }

    /// Set-valued division `self / other`, returning `0`, `1`, or `2` intervals
    /// exactly when the divisor straddles zero. Never panics.
    pub fn extended_div(self, other: Interval) -> Vec<Interval> {
        if self.is_empty() || other.is_empty() {
            return Vec::new();
        }
        if other.lo > 0.0 || other.hi < 0.0 {
            // 0 ∉ divisor: ordinary division.
            let r = other.recip();
            return if r.is_empty() { Vec::new() } else { vec![self.mul(r)] };
        }
        if other.lo == 0.0 && other.hi == 0.0 {
            // Division by the exact zero set → no finite results.
            return Vec::new();
        }
        // 0 ∈ divisor. Break the punctured domain into (−∞,0) and (0,+∞).
        let mut out = Vec::new();
        if other.lo < 0.0 {
            // piece on (−∞, 0):  1/x ∈ (−∞, 1/other.lo]
            let p = Interval::new(f64::NEG_INFINITY, round_up(1.0 / other.lo));
            out.push(self.mul(p));
        }
        if other.hi > 0.0 {
            // piece on (0, +∞):  1/x ∈ [1/other.hi, +∞)
            let p = Interval::new(round_down(1.0 / other.hi), f64::INFINITY);
            out.push(self.mul(p));
        }
        // Collapse duplicate degenerate pieces (e.g. 0 / [−1,1] = {0}).
        out.dedup();
        out
    }

    /// Integer power `selfⁿ`.
    pub fn powi(self, n: i32) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        if n == 0 {
            return ONE;
        }
        if n > 0 {
            let nu = n as u32;
            if nu % 2 == 1 {
                // odd power: monotone increasing
                let lo = self.lo.powi(n);
                let hi = self.hi.powi(n);
                return Interval::new(round_down(lo), round_up(hi));
            }
            // even power: xⁿ ≥ 0, min 0 if 0 ∈ [lo,hi], else min(|lo|,|hi|)ⁿ
            let m = self.lo.abs().min(self.hi.abs());
            let mm = self.lo.abs().max(self.hi.abs());
            let lower = if self.lo <= 0.0 && 0.0 <= self.hi {
                0.0
            } else {
                m.powi(n)
            };
            let upper = mm.powi(n);
            Interval::new(round_down(lower), round_up(upper))
        } else {
            // negative power: xⁿ = 1 / xᵐ with m = −n. Use set-valued division
            // and take the hull so a base straddling 0 yields the correct
            // (possibly unbounded) enclosure instead of an unsound ∅.
            let pm = self.powi(-n);
            let pieces = ONE.extended_div(pm);
            let mut acc = EMPTY;
            for p in pieces {
                acc = acc.hull(p);
            }
            acc
        }
    }

    /// Square root, domain `[0, +∞)`.
    #[inline]
    pub fn sqrt(self) -> Interval {
        if self.is_empty() || self.hi < 0.0 {
            return EMPTY;
        }
        let lo_in = self.lo.max(0.0);
        Interval::new(round_down(lo_in.sqrt()), round_up(self.hi.sqrt()))
    }

    /// `exp([lo, hi])` (monotone increasing).
    #[inline]
    pub fn exp(self) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        Interval::new(round_down(self.lo.exp()), round_up(self.hi.exp()))
    }

    /// Natural logarithm, domain `(0, +∞)`. If the lower endpoint is ≤ 0 the
    /// result's lower bound is `−∞` (since `ln(x) → −∞` as `x → 0⁺`).
    #[inline]
    pub fn ln(self) -> Interval {
        if self.is_empty() || self.hi <= 0.0 {
            return EMPTY;
        }
        let lo_in = self.lo.max(0.0);
        Interval::new(round_down(lo_in.ln()), round_up(self.hi.ln()))
    }

    /// `sin([lo, hi])`, sound over any width.
    pub fn sin(self) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        // A full period covers the whole codomain.
        if round_up(self.hi - self.lo) >= TAU_LO {
            return Interval::new(-1.0, 1.0);
        }
        let mut lo = round_down(self.lo.sin().min(self.hi.sin()));
        let mut hi = round_up(self.lo.sin().max(self.hi.sin()));
        // Extrema of sin occur at π/2 + kπ (k even → +1, k odd → −1).
        for k in critical_ks(self.lo, self.hi, FRAC_PI_2) {
            if k & 1 == 0 {
                hi = 1.0;
            } else {
                lo = -1.0;
            }
        }
        Interval::new(lo, hi)
    }

    /// `cos([lo, hi])`, sound over any width.
    pub fn cos(self) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        if round_up(self.hi - self.lo) >= TAU_LO {
            return Interval::new(-1.0, 1.0);
        }
        let mut lo = round_down(self.lo.cos().min(self.hi.cos()));
        let mut hi = round_up(self.lo.cos().max(self.hi.cos()));
        // Extrema of cos occur at kπ (k even → +1, k odd → −1).
        for k in critical_ks(self.lo, self.hi, 0.0) {
            if k & 1 == 0 {
                hi = 1.0;
            } else {
                lo = -1.0;
            }
        }
        Interval::new(lo, hi)
    }

    /// `|self|`.
    #[inline]
    pub fn abs(self) -> Interval {
        if self.is_empty() {
            return EMPTY;
        }
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Interval::new(0.0, round_up(self.lo.abs().max(self.hi.abs())))
        }
    }

    /// Scalar `k · self`.
    #[inline]
    pub fn scale(self, k: f64) -> Interval {
        self.mul(Interval::point(k))
    }
}

// -------------------------------------------------------------------- helpers

/// Lower bound on `2π` used only as an early-exit optimisation: since the range
/// of sin/cos is always a subset of `[−1, 1]`, returning `[−1, 1]` is *always*
/// sound regardless of width, so the exact value here is immaterial.
const TAU_LO: f64 = 2.0 * std::f64::consts::PI;
/// `π/2` as a plain f64 constant for critical-point indexing.
const FRAC_PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// Return every integer `k` such that the *critical point* `p + kπ` may lie
/// inside `[lo, hi]`. The caller only invokes this once the interval is known
/// to span strictly less than `2π`, so at most three critical points can lie
/// inside; `n0 = floor((lo − p)/π)` anchors the search and only a constant-size
/// window is examined (no unbounded loop for very large coordinates).
///
/// Inclusion of `±1` (the extremal values of sin/cos) is the *sound* direction:
/// at worst the enclosure is a little looser near an exact critical point,
/// never too tight.
fn critical_ks(lo: f64, hi: f64, p: f64) -> Vec<i64> {
    let pi = std::f64::consts::PI;
    let n0 = ((lo - p) / pi).floor() as i64;
    let mut out = Vec::new();
    for k in (n0 - 1)..=(n0 + 3) {
        // Enclosure of the true irrational critical point p + kπ.
        let c = p + (k as f64) * pi;
        let c_lo = round_down(c);
        let c_hi = round_up(c);
        if c_lo <= hi && c_hi >= lo {
            out.push(k);
        }
    }
    out
}

impl fmt::Debug for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "[{}, {}]", self.lo, self.hi)
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "empty")
        } else if self.is_entire() {
            write!(f, "(-inf, +inf)")
        } else {
            write!(f, "[{:.17e}, {:.17e}]", self.lo, self.hi)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_entire() {
        assert!(EMPTY.is_empty());
        assert!(!ENTIRE.is_empty());
        assert!(ENTIRE.is_entire());
        assert!(!Interval::new(0.0, 0.0).is_empty());
        // NaN is contained.
        assert!(Interval::new(f64::NAN, 0.0).is_empty());
        assert!(Interval::new(0.0, f64::NAN).is_empty());
    }

    #[test]
    fn outward_rounding_add_sub() {
        // 0.1 is not exactly representable; outward rounding must widen.
        let a = Interval::point(0.1);
        let b = Interval::point(0.2);
        let s = a.add(b);
        assert!(s.lo <= 0.1 + 0.2 && s.hi >= 0.1 + 0.2);
        assert!(s.contains(0.1 + 0.2));
    }

    #[test]
    fn mul_division_edge_cases() {
        // Exact zero times Entire → a tiny interval containing 0 (outward
        // rounding widens the degenerate result).
        assert!(Interval::ZERO.mul(ENTIRE).contains(0.0));
        // Entire * Entire == Entire
        assert!(ENTIRE.mul(ENTIRE).is_entire());
        // sign cases
        let p = Interval::new(1.0, 2.0);
        let n = Interval::new(-2.0, -1.0);
        let z = Interval::new(-1.0, 1.0);
        assert!(p.mul(n).contains_interval(Interval::new(-4.0, -1.0)));
        assert!(z.mul(z).contains_interval(Interval::new(-1.0, 1.0)));
    }

    #[test]
    fn extended_division_straddling_zero() {
        let num = Interval::point(1.0);
        let den = Interval::new(-2.0, 3.0);
        let pieces = num.extended_div(den);
        assert_eq!(pieces.len(), 2);
        // 1/[-2,0) = (-inf,-0.5], 1/(0,3] = [1/3,+inf)
        assert!(pieces[0].lo == f64::NEG_INFINITY);
        assert!(pieces[0].contains(-0.5));
        assert!(pieces[0].contains(-1.0));
        assert!(pieces[1].hi == f64::INFINITY);
        assert!(pieces[1].contains(1.0 / 3.0));
        assert!(pieces[1].contains(1.0));

        // 0 / [-2,3] == {0}
        let z = Interval::ZERO.extended_div(den);
        assert_eq!(z.len(), 1);
        assert!(z[0].contains(0.0));

        // division by exact zero → empty
        assert!(num.extended_div(Interval::ZERO).is_empty());
    }

    #[test]
    fn transcendentals_domain_clamping() {
        assert!(Interval::new(-1.0, 4.0).sqrt().contains(2.0));
        assert!(!Interval::new(-1.0, 4.0).sqrt().contains(-1.0));
        assert!(Interval::new(-3.0, -1.0).sqrt().is_empty());

        // ln over (0, 1] has lower bound -inf
        let l = Interval::new(0.0, 1.0).ln();
        assert!(l.lo == f64::NEG_INFINITY);
        assert!(l.contains(0.0_f64.ln()));
        assert!(Interval::new(-2.0, 0.0).ln().is_empty());
    }

    #[test]
    fn sin_cos_soundness() {
        // Full period.
        assert_eq!(Interval::new(0.0, TAU_LO).sin(), Interval::new(-1.0, 1.0));
        // Contains a max at π/2.
        let s = Interval::new(0.0, 2.0).sin();
        assert!(s.contains(1.0));
        assert!(s.hi >= 1.0);
        // cos contains 1 at 0.
        let c = Interval::new(-1.0, 1.0).cos();
        assert!(c.contains(1.0));
    }

    #[test]
    fn powi_even_odd() {
        let a = Interval::new(-2.0, 3.0);
        assert!(a.powi(2).contains_interval(Interval::new(0.0, 9.0)));
        let b = Interval::new(-2.0, -1.0);
        assert!(b.powi(2).contains_interval(Interval::new(1.0, 4.0)));
        // odd power is monotone
        let c = Interval::new(-2.0, 3.0);
        assert!(c.powi(3).contains_interval(Interval::new(-8.0, 27.0)));
    }

    #[test]
    fn hull_intersect() {
        let a = Interval::new(0.0, 1.0);
        let b = Interval::new(0.5, 2.0);
        assert_eq!(a.hull(b), Interval::new(0.0, 2.0));
        assert_eq!(a.intersect(b), Interval::new(0.5, 1.0));
        assert!(a.intersect(Interval::new(2.0, 3.0)).is_empty());
        assert_eq!(a.hull(EMPTY), a);
    }
}
