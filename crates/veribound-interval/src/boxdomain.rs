//! An interval box (vector of intervals) used as the search domain of the
//! branch-and-bound solver and the argument set of Taylor models.

use crate::Interval;

/// A Cartesian product of intervals `X₁ × … × Xₙ ⊆ ℝⁿ`.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxDomain {
    dims: Vec<Interval>,
}

impl BoxDomain {
    /// Build a box from per-dimension intervals.
    pub fn new(dims: Vec<Interval>) -> Self {
        BoxDomain { dims }
    }

    /// A degenerate box at a single point.
    pub fn point(p: &[f64]) -> Self {
        BoxDomain {
            dims: p.iter().map(|&x| Interval::point(x)).collect(),
        }
    }

    /// Number of dimensions.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dims.len()
    }

    /// The interval of dimension `i`.
    #[inline]
    pub fn get(&self, i: usize) -> Interval {
        self.dims[i]
    }

    /// Mutable access to dimension `i`.
    #[inline]
    pub fn set(&mut self, i: usize, v: Interval) {
        self.dims[i] = v;
    }

    /// Midpoint vector (component-wise).
    pub fn mid(&self) -> Vec<f64> {
        self.dims.iter().map(|d| d.mid()).collect()
    }

    /// Width of dimension `i`, rounded up.
    #[inline]
    pub fn width(&self, i: usize) -> f64 {
        self.dims[i].width_up()
    }

    /// Maximum width over all dimensions, rounded up.
    pub fn diameter(&self) -> f64 {
        self.dims.iter().map(|d| d.width_up()).fold(0.0, f64::max)
    }

    /// Index of the widest dimension.
    pub fn widest_dim(&self) -> usize {
        let mut best = 0;
        let mut bw = self.dims[0].width_up();
        for (i, d) in self.dims.iter().enumerate().skip(1) {
            let w = d.width_up();
            if w > bw {
                bw = w;
                best = i;
            }
        }
        best
    }

    /// `true` iff `p ∈ self`.
    pub fn contains_point(&self, p: &[f64]) -> bool {
        self.dims
            .iter()
            .zip(p.iter())
            .all(|(d, &x)| d.contains(x))
    }

    /// Split along the widest dimension at its midpoint.
    pub fn split(&self) -> (BoxDomain, BoxDomain) {
        self.split_dim(self.widest_dim())
    }

    /// Split along dimension `i` at its midpoint. For finite `lo < hi` the
    /// midpoint is forced strictly inside so both children make progress.
    pub fn split_dim(&self, i: usize) -> (BoxDomain, BoxDomain) {
        let d = self.dims[i];
        let mut m = d.mid();
        if d.lo.is_finite() && d.hi.is_finite() && d.lo < d.hi {
            if m <= d.lo {
                m = d.hi;
            } else if m >= d.hi {
                m = d.lo;
            }
        }
        let mut left = self.clone();
        let mut right = self.clone();
        left.dims[i] = Interval::new(d.lo, m);
        right.dims[i] = Interval::new(m, d.hi);
        (left, right)
    }

    /// Iterate over the per-dimension intervals.
    pub fn iter(&self) -> impl Iterator<Item = &Interval> {
        self.dims.iter()
    }
}

impl std::fmt::Display for BoxDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, ")")
    }
}
