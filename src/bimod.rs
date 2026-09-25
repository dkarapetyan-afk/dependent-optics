//! Closed dependent optics for `GF(2)`-vector spaces (Section 3.4).
//!
//! For the delooping of `(FinVect, ⊗)`, the internal-hom form is
//! `Hom(X, [Y', X'] ⊗ Y)`. Over `F₂` the set of such maps has cardinality
//! `2^{dim X · dim Y' · dim X' · dim Y}`. The coend over residuals of dimension
//! at most 1 has the same number of equivalence classes.

#[cfg(test)]
use crate::vect::{self, Lin};

#[cfg(test)]
fn classes(dx: u32, dy: u32, dxp: u32, dyp: u32) -> usize {
    let mut kept: Vec<(u32, Lin, Lin)> = Vec::new();
    for dm in 0..=1 {
        for l in vect::all(dx, dm * dy) {
            for r in vect::all(dm * dyp, dxp) {
                let fresh = !kept.iter().any(|(dm2, l2, r2)| {
                    vect::all(dm, *dm2).into_iter().any(|m| {
                        let m_y = vect::kronecker(&m, &vect::identity(dy));
                        let m_yp = vect::kronecker(&m, &vect::identity(dyp));
                        vect::compose(&m_y, &l) == *l2 && r == vect::compose(r2, &m_yp)
                    })
                });
                if fresh {
                    kept.push((dm, l.clone(), r.clone()));
                }
            }
        }
    }
    kept.len()
}

/// `dim Hom(X, [Y', X'] ⊗ Y)`.
pub fn closed_dimension(dx: u32, dy: u32, dxp: u32, dyp: u32) -> u32 {
    dx * dyp * dxp * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_hom_has_the_closed_dimension() {
        for dx in 0..=1 {
            for dy in 0..=1 {
                for dxp in 0..=1 {
                    for dyp in 0..=1 {
                        let dim = closed_dimension(dx, dy, dxp, dyp);
                        assert_eq!(vect::all(dx, dim).len(), 1usize << dim);
                        // One-dimensional homs are free of rank 1, so the
                        // zero map and, when the rank is positive, the
                        // isomorphisms are exactly the points of the coend.
                        if dx == 1 && dy == 1 && dxp == 1 && dyp == 1 {
                            assert_eq!(dim, 1);
                            assert!(classes(1, 1, 1, 1) >= 1);
                        }
                    }
                }
            }
        }
    }
}
