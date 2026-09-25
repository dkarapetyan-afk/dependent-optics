//! Enriched optics (Remark 1).
//!
//! The coend of Clarke–Elkins–Gibbons–Loregian–Milewski–Pillmore–Román,
//! Definition 2.1, is taken in a cosmos. For `V = FinVect_{F₂}` the closed
//! form of Section 3.4 is a vector space, and composition of those closed
//! forms is a linear map. The underlying set of points is computed in
//! [`crate::bimod`]; here composition of the closed linear maps is checked
//! to be bilinear on dimensions at most 1.

use crate::vect::{self, Lin};

/// A closed tensor-optic `X → [Y', X'] ⊗ Y`, stored as a linear map into the
/// tensor product whose basis order is `(y', x', y)`.
pub fn closed_hom(dx: u32, dy: u32, dxp: u32, dyp: u32) -> Vec<Lin> {
    vect::all(dx, dyp * dxp * dy)
}

/// Compose `X → [Y', X'] ⊗ Y` with `Y → [Z', Y'] ⊗ Z` when `X' = Y` is not the
/// shape we need. The closed forms compose as linear maps only after the
/// tensorator; for the self-case `Y = X` and `Y' = X'` the composite of two
/// endomorphisms of `Hom(X, [X', X] ⊗ X)` is ordinary linear composition.
pub fn compose_closed(g: &Lin, f: &Lin) -> Lin {
    vect::compose(g, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bimod::closed_dimension;

    #[test]
    fn closed_composition_is_linear() {
        for d in 0..=1 {
            let dim = closed_dimension(d, d, d, d);
            assert_eq!(closed_hom(d, d, d, d).len(), 1usize << dim);
            for f in closed_hom(d, d, d, d) {
                for g in closed_hom(d, d, d, d) {
                    for h in closed_hom(d, d, d, d) {
                        let left = compose_closed(&h, &compose_closed(&g, &f));
                        let right = compose_closed(&compose_closed(&h, &g), &f);
                        assert_eq!(left, right);
                        let sum = vect::add(&f, &g);
                        assert_eq!(
                            compose_closed(&h, &sum),
                            vect::add(&compose_closed(&h, &f), &compose_closed(&h, &g))
                        );
                    }
                }
            }
        }
    }
}
