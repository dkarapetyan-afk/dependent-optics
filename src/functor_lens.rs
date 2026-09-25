//! Functor lenses are dependent optics for the terminal indexed category
//! (Proposition 2).
//!
//! With `L = •`, an object is a pair `(A, X' → A)`. A morphism is a base map
//! `f: A → B` together with `A ×_B Y' → X'` over `A`. That is the dependent
//! lens whose forward family is the identity `A → A`.

use crate::dlens::{self, DLens};
use crate::finset::{self, Mor};
use crate::optic::Family;
use crate::slice::{self, SliceObj};

/// The identity family `A → A` on the forward leg and `X' → A` backward.
pub fn object(backward: &SliceObj) -> Family {
    Family {
        forward: SliceObj {
            base: backward.base,
            total: backward.base,
            leg: finset::id(backward.base),
        },
        backward: backward.clone(),
    }
}

/// A functor lens is a dependent lens between identity-forward families, so
/// `get: A → B` is the base map and `put` is the backward leg.
pub fn is_functor_lens(lens: &DLens) -> bool {
    lens.dom.forward.leg == finset::id(lens.dom.forward.base)
        && lens.dom.forward.total == lens.dom.forward.base
        && lens.cod.forward.leg == finset::id(lens.cod.forward.base)
        && lens.cod.forward.total == lens.cod.forward.base
}

/// Build the lens from an explicit base map and a put `A ×_B Y' → X'`.
pub fn from_base(dom: &SliceObj, cod: &SliceObj, base_map: &Mor, put: Mor) -> DLens {
    let lens = DLens {
        dom: object(dom),
        cod: object(cod),
        get: base_map.clone(),
        put: slice::SliceMor {
            dom: {
                let pb = finset::pullback(base_map, &cod.leg);
                SliceObj {
                    base: dom.base,
                    total: pb.apex,
                    leg: pb.to_left.clone(),
                }
            },
            cod: dom.clone(),
            map: put,
        },
    };
    lens.check();
    lens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;
    use crate::slice::objects_up_to;

    #[test]
    fn functor_lenses_are_the_identity_forward_dependent_lenses() {
        let objs = objects_up_to(MAX_CARD);
        for dom in &objs {
            for cod in &objs {
                for base in finset::morphisms(dom.base, cod.base) {
                    let fam_d = object(dom);
                    let fam_c = object(cod);
                    let lenses: Vec<_> = dlens::hom(&fam_d, &fam_c)
                        .into_iter()
                        .filter(|l| l.get == base)
                        .collect();
                    for lens in &lenses {
                        assert!(is_functor_lens(lens));
                        assert_eq!(dlens::compose(lens, &dlens::identity(&fam_d)), *lens);
                    }
                    // Every such lens's put domain is the pullback A ×_B Y'.
                    let pb = finset::pullback(&base, &cod.leg);
                    for lens in &lenses {
                        assert_eq!(lens.put.dom.total, pb.apex);
                    }
                }
            }
        }
        // Composition of two functor lenses is a functor lens, and the base
        // map composes.
        for a in &objs {
            for b in &objs {
                for f in dlens::hom(&object(a), &object(b)) {
                    for c in &objs {
                        for g in dlens::hom(&object(b), &object(c)) {
                            let h = dlens::compose(&g, &f);
                            assert!(is_functor_lens(&h));
                            assert_eq!(h.get, finset::compose(&g.get, &f.get));
                        }
                    }
                }
            }
        }
    }
}
