//! Commutative comonoids and comodules in `(FinSet, ×)`.
//!
//! Every object has exactly one commutative comonoid structure, the diagonal.
//! A comodule over that structure is a morphism into the base, and `⊗_B` is a
//! pullback. Dependent monoidal lenses therefore have the same normal forms as
//! [`crate::dlens`] (Section 3.2 on a cartesian base).

use crate::finset::{self, Mor};
use crate::slice::SliceObj;

pub fn diagonal(a: u32) -> Mor {
    let prod = finset::product(a, a);
    Mor {
        dom: a,
        cod: prod.obj,
        map: (0..a).map(|i| finset::pair_index(a, i, i)).collect(),
    }
}

pub fn counit(a: u32) -> Mor {
    finset::to_terminal(a)
}

/// `(f × g): A × C → B × D`.
pub fn times(f: &Mor, g: &Mor) -> Mor {
    let dom = finset::product(f.dom, g.dom);
    let cod = finset::product(f.cod, g.cod);
    Mor {
        dom: dom.obj,
        cod: cod.obj,
        map: (0..dom.obj)
            .map(|k| {
                finset::pair_index(g.cod, f.apply(dom.fst.apply(k)), g.apply(dom.snd.apply(k)))
            })
            .collect(),
    }
}

/// `(f × f) ∘ δ = δ ∘ f` for every `f`, so every function is a comonoid homomorphism.
pub fn preserves_diagonal(f: &Mor) -> bool {
    let left = finset::compose(&times(f, f), &diagonal(f.dom));
    let right = finset::compose(&diagonal(f.cod), f);
    left == right
}

/// The coaction `X → X × A` corresponding to a slice `X → A`.
pub fn coaction(leg: &Mor) -> Mor {
    finset::pair(&finset::id(leg.dom), leg)
}

/// `M ⊗_B N` as the equalizer of the two coactions into `M × B × N`,
/// which on the diagonal comonoid is the pullback of the two legs.
pub fn tensor(m: &SliceObj, n: &SliceObj) -> SliceObj {
    assert_eq!(m.base, n.base);
    let pb = finset::pullback(&m.leg, &n.leg);
    SliceObj {
        base: m.base,
        total: pb.apex,
        leg: finset::compose(&m.leg, &pb.to_left),
    }
}

/// Proposition 5 on this cartesian base: comodules over a coproduct of
/// comonoids are pairs of comodules, which is lextensivity of slices.
pub fn comodules_over(base: u32, max_total: u32) -> Vec<SliceObj> {
    let mut out = Vec::new();
    for total in 0..=max_total {
        for leg in finset::morphisms(total, base) {
            out.push(SliceObj { base, total, leg });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;

    #[test]
    fn unique_diagonal_comonoid_and_every_map_is_a_hom() {
        for a in 0..=MAX_CARD {
            let d = diagonal(a);
            // coassociativity: (δ × id) ∘ δ = (id × δ) ∘ δ
            let prod = finset::product(a, a);
            let delta_id = finset::pair(&finset::compose(&diagonal(a), &prod.fst), &prod.snd);
            let id_delta = finset::pair(&prod.fst, &finset::compose(&diagonal(a), &prod.snd));
            assert_eq!(
                finset::compose(&delta_id, &d),
                finset::compose(&id_delta, &d),
                "coassociativity {a}"
            );
            // counit: (ε × id) ∘ δ is the left unitor A → 1 × A
            let left_unitor = finset::compose(&times(&counit(a), &finset::id(a)), &d);
            assert_eq!(
                left_unitor.map,
                (0..a)
                    .map(|i| finset::pair_index(a, 0, i))
                    .collect::<Vec<_>>()
            );
            for b in 0..=MAX_CARD {
                for f in finset::morphisms(a, b) {
                    assert!(preserves_diagonal(&f), "comonoid hom {f:?}");
                }
            }
        }
    }

    #[test]
    fn tensor_of_diagonal_comodules_is_pullback() {
        for base in 0..=MAX_CARD {
            for m in comodules_over(base, MAX_CARD) {
                for n in comodules_over(base, MAX_CARD) {
                    let t = tensor(&m, &n);
                    let pb = finset::pullback(&m.leg, &n.leg);
                    assert_eq!(t.total, pb.apex);
                    assert_eq!(t.leg, finset::compose(&m.leg, &pb.to_left));
                    // The equalizer presentation agrees: elements of M×N whose
                    // two coactions into M×B×N coincide are the pairs with
                    // equal legs.
                    let prod = finset::product(m.total, n.total);
                    let mut count = 0u32;
                    for k in 0..prod.obj {
                        let x = prod.fst.apply(k);
                        let y = prod.snd.apply(k);
                        if m.leg.apply(x) == n.leg.apply(y) {
                            count += 1;
                        }
                    }
                    assert_eq!(count, t.total);
                }
            }
        }
    }

    #[test]
    fn proposition_5_comodules_split_along_coproducts() {
        for a in 0..=MAX_CARD {
            for b in 0..=MAX_CARD {
                let sum = a + b;
                let whole = comodules_over(sum, MAX_CARD);
                for leg_obj in &whole {
                    let (xa, xb, iso) = crate::slice::lextensive_split(&leg_obj.leg, a);
                    assert!(finset::is_iso(&iso));
                    assert_eq!(xa.base, a);
                    assert_eq!(xb.base, b);
                }
                // Coproduct injections of the underlying sets are comonoid homs.
                let cop = finset::coproduct(a, b);
                assert!(preserves_diagonal(&cop.inl));
                assert!(preserves_diagonal(&cop.inr));
            }
        }
    }
}
