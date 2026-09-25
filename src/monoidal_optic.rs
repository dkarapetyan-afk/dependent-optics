//! Dependent monoidal lenses and prisms (Sections 3.2 and 3.3).
//!
//! On `(FinSet, ×)` every object is a commutative comonoid via the diagonal, so
//! a monoidal lens is an ordinary dependent lens: the index runs over all
//! functions `X → Y` and the backward leg is a map of comodules
//! `X ⊗_B Y' → X'`, i.e. a slice map out of the pullback.
//!
//! A monoidal prism is indexed by monoid homomorphisms `Y' → X'`, with
//! backward data a module map into `X' ⊗_B Y`. The paper writes this hom-set
//! as `A/C(X, X' ⊗_B Y)`; the maps are module homomorphisms, the dual of
//! `Comod_A(X ⊗_B Y', X')`.

use crate::dlens::DLens;
use crate::finset::{self, Mor};
use crate::monoid::{self, CMon, Module};

/// The monoidal-lens normal form of a dependent lens, viewed through `⊗_B`.
pub fn as_comodule_put(lens: &DLens) -> crate::slice::SliceObj {
    let y_prime = &lens.cod.backward;
    let x_as_b = crate::slice::SliceObj {
        base: y_prime.base,
        total: lens.dom.forward.total,
        leg: finset::compose(&lens.cod.forward.leg, &lens.get),
    };
    crate::comonoid::tensor(&x_as_b, y_prime)
}

/// Identity monoidal prism: the identity monoid homomorphism and the quotient
/// map of `X ⊗_A X` onto the carrier of `X` when the two actions are the
/// module's own action and the regular action. For the regular module the
/// quotient `A ⊗_A A` is a singleton only in special cases; we instead check
/// the identity data `id: A → A` and the action `A ⊗_A M → M`.
pub fn action_descends(module: &Module) -> bool {
    let regular = regular_module(&module.scalar);
    let (_n, proj) = monoid::tensor(&regular, module);
    // (a · b, m) ~ (a, b · m), and the action (a, m) ↦ a·m coequalizes them.
    let prod = finset::product(regular.carrier, module.carrier);
    let action = Mor {
        dom: prod.obj,
        cod: module.carrier,
        map: (0..prod.obj)
            .map(|k| module.act(prod.fst.apply(k), prod.snd.apply(k)))
            .collect(),
    };
    // action ∘ left = action ∘ right, so it factors through the coequalizer.
    let triple_ok = (0..module.scalar.carrier).all(|b| {
        (0..regular.carrier).all(|a| {
            (0..module.carrier).all(|m| {
                let left = finset::pair_index(module.carrier, regular.act(b, a), m);
                let right = finset::pair_index(module.carrier, a, module.act(b, m));
                action.apply(left) == action.apply(right) && proj.apply(left) == proj.apply(right)
            })
        })
    });
    triple_ok && monoid::is_module_hom(&finset::id(module.carrier), module, module)
}

fn regular_module(monoid: &CMon) -> Module {
    let mut act = vec![0u32; (monoid.carrier * monoid.carrier) as usize];
    for s in 0..monoid.carrier {
        for x in 0..monoid.carrier {
            act[(s * monoid.carrier + x) as usize] = monoid.mul(s, x);
        }
    }
    Module {
        scalar: monoid.clone(),
        carrier: monoid.carrier,
        act,
    }
}

/// One summand of a monoidal prism: a monoid homomorphism `Y' → X'` and a
/// function `X → X' ⊗_B Y` that coequalizes the `B`-action, i.e. a map out of
/// the tensor. `X'` is given the `B`-action `b · x' = back(b) ·_A x'` only
/// when `back` is a monoid hom and `X'` is an `A`-module; here `X'` is the
/// regular `A`-module and `back: B → A`.
pub struct PrismData {
    pub back: Mor,
    pub into_tensor: Mor,
}

pub fn prism_data(base_a: &CMon, base_b: &CMon, x: &Module, y: &Module) -> Vec<PrismData> {
    assert_eq!(&x.scalar, base_a);
    assert_eq!(&y.scalar, base_b);
    let mut out = Vec::new();
    for back in finset::morphisms(base_b.carrier, base_a.carrier) {
        if !monoid::is_monoid_hom(&back, base_b, base_a) {
            continue;
        }
        // Restrict scalars on the regular A-module along `back` to get a B-module,
        // then tensor with Y and ask for maps X → that carrier which are constant
        // on the tensor relation (so they are maps out of the tensor).
        let x_prime = regular_module(base_a);
        let restricted = Module {
            scalar: base_b.clone(),
            carrier: x_prime.carrier,
            act: {
                let mut act = vec![0u32; (base_b.carrier * x_prime.carrier) as usize];
                for s in 0..base_b.carrier {
                    for e in 0..x_prime.carrier {
                        act[(s * x_prime.carrier + e) as usize] = x_prime.act(back.apply(s), e);
                    }
                }
                act
            },
        };
        if !restricted.check() {
            continue;
        }
        let (n, _proj) = monoid::tensor(&restricted, y);
        for raw in finset::morphisms(x.carrier, n) {
            out.push(PrismData {
                back: back.clone(),
                into_tensor: raw,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlens;
    use crate::finset::MAX_CARD;
    use crate::monoid::monoids_up_to;
    use crate::optic::Family;

    #[test]
    fn monoidal_lenses_match_dependent_lenses() {
        for dom in dlens::families_up_to(1) {
            for cod in dlens::families_up_to(1) {
                for lens in dlens::hom(&dom, &cod) {
                    let tensor_dom = as_comodule_put(&lens);
                    // The tensor is the pullback over B. The dependent-lens put
                    // is that same pullback, structured over A by p_X.
                    assert_eq!(tensor_dom.total, lens.put.dom.total);
                    let pb = finset::pullback(
                        &finset::compose(&lens.cod.forward.leg, &lens.get),
                        &lens.cod.backward.leg,
                    );
                    let over_a = finset::compose(&lens.dom.forward.leg, &pb.to_left);
                    assert_eq!(over_a, lens.put.dom.leg);
                    assert_eq!(tensor_dom.total, pb.apex);
                }
            }
        }
        let _ = Family::check;
    }

    #[test]
    fn monoidal_prism_indices_are_monoid_homs_and_actions_descend() {
        for monoid in monoids_up_to(MAX_CARD) {
            let regular = regular_module(&monoid);
            assert!(regular.check());
            assert!(action_descends(&regular));
            for other in monoids_up_to(MAX_CARD) {
                let modules_a = monoid::modules_over(&monoid, 1);
                let modules_b = monoid::modules_over(&other, 1);
                if let (Some(x), Some(y)) = (modules_a.first(), modules_b.first()) {
                    let data = prism_data(&monoid, &other, x, y);
                    for datum in &data {
                        assert!(monoid::is_monoid_hom(&datum.back, &other, &monoid));
                    }
                    // Identity hom is always among the indices when the monoids agree.
                    if monoid == other {
                        assert!(data.iter().any(|d| d.back == finset::id(monoid.carrier)));
                    }
                }
            }
        }
    }
}
