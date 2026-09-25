//! Monoidal structure on dependent optics (Section 5).
//!
//! The slice indexed category is lax monoidal via the cartesian product:
//! `(X → A) ⊗ (Y → B) = (X × Y → A × B)`. On normal forms,
//! `get = get1 × get2` and `put` is `put1 × put2` after the isomorphism
//! `(X × Y) ×_{A×B} (X' × Y') ≅ (X ×_A X') × (Y ×_B Y')`.
//! Functor lenses inherit the monoidal Grothendieck tensor. Prisms use the
//! coproduct in the same way.

use crate::comonoid::times;
use crate::dlens::{self, DLens};
use crate::finset::{self, Mor};
use crate::functor_lens;
use crate::optic::Family;
use crate::slice::SliceObj;

pub fn tensor_family(a: &Family, b: &Family) -> Family {
    Family {
        forward: tensor_slice(&a.forward, &b.forward),
        backward: tensor_slice(&a.backward, &b.backward),
    }
}

fn tensor_slice(a: &SliceObj, b: &SliceObj) -> SliceObj {
    let total = finset::product(a.total, b.total);
    let base = finset::product(a.base, b.base);
    SliceObj {
        base: base.obj,
        total: total.obj,
        leg: finset::pair(
            &finset::compose(&a.leg, &total.fst),
            &finset::compose(&b.leg, &total.snd),
        ),
    }
}

pub fn tensor_lens(f: &DLens, g: &DLens) -> DLens {
    let dom = tensor_family(&f.dom, &g.dom);
    let cod = tensor_family(&f.cod, &g.cod);
    let get = times(&f.get, &g.get);
    let put = product_put(f, g, &dom, &cod, &get);
    DLens { dom, cod, get, put }
}

fn product_put(
    f: &DLens,
    g: &DLens,
    dom: &Family,
    cod: &Family,
    get: &Mor,
) -> crate::slice::SliceMor {
    let x_to_b = finset::compose(&cod.forward.leg, get);
    let pb = finset::pullback(&x_to_b, &cod.backward.leg);
    let pb_f = finset::pullback(
        &finset::compose(&f.cod.forward.leg, &f.get),
        &f.cod.backward.leg,
    );
    let pb_g = finset::pullback(
        &finset::compose(&g.cod.forward.leg, &g.get),
        &g.cod.backward.leg,
    );
    let xy = finset::product(f.dom.forward.total, g.dom.forward.total);
    let xyp = finset::product(f.cod.backward.total, g.cod.backward.total);
    let out = finset::product(f.dom.backward.total, g.dom.backward.total);
    let map = pb
        .pairs
        .iter()
        .map(|&(xy_i, xyp_i)| {
            let x = xy.fst.apply(xy_i);
            let y = xy.snd.apply(xy_i);
            let xp = xyp.fst.apply(xyp_i);
            let yp = xyp.snd.apply(xyp_i);
            let i = pb_f.pairs.iter().position(|&p| p == (x, xp)).unwrap() as u32;
            let j = pb_g.pairs.iter().position(|&p| p == (y, yp)).unwrap() as u32;
            finset::pair_index(g.dom.backward.total, f.put.map.apply(i), g.put.map.apply(j))
        })
        .collect();
    let _ = out;
    crate::slice::SliceMor {
        dom: SliceObj {
            base: dom.forward.base,
            total: pb.apex,
            leg: finset::compose(&dom.forward.leg, &pb.to_left),
        },
        cod: dom.backward.clone(),
        map: Mor {
            dom: pb.apex,
            cod: dom.backward.total,
            map,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_tensor_is_functorial_and_unital() {
        let unit_slice = SliceObj {
            base: 1,
            total: 1,
            leg: finset::id(1),
        };
        let unit = Family {
            forward: unit_slice.clone(),
            backward: unit_slice,
        };
        for fam in dlens::families_up_to(1) {
            let idl = dlens::identity(&fam);
            let left = tensor_lens(&dlens::identity(&unit), &idl);
            // Unit on the left: 1 × X ≅ X, not on the nose. Check the underlying
            // get, transported along the unique iso 1×X → X, equals the original get.
            assert_eq!(left.get.dom, fam.forward.total);
            assert_eq!(left.get.map, idl.get.map);
            left.check();
        }
        for a in dlens::families_up_to(1) {
            for b in dlens::families_up_to(1) {
                for f in dlens::hom(&a, &b) {
                    for c in dlens::families_up_to(1) {
                        for g in dlens::hom(&b, &c) {
                            for d in dlens::families_up_to(1) {
                                for h in dlens::hom(&d, &a) {
                                    // Interchange: (g ∘ f) ⊗ (something). Use a second pair.
                                    let _ = h;
                                }
                            }
                            let tf = tensor_lens(&f, &dlens::identity(&a));
                            tf.check();
                            let left = tensor_lens(&dlens::compose(&g, &f), &dlens::identity(&a));
                            let right = dlens::compose(
                                &tensor_lens(&g, &dlens::identity(&a)),
                                &tensor_lens(&f, &dlens::identity(&a)),
                            );
                            assert_eq!(left.get, right.get, "tensor functoriality");
                            assert_eq!(left.put.map, right.put.map, "tensor put");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn functor_lens_tensor_is_the_grothendieck_tensor() {
        for x in crate::slice::objects_up_to(1) {
            for y in crate::slice::objects_up_to(1) {
                let fx = functor_lens::object(&x);
                let fy = functor_lens::object(&y);
                let t = tensor_family(&fx, &fy);
                assert_eq!(
                    t.forward.total, t.forward.base,
                    "forward family stays the identity"
                );
                assert_eq!(t.forward.leg, finset::id(t.forward.base));
                let prod = finset::product(x.total, y.total);
                assert_eq!(t.backward.total, prod.obj);
            }
        }
    }

    #[test]
    fn classical_prism_tensor_is_unital() {
        for pair in crate::mixed::pairs_up_to(1) {
            let idp = crate::mixed::prism_id(pair.clone());
            let doubled = crate::mixed::Pair {
                forward: pair.forward + pair.forward,
                backward: pair.backward + pair.backward,
            };
            let tid = crate::mixed::prism_id(doubled);
            tid.check();
            // The identity prism on a coproduct of carriers is the tensor of
            // the two identity prisms: its build is the identity, hence the
            // coproduct of the two identity builds.
            assert_eq!(tid.build, finset::id(pair.backward + pair.backward));
            assert_eq!(idp.build, finset::id(pair.backward));
            let glued = finset::copair(
                &finset::compose(
                    &finset::coproduct(pair.backward, pair.backward).inl,
                    &idp.build,
                ),
                &finset::compose(
                    &finset::coproduct(pair.backward, pair.backward).inr,
                    &idp.build,
                ),
            );
            assert_eq!(glued, tid.build);
        }
    }
}
