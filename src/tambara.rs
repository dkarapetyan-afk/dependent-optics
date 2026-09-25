//! Tambara representations (Section 4).
//!
//! `ι` is Definition 5. Lemmas 2 and 3 are the sliding identities (10) and (11).
//! On a finite full subcategory, representables detect optics (Theorem 3): two
//! dependent lenses are equal exactly when they induce the same precomposition
//! maps. The cartesian strength on classical lenses is the non-dependent case
//! of the same structure.

use crate::dlens::{self, DLens};
use crate::finset::{self, Mor};
use crate::mixed::{self, Lens};
use crate::optic::{self, Family, Witness};
use crate::slice::{self, SliceMor};

/// `ι_A(l, r) = ⟨θ ∘ l | r ∘ θ^{-1}⟩` on the identity span.
/// `l: X0 → X1` and `r: X1' → X0'`.
pub fn iota(l: &SliceMor, r: &SliceMor) -> Witness {
    assert_eq!(l.dom.base, r.dom.base);
    let dom = Family {
        forward: l.dom.clone(),
        backward: r.cod.clone(),
    };
    let cod = Family {
        forward: l.cod.clone(),
        backward: r.dom.clone(),
    };
    Witness {
        dom,
        cod,
        span: crate::span::identity(l.dom.base),
        fwd: slice::compose(&slice::theta_id(&l.cod), l),
        bwd: slice::compose(r, &slice::theta_id_inv(&r.dom)),
    }
}

fn lens_after_domain(lens: &DLens, l: &SliceMor, r: &SliceMor) -> DLens {
    let get = finset::compose(&lens.get, &l.map);
    let x_to_b = finset::compose(&lens.cod.forward.leg, &get);
    let pb = finset::pullback(&x_to_b, &lens.cod.backward.leg);
    let old_to_b = finset::compose(&lens.cod.forward.leg, &lens.get);
    let old_pb = finset::pullback(&old_to_b, &lens.cod.backward.leg);
    let map = pb
        .pairs
        .iter()
        .map(|&(x0, yp)| {
            let x1 = l.map.apply(x0);
            let old = old_pb.pairs.iter().position(|&p| p == (x1, yp)).unwrap() as u32;
            r.map.apply(lens.put.map.apply(old))
        })
        .collect();
    DLens {
        dom: Family {
            forward: l.dom.clone(),
            backward: r.cod.clone(),
        },
        cod: lens.cod.clone(),
        get,
        put: SliceMor {
            dom: slice::SliceObj {
                base: l.dom.base,
                total: pb.apex,
                leg: finset::compose(&l.dom.leg, &pb.to_left),
            },
            cod: r.cod.clone(),
            map: Mor {
                dom: pb.apex,
                cod: r.cod.total,
                map,
            },
        },
    }
}

fn lens_before_codomain(lens: &DLens, l: &SliceMor, r: &SliceMor) -> DLens {
    // l: Y → Z, r: Z' → Y'. Composite get is l ∘ get.
    // put uses r first: (x, z') ↦ put(x, r(z')).
    let get = finset::compose(&l.map, &lens.get);
    let x_to_c = finset::compose(&l.cod.leg, &get);
    let pb = finset::pullback(&x_to_c, &r.dom.leg);
    let old_to_b = finset::compose(&lens.cod.forward.leg, &lens.get);
    let old_pb = finset::pullback(&old_to_b, &lens.cod.backward.leg);
    let map = pb
        .pairs
        .iter()
        .map(|&(x, zp)| {
            let yp = r.map.apply(zp);
            let old = old_pb.pairs.iter().position(|&p| p == (x, yp)).unwrap() as u32;
            lens.put.map.apply(old)
        })
        .collect();
    DLens {
        dom: lens.dom.clone(),
        cod: Family {
            forward: l.cod.clone(),
            backward: r.dom.clone(),
        },
        get,
        put: SliceMor {
            dom: slice::SliceObj {
                base: lens.dom.forward.base,
                total: pb.apex,
                leg: finset::compose(&lens.dom.forward.leg, &pb.to_left),
            },
            cod: lens.dom.backward.clone(),
            map: Mor {
                dom: pb.apex,
                cod: lens.dom.backward.total,
                map,
            },
        },
    }
}

/// Cartesian strength `id_C × lens`, with the backward copy of `C` kept.
pub fn strengthen(lens: &Lens, c: u32) -> Lens {
    let dom = mixed::Pair {
        forward: c * lens.dom.forward,
        backward: c * lens.dom.backward,
    };
    let cod = mixed::Pair {
        forward: c * lens.cod.forward,
        backward: c * lens.cod.backward,
    };
    let get = crate::comonoid::times(&finset::id(c), &lens.get);
    let prod = finset::product(dom.forward, cod.backward);
    let put = Mor {
        dom: prod.obj,
        cod: dom.backward,
        map: (0..prod.obj)
            .map(|k| {
                let cx = prod.fst.apply(k);
                let cy = prod.snd.apply(k);
                let x = if c == 0 {
                    0
                } else {
                    cx % lens.dom.forward.max(1)
                };
                let c_fwd = if lens.dom.forward == 0 {
                    0
                } else {
                    cx / lens.dom.forward
                };
                let y_prime = if c == 0 {
                    0
                } else {
                    cy % lens.cod.backward.max(1)
                };
                let c_bwd = if lens.cod.backward == 0 {
                    0
                } else {
                    cy / lens.cod.backward
                };
                let _ = c_fwd;
                let inner = finset::pair_index(lens.cod.backward, x, y_prime);
                // Empty factors contribute the empty function; the index arithmetic
                // above is only read when the product is nonempty.
                if prod.obj == 0 {
                    return 0;
                }
                let updated = lens.put.apply(inner);
                finset::pair_index(lens.dom.backward, c_bwd, updated)
            })
            .collect(),
    };
    Lens { dom, cod, get, put }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lemma_2_and_proposition_6() {
        for dom in dlens::families_up_to(1) {
            for cod in dlens::families_up_to(1) {
                for lens in dlens::hom(&dom, &cod) {
                    let forwards = slice::objects_up_to(1);
                    for src in forwards.iter().filter(|o| o.base == dom.forward.base) {
                        for l in slice::morphisms(src, &dom.forward) {
                            for bsrc in forwards.iter().filter(|o| o.base == dom.backward.base) {
                                for r in slice::morphisms(&dom.backward, bsrc) {
                                    let slid = iota(&l, &r);
                                    slid.check();
                                    let composed = optic::compose(&dlens::embed(&lens), &slid);
                                    composed.check();
                                    let got = dlens::normalize(&composed);
                                    let expect = lens_after_domain(&lens, &l, &r);
                                    assert_eq!(got, expect, "Lemma 2");
                                }
                            }
                        }
                    }
                }
            }
        }
        // Proposition 6: ι preserves composition.
        for base in 0..=1 {
            let objs: Vec<_> = slice::objects_up_to(1)
                .into_iter()
                .filter(|o| o.base == base)
                .collect();
            for x0 in &objs {
                for x1 in &objs {
                    for l1 in slice::morphisms(x0, x1) {
                        for x2 in &objs {
                            for l2 in slice::morphisms(x1, x2) {
                                for p0 in &objs {
                                    for p1 in &objs {
                                        for r1 in slice::morphisms(p1, p0) {
                                            for p2 in &objs {
                                                for r2 in slice::morphisms(p2, p1) {
                                                    let left = optic::compose(
                                                        &iota(&l2, &r2),
                                                        &iota(&l1, &r1),
                                                    );
                                                    let right = iota(
                                                        &slice::compose(&l2, &l1),
                                                        &slice::compose(&r1, &r2),
                                                    );
                                                    assert_eq!(
                                                        dlens::normalize(&left),
                                                        dlens::normalize(&right),
                                                        "Proposition 6"
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lemma_3_postcomposition() {
        for dom in dlens::families_up_to(1) {
            for mid in dlens::families_up_to(1) {
                for lens in dlens::hom(&dom, &mid) {
                    let objs = slice::objects_up_to(1);
                    for zf in objs.iter().filter(|o| o.base == mid.forward.base) {
                        for l in slice::morphisms(&mid.forward, zf) {
                            for zb in objs.iter().filter(|o| o.base == mid.backward.base) {
                                for r in slice::morphisms(zb, &mid.backward) {
                                    let slid = iota(&l, &r);
                                    let composed = optic::compose(&slid, &dlens::embed(&lens));
                                    let got = dlens::normalize(&composed);
                                    let expect = lens_before_codomain(&lens, &l, &r);
                                    assert_eq!(got, expect, "Lemma 3");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn representables_detect_lenses() {
        let fams = dlens::families_up_to(1);
        for dom in &fams {
            for cod in &fams {
                let homs = dlens::hom(dom, cod);
                for i in 0..homs.len() {
                    for j in 0..homs.len() {
                        let detected = fams.iter().all(|t| {
                            let out_i: Vec<_> = dlens::hom(cod, t)
                                .into_iter()
                                .map(|u| dlens::compose(&u, &homs[i]))
                                .collect();
                            let out_j: Vec<_> = dlens::hom(cod, t)
                                .into_iter()
                                .map(|u| dlens::compose(&u, &homs[j]))
                                .collect();
                            out_i == out_j
                        });
                        assert_eq!(detected, i == j, "Theorem 3 on representables");
                    }
                }
            }
        }
    }

    #[test]
    fn classical_lens_strength_is_functorial() {
        for c in 0..=1 {
            for dom in mixed::pairs_up_to(1) {
                let idl = mixed::lens_id(dom.clone());
                let strengthened = strengthen(&idl, c);
                strengthened.check();
                assert_eq!(strengthened, mixed::lens_id(strengthened.dom.clone()));
                for cod in mixed::pairs_up_to(1) {
                    for lens in mixed::lens_hom(&dom, &cod) {
                        let s = strengthen(&lens, c);
                        s.check();
                        // get/put laws
                        if s.cod.backward == 0 || s.dom.forward == 0 {
                            continue;
                        }
                        for k in 0..s.dom.forward {
                            let idx = finset::pair_index(s.cod.backward, k, s.get.apply(k));
                            assert!(idx < s.put.dom, "put index {idx}");
                            let got = s.put.apply(idx);
                            assert_eq!(got, k, "strength put-get");
                        }
                    }
                }
            }
        }
        for c in 0..=1 {
            for a in mixed::pairs_up_to(1) {
                for b in mixed::pairs_up_to(1) {
                    for f in mixed::lens_hom(&a, &b) {
                        for d in mixed::pairs_up_to(1) {
                            for g in mixed::lens_hom(&b, &d) {
                                let left = strengthen(&mixed::lens_compose(&g, &f), c);
                                let right =
                                    mixed::lens_compose(&strengthen(&g, c), &strengthen(&f, c));
                                assert_eq!(left, right, "strength preserves composition");
                            }
                        }
                    }
                }
            }
        }
    }
}
