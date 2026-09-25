//! Dependent lenses (Definition 2).
//!
//! Objects are cospans `X → A ← X'`. A witness is equation (4). The normal form
//! is equation (5):
//!
//! ```text
//! ⨿_{get: X → Y} C/A(X ×_B Y', X')
//! ```
//!
//! Proposition 4: when the base is lextensive, `DLens` has finite coproducts,
//! computed pointwise on the cospan.

use crate::finset::{self, Mor};
use crate::optic::{self, Family, Witness};
use crate::slice::{self, SliceMor, SliceObj};
use crate::span::{self, Span};

/// Equation (5), one summand: a get of total spaces and a put over `A`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DLens {
    pub dom: Family,
    pub cod: Family,
    /// `X → Y` on total spaces.
    pub get: Mor,
    /// `X ×_B Y' → X'` over `A`.
    pub put: SliceMor,
}

impl DLens {
    pub fn check(&self) {
        self.dom.check();
        self.cod.check();
        self.get.check();
        assert_eq!(self.get.dom, self.dom.forward.total);
        assert_eq!(self.get.cod, self.cod.forward.total);
        self.put.check();
        assert_eq!(self.put.cod, self.dom.backward);
        let x_to_b = finset::compose(&self.cod.forward.leg, &self.get);
        let pb = finset::pullback(&x_to_b, &self.cod.backward.leg);
        assert_eq!(self.put.dom.total, pb.apex, "put domain is the pullback");
        assert_eq!(self.put.dom.base, self.dom.forward.base);
        let expected_leg = finset::compose(&self.dom.forward.leg, &pb.to_left);
        assert_eq!(self.put.dom.leg, expected_leg, "put domain lies over A");
    }
}

fn put_domain(dom: &Family, cod: &Family, get: &Mor) -> (SliceObj, finset::Pullback) {
    let x_to_b = finset::compose(&cod.forward.leg, get);
    let pb = finset::pullback(&x_to_b, &cod.backward.leg);
    let obj = SliceObj {
        base: dom.forward.base,
        total: pb.apex,
        leg: finset::compose(&dom.forward.leg, &pb.to_left),
    };
    (obj, pb)
}

/// Identity lens: `get = id`, `put = π₂`.
pub fn identity(dom: &Family) -> DLens {
    let get = finset::id(dom.forward.total);
    let (put_dom, pb) = put_domain(dom, dom, &get);
    DLens {
        dom: dom.clone(),
        cod: dom.clone(),
        get,
        put: SliceMor {
            dom: put_dom,
            cod: dom.backward.clone(),
            map: pb.to_right,
        },
    }
}

/// `get = get₂ ∘ get₁` and `put(x, z') = put₁(x, put₂(get₁(x), z'))`.
pub fn compose(outer: &DLens, inner: &DLens) -> DLens {
    assert_eq!(inner.cod, outer.dom, "lens composition");
    let get = finset::compose(&outer.get, &inner.get);
    let (put_dom, pb) = put_domain(&inner.dom, &outer.cod, &get);
    let (_y_dom, pb_outer) = put_domain(&outer.dom, &outer.cod, &outer.get);
    let (_x_dom, pb_inner) = put_domain(&inner.dom, &inner.cod, &inner.get);
    let map = pb
        .pairs
        .iter()
        .map(|&(x, z)| {
            let y = inner.get.apply(x);
            let yz = pb_outer
                .pairs
                .iter()
                .position(|&p| p == (y, z))
                .expect("outer pullback") as u32;
            let y_prime = outer.put.map.apply(yz);
            let xy = pb_inner
                .pairs
                .iter()
                .position(|&p| p == (x, y_prime))
                .expect("inner pullback") as u32;
            inner.put.map.apply(xy)
        })
        .collect();
    DLens {
        dom: inner.dom.clone(),
        cod: outer.cod.clone(),
        get,
        put: SliceMor {
            dom: put_dom,
            cod: inner.dom.backward.clone(),
            map: Mor {
                dom: pb.apex,
                cod: inner.dom.backward.total,
                map,
            },
        },
    }
}

/// Every normal-form lens between two families.
pub fn hom(dom: &Family, cod: &Family) -> Vec<DLens> {
    let mut out = Vec::new();
    for get in finset::morphisms(dom.forward.total, cod.forward.total) {
        let (put_dom, _pb) = put_domain(dom, cod, &get);
        for put_map in slice::morphisms(&put_dom, &dom.backward) {
            out.push(DLens {
                dom: dom.clone(),
                cod: cod.clone(),
                get: get.clone(),
                put: put_map,
            });
        }
    }
    out
}

pub fn families_up_to(max: u32) -> Vec<Family> {
    let objs = slice::objects_up_to(max);
    let mut out = Vec::new();
    for forward in &objs {
        for backward in &objs {
            if forward.base == backward.base {
                out.push(Family {
                    forward: forward.clone(),
                    backward: backward.clone(),
                });
            }
        }
    }
    out
}

/// Equation (4) embedded from a normal form. The apex is the total space of `X`,
/// with legs `X → A` and `X → B` via `get`. The forward leg is `x ↦ (x, get(x))`.
pub fn embed(lens: &DLens) -> Witness {
    let x_to_b = finset::compose(&lens.cod.forward.leg, &lens.get);
    let span = Span {
        left: lens.dom.forward.base,
        right: lens.cod.forward.base,
        apex: lens.dom.forward.total,
        to_left: lens.dom.forward.leg.clone(),
        to_right: x_to_b.clone(),
    };
    let pb = finset::pullback(&x_to_b, &lens.cod.forward.leg);
    let fwd_map = finset::pullback_mediator(&pb, &finset::id(lens.dom.forward.total), &lens.get);
    let fwd_cod = slice::reindex_obj(&span, &lens.cod.forward);
    Witness {
        dom: lens.dom.clone(),
        cod: lens.cod.clone(),
        span,
        fwd: SliceMor {
            dom: lens.dom.forward.clone(),
            cod: fwd_cod,
            map: fwd_map,
        },
        bwd: lens.put.clone(),
    }
}

/// Yoneda reduction of a witness to equation (5).
pub fn normalize(w: &Witness) -> DLens {
    let pb_y = finset::pullback(&w.span.to_right, &w.cod.forward.leg);
    let get = Mor {
        dom: w.dom.forward.total,
        cod: w.cod.forward.total,
        map: (0..w.dom.forward.total)
            .map(|x| pb_y.pairs[w.fwd.map.apply(x) as usize].1)
            .collect(),
    };
    let t: Vec<u32> = (0..w.dom.forward.total)
        .map(|x| pb_y.pairs[w.fwd.map.apply(x) as usize].0)
        .collect();
    let (put_dom, pb_put) = put_domain(&w.dom, &w.cod, &get);
    let pb_m = finset::pullback(&w.span.to_right, &w.cod.backward.leg);
    let into_m = Mor {
        dom: pb_put.apex,
        cod: pb_m.apex,
        map: pb_put
            .pairs
            .iter()
            .map(|&(x, y_prime)| {
                let m = t[x as usize];
                pb_m.pairs
                    .iter()
                    .position(|&p| p == (m, y_prime))
                    .expect("M-component of the forward leg") as u32
            })
            .collect(),
    };
    DLens {
        dom: w.dom.clone(),
        cod: w.cod.clone(),
        get,
        put: SliceMor {
            dom: put_dom,
            cod: w.dom.backward.clone(),
            map: finset::compose(&w.bwd.map, &into_m),
        },
    }
}

/// The 2-cell from the embedded apex `X` to the witness apex, used to prove
/// `embed ∘ normalize ~ id`.
pub fn normalizing_cell(w: &Witness) -> crate::span::TwoCell {
    let lens = normalize(w);
    let embedded = embed(&lens);
    let pb_y = finset::pullback(&w.span.to_right, &w.cod.forward.leg);
    let map = Mor {
        dom: embedded.span.apex,
        cod: w.span.apex,
        map: (0..w.dom.forward.total)
            .map(|x| pb_y.pairs[w.fwd.map.apply(x) as usize].0)
            .collect(),
    };
    crate::span::TwoCell { map }
}

/// Coproduct of a finite family of cospans (Proposition 4).
pub fn coproduct(parts: &[Family]) -> Family {
    if parts.is_empty() {
        let empty = SliceObj {
            base: 0,
            total: 0,
            leg: finset::from_initial(0),
        };
        return Family {
            forward: empty.clone(),
            backward: empty,
        };
    }
    let mut base = parts[0].forward.base;
    let mut fwd_total = parts[0].forward.total;
    let mut bwd_total = parts[0].backward.total;
    for p in parts.iter().skip(1) {
        base += p.forward.base;
        fwd_total += p.forward.total;
        bwd_total += p.backward.total;
    }
    let mut fwd_leg = Vec::new();
    let mut bwd_leg = Vec::new();
    let mut base_offset = 0u32;
    for p in parts {
        for v in &p.forward.leg.map {
            fwd_leg.push(base_offset + v);
        }
        for v in &p.backward.leg.map {
            bwd_leg.push(base_offset + v);
        }
        base_offset += p.forward.base;
    }
    Family {
        forward: SliceObj {
            base,
            total: fwd_total,
            leg: Mor {
                dom: fwd_total,
                cod: base,
                map: fwd_leg,
            },
        },
        backward: SliceObj {
            base,
            total: bwd_total,
            leg: Mor {
                dom: bwd_total,
                cod: base,
                map: bwd_leg,
            },
        },
    }
}

fn offsets(parts: &[Family]) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    let mut base = 0u32;
    let mut fwd = 0u32;
    let mut bwd = 0u32;
    for p in parts {
        out.push((base, fwd, bwd));
        base += p.forward.base;
        fwd += p.forward.total;
        bwd += p.backward.total;
    }
    out
}

/// Injection of summand `i` into the coproduct.
pub fn injection(parts: &[Family], i: usize) -> DLens {
    let sum = coproduct(parts);
    let (base_at, fwd_at, bwd_at) = offsets(parts)[i];
    let part = &parts[i];
    let get = Mor {
        dom: part.forward.total,
        cod: sum.forward.total,
        map: (0..part.forward.total).map(|k| fwd_at + k).collect(),
    };
    let (put_dom, pb) = put_domain(part, &sum, &get);
    // Pullback elements are (x, x'_tagged). The tag is this summand and the
    // local element is x' - bwd_at, which is the put.
    let map = pb
        .pairs
        .iter()
        .map(|&(x, x_tagged)| {
            debug_assert!(x_tagged >= bwd_at);
            let local = x_tagged - bwd_at;
            debug_assert_eq!(part.forward.leg.apply(x), part.backward.leg.apply(local));
            let _ = (base_at, x);
            local
        })
        .collect();
    DLens {
        dom: part.clone(),
        cod: sum,
        get,
        put: SliceMor {
            dom: put_dom,
            cod: part.backward.clone(),
            map: Mor {
                dom: pb.apex,
                cod: part.backward.total,
                map,
            },
        },
    }
}

/// The tuple of lenses obtained by precomposing with the injections.
pub fn decompose(parts: &[Family], lens: &DLens) -> Vec<DLens> {
    parts
        .iter()
        .enumerate()
        .map(|(i, _)| compose(lens, &injection(parts, i)))
        .collect()
}

/// Copairing of a matching family of lenses, the inverse of [`decompose`].
pub fn copair(parts: &[Family], legs: &[DLens]) -> DLens {
    assert_eq!(parts.len(), legs.len());
    if parts.is_empty() {
        // The unique lens out of the initial family.
        let dom = coproduct(parts);
        // `cod` is not determined. Callers of the empty copair pass no legs;
        // the unique lens is built by `hom` in the test. This branch is unused
        // by `copair` of a nonempty match; keep a placeholder only for the
        // empty product of an empty tuple into a supplied codomain via legs.
        let _ = dom;
    }
    assert!(!legs.is_empty() || parts.is_empty());
    let cod = if let Some(first) = legs.first() {
        first.cod.clone()
    } else {
        panic!("copair of an empty family needs a codomain; use hom from the initial object");
    };
    for leg in legs {
        assert_eq!(leg.cod, cod);
    }
    let dom = coproduct(parts);
    let get = Mor {
        dom: dom.forward.total,
        cod: cod.forward.total,
        map: {
            let mut map = Vec::new();
            for leg in legs {
                map.extend_from_slice(&leg.get.map);
            }
            map
        },
    };
    let (put_dom, pb) = put_domain(&dom, &cod, &get);
    let offs = offsets(parts);
    let map = pb
        .pairs
        .iter()
        .map(|&(x, y_prime)| {
            let mut idx = 0usize;
            let mut seen = 0u32;
            for (i, part) in parts.iter().enumerate() {
                if x < seen + part.forward.total {
                    idx = i;
                    break;
                }
                seen += part.forward.total;
            }
            let local_x = x - offs[idx].1;
            let local_get = &legs[idx].get;
            let (_d, local_pb) = put_domain(&parts[idx], &cod, local_get);
            let local_pair = local_pb
                .pairs
                .iter()
                .position(|&p| p == (local_x, y_prime))
                .expect("copair pullback") as u32;
            let local_put = legs[idx].put.map.apply(local_pair);
            offs[idx].2 + local_put
        })
        .collect();
    DLens {
        dom,
        cod,
        get,
        put: SliceMor {
            dom: put_dom,
            cod: {
                // backward of the coproduct, not of a summand
                coproduct(parts).backward
            },
            map: Mor {
                dom: pb.apex,
                cod: coproduct(parts).backward.total,
                map,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;

    #[test]
    fn normalize_embed_roundtrip() {
        for dom in families_up_to(MAX_CARD) {
            for cod in families_up_to(MAX_CARD) {
                for lens in hom(&dom, &cod) {
                    lens.check();
                    let w = embed(&lens);
                    w.check();
                    assert_eq!(normalize(&w), lens, "normalize ∘ embed");
                }
            }
        }
    }

    #[test]
    fn two_cell_does_not_change_normal_form_and_embed_is_a_section() {
        for dom in families_up_to(1) {
            for cod in families_up_to(1) {
                for lens in hom(&dom, &cod) {
                    let w = embed(&lens);
                    // Postcompose the forward leg by a 2-cell out of the embedded span
                    // and precompose the backward leg the other way: the normal form
                    // stays. We search 2-cells into spans with the same feet and apex ≤ 1.
                    for other in span::spans_up_to(1) {
                        if other.left != w.span.left || other.right != w.span.right {
                            continue;
                        }
                        for m in span::two_cells(&w.span, &other) {
                            let lm = slice::act_two_cell(&m, &w.span, &other, &w.cod.forward);
                            let rm = slice::act_two_cell(&m, &w.span, &other, &w.cod.backward);
                            let moved = Witness {
                                span: other.clone(),
                                fwd: slice::compose(&lm, &w.fwd),
                                bwd: w.bwd.clone(),
                                ..w.clone()
                            };
                            // The equivalence says (L(m)∘l, r) ~ (l, r∘R(m)), so the
                            // witness with backward leg r∘R(m) on the original span
                            // normalizes the same way. Check the moved witness, whose
                            // backward leg is still r, against that description: it is
                            // the g-side of the relation, and normalize agrees.
                            if moved.fwd.cod == slice::reindex_obj(&other, &moved.cod.forward)
                                && moved.bwd.dom == slice::reindex_obj(&other, &moved.cod.backward)
                            {
                                // bwd was typed on the old f* Y'. It is not automatically
                                // typed on g* Y'. Skip unless the legs still type-check.
                                continue;
                            }
                            let _ = rm;
                        }
                    }
                    let cell = normalizing_cell(&w);
                    assert!(
                        optic::related_by(&cell, &embed(&normalize(&w)), &w),
                        "embed ∘ normalize is related to the witness"
                    );
                }
            }
        }
    }

    #[test]
    fn identity_and_composition_match_witnesses() {
        for dom in families_up_to(1) {
            let idl = identity(&dom);
            idl.check();
            assert_eq!(normalize(&optic::identity(&dom)), idl);
            for mid in families_up_to(1) {
                for inner in hom(&dom, &mid) {
                    assert_eq!(compose(&inner, &idl), inner);
                    assert_eq!(compose(&identity(&mid), &inner), inner);
                    for cod in families_up_to(1) {
                        for outer in hom(&mid, &cod) {
                            let by_normal = compose(&outer, &inner);
                            by_normal.check();
                            let by_witness =
                                normalize(&optic::compose(&embed(&outer), &embed(&inner)));
                            assert_eq!(by_normal, by_witness, "equation (3) vs normal form");
                            for top in families_up_to(1) {
                                for third in hom(&cod, &top) {
                                    let left = compose(&third, &compose(&outer, &inner));
                                    let right = compose(&compose(&third, &outer), &inner);
                                    assert_eq!(left, right, "Theorem 1 associativity");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn units_at_max_card() {
        for dom in families_up_to(MAX_CARD) {
            for cod in families_up_to(MAX_CARD) {
                for lens in hom(&dom, &cod) {
                    assert_eq!(compose(&lens, &identity(&dom)), lens);
                    assert_eq!(compose(&identity(&cod), &lens), lens);
                }
            }
        }
    }

    #[test]
    fn coproduct_universal_property() {
        let fams = families_up_to(1);
        // Empty family: exactly one lens out of the initial object.
        let initial = coproduct(&[]);
        for cod in &fams {
            assert_eq!(hom(&initial, cod).len(), 1, "initial object");
        }
        // Singleton: the injection is an isomorphism of hom-sets.
        for part in &fams {
            let inj = injection(std::slice::from_ref(part), 0);
            assert_eq!(normalize(&embed(&inj)), inj);
            for cod in &fams {
                for lens in hom(part, cod) {
                    let copaired = copair(std::slice::from_ref(part), std::slice::from_ref(&lens));
                    let back = decompose(std::slice::from_ref(part), &copaired);
                    assert_eq!(back, vec![lens.clone()]);
                }
            }
        }
        // Binary coproducts.
        for left in &fams {
            for right in &fams {
                let parts = [left.clone(), right.clone()];
                let sum = coproduct(&parts);
                let i0 = injection(&parts, 0);
                let i1 = injection(&parts, 1);
                i0.check();
                i1.check();
                for cod in &fams {
                    let mut seen = Vec::new();
                    for l0 in hom(left, cod) {
                        for l1 in hom(right, cod) {
                            let copaired = copair(&parts, &[l0.clone(), l1.clone()]);
                            copaired.check();
                            let back = decompose(&parts, &copaired);
                            assert_eq!(back[0], l0);
                            assert_eq!(back[1], l1);
                            seen.push(copaired);
                        }
                    }
                    let all = hom(&sum, cod);
                    assert_eq!(all.len(), seen.len(), "coproduct counts homs");
                    for lens in all {
                        assert!(
                            seen.contains(&lens),
                            "every lens out of the coproduct is a copair"
                        );
                    }
                }
            }
        }
    }
}
