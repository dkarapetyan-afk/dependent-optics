//! Dependent prisms (Section 3.3).
//!
//! Objects are spans `X ← A → X'`. Equation (8) is the witness coend. The
//! explicit form is
//!
//! ```text
//! ⨿_{Y' → X'} A/C(X, X' ⊔_B Y)
//! ```
//!
//! with `Y' → X'` a map of total spaces. Composition is equation (3) on
//! witnesses, transported across the Yoneda isomorphism.

use crate::finset::{self, Mor};
use crate::slice::{self, CosliceMor, CosliceObj};
use crate::span::{self, Cospan};

/// `(X, X')_A` with both legs coslices `A → X` and `A → X'`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CoFamily {
    pub forward: CosliceObj,
    pub backward: CosliceObj,
}

impl CoFamily {
    pub fn check(&self) {
        self.forward.check();
        self.backward.check();
        assert_eq!(self.forward.base, self.backward.base);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PrismWitness {
    pub dom: CoFamily,
    pub cod: CoFamily,
    pub cospan: Cospan,
    pub fwd: CosliceMor,
    pub bwd: CosliceMor,
}

impl PrismWitness {
    pub fn check(&self) {
        self.dom.check();
        self.cod.check();
        self.cospan.check();
        assert_eq!(self.cospan.left, self.dom.forward.base);
        assert_eq!(self.cospan.right, self.cod.forward.base);
        self.fwd.check();
        self.bwd.check();
        assert_eq!(self.fwd.dom, self.dom.forward);
        assert_eq!(self.bwd.cod, self.dom.backward);
        assert_eq!(
            self.fwd.cod,
            slice::reindex_coslice(&self.cospan, &self.cod.forward)
        );
        assert_eq!(
            self.bwd.dom,
            slice::reindex_coslice(&self.cospan, &self.cod.backward)
        );
    }
}

pub fn witness_id(dom: &CoFamily) -> PrismWitness {
    PrismWitness {
        dom: dom.clone(),
        cod: dom.clone(),
        cospan: span::coidentity(dom.forward.base),
        fwd: slice::co_theta_id(&dom.forward),
        bwd: slice::co_theta_id_inv(&dom.backward),
    }
}

pub fn witness_compose(outer: &PrismWitness, inner: &PrismWitness) -> PrismWitness {
    assert_eq!(inner.cod, outer.dom);
    let f = &inner.cospan;
    let g = &outer.cospan;
    let f_l2 = slice::reindex_coslice_mor(f, &outer.fwd);
    let theta = slice::co_theta_comp(f, g, &outer.cod.forward);
    let fwd = slice::cocompose(&theta, &slice::cocompose(&f_l2, &inner.fwd));
    let theta_b = slice::co_theta_comp(f, g, &outer.cod.backward);
    assert!(finset::is_iso(&theta_b.map), "backward theta");
    let theta_inv = CosliceMor {
        dom: theta_b.cod.clone(),
        cod: theta_b.dom.clone(),
        map: finset::inverse(&theta_b.map),
    };
    let f_r2 = slice::reindex_coslice_mor(f, &outer.bwd);
    let bwd = slice::cocompose(&inner.bwd, &slice::cocompose(&f_r2, &theta_inv));
    PrismWitness {
        dom: inner.dom.clone(),
        cod: outer.cod.clone(),
        cospan: span::cocompose(g, f),
        fwd,
        bwd,
    }
}

/// A normal-form prism: `back: Y' → X'` and `fwd: X → X' ⊔_B Y` under `A`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DPrism {
    pub dom: CoFamily,
    pub cod: CoFamily,
    pub back: Mor,
    pub fwd: CosliceMor,
}

impl DPrism {
    pub fn check(&self) {
        self.dom.check();
        self.cod.check();
        self.back.check();
        assert_eq!(self.back.dom, self.cod.backward.total);
        assert_eq!(self.back.cod, self.dom.backward.total);
        self.fwd.check();
        assert_eq!(self.fwd.dom, self.dom.forward);
        assert_eq!(
            self.fwd.cod,
            match_codomain(&self.dom, &self.cod, &self.back)
        );
    }
}

fn embedded_cospan(dom: &CoFamily, cod: &CoFamily, back: &Mor) -> Cospan {
    Cospan {
        left: dom.forward.base,
        right: cod.forward.base,
        apex: dom.backward.total,
        from_left: dom.backward.leg.clone(),
        from_right: finset::compose(back, &cod.backward.leg),
    }
}

/// `X' ⊔_B Y`, with the same element order as reindexing along [`embedded_cospan`].
fn match_codomain(dom: &CoFamily, cod: &CoFamily, back: &Mor) -> CosliceObj {
    slice::reindex_coslice(&embedded_cospan(dom, cod, back), &cod.forward)
}

pub fn identity(dom: &CoFamily) -> DPrism {
    let back = finset::id(dom.backward.total);
    let cod_obj = match_codomain(dom, dom, &back);
    // Reindexing pushes out `A → X` against `A → X'`, and `from_left` includes `X`.
    let po = finset::pushout(&dom.forward.leg, &finset::compose(&back, &dom.backward.leg));
    DPrism {
        dom: dom.clone(),
        cod: dom.clone(),
        back,
        fwd: CosliceMor {
            dom: dom.forward.clone(),
            cod: cod_obj,
            map: po.from_left,
        },
    }
}

pub fn embed(prism: &DPrism) -> PrismWitness {
    let cospan = embedded_cospan(&prism.dom, &prism.cod, &prism.back);
    let po_bwd = finset::pushout(&prism.cod.backward.leg, &cospan.from_right);
    let bwd_map = finset::pushout_mediator(
        &po_bwd,
        &prism.back,
        &finset::id(prism.dom.backward.total),
        prism.cod.backward.total,
    );
    let bwd_dom = slice::reindex_coslice(&cospan, &prism.cod.backward);
    PrismWitness {
        dom: prism.dom.clone(),
        cod: prism.cod.clone(),
        cospan,
        fwd: prism.fwd.clone(),
        bwd: CosliceMor {
            dom: bwd_dom,
            cod: prism.dom.backward.clone(),
            map: bwd_map,
        },
    }
}

pub fn normalize(w: &PrismWitness) -> DPrism {
    let po_bwd = finset::pushout(&w.cod.backward.leg, &w.cospan.from_right);
    let back = finset::compose(&w.bwd.map, &po_bwd.from_left);
    let alpha = finset::compose(&w.bwd.map, &po_bwd.from_right);
    let target = finset::pushout(
        &w.cod.forward.leg,
        &finset::compose(&back, &w.cod.backward.leg),
    );
    let source = finset::pushout(&w.cod.forward.leg, &w.cospan.from_right);
    let y_to_target = target.from_left.clone();
    let m_to_target = finset::compose(&target.from_right, &alpha);
    let fold = finset::pushout_mediator(&source, &y_to_target, &m_to_target, w.cod.forward.total);
    let fwd_map = finset::compose(&fold, &w.fwd.map);
    let fwd_cod = match_codomain(&w.dom, &w.cod, &back);
    DPrism {
        dom: w.dom.clone(),
        cod: w.cod.clone(),
        back,
        fwd: CosliceMor {
            dom: w.dom.forward.clone(),
            cod: fwd_cod,
            map: fwd_map,
        },
    }
}

pub fn compose(outer: &DPrism, inner: &DPrism) -> DPrism {
    normalize(&witness_compose(&embed(outer), &embed(inner)))
}

pub fn families_up_to(max: u32) -> Vec<CoFamily> {
    let objs = slice::coslice_objects_up_to(max);
    let mut out = Vec::new();
    for forward in &objs {
        for backward in &objs {
            if forward.base == backward.base {
                out.push(CoFamily {
                    forward: forward.clone(),
                    backward: backward.clone(),
                });
            }
        }
    }
    out
}

pub fn hom(dom: &CoFamily, cod: &CoFamily) -> Vec<DPrism> {
    let mut out = Vec::new();
    for back in finset::morphisms(cod.backward.total, dom.backward.total) {
        let fwd_cod = match_codomain(dom, cod, &back);
        for fwd in slice::coslice_morphisms(&dom.forward, &fwd_cod) {
            out.push(DPrism {
                dom: dom.clone(),
                cod: cod.clone(),
                back: back.clone(),
                fwd,
            });
        }
    }
    out
}

/// Pointwise coproduct of spans `X ← A → X'`.
pub fn coproduct(parts: &[CoFamily]) -> CoFamily {
    if parts.is_empty() {
        // Over the initial base the forward fibre is initial and the backward
        // fibre is terminal, so there is exactly one prism out (Proposition 3
        // for the empty family).
        return CoFamily {
            forward: CosliceObj {
                base: 0,
                total: 0,
                leg: finset::from_initial(0),
            },
            backward: CosliceObj {
                base: 0,
                total: 1,
                leg: finset::from_initial(1),
            },
        };
    }
    let mut maps_f = Vec::new();
    let mut maps_b = Vec::new();
    let mut base = 0u32;
    let mut fwd = 0u32;
    let mut bwd = 0u32;
    for p in parts {
        for i in 0..p.forward.base {
            // leg after injecting the total
            maps_f.push((fwd, p.forward.leg.apply(i)));
            maps_b.push((bwd, p.backward.leg.apply(i)));
        }
        base += p.forward.base;
        fwd += p.forward.total;
        bwd += p.backward.total;
    }
    // Rebuild legs as copair: domain is the coproduct of bases, in the same order.
    let mut leg_f = Vec::new();
    let mut leg_b = Vec::new();
    let mut fwd_at = 0u32;
    let mut bwd_at = 0u32;
    for p in parts {
        for i in 0..p.forward.base {
            leg_f.push(fwd_at + p.forward.leg.apply(i));
            leg_b.push(bwd_at + p.backward.leg.apply(i));
        }
        fwd_at += p.forward.total;
        bwd_at += p.backward.total;
    }
    let _ = (maps_f, maps_b, base);
    let base: u32 = parts.iter().map(|p| p.forward.base).sum();
    let fwd: u32 = parts.iter().map(|p| p.forward.total).sum();
    let bwd: u32 = parts.iter().map(|p| p.backward.total).sum();
    CoFamily {
        forward: CosliceObj {
            base,
            total: fwd,
            leg: Mor {
                dom: base,
                cod: fwd,
                map: leg_f,
            },
        },
        backward: CosliceObj {
            base,
            total: bwd,
            leg: Mor {
                dom: base,
                cod: bwd,
                map: leg_b,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;

    #[test]
    fn roundtrip_units_and_associativity() {
        for dom in families_up_to(1) {
            let idp = identity(&dom);
            idp.check();
            assert_eq!(normalize(&witness_id(&dom)), idp);
            let embedded = embed(&idp);
            embedded.check();
            assert_eq!(normalize(&embedded), idp);
            for cod in families_up_to(1) {
                for prism in hom(&dom, &cod) {
                    prism.check();
                    let w = embed(&prism);
                    w.check();
                    assert_eq!(normalize(&w), prism, "normalize ∘ embed");
                    assert_eq!(compose(&prism, &identity(&dom)), prism);
                    assert_eq!(compose(&identity(&cod), &prism), prism);
                }
            }
        }
        for a in families_up_to(1) {
            for b in families_up_to(1) {
                for f in hom(&a, &b) {
                    for c in families_up_to(1) {
                        for g in hom(&b, &c) {
                            let gf = compose(&g, &f);
                            gf.check();
                            for d in families_up_to(1) {
                                for h in hom(&c, &d) {
                                    assert_eq!(
                                        compose(&h, &compose(&g, &f)),
                                        compose(&compose(&h, &g), &f),
                                        "prism associativity"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn roundtrip_at_max_card() {
        for dom in families_up_to(MAX_CARD) {
            for cod in families_up_to(MAX_CARD) {
                for prism in hom(&dom, &cod) {
                    assert_eq!(normalize(&embed(&prism)), prism);
                    assert_eq!(compose(&prism, &identity(&dom)), prism);
                    assert_eq!(compose(&identity(&cod), &prism), prism);
                }
            }
        }
    }

    #[test]
    fn coproduct_injections_reindex_to_summands() {
        let fams = families_up_to(1);
        let initial = coproduct(&[]);
        for cod in &fams {
            let n = hom(&initial, cod).len();
            assert_eq!(n, 1, "initial prism object");
        }
        for left in &fams {
            for right in &fams {
                let sum = coproduct(&[left.clone(), right.clone()]);
                assert_eq!(sum.forward.total, left.forward.total + right.forward.total);
                assert_eq!(
                    sum.backward.total,
                    left.backward.total + right.backward.total
                );
                for i in 0..sum.forward.base {
                    let in_left = i < left.forward.base;
                    let image = sum.forward.leg.apply(i);
                    if in_left {
                        assert!(image < left.forward.total);
                        assert_eq!(image, left.forward.leg.apply(i));
                    } else {
                        let j = i - left.forward.base;
                        assert_eq!(image, left.forward.total + right.forward.leg.apply(j));
                    }
                }
            }
        }
    }
}
