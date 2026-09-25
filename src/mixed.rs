//! Mixed optics as dependent optics on a one-object bicategory (Proposition 1).
//!
//! Delooping `(FinSet, ×)` recovers classical lenses
//! `get: X → Y`, `put: X × Y' → X'`.
//! Delooping `(FinSet, ⊔)` recovers classical prisms
//! `build: Y' → X'`, `match: X → X' ⊔ Y`.

use crate::finset::{self, Mor};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pair {
    pub forward: u32,
    pub backward: u32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lens {
    pub dom: Pair,
    pub cod: Pair,
    pub get: Mor,
    pub put: Mor,
}

impl Lens {
    pub fn check(&self) {
        assert_eq!(self.get.dom, self.dom.forward);
        assert_eq!(self.get.cod, self.cod.forward);
        let prod = finset::product(self.dom.forward, self.cod.backward);
        assert_eq!(self.put.dom, prod.obj);
        assert_eq!(self.put.cod, self.dom.backward);
        self.get.check();
        self.put.check();
    }
}

pub fn lens_id(dom: Pair) -> Lens {
    let prod = finset::product(dom.forward, dom.backward);
    Lens {
        cod: dom.clone(),
        get: finset::id(dom.forward),
        put: prod.snd,
        dom,
    }
}

pub fn lens_compose(outer: &Lens, inner: &Lens) -> Lens {
    assert_eq!(inner.cod, outer.dom);
    let get = finset::compose(&outer.get, &inner.get);
    let prod = finset::product(inner.dom.forward, outer.cod.backward);
    let outer_prod = finset::product(outer.dom.forward, outer.cod.backward);
    let inner_prod = finset::product(inner.dom.forward, inner.cod.backward);
    let map = (0..prod.obj)
        .map(|k| {
            let x = prod.fst.apply(k);
            let z = prod.snd.apply(k);
            let y = inner.get.apply(x);
            let yz = finset::pair_index(outer.cod.backward, y, z);
            debug_assert!(yz < outer_prod.obj);
            let y_prime = outer.put.apply(yz);
            let xy = finset::pair_index(inner.cod.backward, x, y_prime);
            debug_assert!(xy < inner_prod.obj);
            inner.put.apply(xy)
        })
        .collect();
    Lens {
        dom: inner.dom.clone(),
        cod: outer.cod.clone(),
        get,
        put: Mor {
            dom: prod.obj,
            cod: inner.dom.backward,
            map,
        },
    }
}

/// A witness for the cartesian action: residual `M`, `l: X → M × Y`, `r: M × Y' → X'`.
#[derive(Clone, Debug)]
pub struct LensWitness {
    pub residual: u32,
    pub fwd: Mor,
    pub bwd: Mor,
}

pub fn embed_lens(lens: &Lens) -> LensWitness {
    // Residual is X. fwd(x) = (x, get(x)). bwd is put.
    let prod = finset::product(lens.dom.forward, lens.cod.forward);
    let fwd = finset::pair(&finset::id(lens.dom.forward), &lens.get);
    debug_assert_eq!(fwd.cod, prod.obj);
    LensWitness {
        residual: lens.dom.forward,
        fwd,
        bwd: lens.put.clone(),
    }
}

pub fn normalize_lens(dom: &Pair, cod: &Pair, w: &LensWitness) -> Lens {
    let prod_y = finset::product(w.residual, cod.forward);
    assert_eq!(w.fwd.cod, prod_y.obj);
    let get = finset::compose(&prod_y.snd, &w.fwd);
    let t = finset::compose(&prod_y.fst, &w.fwd);
    let prod_put = finset::product(dom.forward, cod.backward);
    let prod_m = finset::product(w.residual, cod.backward);
    // (x, y') ↦ (t(x), y')
    let into = Mor {
        dom: prod_put.obj,
        cod: prod_m.obj,
        map: (0..prod_put.obj)
            .map(|k| {
                finset::pair_index(
                    cod.backward,
                    t.apply(prod_put.fst.apply(k)),
                    prod_put.snd.apply(k),
                )
            })
            .collect(),
    };
    Lens {
        dom: dom.clone(),
        cod: cod.clone(),
        get,
        put: finset::compose(&w.bwd, &into),
    }
}

pub fn lens_hom(dom: &Pair, cod: &Pair) -> Vec<Lens> {
    let mut out = Vec::new();
    for get in finset::morphisms(dom.forward, cod.forward) {
        let prod = finset::product(dom.forward, cod.backward);
        for put in finset::morphisms(prod.obj, dom.backward) {
            out.push(Lens {
                dom: dom.clone(),
                cod: cod.clone(),
                get: get.clone(),
                put,
            });
        }
    }
    out
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Prism {
    pub dom: Pair,
    pub cod: Pair,
    /// `Y' → X'`.
    pub build: Mor,
    /// `X → X' ⊔ Y`.
    pub match_: Mor,
}

impl Prism {
    pub fn check(&self) {
        assert_eq!(self.build.dom, self.cod.backward);
        assert_eq!(self.build.cod, self.dom.backward);
        let sum = finset::coproduct(self.dom.backward, self.cod.forward);
        assert_eq!(self.match_.dom, self.dom.forward);
        assert_eq!(self.match_.cod, sum.obj);
    }
}

pub fn prism_id(dom: Pair) -> Prism {
    let sum = finset::coproduct(dom.backward, dom.forward);
    Prism {
        cod: dom.clone(),
        build: finset::id(dom.backward),
        match_: sum.inr,
        dom,
    }
}

pub fn prism_compose(outer: &Prism, inner: &Prism) -> Prism {
    assert_eq!(inner.cod, outer.dom);
    let build = finset::compose(&inner.build, &outer.build);
    let target = finset::coproduct(inner.dom.backward, outer.cod.forward);
    // Y → Y' ⊔ Z → X' ⊔ Z, then copair with X' → X' ⊔ Z.
    let fold = finset::copair(&finset::compose(&target.inl, &inner.build), &target.inr);
    let y_to = finset::compose(&fold, &outer.match_);
    let step = finset::copair(&target.inl, &y_to);
    Prism {
        dom: inner.dom.clone(),
        cod: outer.cod.clone(),
        build,
        match_: finset::compose(&step, &inner.match_),
    }
}

pub fn pairs_up_to(max: u32) -> Vec<Pair> {
    let mut out = Vec::new();
    for forward in 0..=max {
        for backward in 0..=max {
            out.push(Pair { forward, backward });
        }
    }
    out
}

pub fn prism_hom(dom: &Pair, cod: &Pair) -> Vec<Prism> {
    let sum = finset::coproduct(dom.backward, cod.forward);
    let mut out = Vec::new();
    for build in finset::morphisms(cod.backward, dom.backward) {
        for match_ in finset::morphisms(dom.forward, sum.obj) {
            out.push(Prism {
                dom: dom.clone(),
                cod: cod.clone(),
                build: build.clone(),
                match_,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;

    #[test]
    fn classical_lens_laws() {
        for dom in pairs_up_to(MAX_CARD) {
            for cod in pairs_up_to(MAX_CARD) {
                for lens in lens_hom(&dom, &cod) {
                    lens.check();
                    assert_eq!(lens_compose(&lens, &lens_id(dom.clone())), lens);
                    assert_eq!(lens_compose(&lens_id(cod.clone()), &lens), lens);
                    assert_eq!(normalize_lens(&dom, &cod, &embed_lens(&lens)), lens);
                }
            }
        }
        for a in pairs_up_to(1) {
            for b in pairs_up_to(1) {
                for f in lens_hom(&a, &b) {
                    for c in pairs_up_to(1) {
                        for g in lens_hom(&b, &c) {
                            for d in pairs_up_to(1) {
                                for h in lens_hom(&c, &d) {
                                    assert_eq!(
                                        lens_compose(&h, &lens_compose(&g, &f)),
                                        lens_compose(&lens_compose(&h, &g), &f)
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
    fn classical_prism_laws() {
        for dom in pairs_up_to(1) {
            let idp = prism_id(dom.clone());
            idp.check();
            for cod in pairs_up_to(1) {
                for prism in prism_hom(&dom, &cod) {
                    prism.check();
                    assert_eq!(prism_compose(&prism, &prism_id(dom.clone())), prism);
                    assert_eq!(prism_compose(&prism_id(cod.clone()), &prism), prism);
                }
            }
        }
        for a in pairs_up_to(1) {
            for b in pairs_up_to(1) {
                for f in prism_hom(&a, &b) {
                    for c in pairs_up_to(1) {
                        for g in prism_hom(&b, &c) {
                            for d in pairs_up_to(1) {
                                for h in prism_hom(&c, &d) {
                                    assert_eq!(
                                        prism_compose(&h, &prism_compose(&g, &f)),
                                        prism_compose(&prism_compose(&h, &g), &f)
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
