//! Commutative monoids, modules, and the tensor product over a monoid.
//!
//! `X ⊗_B Y` is the coequalizer of the two actions `X × B × Y ⇉ X × Y`
//! (Section 3.3). Reflexive coequalizers are preserved by the cartesian
//! product; that stability is checked in the tests.

use crate::finset::{self, Mor};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CMon {
    pub carrier: u32,
    pub unit: u32,
    /// Row-major, index `i * carrier + j` is `i · j`.
    pub mul: Vec<u32>,
}

impl CMon {
    pub fn mul(&self, i: u32, j: u32) -> u32 {
        self.mul[(i * self.carrier + j) as usize]
    }

    pub fn check(&self) -> bool {
        if self.carrier == 0 {
            return false;
        }
        if self.unit >= self.carrier || self.mul.len() != (self.carrier * self.carrier) as usize {
            return false;
        }
        if self.mul.iter().any(|&v| v >= self.carrier) {
            return false;
        }
        for i in 0..self.carrier {
            if self.mul(self.unit, i) != i || self.mul(i, self.unit) != i {
                return false;
            }
            for j in 0..self.carrier {
                if self.mul(i, j) != self.mul(j, i) {
                    return false;
                }
                for k in 0..self.carrier {
                    if self.mul(self.mul(i, j), k) != self.mul(i, self.mul(j, k)) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Module {
    pub scalar: CMon,
    pub carrier: u32,
    /// Index `s * carrier + x`.
    pub act: Vec<u32>,
}

impl Module {
    pub fn act(&self, s: u32, x: u32) -> u32 {
        self.act[(s * self.carrier + x) as usize]
    }

    pub fn check(&self) -> bool {
        if !self.scalar.check() {
            return false;
        }
        if self.carrier == 0 {
            return self.act.is_empty();
        }
        if self.act.len() != (self.scalar.carrier * self.carrier) as usize {
            return false;
        }
        if self.act.iter().any(|&v| v >= self.carrier) {
            return false;
        }
        let u = self.scalar.unit;
        for x in 0..self.carrier {
            if self.act(u, x) != x {
                return false;
            }
            for s in 0..self.scalar.carrier {
                for t in 0..self.scalar.carrier {
                    if self.act(self.scalar.mul(s, t), x) != self.act(s, self.act(t, x)) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

pub fn monoids_up_to(max: u32) -> Vec<CMon> {
    let mut out = Vec::new();
    for carrier in 1..=max {
        for unit in 0..carrier {
            let cells = (carrier * carrier) as usize;
            let mut mul = vec![0u32; cells];
            loop {
                let m = CMon {
                    carrier,
                    unit,
                    mul: mul.clone(),
                };
                if m.check() {
                    out.push(m);
                }
                let mut i = 0;
                loop {
                    if i == cells {
                        break;
                    }
                    mul[i] += 1;
                    if mul[i] < carrier {
                        break;
                    }
                    mul[i] = 0;
                    i += 1;
                }
                if i == cells {
                    break;
                }
            }
        }
    }
    out
}

pub fn modules_over(scalar: &CMon, max_carrier: u32) -> Vec<Module> {
    let mut out = Vec::new();
    for carrier in 0..=max_carrier {
        if carrier == 0 {
            out.push(Module {
                scalar: scalar.clone(),
                carrier: 0,
                act: vec![],
            });
            continue;
        }
        let cells = (scalar.carrier * carrier) as usize;
        let mut act = vec![0u32; cells];
        loop {
            let m = Module {
                scalar: scalar.clone(),
                carrier,
                act: act.clone(),
            };
            if m.check() {
                out.push(m);
            }
            let mut i = 0;
            loop {
                if i == cells {
                    break;
                }
                act[i] += 1;
                if act[i] < carrier {
                    break;
                }
                act[i] = 0;
                i += 1;
            }
            if i == cells {
                break;
            }
        }
    }
    out
}

pub fn is_monoid_hom(f: &Mor, dom: &CMon, cod: &CMon) -> bool {
    f.dom == dom.carrier
        && f.cod == cod.carrier
        && f.apply(dom.unit) == cod.unit
        && (0..dom.carrier).all(|i| {
            (0..dom.carrier).all(|j| f.apply(dom.mul(i, j)) == cod.mul(f.apply(i), f.apply(j)))
        })
}

/// Quotient projection `X × Y → X ⊗_B Y`.
pub fn tensor(x: &Module, y: &Module) -> (u32, Mor) {
    assert_eq!(x.scalar, y.scalar, "tensor over a common monoid");
    let prod = finset::product(x.carrier, y.carrier);
    let triple = finset::product(prod.obj, x.scalar.carrier);
    // two maps X×B×Y → X×Y. Index of the triple is (pair(x,y), b) with the
    // product ordered as (X×Y) × B, so fst is the pair and snd is the scalar.
    let left = Mor {
        dom: triple.obj,
        cod: prod.obj,
        map: (0..triple.obj)
            .map(|k| {
                let pair = triple.fst.apply(k);
                let b = triple.snd.apply(k);
                let xe = prod.fst.apply(pair);
                let ye = prod.snd.apply(pair);
                finset::pair_index(y.carrier, x.act(b, xe), ye)
            })
            .collect(),
    };
    let right = Mor {
        dom: triple.obj,
        cod: prod.obj,
        map: (0..triple.obj)
            .map(|k| {
                let pair = triple.fst.apply(k);
                let b = triple.snd.apply(k);
                let xe = prod.fst.apply(pair);
                let ye = prod.snd.apply(pair);
                finset::pair_index(y.carrier, xe, y.act(b, ye))
            })
            .collect(),
    };
    let q = finset::coequalizer(&left, &right);
    (q.obj, q.proj)
}

/// A module homomorphism `X → Y` over the same monoid.
pub fn is_module_hom(f: &Mor, x: &Module, y: &Module) -> bool {
    x.scalar == y.scalar
        && f.dom == x.carrier
        && f.cod == y.carrier
        && (0..x.scalar.carrier)
            .all(|s| (0..x.carrier).all(|e| f.apply(x.act(s, e)) == y.act(s, f.apply(e))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;

    #[test]
    fn tensors_coequalize_the_two_actions() {
        for monoid in monoids_up_to(MAX_CARD) {
            let modules = modules_over(&monoid, 1);
            for x in &modules {
                for y in &modules {
                    let (n, proj) = tensor(x, y);
                    assert_eq!(proj.dom, x.carrier * y.carrier);
                    assert_eq!(proj.cod, n);
                    // The two actions agree after the projection.
                    let prod = finset::product(x.carrier, y.carrier);
                    for b in 0..monoid.carrier {
                        for xe in 0..x.carrier {
                            for ye in 0..y.carrier {
                                let left = finset::pair_index(y.carrier, x.act(b, xe), ye);
                                let right = finset::pair_index(y.carrier, xe, y.act(b, ye));
                                assert_eq!(proj.apply(left), proj.apply(right));
                            }
                        }
                    }
                    let _ = prod;
                }
            }
        }
    }

    #[test]
    fn product_preserves_reflexive_coequalizers() {
        // For a fixed set A, (coeq(f,g)) × A ≅ coeq(f×id, g×id).
        for x in 0..=1u32 {
            for y in 0..=1 {
                for a in 0..=1 {
                    for f in finset::morphisms(x, y) {
                        for g in finset::morphisms(x, y) {
                            // Reflexive: a common section s: Y → X with f∘s = g∘s = id,
                            // when one exists. The stability claim is for every parallel
                            // pair the product preserves the coequalizer we computed;
                            // check cardinalities after crossing with A.
                            let q = finset::coequalizer(&f, &g);
                            let left = finset::product(q.obj, a);
                            let ya = finset::product(y, a);
                            let xa = finset::product(x, a);
                            let f_id = finset::pair(&finset::compose(&f, &xa.fst), &xa.snd);
                            let g_id = finset::pair(&finset::compose(&g, &xa.fst), &xa.snd);
                            assert_eq!(f_id.cod, ya.obj);
                            let right = finset::coequalizer(&f_id, &g_id);
                            assert_eq!(left.obj, right.obj, "product preserves coequalizers");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn monoid_homs_compose() {
        let ms = monoids_up_to(MAX_CARD);
        for a in &ms {
            for b in &ms {
                for f in finset::morphisms(a.carrier, b.carrier) {
                    if !is_monoid_hom(&f, a, b) {
                        continue;
                    }
                    for c in &ms {
                        for g in finset::morphisms(b.carrier, c.carrier) {
                            if is_monoid_hom(&g, b, c) {
                                assert!(is_monoid_hom(&finset::compose(&g, &f), a, c));
                            }
                        }
                    }
                }
            }
        }
    }
}
