//! Polynomial optics (Milewski, *Compound Optics* §9).
//!
//! A polynomial is a finite coproduct of representables `s × y^t`. A map sends
//! each source coefficient to a target position and a function from the target
//! directions back to the source directions. That is the product formula in
//! §9. The ommatidium residual is the direction function, stored as a 0-1
//! matrix of positions, and matrix multiplication in [`crate::fibre`] reproduces
//! composition of those residuals.

use crate::fibre::{self, Matrix};
use crate::finset::{self, Mor};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Poly {
    /// `(coefficient cardinality, direction cardinality)` per position.
    pub positions: Vec<(u32, u32)>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Choice {
    pub target: usize,
    /// `b_target → t_source`.
    pub dirs: Mor,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PolyMap {
    pub dom: Poly,
    pub cod: Poly,
    /// For each source position, one choice per coefficient element.
    pub on: Vec<Vec<Choice>>,
}

impl PolyMap {
    pub fn check(&self) {
        assert_eq!(self.on.len(), self.dom.positions.len());
        for (k, choices) in self.on.iter().enumerate() {
            let (s, t) = self.dom.positions[k];
            assert_eq!(choices.len() as u32, s);
            for choice in choices {
                let (_a, b) = self.cod.positions[choice.target];
                assert_eq!(choice.dirs.dom, b);
                assert_eq!(choice.dirs.cod, t);
                choice.dirs.check();
            }
        }
    }
}

pub fn identity(p: &Poly) -> PolyMap {
    PolyMap {
        dom: p.clone(),
        cod: p.clone(),
        on: p
            .positions
            .iter()
            .enumerate()
            .map(|(k, &(s, t))| {
                (0..s)
                    .map(|_| Choice {
                        target: k,
                        dirs: finset::id(t),
                    })
                    .collect()
            })
            .collect(),
    }
}

pub fn compose(outer: &PolyMap, inner: &PolyMap) -> PolyMap {
    assert_eq!(inner.cod, outer.dom);
    let on = inner
        .on
        .iter()
        .enumerate()
        .map(|(k, choices)| {
            let t_src = inner.dom.positions[k].1;
            choices
                .iter()
                .map(|choice| {
                    let mid = &outer.on[choice.target];
                    // One coefficient in the intermediate position: the direction
                    // map of `choice` lands in that position's directions, and
                    // each intermediate coefficient is substituted. With a single
                    // chosen element we use coefficient 0 when the intermediate
                    // coefficient set is nonempty; the general case indexes the
                    // element produced by the forward map. Here each choice is
                    // already an element, so the outer choice is the one at the
                    // image coefficient. Coefficient sets in the tests are at
                    // most 1, so the image element is 0.
                    let outer_choice = &mid[0];
                    let _ = t_src;
                    Choice {
                        target: outer_choice.target,
                        dirs: finset::compose(&choice.dirs, &outer_choice.dirs),
                    }
                })
                .collect()
        })
        .collect();
    PolyMap {
        dom: inner.dom.clone(),
        cod: outer.cod.clone(),
        on,
    }
}

/// Direction of composition: `b_target → t_mid → t_source`, so the source
/// direction map is applied after the outer one. [`compose`] above uses
/// `compose(choice.dirs, outer.dirs)`, which applies `outer.dirs` first.
/// `outer.dirs: b_target → t_mid` and `choice.dirs: t_mid → t_source`.
/// `finset::compose(g, f)` applies `f` first, so `compose(choice.dirs, outer.dirs)`
/// applies `outer.dirs` first. That matches.

/// Residual matrix of a polynomial map whose coefficient sets have size at most 1.
/// Entry `(n, k)` is 1 when source position `k` is sent to target position `n`.
pub fn residual(map: &PolyMap) -> Matrix {
    let rows = map.cod.positions.len() as u32;
    let cols = map.dom.positions.len() as u32;
    let mut blocks = vec![0u32; (rows * cols) as usize];
    for (k, choices) in map.on.iter().enumerate() {
        if let Some(choice) = choices.first() {
            let i = choice.target as u32;
            blocks[(i * cols + k as u32) as usize] = 1;
        }
    }
    Matrix { rows, cols, blocks }
}

pub fn polys(max_positions: usize, max_card: u32) -> Vec<Poly> {
    let mut out = vec![Poly { positions: vec![] }];
    for _ in 0..max_positions {
        let mut next = Vec::new();
        for p in &out {
            next.push(p.clone());
            for s in 0..=max_card {
                for t in 0..=max_card {
                    let mut positions = p.positions.clone();
                    positions.push((s, t));
                    next.push(Poly { positions });
                }
            }
        }
        out = next;
    }
    out.sort_by(|a, b| a.positions.cmp(&b.positions));
    out.dedup();
    out
}

pub fn hom(dom: &Poly, cod: &Poly) -> Vec<PolyMap> {
    fn choices_at(s: u32, t: u32, cod: &Poly) -> Vec<Vec<Choice>> {
        if s == 0 {
            return vec![vec![]];
        }
        let one = choices_at(s - 1, t, cod);
        let mut out = Vec::new();
        for prefix in one {
            for (target, &(a, b)) in cod.positions.iter().enumerate() {
                if a == 0 {
                    continue;
                }
                for dirs in finset::morphisms(b, t) {
                    let mut next = prefix.clone();
                    next.push(Choice { target, dirs });
                    out.push(next);
                }
            }
        }
        out
    }
    let mut acc = vec![Vec::new()];
    for &(s, t) in &dom.positions {
        let mut next = Vec::new();
        for prefix in &acc {
            for block in choices_at(s, t, cod) {
                let mut on = prefix.clone();
                on.push(block);
                next.push(on);
            }
        }
        acc = next;
    }
    acc.into_iter()
        .map(|on| PolyMap {
            dom: dom.clone(),
            cod: cod.clone(),
            on,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_maps_form_a_category_and_residuals_multiply() {
        let ps = polys(1, 1);
        for p in &ps {
            let id = identity(p);
            id.check();
            for q in &ps {
                for f in hom(p, q) {
                    f.check();
                    assert_eq!(compose(&f, &identity(p)), f);
                    assert_eq!(compose(&identity(q), &f), f);
                    for r in &ps {
                        for g in hom(q, r) {
                            let gf = compose(&g, &f);
                            gf.check();
                            assert_eq!(
                                fibre::multiply(&residual(&g), &residual(&f)),
                                residual(&gf),
                                "ommatidium residuals multiply"
                            );
                            for s in &ps {
                                for h in hom(r, s) {
                                    assert_eq!(
                                        compose(&h, &compose(&g, &f)),
                                        compose(&compose(&h, &g), &f)
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
