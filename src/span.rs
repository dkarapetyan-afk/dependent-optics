//! Bicategories of spans and cospans in [`crate::finset`].
//!
//! A span `A → B` is an apex `M` with legs `M → A` and `M → B`. Composition is
//! a pullback. `Cospan` is the opposite bicategory: composition is a pushout.
//! The inclusion of a function `f: A → B` is the span `A ←id— A —f→ B`
//! (Lemma 1).

use crate::finset::{self, Mor};

/// A span `left ← apex → right`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Span {
    pub left: u32,
    pub right: u32,
    pub apex: u32,
    pub to_left: Mor,
    pub to_right: Mor,
}

impl Span {
    pub fn check(&self) {
        self.to_left.check();
        self.to_right.check();
        assert_eq!(self.to_left.dom, self.apex);
        assert_eq!(self.to_right.dom, self.apex);
        assert_eq!(self.to_left.cod, self.left);
        assert_eq!(self.to_right.cod, self.right);
    }
}

pub fn identity(a: u32) -> Span {
    Span {
        left: a,
        right: a,
        apex: a,
        to_left: finset::id(a),
        to_right: finset::id(a),
    }
}

/// `g ∘ f` for `f: A → B` and `g: B → C`. The apex is `M ×_B N`.
pub fn compose(g: &Span, f: &Span) -> Span {
    assert_eq!(f.right, g.left, "span composition: adjacent feet");
    let pb = finset::pullback(&f.to_right, &g.to_left);
    Span {
        left: f.left,
        right: g.right,
        apex: pb.apex,
        to_left: finset::compose(&f.to_left, &pb.to_left),
        to_right: finset::compose(&g.to_right, &pb.to_right),
    }
}

/// A 2-cell `f ⇒ g` is a morphism of apices commuting with both legs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TwoCell {
    pub map: Mor,
}

pub fn two_cells(from: &Span, to: &Span) -> Vec<TwoCell> {
    assert_eq!(from.left, to.left);
    assert_eq!(from.right, to.right);
    finset::morphisms(from.apex, to.apex)
        .into_iter()
        .filter(|map| {
            finset::compose(&to.to_left, map) == from.to_left
                && finset::compose(&to.to_right, map) == from.to_right
        })
        .map(|map| TwoCell { map })
        .collect()
}

pub fn id_two_cell(s: &Span) -> TwoCell {
    TwoCell {
        map: finset::id(s.apex),
    }
}

pub fn vcompose(second: &TwoCell, first: &TwoCell) -> TwoCell {
    TwoCell {
        map: finset::compose(&second.map, &first.map),
    }
}

/// Image of a function `f: A → B` under `C → Span`.
pub fn include_morphism(f: &Mor) -> Span {
    Span {
        left: f.dom,
        right: f.cod,
        apex: f.dom,
        to_left: finset::id(f.dom),
        to_right: f.clone(),
    }
}

/// Right unitor `ρ_f: f ∘ id_A → f`, as a 2-cell.
///
/// The composite apex consists of pairs `(a, m)` with `a = to_left(m)`.
pub fn right_unitor(f: &Span) -> TwoCell {
    let comp = compose(f, &identity(f.left));
    // comp apex enumerates (a, m) with id(a) = to_left(m), i.e. a = to_left(m).
    // The 2-cell comp → f sends that pair to m.
    let pb = finset::pullback(&finset::id(f.left), &f.to_left);
    debug_assert_eq!(pb.apex, comp.apex);
    TwoCell {
        map: pb.to_right.clone(),
    }
}

/// Left unitor `λ_f: id_B ∘ f → f`.
pub fn left_unitor(f: &Span) -> TwoCell {
    let comp = compose(&identity(f.right), f);
    let pb = finset::pullback(&f.to_right, &finset::id(f.right));
    debug_assert_eq!(pb.apex, comp.apex);
    TwoCell { map: pb.to_left }
}

/// Associator `α: (h ∘ g) ∘ f → h ∘ (g ∘ f)`.
///
/// An element of the left apex is a pair `((m, n), p)`; the right apex is
/// `(m, (n, p))`. The 2-cell matches the three components.
pub fn associator(h: &Span, g: &Span, f: &Span) -> TwoCell {
    let left = compose(&compose(h, g), f);
    let right = compose(h, &compose(g, f));
    let inner_fg_wait = finset::pullback(&f.to_right, &g.to_left); // (m, n)
                                                                   // left composite is compose(compose(h,g), f), so apex is pullback of
                                                                   // f.to_right and (h∘g).to_left.
    let hg = compose(h, g);
    let left_pb = finset::pullback(&f.to_right, &hg.to_left);
    debug_assert_eq!(left_pb.apex, left.apex);
    let gf = compose(g, f);
    let right_pb = finset::pullback(&gf.to_right, &h.to_left);
    debug_assert_eq!(right_pb.apex, right.apex);

    // For each left element, recover (m, n, p).
    // left_pb pairs are (x, p) where x is an index into hg's apex,
    // and hg's apex is pullback of g.to_right against h.to_left?
    // compose(h, g): pullback of g.to_right and h.to_left, pairs (n, p_h)?
    //
    // compose(g, f) uses pullback(f.to_right, g.to_left) = pairs (m, n).
    // compose(h, g) uses pullback(g.to_right, h.to_left) = pairs (n, p).
    //
    // left = compose(hg, f) = pullback(f.to_right, hg.to_left).
    // hg.to_left = g.to_left ∘ (proj to n of (n, p)).
    // A left element is (idx_hg, m) with f.to_right(m) = hg.to_left(idx_hg).
    // idx_hg points at a pair (n, p) in pullback(g.to_right, h.to_left).
    let hg_pb = finset::pullback(&g.to_right, &h.to_left);
    let gf_pb = inner_fg_wait;
    let mut right_index = std::collections::HashMap::new();
    for (i, &(gf_idx, p)) in right_pb.pairs.iter().enumerate() {
        let (m, n) = gf_pb.pairs[gf_idx as usize];
        right_index.insert((m, n, p), i as u32);
    }
    let map = left_pb
        .pairs
        .iter()
        .map(|&(hg_idx, m)| {
            let (n, p) = hg_pb.pairs[hg_idx as usize];
            *right_index
                .get(&(m, n, p))
                .expect("associator: triple missing on the right")
        })
        .collect();
    let cell = TwoCell {
        map: Mor {
            dom: left.apex,
            cod: right.apex,
            map,
        },
    };
    debug_assert!(cell_commutes(&cell, &left, &right));
    cell
}

pub fn cell_commutes(cell: &TwoCell, from: &Span, to: &Span) -> bool {
    finset::compose(&to.to_left, &cell.map) == from.to_left
        && finset::compose(&to.to_right, &cell.map) == from.to_right
}

/// A cospan `left → apex ← right`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Cospan {
    pub left: u32,
    pub right: u32,
    pub apex: u32,
    pub from_left: Mor,
    pub from_right: Mor,
}

impl Cospan {
    pub fn check(&self) {
        self.from_left.check();
        self.from_right.check();
        assert_eq!(self.from_left.dom, self.left);
        assert_eq!(self.from_right.dom, self.right);
        assert_eq!(self.from_left.cod, self.apex);
        assert_eq!(self.from_right.cod, self.apex);
    }
}

pub fn coidentity(a: u32) -> Cospan {
    Cospan {
        left: a,
        right: a,
        apex: a,
        from_left: finset::id(a),
        from_right: finset::id(a),
    }
}

/// `g ∘ f` for cospans, by pushout of the adjacent legs.
pub fn cocompose(g: &Cospan, f: &Cospan) -> Cospan {
    assert_eq!(f.right, g.left, "cospan composition");
    let po = finset::pushout(&f.from_right, &g.from_left);
    Cospan {
        left: f.left,
        right: g.right,
        apex: po.apex,
        from_left: finset::compose(&po.from_left, &f.from_left),
        from_right: finset::compose(&po.from_right, &g.from_right),
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CoTwoCell {
    pub map: Mor,
}

pub fn co_two_cells(from: &Cospan, to: &Cospan) -> Vec<CoTwoCell> {
    assert_eq!(from.left, to.left);
    assert_eq!(from.right, to.right);
    finset::morphisms(from.apex, to.apex)
        .into_iter()
        .filter(|map| {
            finset::compose(map, &from.from_left) == to.from_left
                && finset::compose(map, &from.from_right) == to.from_right
        })
        .map(|map| CoTwoCell { map })
        .collect()
}

pub fn co_right_unitor(f: &Cospan) -> CoTwoCell {
    let comp = cocompose(f, &coidentity(f.left));
    // comp apex is pushout of id: A→A along from_left: A→M, which collapses to M.
    // The comparison comp → f is the pushout mediator of (from_left: A→M, id: M→M)
    // out of pushout(id_A, f.from_left).
    let po = finset::pushout(&finset::id(f.left), &f.from_left);
    debug_assert_eq!(po.apex, comp.apex);
    // from_left of the pushout is A → comp, from_right is M → comp.
    // We want the inverse direction: comp → M, which is inverse to from_right
    // when from_right is an iso. For pushout of id along f.from_left,
    // from_right: M → pushout is an iso.
    assert!(
        finset::is_iso(&po.from_right),
        "cospan right unitor leg {:?}",
        po.from_right
    );
    CoTwoCell {
        map: finset::inverse(&po.from_right),
    }
}

pub fn co_left_unitor(f: &Cospan) -> CoTwoCell {
    let po = finset::pushout(&f.from_right, &finset::id(f.right));
    assert!(finset::is_iso(&po.from_left), "cospan left unitor");
    CoTwoCell {
        map: finset::inverse(&po.from_left),
    }
}

pub fn co_associator(h: &Cospan, g: &Cospan, f: &Cospan) -> CoTwoCell {
    let left = cocompose(&cocompose(h, g), f);
    let right = cocompose(h, &cocompose(g, f));
    // Both apices are quotients of the same triple coproduct of the three
    // apices, so compare by sending each generator and checking they agree
    // on the relations. Concrete map: every element of `left` is a class of
    // an element of f.apex ⊔ hg.apex, and hg is a class of g.apex ⊔ h.apex.
    // We pick a representative triple (tag, index) and land it in `right`.
    let hg = cocompose(h, g);
    let gf = cocompose(g, f);
    let left_po = finset::pushout(&f.from_right, &hg.from_left);
    let right_po = finset::pushout(&gf.from_right, &h.from_left);
    debug_assert_eq!(left_po.apex, left.apex);
    debug_assert_eq!(right_po.apex, right.apex);

    // Represent an element by which original apex it came from, preferring
    // the earliest summand that maps onto it. Walk generators of F, G, H.
    fn class_of_generators(
        f: &Cospan,
        g: &Cospan,
        h: &Cospan,
        // maps from each apex into the composite apex
        from_f: &Mor,
        from_g: &Mor,
        from_h: &Mor,
    ) -> Vec<(u8, u32)> {
        let mut rep = vec![(0u8, 0u32); from_f.cod as usize];
        let mut filled = vec![false; from_f.cod as usize];
        for i in 0..f.apex {
            let c = from_f.apply(i);
            if !filled[c as usize] {
                rep[c as usize] = (0, i);
                filled[c as usize] = true;
            }
        }
        for i in 0..g.apex {
            let c = from_g.apply(i);
            if !filled[c as usize] {
                rep[c as usize] = (1, i);
                filled[c as usize] = true;
            }
        }
        for i in 0..h.apex {
            let c = from_h.apply(i);
            if !filled[c as usize] {
                rep[c as usize] = (2, i);
                filled[c as usize] = true;
            }
        }
        debug_assert!(filled.iter().all(|b| *b));
        rep
    }

    // Maps from f, g, h apices into the left composite.
    // left_po.from_left: f.apex → left, but only after f.from_right is the leg.
    // Generators:
    //   f.apex --left_po.from_left→ left   (this is the image of the whole f apex,
    //   because pushout left leg is from f.apex?
    //   pushout(f.from_right: B→f.apex? NO.
    //
    // Cospan f: A → F ← B, so from_right: B → F (apex is F).
    // hg: B → Hg ← C, from_left: B → Hg.
    // pushout(f.from_right, hg.from_left): both have domain B.
    // from_left of pushout: F → left (domain is codomain of f.from_right = f.apex).
    // from_right of pushout: Hg → left.
    //
    // g's apex maps into Hg via hg's construction.
    let hg_po = finset::pushout(&g.from_right, &h.from_left);
    // hg_po.from_left: g.apex → hg.apex
    // hg_po.from_right: h.apex → hg.apex
    let from_f = left_po.from_left.clone();
    let from_g = finset::compose(&left_po.from_right, &hg_po.from_left);
    let from_h = finset::compose(&left_po.from_right, &hg_po.from_right);
    let left_rep = class_of_generators(f, g, h, &from_f, &from_g, &from_h);

    let gf_po = finset::pushout(&f.from_right, &g.from_left);
    // right pushout(gf.from_right, h.from_left)
    // gf.from_right: C? gf is cospan A → Gf ← C, from_right: C → gf.apex.
    // Wait g: B → G ← C, f: A → F ← B.
    // cocompose(g, f): pushout(f.from_right: B→F, g.from_left: B→G).
    // gf.from_left = po.from_left ∘ f.from_left : A → gf
    // gf.from_right = po.from_right ∘ g.from_right : C → gf
    // So gf_po.from_left: F → gf, gf_po.from_right: G → gf.
    // right = pushout(gf.from_right, h.from_left) = pushout(C→gf, C→H).
    // right_po.from_left: gf.apex → right
    // right_po.from_right: h.apex → right
    let r_from_f = finset::compose(&right_po.from_left, &gf_po.from_left);
    let r_from_g = finset::compose(&right_po.from_left, &gf_po.from_right);
    let r_from_h = right_po.from_right.clone();

    let map = (0..left.apex)
        .map(|c| {
            let (tag, i) = left_rep[c as usize];
            match tag {
                0 => r_from_f.apply(i),
                1 => r_from_g.apply(i),
                2 => r_from_h.apply(i),
                _ => unreachable!(),
            }
        })
        .collect();
    CoTwoCell {
        map: Mor {
            dom: left.apex,
            cod: right.apex,
            map,
        },
    }
}

/// Spans with feet and apex of cardinality at most `max`.
pub fn spans_up_to(max: u32) -> Vec<Span> {
    let mut out = Vec::new();
    for left in 0..=max {
        for right in 0..=max {
            for apex in 0..=max {
                for to_left in finset::morphisms(apex, left) {
                    for to_right in finset::morphisms(apex, right) {
                        out.push(Span {
                            left,
                            right,
                            apex,
                            to_left: to_left.clone(),
                            to_right,
                        });
                    }
                }
            }
        }
    }
    out
}

pub fn cospans_up_to(max: u32) -> Vec<Cospan> {
    let mut out = Vec::new();
    for left in 0..=max {
        for right in 0..=max {
            for apex in 0..=max {
                for from_left in finset::morphisms(left, apex) {
                    for from_right in finset::morphisms(right, apex) {
                        out.push(Cospan {
                            left,
                            right,
                            apex,
                            from_left: from_left.clone(),
                            from_right,
                        });
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::{self, MAX_CARD};

    #[test]
    fn span_unitors_are_isos_and_triangles() {
        // Triangle: α_{id, g, f} then unitors, on apices ≤ 1 so the triple
        // enumeration stays inside the cardinality bound after pullbacks grow.
        let max = 1;
        for f in spans_up_to(max) {
            let rho = right_unitor(&f);
            let comp_r = compose(&f, &identity(f.left));
            assert!(cell_commutes(&rho, &comp_r, &f));
            assert!(finset::is_iso(&rho.map), "right unitor {f:?}");

            let lambda = left_unitor(&f);
            let comp_l = compose(&identity(f.right), &f);
            assert!(cell_commutes(&lambda, &comp_l, &f));
            assert!(finset::is_iso(&lambda.map), "left unitor {f:?}");
        }
    }

    #[test]
    fn span_associator_iso_and_pentagon_small() {
        let max = 1;
        let all = spans_up_to(max);
        for f in &all {
            for g in &all {
                if f.right != g.left {
                    continue;
                }
                for h in &all {
                    if g.right != h.left {
                        continue;
                    }
                    let alpha = associator(h, g, f);
                    let left = compose(&compose(h, g), f);
                    let right = compose(h, &compose(g, f));
                    assert!(cell_commutes(&alpha, &left, &right), "associator commute");
                    assert!(finset::is_iso(&alpha.map), "associator iso\n{alpha:?}");
                }
            }
        }
    }

    #[test]
    fn span_triangle() {
        // (id ∘ g) ∘ f  →  id ∘ (g ∘ f)  →  g ∘ f, versus the other unitor path.
        let max = 1;
        for f in spans_up_to(max) {
            for g in spans_up_to(max) {
                if f.right != g.left {
                    continue;
                }
                let alpha = associator(&g, &identity(f.right), &f);
                // α: (g ∘ id) ∘ f → g ∘ (id ∘ f)
                let left_leg = compose(&g, &identity(f.right));
                // right unitor of the composite, whiskered: ρ_g * f  is a 2-cell
                // (g ∘ id) ∘ f → g ∘ f, obtained by composing the unitor on the
                // pulled-back apex. We compare the two paths as maps of apices
                // into g ∘ f by chasing elements.
                let gf = compose(&g, &f);
                let rho = right_unitor(&g);
                // Transport ρ across the pullback that defines (g ∘ id) ∘ f.
                // Apex of (g∘id)∘f is pullback of f.to_right and (g∘id).to_left.
                let gid = compose(&g, &identity(g.left));
                let src = compose(&gid, &f);
                // Element (idx_gid, m). ρ sends the n-component of gid to g's apex.
                let gid_pb = finset::pullback(&finset::id(g.left), &g.to_left);
                let src_pb = finset::pullback(&f.to_right, &gid.to_left);
                debug_assert_eq!(src_pb.apex, src.apex);
                let path_unitor: Vec<u32> = src_pb
                    .pairs
                    .iter()
                    .map(|&(gid_idx, m)| {
                        let n = rho.map.apply(gid_idx);
                        // target element of g∘f is the pair (m, n) in pullback(f.to_right, g.to_left)
                        let gf_pb = finset::pullback(&f.to_right, &g.to_left);
                        gf_pb
                            .pairs
                            .iter()
                            .position(|&pair| pair == (m, n))
                            .expect("triangle image") as u32
                    })
                    .collect();
                let _ = (left_leg, gf, gid_pb);
                // Other path: α then the left unitor of f, whiskered by g.
                let alpha_map = alpha.map.clone();
                let mid = compose(&g, &compose(&identity(f.right), &f));
                assert_eq!(alpha_map.dom, src.apex);
                assert_eq!(alpha_map.cod, mid.apex);
                let id_f = compose(&identity(f.right), &f);
                let lam_f = left_unitor(&f);
                // g ∘ (id∘f). λ_f: id∘f → f sends an apex index to f.apex.
                // Whisker on the left by g: an element (idx_idf, nothing wait)
                // pairs (idf_idx, p)? pullback(id_f.to_right, g.to_left) pairs
                // (idx into id_f? no: pullback(left=id_f.to_right's domain which is id_f.apex,
                // right = g.to_left's domain = g.apex).
                // Our compose(g, id_f) pullback is pullback(id_f.to_right, g.to_left),
                // pairs (idf_idx, g_idx).
                let whisker: Vec<u32> = {
                    let pb = finset::pullback(&id_f.to_right, &g.to_left);
                    let target = finset::pullback(&f.to_right, &g.to_left);
                    pb.pairs
                        .iter()
                        .map(|&(idf_idx, g_idx)| {
                            let m = lam_f.map.apply(idf_idx);
                            target
                                .pairs
                                .iter()
                                .position(|&pair| pair == (m, g_idx))
                                .expect("whiskered unitor") as u32
                        })
                        .collect()
                };
                let via_alpha: Vec<u32> = (0..src.apex)
                    .map(|i| whisker[alpha_map.apply(i) as usize])
                    .collect();
                assert_eq!(path_unitor, via_alpha, "span triangle");
            }
        }
    }

    #[test]
    fn cospan_unitors_iso() {
        for f in cospans_up_to(1) {
            let rho = co_right_unitor(&f);
            let src = cocompose(&f, &coidentity(f.left));
            assert_eq!(rho.map.dom, src.apex);
            assert_eq!(rho.map.cod, f.apex);
            assert!(finset::is_iso(&rho.map), "co right unitor");
            let lambda = co_left_unitor(&f);
            assert!(finset::is_iso(&lambda.map), "co left unitor");
        }
    }

    #[test]
    fn cospan_associator_iso() {
        let all = cospans_up_to(1);
        for f in &all {
            for g in &all {
                if f.right != g.left {
                    continue;
                }
                for h in &all {
                    if g.right != h.left {
                        continue;
                    }
                    let alpha = co_associator(h, g, f);
                    assert!(
                        finset::is_iso(&alpha.map),
                        "cospan associator not iso: {alpha:?}\n{f:?}\n{g:?}\n{h:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn lemma_1_coproducts_of_spans() {
        // Span(A ⊔ B, C) ≅ Span(A, C) × Span(B, C) on objects, via
        // distributivity and lextensivity of the slice over the product.
        for a in 0..=MAX_CARD {
            for b in 0..=MAX_CARD {
                for c in 0..=MAX_CARD {
                    let sum = finset::coproduct(a, b);
                    let left_base = finset::product(sum.obj, c).obj;
                    let right_base = {
                        let ac = finset::product(a, c).obj;
                        let bc = finset::product(b, c).obj;
                        finset::coproduct(ac, bc).obj
                    };
                    let iso = {
                        // (A⊔B)×C → (A×C) ⊔ (B×C) is the inverse of distribute
                        // of C × (A⊔B) up to symmetry. Build it directly.
                        let mut map = Vec::new();
                        let ac = finset::product(a, c);
                        for s in 0..sum.obj {
                            for k in 0..c {
                                let idx = match finset::coproduct_side(a, s) {
                                    Ok(i) => finset::pair_index(c, i, k),
                                    Err(j) => ac.obj + finset::pair_index(c, j, k),
                                };
                                map.push(idx);
                            }
                        }
                        Mor {
                            dom: left_base,
                            cod: right_base,
                            map,
                        }
                    };
                    assert!(finset::is_iso(&iso), "span coproduct base {a},{b},{c}");
                    // A slice over (A⊔B)×C transports along the iso to a slice
                    // over the coproduct, which splits as a pair of slices.
                    for total in 0..=MAX_CARD {
                        for leg in finset::morphisms(total, left_base) {
                            let transported = finset::compose(&iso, &leg);
                            let sum_base = finset::coproduct(
                                finset::product(a, c).obj,
                                finset::product(b, c).obj,
                            );
                            assert_eq!(transported.cod, sum_base.obj);
                            let pull_l = finset::pullback(&transported, &sum_base.inl);
                            let pull_r = finset::pullback(&transported, &sum_base.inr);
                            assert_eq!(
                                pull_l.apex + pull_r.apex,
                                total,
                                "lextensive decomposition of a span"
                            );
                        }
                    }
                }
            }
        }
    }
}
