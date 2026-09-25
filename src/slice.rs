//! The indexed categories of slices and coslices.
//!
//! `C/– : Span^op → Cat` (Definition 2) sends `A` to the slice `C/A` and a span
//! `A ← M → B` to pullback along `M → B` followed by the leg `M → A`.
//! `–/C` is the dual, reindexing coslices along cospans by pushout.
//!
//! `θ_A : Id ⇒ id*` and `θ_{f,g} : f* ∘ g* ⇒ (g ∘ f)*` are the canonical
//! comparisons of those pullbacks. `act_two_cell` is the action of a span
//! 2-cell on a slice, covariant in the 2-cell as in the equivalence under
//! Definition 1.

use crate::finset::{self, Mor};
use crate::span::{self, Cospan, Span, TwoCell};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SliceObj {
    pub base: u32,
    pub total: u32,
    pub leg: Mor,
}

impl SliceObj {
    pub fn check(&self) {
        self.leg.check();
        assert_eq!(self.leg.dom, self.total);
        assert_eq!(self.leg.cod, self.base);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SliceMor {
    pub dom: SliceObj,
    pub cod: SliceObj,
    pub map: Mor,
}

impl SliceMor {
    pub fn check(&self) {
        self.dom.check();
        self.cod.check();
        self.map.check();
        assert_eq!(
            self.dom.base, self.cod.base,
            "slice morphisms stay over one base"
        );
        assert_eq!(self.map.dom, self.dom.total);
        assert_eq!(self.map.cod, self.cod.total);
        assert_eq!(
            finset::compose(&self.cod.leg, &self.map),
            self.dom.leg,
            "slice triangle"
        );
    }
}

pub fn slice_id(obj: &SliceObj) -> SliceMor {
    SliceMor {
        dom: obj.clone(),
        cod: obj.clone(),
        map: finset::id(obj.total),
    }
}

pub fn compose(g: &SliceMor, f: &SliceMor) -> SliceMor {
    assert_eq!(f.cod, g.dom, "slice composition");
    SliceMor {
        dom: f.dom.clone(),
        cod: g.cod.clone(),
        map: finset::compose(&g.map, &f.map),
    }
}

pub fn objects_up_to(max: u32) -> Vec<SliceObj> {
    let mut out = Vec::new();
    for base in 0..=max {
        for total in 0..=max {
            for leg in finset::morphisms(total, base) {
                out.push(SliceObj { base, total, leg });
            }
        }
    }
    out
}

pub fn morphisms(dom: &SliceObj, cod: &SliceObj) -> Vec<SliceMor> {
    assert_eq!(dom.base, cod.base);
    finset::morphisms(dom.total, cod.total)
        .into_iter()
        .filter(|map| finset::compose(&cod.leg, map) == dom.leg)
        .map(|map| SliceMor {
            dom: dom.clone(),
            cod: cod.clone(),
            map,
        })
        .collect()
}

/// `f* Y` for a span `f: A → B` and `Y → B`.
pub fn reindex_obj(f: &Span, y: &SliceObj) -> SliceObj {
    assert_eq!(f.right, y.base, "reindex: span codomain is the slice base");
    let pb = finset::pullback(&f.to_right, &y.leg);
    SliceObj {
        base: f.left,
        total: pb.apex,
        leg: finset::compose(&f.to_left, &pb.to_left),
    }
}

pub fn reindex_mor(f: &Span, mor: &SliceMor) -> SliceMor {
    let dom = reindex_obj(f, &mor.dom);
    let cod = reindex_obj(f, &mor.cod);
    let pb_dom = finset::pullback(&f.to_right, &mor.dom.leg);
    let pb_cod = finset::pullback(&f.to_right, &mor.cod.leg);
    let to_y = finset::compose(&mor.map, &pb_dom.to_right);
    let map = finset::pullback_mediator(&pb_cod, &pb_dom.to_left, &to_y);
    SliceMor { dom, cod, map }
}

/// `θ_A(Y): Y → id_A* Y`.
pub fn theta_id(y: &SliceObj) -> SliceMor {
    let cod = reindex_obj(&span::identity(y.base), y);
    let pb = finset::pullback(&finset::id(y.base), &y.leg);
    let map = finset::pullback_mediator(&pb, &y.leg, &finset::id(y.total));
    SliceMor {
        dom: y.clone(),
        cod,
        map,
    }
}

/// Inverse of [`theta_id`].
pub fn theta_id_inv(y: &SliceObj) -> SliceMor {
    let dom = reindex_obj(&span::identity(y.base), y);
    let pb = finset::pullback(&finset::id(y.base), &y.leg);
    SliceMor {
        dom,
        cod: y.clone(),
        map: pb.to_right,
    }
}

/// `θ_{f,g}(Z): f*(g* Z) → (g ∘ f)* Z`.
pub fn theta_comp(f: &Span, g: &Span, z: &SliceObj) -> SliceMor {
    assert_eq!(f.right, g.left, "theta_comp spans");
    assert_eq!(g.right, z.base, "theta_comp object");
    let gz = reindex_obj(g, z);
    let dom = reindex_obj(f, &gz);
    let gf = span::compose(g, f);
    let cod = reindex_obj(&gf, z);

    let gz_pb = finset::pullback(&g.to_right, &z.leg);
    let dom_pb = finset::pullback(&f.to_right, &gz.leg);
    let gf_pb = finset::pullback(&f.to_right, &g.to_left);
    let cod_pb = finset::pullback(&gf.to_right, &z.leg);

    let mut cod_index = std::collections::HashMap::new();
    for (i, &(gf_idx, z_elem)) in cod_pb.pairs.iter().enumerate() {
        let (m, n) = gf_pb.pairs[gf_idx as usize];
        cod_index.insert((m, n, z_elem), i as u32);
    }
    let map = dom_pb
        .pairs
        .iter()
        .map(|&(m, gz_idx)| {
            let (n, z_elem) = gz_pb.pairs[gz_idx as usize];
            *cod_index
                .get(&(m, n, z_elem))
                .expect("theta_comp: triple not in the composite pullback")
        })
        .collect();
    SliceMor {
        dom,
        cod,
        map: Mor {
            dom: dom_pb.apex,
            cod: cod_pb.apex,
            map,
        },
    }
}

pub fn theta_comp_inv(f: &Span, g: &Span, z: &SliceObj) -> SliceMor {
    let fwd = theta_comp(f, g, z);
    assert!(
        finset::is_iso(&fwd.map),
        "theta_comp is not invertible: {fwd:?}"
    );
    SliceMor {
        dom: fwd.cod.clone(),
        cod: fwd.dom.clone(),
        map: finset::inverse(&fwd.map),
    }
}

/// `L(m)_Y: f* Y → g* Y` for a 2-cell `m: f ⇒ g`.
pub fn act_two_cell(m: &TwoCell, from: &Span, to: &Span, y: &SliceObj) -> SliceMor {
    assert!(span::cell_commutes(m, from, to), "not a 2-cell");
    let dom = reindex_obj(from, y);
    let cod = reindex_obj(to, y);
    let pb_from = finset::pullback(&from.to_right, &y.leg);
    let pb_to = finset::pullback(&to.to_right, &y.leg);
    let to_apex = finset::compose(&m.map, &pb_from.to_left);
    let map = finset::pullback_mediator(&pb_to, &to_apex, &pb_from.to_right);
    SliceMor { dom, cod, map }
}

// ----- coslices -----

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CosliceObj {
    pub base: u32,
    pub total: u32,
    /// `base → total`.
    pub leg: Mor,
}

impl CosliceObj {
    pub fn check(&self) {
        self.leg.check();
        assert_eq!(self.leg.dom, self.base);
        assert_eq!(self.leg.cod, self.total);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CosliceMor {
    pub dom: CosliceObj,
    pub cod: CosliceObj,
    pub map: Mor,
}

impl CosliceMor {
    pub fn check(&self) {
        assert_eq!(self.dom.base, self.cod.base);
        assert_eq!(
            finset::compose(&self.map, &self.dom.leg),
            self.cod.leg,
            "coslice triangle"
        );
    }
}

pub fn coslice_id(obj: &CosliceObj) -> CosliceMor {
    CosliceMor {
        dom: obj.clone(),
        cod: obj.clone(),
        map: finset::id(obj.total),
    }
}

pub fn cocompose(g: &CosliceMor, f: &CosliceMor) -> CosliceMor {
    assert_eq!(f.cod, g.dom, "coslice composition");
    CosliceMor {
        dom: f.dom.clone(),
        cod: g.cod.clone(),
        map: finset::compose(&g.map, &f.map),
    }
}

pub fn coslice_objects_up_to(max: u32) -> Vec<CosliceObj> {
    let mut out = Vec::new();
    for base in 0..=max {
        for total in 0..=max {
            for leg in finset::morphisms(base, total) {
                out.push(CosliceObj { base, total, leg });
            }
        }
    }
    out
}

pub fn coslice_morphisms(dom: &CosliceObj, cod: &CosliceObj) -> Vec<CosliceMor> {
    assert_eq!(dom.base, cod.base);
    finset::morphisms(dom.total, cod.total)
        .into_iter()
        .filter(|map| finset::compose(map, &dom.leg) == cod.leg)
        .map(|map| CosliceMor {
            dom: dom.clone(),
            cod: cod.clone(),
            map,
        })
        .collect()
}

/// Reindex a coslice `B → Y` along a cospan `A → M ← B`.
pub fn reindex_coslice(f: &Cospan, y: &CosliceObj) -> CosliceObj {
    assert_eq!(f.right, y.base, "coslice reindex base");
    let po = finset::pushout(&y.leg, &f.from_right);
    CosliceObj {
        base: f.left,
        total: po.apex,
        leg: finset::compose(&po.from_right, &f.from_left),
    }
}

pub fn reindex_coslice_mor(f: &Cospan, mor: &CosliceMor) -> CosliceMor {
    let dom = reindex_coslice(f, &mor.dom);
    let cod = reindex_coslice(f, &mor.cod);
    let po_dom = finset::pushout(&mor.dom.leg, &f.from_right);
    let po_cod = finset::pushout(&mor.cod.leg, &f.from_right);
    // Y → pushout_cod via the morphism, and M → pushout_cod via the shared right leg.
    let from_y = finset::compose(&po_cod.from_left, &mor.map);
    let map = finset::pushout_mediator(&po_dom, &from_y, &po_cod.from_right, mor.dom.total);
    CosliceMor { dom, cod, map }
}

/// `θ` for coslices: `id* Y → Y`, the direction used by the backward leg of an identity optic.
/// The paper's `θ'_A` for `R` has the same variance as `θ_A`. For a covariant description of
/// the pushout comparison, `co_theta_id` is `Y → id* Y` and is an isomorphism.
pub fn co_theta_id(y: &CosliceObj) -> CosliceMor {
    let cod = reindex_coslice(&span::coidentity(y.base), y);
    let po = finset::pushout(&y.leg, &finset::id(y.base));
    // po.from_left: Y → pushout. That is θ: Y → id* Y.
    debug_assert_eq!(po.apex, cod.total);
    CosliceMor {
        dom: y.clone(),
        cod,
        map: po.from_left,
    }
}

pub fn co_theta_id_inv(y: &CosliceObj) -> CosliceMor {
    let fwd = co_theta_id(y);
    assert!(finset::is_iso(&fwd.map), "coslice theta_id {fwd:?}");
    CosliceMor {
        dom: fwd.cod,
        cod: fwd.dom,
        map: finset::inverse(&fwd.map),
    }
}

/// `θ_{f,g}: f*(g* Z) → (g ∘ f)* Z` for coslice reindexing.
pub fn co_theta_comp(f: &Cospan, g: &Cospan, z: &CosliceObj) -> CosliceMor {
    let gz = reindex_coslice(g, z);
    let dom = reindex_coslice(f, &gz);
    let gf = span::cocompose(g, f);
    let cod = reindex_coslice(&gf, z);
    // Both sides are pushouts of Z, the g-apex and the f-apex. Compare by
    // where the three generators land, using the same representative convention
    // as the cospan associator: prefer the Z summand, then g's apex, then f's.
    let g_po = finset::pushout(&z.leg, &g.from_right); // Z ⊔_B N
    let dom_po = finset::pushout(&gz.leg, &f.from_right);
    let gf_po_apex = finset::pushout(&f.from_right, &g.from_left);
    let cod_po = finset::pushout(&z.leg, &gf.from_right);

    // Generators into dom = f*(g* Z):
    // Z → g*Z → f*(g*Z), N → g*Z → f*(g*Z), M → f*(g*Z).
    let z_into_dom = finset::compose(&dom_po.from_left, &g_po.from_left);
    let n_into_dom = finset::compose(&dom_po.from_left, &g_po.from_right);
    let m_into_dom = dom_po.from_right.clone();

    // Generators into (g∘f)* Z.
    // gf apex is pushout of f.from_right: B→M? Cospan f: A→M←B so from_right: B→ apex_f = M.
    // cocompose(g, f) pushout(f.from_right, g.from_left), both domain B.
    // gf.from_right = po.from_right ∘ g.from_right, so N maps?
    // We need Z, N, M into the composite pushout.
    // cod_po = pushout(z.leg: B→Z, gf.from_right: C? gf.from_right has domain g.right = C).
    // z.leg domain is z.base = g.right = C. Good, both domain C?
    // z.leg: base → total, base is C. gf.from_right: C → gf.apex.
    // pushout of those has domain C. Generators: Z and gf.apex.
    // gf.apex is generated by M and N.
    let z_into_cod = cod_po.from_left.clone();
    let gf_into_cod = cod_po.from_right.clone();
    let m_into_gf = gf_po_apex.from_left.clone(); // M → gf
    let n_into_gf = gf_po_apex.from_right.clone(); // N → gf?
                                                   // pushout(f.from_right: B→M, g.from_left: B→N): from_left is M's codomain side, so from_left: M → po.
                                                   // from_right: N → po. Yes if g.from_left: B → N = g.apex.
    let m_into_cod = finset::compose(&gf_into_cod, &m_into_gf);
    let n_into_cod = finset::compose(&gf_into_cod, &n_into_gf);

    // For each class in dom, pick a generator that hits it and send that generator to cod.
    let mut map = vec![0u32; dom.total as usize];
    let mut filled = vec![false; dom.total as usize];
    for i in 0..z.total {
        let c = z_into_dom.apply(i);
        if !filled[c as usize] {
            map[c as usize] = z_into_cod.apply(i);
            filled[c as usize] = true;
        }
    }
    for i in 0..g.apex {
        let c = n_into_dom.apply(i);
        if !filled[c as usize] {
            map[c as usize] = n_into_cod.apply(i);
            filled[c as usize] = true;
        }
    }
    for i in 0..f.apex {
        let c = m_into_dom.apply(i);
        if !filled[c as usize] {
            map[c as usize] = m_into_cod.apply(i);
            filled[c as usize] = true;
        }
    }
    assert!(
        filled.iter().all(|b| *b),
        "coslice theta_comp missed a class"
    );
    CosliceMor {
        dom,
        cod,
        map: Mor {
            dom: dom_po.apex,
            cod: cod_po.apex,
            map,
        },
    }
}

pub fn co_act_two_cell(
    m: &span::CoTwoCell,
    from: &Cospan,
    to: &Cospan,
    y: &CosliceObj,
) -> CosliceMor {
    let dom = reindex_coslice(from, y);
    let cod = reindex_coslice(to, y);
    let po_from = finset::pushout(&y.leg, &from.from_right);
    let po_to = finset::pushout(&y.leg, &to.from_right);
    // Y stays, the apex of the cospan is sent along the 2-cell.
    let to_apex = finset::compose(&po_to.from_right, &m.map);
    let map = finset::pushout_mediator(&po_from, &po_to.from_left, &to_apex, y.total);
    CosliceMor { dom, cod, map }
}

/// Decompose `X → A ⊔ B` into the pair of pullbacks along the injections.
/// Returns `(X_A, X_B, iso: X → X_A ⊔ X_B)`.
pub fn lextensive_split(leg: &Mor, left: u32) -> (SliceObj, SliceObj, Mor) {
    let sum = leg.cod;
    let right = sum - left;
    let cop = finset::coproduct(left, right);
    let pb_l = finset::pullback(leg, &cop.inl);
    let pb_r = finset::pullback(leg, &cop.inr);
    let x_a = SliceObj {
        base: left,
        total: pb_l.apex,
        leg: pb_l.to_right.clone(),
    };
    let x_b = SliceObj {
        base: right,
        total: pb_r.apex,
        leg: pb_r.to_right.clone(),
    };
    // iso sends x to the side it lands on.
    let mut map = vec![0u32; leg.dom as usize];
    for (i, &(x, _)) in pb_l.pairs.iter().enumerate() {
        map[x as usize] = i as u32;
    }
    for (i, &(x, _)) in pb_r.pairs.iter().enumerate() {
        map[x as usize] = pb_l.apex + i as u32;
    }
    let iso = Mor {
        dom: leg.dom,
        cod: pb_l.apex + pb_r.apex,
        map,
    };
    (x_a, x_b, iso)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finset::MAX_CARD;
    use crate::span::{cospans_up_to, spans_up_to};

    #[test]
    fn reindex_functorial_on_morphisms() {
        for f in spans_up_to(MAX_CARD) {
            for base_objs in objects_up_to(MAX_CARD)
                .into_iter()
                .filter(|o| o.base == f.right)
            {
                let id_img = reindex_mor(&f, &slice_id(&base_objs));
                assert_eq!(id_img.map, finset::id(id_img.dom.total));
            }
            let overs: Vec<_> = objects_up_to(MAX_CARD)
                .into_iter()
                .filter(|o| o.base == f.right)
                .collect();
            for a in &overs {
                for b in &overs {
                    for mor in morphisms(a, b) {
                        for c in &overs {
                            for nu in morphisms(b, c) {
                                let left = reindex_mor(&f, &compose(&nu, &mor));
                                let right = compose(&reindex_mor(&f, &nu), &reindex_mor(&f, &mor));
                                assert_eq!(left, right, "reindex preserves composition");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn theta_id_is_iso_and_natural() {
        for y in objects_up_to(MAX_CARD) {
            let t = theta_id(&y);
            let inv = theta_id_inv(&y);
            t.check();
            inv.check();
            assert_eq!(compose(&inv, &t), slice_id(&y));
            assert_eq!(compose(&t, &inv), slice_id(&t.cod));
        }
        for base in 0..=MAX_CARD {
            let overs: Vec<_> = objects_up_to(MAX_CARD)
                .into_iter()
                .filter(|o| o.base == base)
                .collect();
            for a in &overs {
                for b in &overs {
                    for mor in morphisms(a, b) {
                        let left = compose(&theta_id(b), &mor);
                        let right =
                            compose(&reindex_mor(&span::identity(base), &mor), &theta_id(a));
                        assert_eq!(left, right, "theta_id natural");
                    }
                }
            }
        }
    }

    #[test]
    fn theta_comp_is_iso_and_natural() {
        for f in spans_up_to(1) {
            for g in spans_up_to(1) {
                if f.right != g.left {
                    continue;
                }
                for z in objects_up_to(1).into_iter().filter(|z| z.base == g.right) {
                    let t = theta_comp(&f, &g, &z);
                    t.check();
                    assert!(finset::is_iso(&t.map), "theta_comp iso");
                    let inv = theta_comp_inv(&f, &g, &z);
                    assert_eq!(compose(&inv, &t).map, finset::id(t.dom.total));
                }
            }
        }
    }

    #[test]
    fn pseudofunctor_identity_coherence() {
        // L(ρ_f) ∘ θ_{id, f} ∘ θ_A(f* Y) = id, and the left-unit twin.
        for f in spans_up_to(MAX_CARD) {
            for y in objects_up_to(MAX_CARD)
                .into_iter()
                .filter(|y| y.base == f.right)
            {
                let fy = reindex_obj(&f, &y);
                let theta_a = theta_id(&fy);
                let theta_id_f = theta_comp(&span::identity(f.left), &f, &y);
                let rho = span::right_unitor(&f);
                let comp = span::compose(&f, &span::identity(f.left));
                let act = act_two_cell(&rho, &comp, &f, &y);
                let mid = compose(&theta_id_f, &theta_a);
                let whole = compose(&act, &mid);
                assert_eq!(whole.map, finset::id(fy.total), "right identity coherence");

                let theta_b = reindex_mor(&f, &theta_id(&y));
                let theta_f_id = theta_comp(&f, &span::identity(f.right), &y);
                let lambda = span::left_unitor(&f);
                let comp_l = span::compose(&span::identity(f.right), &f);
                let act_l = act_two_cell(&lambda, &comp_l, &f, &y);
                let whole_l = compose(&act_l, &compose(&theta_f_id, &theta_b));
                assert_eq!(whole_l.map, finset::id(fy.total), "left identity coherence");
            }
        }
    }

    #[test]
    fn pseudofunctor_associativity_coherence() {
        // L(α) ∘ θ_{f, g∘? wait} as in the proof of Theorem 1:
        // L(α_{f,g,h}) ∘ θ_{(g∘f), h} ∘ (θ_{f,g}) on h*   =  θ_{f, (h∘g)} ∘ f*(θ_{g,h})
        // with α: (h ∘ g) ∘ f ⇒ h ∘ (g ∘ f).
        let all = spans_up_to(1);
        for f in &all {
            for g in &all {
                if f.right != g.left {
                    continue;
                }
                for h in &all {
                    if g.right != h.left {
                        continue;
                    }
                    for z in objects_up_to(1).into_iter().filter(|z| z.base == h.right) {
                        let gf = span::compose(g, f);
                        let hg = span::compose(h, g);
                        // left: θ_{f,g} at h* Z, then θ_{g∘f, h}, then L(α)
                        let theta_fg = theta_comp(f, g, &reindex_obj(h, &z));
                        let theta_gf_h = theta_comp(&gf, h, &z);
                        let alpha = span::associator(h, g, f);
                        let left_span = span::compose(&hg, f);
                        let right_span = span::compose(h, &span::compose(g, f));
                        let act = act_two_cell(&alpha, &left_span, &right_span, &z);
                        let left = compose(&act, &compose(&theta_gf_h, &theta_fg));

                        let theta_gh = theta_comp(g, h, &z);
                        let f_theta = reindex_mor(f, &theta_gh);
                        let theta_f_hg = theta_comp(f, &hg, &z);
                        let right = compose(&theta_f_hg, &f_theta);
                        assert_eq!(left, right, "associativity coherence");
                    }
                }
            }
        }
    }

    #[test]
    fn two_cell_action_is_functorial() {
        for f in spans_up_to(1) {
            for g in spans_up_to(1) {
                if f.left != g.left || f.right != g.right {
                    continue;
                }
                for m in span::two_cells(&f, &g) {
                    for h in spans_up_to(1) {
                        if g.left != h.left || g.right != h.right {
                            continue;
                        }
                        for n in span::two_cells(&g, &h) {
                            for y in objects_up_to(1).into_iter().filter(|y| y.base == f.right) {
                                let left = act_two_cell(&span::vcompose(&n, &m), &f, &h, &y);
                                let right = compose(
                                    &act_two_cell(&n, &g, &h, &y),
                                    &act_two_cell(&m, &f, &g, &y),
                                );
                                assert_eq!(left.map, right.map, "2-cell functoriality");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lextensive_slice_decomposition() {
        for a in 0..=MAX_CARD {
            for b in 0..=MAX_CARD {
                let sum = a + b;
                for total in 0..=MAX_CARD {
                    for leg in finset::morphisms(total, sum) {
                        let (xa, xb, iso) = lextensive_split(&leg, a);
                        assert!(finset::is_iso(&iso), "lextensive iso");
                        assert_eq!(xa.total + xb.total, total);
                        // recomposed leg matches
                        let cop = finset::coproduct(a, b);
                        let leg_a = finset::compose(&cop.inl, &xa.leg);
                        let leg_b = finset::compose(&cop.inr, &xb.leg);
                        let recomposed = finset::copair(&leg_a, &leg_b);
                        assert_eq!(finset::compose(&recomposed, &iso), leg);
                    }
                }
            }
        }
    }

    #[test]
    fn coslice_theta_id_iso() {
        for y in coslice_objects_up_to(MAX_CARD) {
            let t = co_theta_id(&y);
            t.check();
            let inv = co_theta_id_inv(&y);
            assert_eq!(cocompose(&inv, &t).map, finset::id(y.total));
        }
    }

    #[test]
    fn coslice_identity_coherence() {
        for f in cospans_up_to(MAX_CARD) {
            for y in coslice_objects_up_to(MAX_CARD)
                .into_iter()
                .filter(|y| y.base == f.right)
            {
                let fy = reindex_coslice(&f, &y);
                let theta = co_theta_comp(&span::coidentity(f.left), &f, &y);
                theta.check();
                assert!(
                    finset::is_iso(&theta.map),
                    "co theta_comp not iso {theta:?}\n{f:?}\n{y:?}"
                );
                let rho = span::co_right_unitor(&f);
                let comp = span::cocompose(&f, &span::coidentity(f.left));
                let act = co_act_two_cell(&rho, &comp, &f, &y);
                let theta_a = co_theta_id(&fy);
                let whole = cocompose(&act, &cocompose(&theta, &theta_a));
                assert_eq!(
                    whole.map,
                    finset::id(fy.total),
                    "coslice right identity coherence"
                );
            }
        }
    }

    #[test]
    fn coslice_associativity_coherence() {
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
                    for z in coslice_objects_up_to(1)
                        .into_iter()
                        .filter(|z| z.base == h.right)
                    {
                        let gf = span::cocompose(g, f);
                        let hg = span::cocompose(h, g);
                        let theta_fg = co_theta_comp(f, g, &reindex_coslice(h, &z));
                        let theta_gf_h = co_theta_comp(&gf, h, &z);
                        let alpha = span::co_associator(h, g, f);
                        let left_span = span::cocompose(&hg, f);
                        let right_span = span::cocompose(h, &span::cocompose(g, f));
                        let act = co_act_two_cell(&alpha, &left_span, &right_span, &z);
                        let left = cocompose(&act, &cocompose(&theta_gf_h, &theta_fg));
                        let theta_gh = co_theta_comp(g, h, &z);
                        let f_theta = reindex_coslice_mor(f, &theta_gh);
                        let theta_f_hg = co_theta_comp(f, &hg, &z);
                        let right = cocompose(&theta_f_hg, &f_theta);
                        assert_eq!(left.map, right.map, "coslice associativity");
                    }
                }
            }
        }
    }

    #[test]
    fn coslice_reindex_preserves_composition() {
        for f in cospans_up_to(1) {
            let overs: Vec<_> = coslice_objects_up_to(1)
                .into_iter()
                .filter(|o| o.base == f.right)
                .collect();
            for a in &overs {
                for b in &overs {
                    for mor in coslice_morphisms(a, b) {
                        for c in &overs {
                            for nu in coslice_morphisms(b, c) {
                                let left = reindex_coslice_mor(&f, &cocompose(&nu, &mor));
                                let right = cocompose(
                                    &reindex_coslice_mor(&f, &nu),
                                    &reindex_coslice_mor(&f, &mor),
                                );
                                assert_eq!(left.map, right.map);
                            }
                        }
                    }
                }
            }
        }
    }
}
