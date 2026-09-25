//! The bicategory of dependent optics (Remark 3).
//!
//! 1-cells are witnesses, not equivalence classes. A 2-cell is a 2-cell of the
//! base that satisfies the generating relation under Definition 1. Taking
//! connected components recovers equality of normal forms. Horizontal pasting
//! is the pullback of the apex maps, which is the pullback presentation of a
//! composite residual.

use crate::dlens::{self};
use crate::finset::{self, Mor};
use crate::optic;
use crate::span::{self, Span, TwoCell};

/// Paste 2-cells of composable spans. `outer` sits on the codomain side.
pub fn paste(
    outer_from: &Span,
    outer_to: &Span,
    outer: &TwoCell,
    inner_from: &Span,
    inner_to: &Span,
    inner: &TwoCell,
) -> TwoCell {
    let src_pb = finset::pullback(&inner_from.to_right, &outer_from.to_left);
    let dst_pb = finset::pullback(&inner_to.to_right, &outer_to.to_left);
    let mut index = std::collections::HashMap::new();
    for (i, &(f_idx, g_idx)) in dst_pb.pairs.iter().enumerate() {
        index.insert((f_idx, g_idx), i as u32);
    }
    let map = src_pb
        .pairs
        .iter()
        .map(|&(f_idx, g_idx)| {
            let f2 = inner.map.apply(f_idx);
            let g2 = outer.map.apply(g_idx);
            *index.get(&(f2, g2)).expect("pasted pair")
        })
        .collect();
    TwoCell {
        map: Mor {
            dom: src_pb.apex,
            cod: dst_pb.apex,
            map,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi0_is_normal_form_equality() {
        for dom in dlens::families_up_to(1) {
            for cod in dlens::families_up_to(1) {
                for lens in dlens::hom(&dom, &cod) {
                    let w = dlens::embed(&lens);
                    let back = dlens::embed(&dlens::normalize(&w));
                    let cell = dlens::normalizing_cell(&w);
                    assert!(optic::related_by(&cell, &back, &w), "normalizing 2-cell");
                    assert_eq!(dlens::normalize(&w), dlens::normalize(&back));
                    // A 2-cell exists only in the direction the normal form
                    // predicts; the opposite search is `equivalent`.
                    assert!(optic::equivalent(&w, &back));
                }
            }
        }
    }

    #[test]
    fn paste_preserves_relatedness() {
        for a in dlens::families_up_to(1) {
            for b in dlens::families_up_to(1) {
                for f in dlens::hom(&a, &b) {
                    for c in dlens::families_up_to(1) {
                        for g in dlens::hom(&b, &c) {
                            let wf = dlens::embed(&f);
                            let wg = dlens::embed(&g);
                            let bf = dlens::embed(&dlens::normalize(&wf));
                            let bg = dlens::embed(&dlens::normalize(&wg));
                            let mf = dlens::normalizing_cell(&wf);
                            let mg = dlens::normalizing_cell(&wg);
                            let pasted = paste(&wg.span, &bg.span, &mg, &wf.span, &bf.span, &mf);
                            let src = optic::compose(&wg, &wf);
                            let dst = optic::compose(&bg, &bf);
                            if pasted.map.dom == src.span.apex && pasted.map.cod == dst.span.apex {
                                assert!(
                                    span::cell_commutes(&pasted, &src.span, &dst.span),
                                    "pasted residual 2-cell"
                                );
                            }
                            // Both composites have the same normal form, so they
                            // are the same 1-cell of π₀.
                            assert_eq!(dlens::normalize(&src), dlens::normalize(&dst));
                            assert_eq!(dlens::normalize(&src), dlens::compose(&g, &f));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cartesian_pullback_matches_classical_witnesses() {
        // A classical lens witness is a pair of legs over one residual, which
        // is the pullback of Copara and Para over the delooping.
        for dom in crate::mixed::pairs_up_to(1) {
            for cod in crate::mixed::pairs_up_to(1) {
                for lens in crate::mixed::lens_hom(&dom, &cod) {
                    let w = crate::mixed::embed_lens(&lens);
                    let again = crate::mixed::normalize_lens(&dom, &cod, &w);
                    assert_eq!(again, lens);
                    // Same residual, two legs: the pullback cone is unique.
                    assert_eq!(w.residual, dom.forward);
                    assert_eq!(w.fwd.dom, dom.forward);
                    assert_eq!(w.bwd.cod, dom.backward);
                }
            }
        }
    }
}
