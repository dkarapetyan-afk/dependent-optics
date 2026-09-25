//! Closed dependent optics (Section 3.4).
//!
//! For slices, `M ↦ M ×_B Y'` has right adjoint `Y' ⊲ –`, the dependent
//! function space. The fibre over `(a, b)` is the set of functions
//! `Y'_b → X'_a`. Equation (1) collapses to a single slice hom
//! `X → (Y' ⊲ X') ×_B Y`.

use crate::dlens::{self, DLens};
use crate::finset::{self, Mor};
use crate::optic::Family;
use crate::slice::{self, SliceMor, SliceObj};

/// An element of `Y' ⊲ X'`: a pair `(a, b)` and a table `fiber(Y', b) → fiber(X', a)`.
#[derive(Clone, Debug)]
struct Exponential {
    /// Base `A × B`, index `a * b_card + b`.
    pub obj: SliceObj,
    pub a_card: u32,
    pub b_card: u32,
    /// For each element of the total space, the function table on the ordered fibre of `Y'`.
    pub tables: Vec<Vec<u32>>,
    pub y_prime: SliceObj,
    pub x_prime: SliceObj,
}

fn fiber(leg: &Mor, p: u32) -> Vec<u32> {
    (0..leg.dom).filter(|&i| leg.apply(i) == p).collect()
}

fn exponential(y_prime: &SliceObj, x_prime: &SliceObj) -> Exponential {
    let a_card = x_prime.base;
    let b_card = y_prime.base;
    let base = finset::product(a_card, b_card).obj;
    let mut tables = Vec::new();
    let mut leg_map = Vec::new();
    for a in 0..a_card {
        for b in 0..b_card {
            let dom_f = fiber(&y_prime.leg, b);
            let cod_f = fiber(&x_prime.leg, a);
            if dom_f.is_empty() {
                // One empty function, provided the codomain fibre may be empty.
                tables.push(vec![]);
                leg_map.push(finset::pair_index(b_card, a, b));
                continue;
            }
            if cod_f.is_empty() {
                continue;
            }
            let mut cur = vec![0u32; dom_f.len()];
            loop {
                tables.push(cur.iter().map(|&i| cod_f[i as usize]).collect());
                leg_map.push(finset::pair_index(b_card, a, b));
                let mut i = 0;
                loop {
                    if i == cur.len() {
                        break;
                    }
                    cur[i] += 1;
                    if (cur[i] as usize) < cod_f.len() {
                        break;
                    }
                    cur[i] = 0;
                    i += 1;
                }
                if i == cur.len() {
                    break;
                }
            }
        }
    }
    let total = tables.len() as u32;
    Exponential {
        obj: SliceObj {
            base,
            total,
            leg: Mor {
                dom: total,
                cod: base,
                map: leg_map,
            },
        },
        a_card,
        b_card,
        tables,
        y_prime: y_prime.clone(),
        x_prime: x_prime.clone(),
    }
}

/// Evaluation `(Y' ⊲ X') ×_B Y' → X'`.
fn counit(exp: &Exponential) -> SliceMor {
    let to_b = Mor {
        dom: exp.obj.total,
        cod: exp.b_card,
        map: (0..exp.obj.total)
            .map(|e| {
                let ab = exp.obj.leg.apply(e);
                let prod = finset::product(exp.a_card, exp.b_card);
                prod.snd.apply(ab)
            })
            .collect(),
    };
    let pb = finset::pullback(&to_b, &exp.y_prime.leg);
    let map = pb
        .pairs
        .iter()
        .map(|&(e, y)| {
            let b = exp.y_prime.leg.apply(y);
            let fibre = fiber(&exp.y_prime.leg, b);
            let pos = fibre.iter().position(|&v| v == y).unwrap();
            let table = &exp.tables[e as usize];
            if fibre.is_empty() {
                // No element of an empty fibre is ever paired.
                unreachable!("empty fibre element");
            }
            table[pos]
        })
        .collect();
    let _ = pb;
    let pb = finset::pullback(&to_b, &exp.y_prime.leg);
    SliceMor {
        dom: SliceObj {
            base: exp.x_prime.base,
            total: pb.apex,
            leg: Mor {
                dom: pb.apex,
                cod: exp.x_prime.base,
                map: pb
                    .pairs
                    .iter()
                    .map(|&(e, _)| {
                        let ab = exp.obj.leg.apply(e);
                        let prod = finset::product(exp.a_card, exp.b_card);
                        prod.fst.apply(ab)
                    })
                    .collect(),
            },
        },
        cod: exp.x_prime.clone(),
        map: Mor {
            dom: pb.apex,
            cod: exp.x_prime.total,
            map,
        },
    }
}

/// A closed dependent lens is a slice map `X → (Y' ⊲ X') ×_B Y` over `A`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Closed {
    pub dom: Family,
    pub cod: Family,
    pub map: SliceMor,
}

fn closed_codomain(dom: &Family, cod: &Family) -> SliceObj {
    let exp = exponential(&cod.backward, &dom.backward);
    // (Y' ⊲ X') ×_B Y, structured over A by the A-component of the exponential.
    let to_b = Mor {
        dom: exp.obj.total,
        cod: cod.forward.base,
        map: (0..exp.obj.total)
            .map(|e| {
                let ab = exp.obj.leg.apply(e);
                finset::product(exp.a_card, exp.b_card).snd.apply(ab)
            })
            .collect(),
    };
    // Y lies over B. Pull back along Y → B.
    assert_eq!(cod.forward.base, cod.backward.base);
    let pb = finset::pullback(&to_b, &cod.forward.leg);
    SliceObj {
        base: dom.forward.base,
        total: pb.apex,
        leg: Mor {
            dom: pb.apex,
            cod: dom.forward.base,
            map: pb
                .pairs
                .iter()
                .map(|&(e, _)| {
                    let ab = exp.obj.leg.apply(e);
                    finset::product(dom.backward.base, cod.backward.base)
                        .fst
                        .apply(ab)
                })
                .collect(),
        },
    }
}

/// Send a dependent lens to the slice map that packages its get and its put.
pub fn from_lens(lens: &DLens) -> Closed {
    let exp = exponential(&lens.cod.backward, &lens.dom.backward);
    let cod_obj = closed_codomain(&lens.dom, &lens.cod);
    // For each x, the element of the pullback is (function element, get(x)),
    // where the function at b = p_Y(get(x)) sends y' in that fibre to put(x, y').
    let to_b_exp = Mor {
        dom: exp.obj.total,
        cod: lens.cod.forward.base,
        map: (0..exp.obj.total)
            .map(|e| {
                finset::product(exp.a_card, exp.b_card)
                    .snd
                    .apply(exp.obj.leg.apply(e))
            })
            .collect(),
    };
    let pb = finset::pullback(&to_b_exp, &lens.cod.forward.leg);
    let x_to_b = finset::compose(&lens.cod.forward.leg, &lens.get);
    let put_pb = finset::pullback(&x_to_b, &lens.cod.backward.leg);
    let map = (0..lens.dom.forward.total)
        .map(|x| {
            let y = lens.get.apply(x);
            let b = lens.cod.forward.leg.apply(y);
            let a = lens.dom.forward.leg.apply(x);
            let fibre = fiber(&lens.cod.backward.leg, b);
            let mut table = Vec::new();
            for &yp in &fibre {
                let pair = put_pb
                    .pairs
                    .iter()
                    .position(|&p| p == (x, yp))
                    .expect("put pair") as u32;
                table.push(lens.put.map.apply(pair));
            }
            let e = (0..exp.obj.total)
                .find(|&e| {
                    let ab = exp.obj.leg.apply(e);
                    let prod = finset::product(exp.a_card, exp.b_card);
                    prod.fst.apply(ab) == a
                        && prod.snd.apply(ab) == b
                        && exp.tables[e as usize] == table
                })
                .expect("function lives in the exponential");
            pb.pairs
                .iter()
                .position(|&p| p == (e, y))
                .expect("closed pair") as u32
        })
        .collect();
    let _ = (cod_obj, pb);
    let pb = finset::pullback(&to_b_exp, &lens.cod.forward.leg);
    Closed {
        dom: lens.dom.clone(),
        cod: lens.cod.clone(),
        map: SliceMor {
            dom: lens.dom.forward.clone(),
            cod: SliceObj {
                base: lens.dom.forward.base,
                total: pb.apex,
                leg: Mor {
                    dom: pb.apex,
                    cod: lens.dom.forward.base,
                    map: pb
                        .pairs
                        .iter()
                        .map(|&(e, _)| {
                            let ab = exp.obj.leg.apply(e);
                            finset::product(exp.a_card, exp.b_card).fst.apply(ab)
                        })
                        .collect(),
                },
            },
            map: Mor {
                dom: lens.dom.forward.total,
                cod: pb.apex,
                map,
            },
        },
    }
}

pub fn to_lens(closed: &Closed) -> DLens {
    let exp = exponential(&closed.cod.backward, &closed.dom.backward);
    let to_b_exp = Mor {
        dom: exp.obj.total,
        cod: closed.cod.forward.base,
        map: (0..exp.obj.total)
            .map(|e| {
                finset::product(exp.a_card, exp.b_card)
                    .snd
                    .apply(exp.obj.leg.apply(e))
            })
            .collect(),
    };
    let pb = finset::pullback(&to_b_exp, &closed.cod.forward.leg);
    let get = Mor {
        dom: closed.dom.forward.total,
        cod: closed.cod.forward.total,
        map: (0..closed.dom.forward.total)
            .map(|x| pb.pairs[closed.map.map.apply(x) as usize].1)
            .collect(),
    };
    let x_to_b = finset::compose(&closed.cod.forward.leg, &get);
    let put_pb = finset::pullback(&x_to_b, &closed.cod.backward.leg);
    let put_map = put_pb
        .pairs
        .iter()
        .map(|&(x, yp)| {
            let idx = closed.map.map.apply(x);
            let (e, _y) = pb.pairs[idx as usize];
            let b = closed.cod.backward.leg.apply(yp);
            let fibre = fiber(&closed.cod.backward.leg, b);
            let pos = fibre.iter().position(|&v| v == yp).unwrap();
            exp.tables[e as usize][pos]
        })
        .collect();
    let lens = DLens {
        dom: closed.dom.clone(),
        cod: closed.cod.clone(),
        get,
        put: slice::SliceMor {
            dom: slice::SliceObj {
                base: closed.dom.forward.base,
                total: put_pb.apex,
                leg: finset::compose(&closed.dom.forward.leg, &put_pb.to_left),
            },
            cod: closed.dom.backward.clone(),
            map: Mor {
                dom: put_pb.apex,
                cod: closed.dom.backward.total,
                map: put_map,
            },
        },
    };
    lens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_bijects_with_dependent_lenses() {
        for dom in dlens::families_up_to(1) {
            for cod in dlens::families_up_to(1) {
                let lenses = dlens::hom(&dom, &cod);
                let mut images = Vec::new();
                for lens in &lenses {
                    let closed = from_lens(lens);
                    closed.map.check();
                    let back = to_lens(&closed);
                    assert_eq!(&back, lens, "counit/unit of the closed adjunction");
                    images.push(closed.map.map.clone());
                }
                images.sort_by(|a, b| a.map.cmp(&b.map));
                images.dedup();
                assert_eq!(images.len(), lenses.len(), "the adjunct is injective");
            }
        }
    }

    #[test]
    fn counit_is_a_slice_map() {
        for y in slice::objects_up_to(1) {
            for x in slice::objects_up_to(1) {
                let exp = exponential(&y, &x);
                let ev = counit(&exp);
                ev.check();
            }
        }
    }
}
