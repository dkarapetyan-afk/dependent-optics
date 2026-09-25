//! Skeletal finite sets.
//!
//! Objects are cardinalities. A morphism `n → m` is a function, stored as a
//! table of length `n` with values in `0..m`. Product indices use
//! `i * rhs + j`. Coproduct indices lay the left summand in `0..left` and the
//! right summand after it.
//!
//! Cardinality bound used by the exhaustive tests: [`MAX_CARD`].

/// Largest cardinality iterated by the exhaustive tests.
pub const MAX_CARD: u32 = 2;

/// An object of the skeletal category of finite sets: a cardinality.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Obj(pub u32);

/// A function between finite sets.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Mor {
    pub dom: u32,
    pub cod: u32,
    pub map: Vec<u32>,
}

impl Mor {
    pub fn apply(&self, i: u32) -> u32 {
        debug_assert!(i < self.dom, "index {i} out of domain {}", self.dom);
        self.map[i as usize]
    }

    pub fn check(&self) {
        assert_eq!(self.map.len() as u32, self.dom, "table length");
        if self.dom > 0 {
            assert!(
                self.cod > 0 || self.dom == 0,
                "nonempty map into the empty set"
            );
        }
        for &v in &self.map {
            assert!(v < self.cod, "value {v} not in 0..{}", self.cod);
        }
    }
}

pub fn id(a: u32) -> Mor {
    Mor {
        dom: a,
        cod: a,
        map: (0..a).collect(),
    }
}

/// `g ∘ f`, applying `f` first.
pub fn compose(g: &Mor, f: &Mor) -> Mor {
    assert_eq!(f.cod, g.dom, "compose: codomain of f must be domain of g");
    Mor {
        dom: f.dom,
        cod: g.cod,
        map: f.map.iter().map(|&i| g.apply(i)).collect(),
    }
}

pub fn is_iso(f: &Mor) -> bool {
    if f.dom != f.cod {
        return false;
    }
    let mut seen = vec![false; f.cod as usize];
    for &v in &f.map {
        if seen[v as usize] {
            return false;
        }
        seen[v as usize] = true;
    }
    true
}

pub fn inverse(f: &Mor) -> Mor {
    assert!(is_iso(f), "inverse of a non-bijection");
    let mut map = vec![0; f.cod as usize];
    for (i, &v) in f.map.iter().enumerate() {
        map[v as usize] = i as u32;
    }
    Mor {
        dom: f.cod,
        cod: f.dom,
        map,
    }
}

/// Every function `dom → cod`, in lexicographic order of the table.
pub fn morphisms(dom: u32, cod: u32) -> Vec<Mor> {
    if dom == 0 {
        return vec![Mor {
            dom: 0,
            cod,
            map: vec![],
        }];
    }
    if cod == 0 {
        return vec![];
    }
    let mut out = Vec::new();
    let mut cur = vec![0u32; dom as usize];
    loop {
        out.push(Mor {
            dom,
            cod,
            map: cur.clone(),
        });
        let mut i = 0;
        loop {
            if i == dom as usize {
                return out;
            }
            cur[i] += 1;
            if cur[i] < cod {
                break;
            }
            cur[i] = 0;
            i += 1;
        }
    }
}

pub fn objects_up_to(max: u32) -> impl Iterator<Item = u32> {
    0..=max
}

// ----- limits -----

pub fn terminal() -> u32 {
    1
}

pub fn to_terminal(dom: u32) -> Mor {
    Mor {
        dom,
        cod: 1,
        map: vec![0; dom as usize],
    }
}

/// `A × B`, with `index(i, j) = i * b + j`.
pub struct Product {
    pub obj: u32,
    pub fst: Mor,
    pub snd: Mor,
    pub left: u32,
    pub right: u32,
}

pub fn product(a: u32, b: u32) -> Product {
    let obj = a.saturating_mul(b);
    let fst = Mor {
        dom: obj,
        cod: a,
        map: (0..obj).map(|k| if b == 0 { 0 } else { k / b }).collect(),
    };
    let snd = Mor {
        dom: obj,
        cod: b,
        map: (0..obj).map(|k| if b == 0 { 0 } else { k % b }).collect(),
    };
    Product {
        obj,
        fst,
        snd,
        left: a,
        right: b,
    }
}

pub fn pair_index(b_card: u32, i: u32, j: u32) -> u32 {
    i * b_card + j
}

/// Mediator of `f: Z → A` and `g: Z → B` into `A × B`.
pub fn pair(f: &Mor, g: &Mor) -> Mor {
    assert_eq!(f.dom, g.dom, "pair: common domain");
    let b = g.cod;
    Mor {
        dom: f.dom,
        cod: f.cod.saturating_mul(b),
        map: (0..f.dom)
            .map(|z| pair_index(b, f.apply(z), g.apply(z)))
            .collect(),
    }
}

pub struct Pullback {
    pub apex: u32,
    /// Apex → domain of `left_leg`.
    pub to_left: Mor,
    /// Apex → domain of `right_leg`.
    pub to_right: Mor,
    pub pairs: Vec<(u32, u32)>,
}

/// Pullback of `left_leg: X → Z` and `right_leg: Y → Z`.
pub fn pullback(left_leg: &Mor, right_leg: &Mor) -> Pullback {
    assert_eq!(left_leg.cod, right_leg.cod, "pullback: common codomain");
    let mut pairs = Vec::new();
    for x in 0..left_leg.dom {
        for y in 0..right_leg.dom {
            if left_leg.apply(x) == right_leg.apply(y) {
                pairs.push((x, y));
            }
        }
    }
    let apex = pairs.len() as u32;
    Pullback {
        apex,
        to_left: Mor {
            dom: apex,
            cod: left_leg.dom,
            map: pairs.iter().map(|p| p.0).collect(),
        },
        to_right: Mor {
            dom: apex,
            cod: right_leg.dom,
            map: pairs.iter().map(|p| p.1).collect(),
        },
        pairs,
    }
}

/// Inclusion of the pullback into the product, for the chosen index order.
pub fn pullback_into_product(pb: &Pullback, right_card: u32) -> Mor {
    Mor {
        dom: pb.apex,
        cod: pb.to_left.cod.saturating_mul(right_card),
        map: pb
            .pairs
            .iter()
            .map(|&(x, y)| pair_index(right_card, x, y))
            .collect(),
    }
}

/// Mediator into a pullback. `p` lands in the left object, `q` in the right,
/// and the two composites into the cospan codomain agree.
pub fn pullback_mediator(pb: &Pullback, p: &Mor, q: &Mor) -> Mor {
    assert_eq!(p.dom, q.dom, "pullback mediator: common domain");
    let mut index_of = std::collections::HashMap::with_capacity(pb.pairs.len());
    for (i, &(x, y)) in pb.pairs.iter().enumerate() {
        index_of.insert((x, y), i as u32);
    }
    Mor {
        dom: p.dom,
        cod: pb.apex,
        map: (0..p.dom)
            .map(|w| {
                let key = (p.apply(w), q.apply(w));
                *index_of.get(&key).expect("pair is not in the pullback")
            })
            .collect(),
    }
}

pub struct Equalizer {
    pub obj: u32,
    pub incl: Mor,
    pub elements: Vec<u32>,
}

/// Equalizer of a parallel pair `f, g: X → Y`.
pub fn equalizer(f: &Mor, g: &Mor) -> Equalizer {
    assert_eq!(f.dom, g.dom, "equalizer domain");
    assert_eq!(f.cod, g.cod, "equalizer codomain");
    let elements: Vec<u32> = (0..f.dom).filter(|&x| f.apply(x) == g.apply(x)).collect();
    let obj = elements.len() as u32;
    Equalizer {
        obj,
        incl: Mor {
            dom: obj,
            cod: f.dom,
            map: elements.clone(),
        },
        elements,
    }
}

pub fn equalizer_mediator(eq: &Equalizer, h: &Mor) -> Mor {
    let mut index_of = vec![None; eq.incl.cod as usize];
    for (i, &x) in eq.elements.iter().enumerate() {
        index_of[x as usize] = Some(i as u32);
    }
    Mor {
        dom: h.dom,
        cod: eq.obj,
        map: (0..h.dom)
            .map(|w| {
                index_of[h.apply(w) as usize]
                    .expect("morphism does not factor through the equalizer")
            })
            .collect(),
    }
}

// ----- colimits -----

pub fn initial() -> u32 {
    0
}

pub fn from_initial(cod: u32) -> Mor {
    Mor {
        dom: 0,
        cod,
        map: vec![],
    }
}

pub struct Coproduct {
    pub obj: u32,
    pub inl: Mor,
    pub inr: Mor,
    pub left: u32,
    pub right: u32,
}

pub fn coproduct(a: u32, b: u32) -> Coproduct {
    let obj = a + b;
    Coproduct {
        obj,
        inl: Mor {
            dom: a,
            cod: obj,
            map: (0..a).collect(),
        },
        inr: Mor {
            dom: b,
            cod: obj,
            map: (0..b).map(|j| a + j).collect(),
        },
        left: a,
        right: b,
    }
}

/// Mediator out of `A ⊔ B`.
pub fn copair(f: &Mor, g: &Mor) -> Mor {
    assert_eq!(f.cod, g.cod, "copair: common codomain");
    let mut map = f.map.clone();
    map.extend_from_slice(&g.map);
    Mor {
        dom: f.dom + g.dom,
        cod: f.cod,
        map,
    }
}

/// Tag of an element of `A ⊔ B`: `Ok(i)` lies in the left summand.
pub fn coproduct_side(left: u32, index: u32) -> Result<u32, u32> {
    if index < left {
        Ok(index)
    } else {
        Err(index - left)
    }
}

struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(n: u32) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32) {
        let pa = self.find(a);
        let pb = self.find(b);
        if pa != pb {
            self.parent[pb as usize] = pa;
        }
    }
}

pub struct Coequalizer {
    pub obj: u32,
    pub proj: Mor,
}

/// Coequalizer of a parallel pair `f, g: X → Y`, as a quotient of `Y`.
pub fn coequalizer(f: &Mor, g: &Mor) -> Coequalizer {
    assert_eq!(f.dom, g.dom, "coequalizer domain");
    assert_eq!(f.cod, g.cod, "coequalizer codomain");
    let mut uf = UnionFind::new(f.cod);
    for x in 0..f.dom {
        uf.union(f.apply(x), g.apply(x));
    }
    let mut root_index = vec![None; f.cod as usize];
    let mut next = 0u32;
    let mut map = Vec::with_capacity(f.cod as usize);
    for y in 0..f.cod {
        let r = uf.find(y);
        let idx = *root_index[r as usize].get_or_insert_with(|| {
            let i = next;
            next += 1;
            i
        });
        map.push(idx);
    }
    Coequalizer {
        obj: next,
        proj: Mor {
            dom: f.cod,
            cod: next,
            map,
        },
    }
}

/// Mediator out of a coequalizer. `h: Y → W` must identify `f` and `g`.
pub fn coequalizer_mediator(q: &Coequalizer, h: &Mor) -> Mor {
    assert_eq!(
        h.dom, q.proj.dom,
        "mediator domain is the coequalizer's source"
    );
    let mut class_image = vec![None; q.obj as usize];
    for y in 0..h.dom {
        let c = q.proj.apply(y);
        let image = h.apply(y);
        match class_image[c as usize] {
            None => class_image[c as usize] = Some(image),
            Some(prev) => assert_eq!(prev, image, "h does not respect the quotient"),
        }
    }
    Mor {
        dom: q.obj,
        cod: h.cod,
        map: class_image.into_iter().map(|v| v.unwrap()).collect(),
    }
}

pub struct Pushout {
    pub apex: u32,
    pub from_left: Mor,
    pub from_right: Mor,
}

/// Pushout of `left_leg: Z → X` and `right_leg: Z → Y`.
pub fn pushout(left_leg: &Mor, right_leg: &Mor) -> Pushout {
    assert_eq!(left_leg.dom, right_leg.dom, "pushout: common domain");
    let sum = coproduct(left_leg.cod, right_leg.cod);
    let to_sum_l = compose(&sum.inl, left_leg);
    let to_sum_r = compose(&sum.inr, right_leg);
    let q = coequalizer(&to_sum_l, &to_sum_r);
    Pushout {
        apex: q.obj,
        from_left: compose(&q.proj, &sum.inl),
        from_right: compose(&q.proj, &sum.inr),
    }
}

pub fn pushout_mediator(po: &Pushout, p: &Mor, r: &Mor, left_card: u32) -> Mor {
    assert_eq!(p.cod, r.cod, "pushout mediator: common codomain");
    let sum = coproduct(left_card, r.dom);
    let h = copair(p, r);
    // Rebuild the quotient projection from the pushout legs: from_left and
    // from_right together are q ∘ inl and q ∘ inr.
    let proj = copair(&po.from_left, &po.from_right);
    assert_eq!(proj.dom, sum.obj);
    let q = Coequalizer { obj: po.apex, proj };
    coequalizer_mediator(&q, &h)
}

/// Canonical distributivity isomorphism `A × (B ⊔ C) → (A × B) ⊔ (A × C)`.
pub fn distribute_product_coproduct(a: u32, b: u32, c: u32) -> Mor {
    let left = product(a, b + c);
    let ab = product(a, b);
    let ac = product(a, c);
    let right = coproduct(ab.obj, ac.obj);
    let mut map = Vec::with_capacity(left.obj as usize);
    for i in 0..a {
        for k in 0..(b + c) {
            let idx = if k < b {
                pair_index(b, i, k)
            } else {
                ab.obj + pair_index(c, i, k - b)
            };
            let _ = right;
            map.push(idx);
        }
    }
    Mor {
        dom: left.obj,
        cod: ab.obj + ac.obj,
        map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_commutes(left: &Mor, right: &Mor) {
        assert_eq!(
            left, right,
            "diagram failed\n left {left:?}\n right {right:?}"
        );
    }

    #[test]
    fn category_laws_up_to_max() {
        for a in objects_up_to(MAX_CARD) {
            for b in objects_up_to(MAX_CARD) {
                for f in morphisms(a, b) {
                    assert_commutes(&compose(&f, &id(a)), &f);
                    assert_commutes(&compose(&id(b), &f), &f);
                    for c in objects_up_to(MAX_CARD) {
                        for g in morphisms(b, c) {
                            for d in objects_up_to(MAX_CARD) {
                                for h in morphisms(c, d) {
                                    let left = compose(&h, &compose(&g, &f));
                                    let right = compose(&compose(&h, &g), &f);
                                    assert_commutes(&left, &right);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn product_universal() {
        for a in objects_up_to(MAX_CARD) {
            for b in objects_up_to(MAX_CARD) {
                let p = product(a, b);
                assert_eq!(compose(&p.fst, &pair(&p.fst, &p.snd)), p.fst);
                assert_eq!(compose(&p.snd, &pair(&p.fst, &p.snd)), p.snd);
                for z in objects_up_to(MAX_CARD) {
                    for f in morphisms(z, a) {
                        for g in morphisms(z, b) {
                            let m = pair(&f, &g);
                            assert_commutes(&compose(&p.fst, &m), &f);
                            assert_commutes(&compose(&p.snd, &m), &g);
                            // uniqueness: any other mediator agrees
                            for other in morphisms(z, p.obj) {
                                if compose(&p.fst, &other) == f && compose(&p.snd, &other) == g {
                                    assert_eq!(other, m);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn coproduct_universal() {
        for a in objects_up_to(MAX_CARD) {
            for b in objects_up_to(MAX_CARD) {
                let s = coproduct(a, b);
                assert_eq!(compose(&copair(&s.inl, &s.inr), &s.inl), s.inl);
                for c in objects_up_to(MAX_CARD) {
                    for f in morphisms(a, c) {
                        for g in morphisms(b, c) {
                            let m = copair(&f, &g);
                            assert_commutes(&compose(&m, &s.inl), &f);
                            assert_commutes(&compose(&m, &s.inr), &g);
                            for other in morphisms(s.obj, c) {
                                if compose(&other, &s.inl) == f && compose(&other, &s.inr) == g {
                                    assert_eq!(other, m);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pullback_universal() {
        for x in objects_up_to(MAX_CARD) {
            for y in objects_up_to(MAX_CARD) {
                for z in objects_up_to(MAX_CARD) {
                    for f in morphisms(x, z) {
                        for g in morphisms(y, z) {
                            let pb = pullback(&f, &g);
                            assert_commutes(&compose(&f, &pb.to_left), &compose(&g, &pb.to_right));
                            let prod = product(x, y);
                            let incl = pullback_into_product(&pb, y);
                            assert_commutes(&compose(&prod.fst, &incl), &pb.to_left);
                            assert_commutes(&compose(&prod.snd, &incl), &pb.to_right);
                            for w in objects_up_to(MAX_CARD) {
                                for p in morphisms(w, x) {
                                    for q in morphisms(w, y) {
                                        if compose(&f, &p) != compose(&g, &q) {
                                            continue;
                                        }
                                        let m = pullback_mediator(&pb, &p, &q);
                                        assert_commutes(&compose(&pb.to_left, &m), &p);
                                        assert_commutes(&compose(&pb.to_right, &m), &q);
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
    fn equalizer_and_coequalizer_universal() {
        for x in objects_up_to(MAX_CARD) {
            for y in objects_up_to(MAX_CARD) {
                for f in morphisms(x, y) {
                    for g in morphisms(x, y) {
                        let eq = equalizer(&f, &g);
                        assert_commutes(&compose(&f, &eq.incl), &compose(&g, &eq.incl));
                        let co = coequalizer(&f, &g);
                        assert_commutes(&compose(&co.proj, &f), &compose(&co.proj, &g));
                        for w in objects_up_to(MAX_CARD) {
                            for h in morphisms(w, x) {
                                if compose(&f, &h) == compose(&g, &h) {
                                    let m = equalizer_mediator(&eq, &h);
                                    assert_commutes(&compose(&eq.incl, &m), &h);
                                }
                            }
                            for h in morphisms(y, w) {
                                if compose(&h, &f) == compose(&h, &g) {
                                    let m = coequalizer_mediator(&co, &h);
                                    assert_commutes(&compose(&m, &co.proj), &h);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pushout_square_commutes() {
        for z in objects_up_to(MAX_CARD) {
            for x in objects_up_to(MAX_CARD) {
                for y in objects_up_to(MAX_CARD) {
                    for f in morphisms(z, x) {
                        for g in morphisms(z, y) {
                            let po = pushout(&f, &g);
                            assert_commutes(
                                &compose(&po.from_left, &f),
                                &compose(&po.from_right, &g),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn product_distributes_over_coproduct() {
        for a in objects_up_to(MAX_CARD) {
            for b in objects_up_to(MAX_CARD) {
                for c in objects_up_to(MAX_CARD) {
                    let iso = distribute_product_coproduct(a, b, c);
                    assert!(is_iso(&iso), "distributivity {a},{b},{c}: {iso:?}");
                    let inv = inverse(&iso);
                    let left = product(a, b + c);
                    let ab = product(a, b);
                    let ac = product(a, c);
                    let right = coproduct(ab.obj, ac.obj);
                    // projection to A agrees
                    let to_a_left = left.fst.clone();
                    let to_a_right = copair(&ab.fst, &ac.fst);
                    assert_commutes(
                        &compose(&to_a_left, &inv),
                        &compose(&to_a_right, &id(right.obj)),
                    );
                    let _ = to_a_right;
                }
            }
        }
    }

    #[test]
    fn terminal_and_initial() {
        assert_eq!(terminal(), 1);
        assert_eq!(initial(), 0);
        for a in objects_up_to(MAX_CARD) {
            assert_eq!(morphisms(a, 1).len(), 1);
            assert_eq!(morphisms(0, a).len(), 1);
            if a > 0 {
                assert!(morphisms(a, 0).is_empty());
            }
        }
    }

    #[test]
    fn product_preserves_equalizers() {
        // (eq(f, g)) × A  ≅  eq(f × id, g × id)
        for x in objects_up_to(MAX_CARD) {
            for y in objects_up_to(MAX_CARD) {
                for a in objects_up_to(MAX_CARD) {
                    for f in morphisms(x, y) {
                        for g in morphisms(x, y) {
                            let eq = equalizer(&f, &g);
                            let left = product(eq.obj, a);
                            let xa = product(x, a);
                            let ya = product(y, a);
                            let f_id = pair(&compose(&f, &xa.fst), &xa.snd);
                            let g_id = pair(&compose(&g, &xa.fst), &xa.snd);
                            assert_eq!(f_id.cod, ya.obj);
                            let right = equalizer(&f_id, &g_id);
                            // inclusion of left into X×A
                            let incl_left = pair(&compose(&eq.incl, &left.fst), &left.snd);
                            assert_eq!(incl_left.dom, left.obj);
                            assert_eq!(right.obj, left.obj, "equalizer card after product");
                            // the underlying elements match via the product order
                            let mut from_right = vec![false; xa.obj as usize];
                            for &e in &right.elements {
                                from_right[e as usize] = true;
                            }
                            for i in 0..left.obj {
                                assert!(from_right[incl_left.apply(i) as usize]);
                            }
                        }
                    }
                }
            }
        }
    }
}
