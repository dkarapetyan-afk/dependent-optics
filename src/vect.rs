//! Finite-dimensional vector spaces over `GF(2)`.
//!
//! Maps are stored by the images of basis vectors, each a bitset. Dimension 0
//! has a single map, the zero map, including into or out of the zero space.

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lin {
    pub dom: u32,
    pub cod: u32,
    pub columns: Vec<u32>,
}

impl Lin {
    pub fn apply_vec(&self, vector: u32) -> u32 {
        let mut acc = 0u32;
        for (j, col) in self.columns.iter().enumerate() {
            if (vector >> j) & 1 == 1 {
                acc ^= col;
            }
        }
        acc
    }
}

pub fn zero(dom: u32, cod: u32) -> Lin {
    Lin {
        dom,
        cod,
        columns: vec![0; dom as usize],
    }
}

pub fn identity(n: u32) -> Lin {
    Lin {
        dom: n,
        cod: n,
        columns: (0..n).map(|i| 1u32 << i).collect(),
    }
}

pub fn compose(g: &Lin, f: &Lin) -> Lin {
    assert_eq!(f.cod, g.dom, "linear composition");
    Lin {
        dom: f.dom,
        cod: g.cod,
        columns: f.columns.iter().map(|&col| g.apply_vec(col)).collect(),
    }
}

pub fn add(f: &Lin, g: &Lin) -> Lin {
    assert_eq!(f.dom, g.dom);
    assert_eq!(f.cod, g.cod);
    Lin {
        dom: f.dom,
        cod: f.cod,
        columns: f
            .columns
            .iter()
            .zip(&g.columns)
            .map(|(a, b)| a ^ b)
            .collect(),
    }
}

pub fn transpose(f: &Lin) -> Lin {
    let mut columns = vec![0u32; f.cod as usize];
    for (j, col) in f.columns.iter().enumerate() {
        for i in 0..f.cod {
            if (col >> i) & 1 == 1 {
                columns[i as usize] ^= 1u32 << j;
            }
        }
    }
    Lin {
        dom: f.cod,
        cod: f.dom,
        columns,
    }
}

/// Kronecker product. Basis of `A ⊗ B` is ordered `a * dim_b + b`.
pub fn kronecker(f: &Lin, g: &Lin) -> Lin {
    let dom = f.dom * g.dom;
    let cod = f.cod * g.cod;
    let mut columns = vec![0u32; dom as usize];
    for ja in 0..f.dom {
        for jb in 0..g.dom {
            let mut image = 0u32;
            let fa = f.columns[ja as usize];
            let gb = g.columns[jb as usize];
            for ia in 0..f.cod {
                if (fa >> ia) & 1 == 0 {
                    continue;
                }
                for ib in 0..g.cod {
                    if (gb >> ib) & 1 == 1 {
                        image ^= 1u32 << (ia * g.cod + ib);
                    }
                }
            }
            columns[(ja * g.dom + jb) as usize] = image;
        }
    }
    Lin { dom, cod, columns }
}

pub fn all(dom: u32, cod: u32) -> Vec<Lin> {
    let bits = (dom * cod) as usize;
    let total = 1u32 << bits;
    let mut out = Vec::with_capacity(total as usize);
    for mask in 0..total {
        let mut columns = vec![0u32; dom as usize];
        for j in 0..dom {
            let mut col = 0u32;
            for i in 0..cod {
                let bit = (j * cod + i) as u32;
                if (mask >> bit) & 1 == 1 {
                    col |= 1 << i;
                }
            }
            columns[j as usize] = col;
        }
        out.push(Lin { dom, cod, columns });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_category_and_tensor() {
        for n in 0..=2 {
            let id = identity(n);
            for f in all(n, n) {
                assert_eq!(compose(&id, &f), f);
                assert_eq!(compose(&f, &id), f);
                assert_eq!(add(&f, &f), zero(n, n));
                assert_eq!(compose(&transpose(&transpose(&f)), &identity(n)), f);
            }
        }
        for a in 0..=1 {
            for b in 0..=1 {
                for f in all(a, a) {
                    for g in all(b, b) {
                        let t = kronecker(&f, &g);
                        assert_eq!(t.dom, a * b);
                        assert_eq!(t.cod, a * b);
                        assert_eq!(kronecker(&identity(a), &identity(b)), identity(a * b));
                    }
                }
            }
        }
    }
}
