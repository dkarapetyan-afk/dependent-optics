//! Fibre optics as dependent optics on a matrix bicategory (Section 5,
//! citing Braithwaite–Capucci–Gavranović–Hedges–Rischel §4.3).
//!
//! A 1-cell `I → J` is an `I × J` matrix of cardinalities. Composition sums
//! the products of entries. Reindexing a family is matrix–vector multiplication.
//! A terminal index set is the one-object delooping, whose blocks are ordinary
//! cartesian residuals.

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Matrix {
    pub rows: u32,
    pub cols: u32,
    pub blocks: Vec<u32>,
}

impl Matrix {
    pub fn get(&self, i: u32, j: u32) -> u32 {
        self.blocks[(i * self.cols + j) as usize]
    }
}

pub fn identity(n: u32) -> Matrix {
    let mut blocks = vec![0; (n * n) as usize];
    for i in 0..n {
        blocks[(i * n + i) as usize] = 1;
    }
    Matrix {
        rows: n,
        cols: n,
        blocks,
    }
}

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "matrix composition");
    let mut blocks = vec![0u32; (a.rows * b.cols) as usize];
    for i in 0..a.rows {
        for k in 0..b.cols {
            let mut sum = 0u32;
            for j in 0..a.cols {
                sum += a.get(i, j) * b.get(j, k);
            }
            blocks[(i * b.cols + k) as usize] = sum;
        }
    }
    Matrix {
        rows: a.rows,
        cols: b.cols,
        blocks,
    }
}

/// `(M • Y)_i = Σ_j |M_ij| · |Y_j|`.
pub fn act(m: &Matrix, family: &[u32]) -> Vec<u32> {
    assert_eq!(family.len() as u32, m.cols);
    (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m.get(i, j) * family[j as usize]).sum())
        .collect()
}

pub fn matrices(rows: u32, cols: u32, max_entry: u32) -> Vec<Matrix> {
    let cells = (rows * cols) as usize;
    if cells == 0 {
        return vec![Matrix {
            rows,
            cols,
            blocks: vec![],
        }];
    }
    let mut cur = vec![0u32; cells];
    let mut out = Vec::new();
    loop {
        out.push(Matrix {
            rows,
            cols,
            blocks: cur.clone(),
        });
        let mut i = 0;
        loop {
            if i == cells {
                return out;
            }
            cur[i] += 1;
            if cur[i] <= max_entry {
                break;
            }
            cur[i] = 0;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_multiplication_act_as_a_pseudofunctor_on_cardinalities() {
        for n in 0..=2 {
            let id = identity(n);
            for family in families(n, 2) {
                assert_eq!(act(&id, &family), family, "identity matrix");
            }
        }
        for rows in 0..=2 {
            for mid in 0..=2 {
                for cols in 0..=2 {
                    for a in matrices(rows, mid, 1) {
                        for b in matrices(mid, cols, 1) {
                            let ab = multiply(&a, &b);
                            for y in families(cols, 1) {
                                let via_mul = act(&ab, &y);
                                let via_act = act(&a, &act(&b, &y));
                                assert_eq!(via_mul, via_act, "matrix action");
                            }
                            // Identity laws for the bicategory on cardinalities.
                            if mid == rows {
                                assert_eq!(multiply(&a, &identity(mid)), a);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn terminal_index_is_a_cartesian_residual() {
        for m in 0..=2 {
            for y in 0..=2 {
                let matrix = Matrix {
                    rows: 1,
                    cols: 1,
                    blocks: vec![m],
                };
                assert_eq!(act(&matrix, &[y]), vec![m * y]);
            }
        }
    }

    fn families(n: u32, max_part: u32) -> Vec<Vec<u32>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut cur = vec![0u32; n as usize];
        let mut out = Vec::new();
        loop {
            out.push(cur.clone());
            let mut i = 0;
            loop {
                if i == n as usize {
                    return out;
                }
                cur[i] += 1;
                if cur[i] <= max_part {
                    break;
                }
                cur[i] = 0;
                i += 1;
            }
        }
    }
}
