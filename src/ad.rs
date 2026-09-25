//! Reverse-mode differentiation with an explicit residual (Section 3.1).
//!
//! Each primitive is a lens whose residual stores only what the backward map
//! reads. Addition stores nothing about its inputs. Multiplication stores both
//! factors and not the product. A conditional stores only the branch that was
//! taken. Cotangents are exact rationals and agree with forward-mode dual numbers.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q {
    n: i64,
    d: i64,
}

impl Q {
    pub fn new(n: i64, d: i64) -> Self {
        assert!(d != 0);
        let mut n = n;
        let mut d = d;
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = gcd(n.abs(), d);
        Self { n: n / g, d: d / g }
    }

    pub fn add(self, o: Self) -> Self {
        Self::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }

    pub fn mul(self, o: Self) -> Self {
        Self::new(self.n * o.n, self.d * o.d)
    }

    pub fn neg(self) -> Self {
        Self::new(-self.n, self.d)
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var(usize),
    Const(Q),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    /// `let x = e1 in e2`, with `x` at de Bruijn index 0 in `e2`.
    Let(Box<Expr>, Box<Expr>),
    IfEq(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
pub enum Tape {
    Var(usize),
    Const,
    Neg(Box<Tape>),
    Add(Box<Tape>, Box<Tape>),
    Mul(Q, Q, Box<Tape>, Box<Tape>),
    Let(Box<Tape>, Box<Tape>),
    IfEq {
        then_branch: bool,
        branch: Box<Tape>,
    },
}

pub fn forward(expr: &Expr, env: &[Q]) -> (Q, Tape) {
    match expr {
        Expr::Var(i) => (env[*i], Tape::Var(*i)),
        Expr::Const(q) => (*q, Tape::Const),
        Expr::Neg(e) => {
            let (v, t) = forward(e, env);
            (v.neg(), Tape::Neg(Box::new(t)))
        }
        Expr::Add(a, b) => {
            let (va, ta) = forward(a, env);
            let (vb, tb) = forward(b, env);
            (va.add(vb), Tape::Add(Box::new(ta), Box::new(tb)))
        }
        Expr::Mul(a, b) => {
            let (va, ta) = forward(a, env);
            let (vb, tb) = forward(b, env);
            (va.mul(vb), Tape::Mul(va, vb, Box::new(ta), Box::new(tb)))
        }
        Expr::Let(bound, body) => {
            let (vb, tb) = forward(bound, env);
            let mut ext = Vec::with_capacity(env.len() + 1);
            ext.push(vb);
            ext.extend_from_slice(env);
            let (v, tbody) = forward(body, &ext);
            (v, Tape::Let(Box::new(tb), Box::new(tbody)))
        }
        Expr::IfEq(c1, c2, t, e) => {
            let (v1, _) = forward(c1, env);
            let (v2, _) = forward(c2, env);
            if v1 == v2 {
                let (v, tape) = forward(t, env);
                (
                    v,
                    Tape::IfEq {
                        then_branch: true,
                        branch: Box::new(tape),
                    },
                )
            } else {
                let (v, tape) = forward(e, env);
                (
                    v,
                    Tape::IfEq {
                        then_branch: false,
                        branch: Box::new(tape),
                    },
                )
            }
        }
    }
}

pub fn backward(expr: &Expr, tape: &Tape, cot: Q, env_len: usize) -> Vec<Q> {
    let mut acc = vec![Q::new(0, 1); env_len];
    fn add_at(acc: &mut [Q], i: usize, v: Q) {
        acc[i] = acc[i].add(v);
    }
    fn go(expr: &Expr, tape: &Tape, cot: Q, acc: &mut [Q]) {
        match (expr, tape) {
            (Expr::Var(i), Tape::Var(j)) => {
                assert_eq!(i, j);
                add_at(acc, *i, cot);
            }
            (Expr::Const(_), Tape::Const) => {}
            (Expr::Neg(e), Tape::Neg(t)) => go(e, t, cot.neg(), acc),
            (Expr::Add(a, b), Tape::Add(ta, tb)) => {
                go(a, ta, cot, acc);
                go(b, tb, cot, acc);
            }
            (Expr::Mul(a, b), Tape::Mul(x, y, ta, tb)) => {
                go(a, ta, cot.mul(*y), acc);
                go(b, tb, cot.mul(*x), acc);
            }
            (Expr::Let(bound, body), Tape::Let(tb, tbody)) => {
                let mut body_acc = vec![Q::new(0, 1); acc.len() + 1];
                go(body, tbody, cot, &mut body_acc);
                let bound_cot = body_acc[0];
                for (i, v) in body_acc.into_iter().skip(1).enumerate() {
                    add_at(acc, i, v);
                }
                go(bound, tb, bound_cot, acc);
            }
            (
                Expr::IfEq(_, _, then_e, else_e),
                Tape::IfEq {
                    then_branch,
                    branch,
                },
            ) => {
                if *then_branch {
                    go(then_e, branch, cot, acc);
                } else {
                    go(else_e, branch, cot, acc);
                }
            }
            _ => panic!("tape does not match expression"),
        }
    }
    go(expr, tape, cot, &mut acc);
    acc
}

/// Rational scalars stored for the backward pass, not the shape of the tape.
pub fn stored_scalars(tape: &Tape) -> Vec<Q> {
    match tape {
        Tape::Var(_) | Tape::Const => vec![],
        Tape::Neg(t) => stored_scalars(t),
        Tape::Add(a, b) | Tape::Let(a, b) => {
            let mut v = stored_scalars(a);
            v.extend(stored_scalars(b));
            v
        }
        Tape::Mul(x, y, a, b) => {
            let mut v = vec![*x, *y];
            v.extend(stored_scalars(a));
            v.extend(stored_scalars(b));
            v
        }
        Tape::IfEq { branch, .. } => stored_scalars(branch),
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct Dual {
    re: Q,
    eps: Q,
}

#[cfg(test)]
fn dual_eval(expr: &Expr, env: &[Dual]) -> Dual {
    match expr {
        Expr::Var(i) => env[*i],
        Expr::Const(q) => Dual {
            re: *q,
            eps: Q::new(0, 1),
        },
        Expr::Neg(e) => {
            let v = dual_eval(e, env);
            Dual {
                re: v.re.neg(),
                eps: v.eps.neg(),
            }
        }
        Expr::Add(a, b) => {
            let va = dual_eval(a, env);
            let vb = dual_eval(b, env);
            Dual {
                re: va.re.add(vb.re),
                eps: va.eps.add(vb.eps),
            }
        }
        Expr::Mul(a, b) => {
            let va = dual_eval(a, env);
            let vb = dual_eval(b, env);
            Dual {
                re: va.re.mul(vb.re),
                eps: va.re.mul(vb.eps).add(va.eps.mul(vb.re)),
            }
        }
        Expr::Let(bound, body) => {
            let vb = dual_eval(bound, env);
            let mut ext = Vec::with_capacity(env.len() + 1);
            ext.push(vb);
            ext.extend_from_slice(env);
            dual_eval(body, &ext)
        }
        Expr::IfEq(c1, c2, t, e) => {
            let v1 = dual_eval(c1, env);
            let v2 = dual_eval(c2, env);
            if v1.re == v2.re {
                dual_eval(t, env)
            } else {
                dual_eval(e, env)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Q {
        Q::new(n, 1)
    }

    fn catalogue() -> Vec<(Expr, usize)> {
        let x = || Expr::Var(0);
        let y = || Expr::Var(1);
        vec![
            (
                Expr::Add(
                    Box::new(Expr::Mul(Box::new(x()), Box::new(x()))),
                    Box::new(y()),
                ),
                2,
            ),
            (
                Expr::Let(
                    Box::new(Expr::Mul(Box::new(Expr::Var(0)), Box::new(Expr::Var(1)))),
                    Box::new(Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
                ),
                2,
            ),
            (Expr::Var(0), 2),
            (
                Expr::IfEq(
                    Box::new(x()),
                    Box::new(Expr::Const(q(0))),
                    Box::new(y()),
                    Box::new(Expr::Mul(Box::new(x()), Box::new(y()))),
                ),
                2,
            ),
            (Expr::Add(Box::new(x()), Box::new(y())), 2),
            (Expr::Mul(Box::new(x()), Box::new(y())), 2),
            (Expr::Neg(Box::new(x())), 1),
        ]
    }

    #[test]
    fn residuals_store_only_what_the_backward_pass_reads() {
        let env = [q(3), q(4)];
        let (_v, add) = forward(
            &Expr::Add(Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            &env,
        );
        assert!(stored_scalars(&add).is_empty(), "addition stores no primal");
        let (_v, neg) = forward(&Expr::Neg(Box::new(Expr::Var(0))), &env);
        assert!(stored_scalars(&neg).is_empty());
        let (_v, mul) = forward(
            &Expr::Mul(Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            &env,
        );
        assert_eq!(stored_scalars(&mul), vec![q(3), q(4)]);
        let (_v, proj) = forward(&Expr::Var(0), &env);
        assert!(
            stored_scalars(&proj).is_empty(),
            "a projection stores nothing"
        );
        let (_v, branch_then) = forward(
            &Expr::IfEq(
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(q(3))),
                Box::new(Expr::Var(1)),
                Box::new(Expr::Mul(Box::new(Expr::Var(0)), Box::new(Expr::Var(1)))),
            ),
            &env,
        );
        assert!(
            stored_scalars(&branch_then).is_empty(),
            "taken branch is a projection"
        );
        let (_v, branch_else) = forward(
            &Expr::IfEq(
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(q(0))),
                Box::new(Expr::Var(1)),
                Box::new(Expr::Mul(Box::new(Expr::Var(0)), Box::new(Expr::Var(1)))),
            ),
            &env,
        );
        assert_eq!(stored_scalars(&branch_else), vec![q(3), q(4)]);
    }

    #[test]
    fn cotangents_match_dual_numbers() {
        for (expr, arity) in catalogue() {
            let coords = [-2, -1, 0, 1, 2];
            let points: Vec<Vec<Q>> = if arity == 1 {
                coords.iter().map(|x| vec![q(*x)]).collect()
            } else {
                coords
                    .iter()
                    .flat_map(|x| coords.iter().map(move |y| vec![q(*x), q(*y)]))
                    .collect()
            };
            for point in points {
                for dir in 0..arity {
                    let mut dual_env = Vec::new();
                    for (i, v) in point.iter().enumerate() {
                        dual_env.push(Dual {
                            re: *v,
                            eps: if i == dir { q(1) } else { q(0) },
                        });
                    }
                    let dual = dual_eval(&expr, &dual_env);
                    let (value, tape) = forward(&expr, &point);
                    assert_eq!(value, dual.re);
                    let cot = backward(&expr, &tape, q(1), arity);
                    assert_eq!(cot[dir], dual.eps, "direction {dir} at {point:?}");
                    // The backward pass is determined by the residual and the
                    // cotangent, which is the representative-free interface.
                    let again = backward(&expr, &tape, q(1), arity);
                    assert_eq!(again, cot);
                }
            }
        }
    }
}
