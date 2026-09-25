//! Neural architectures as dependent lenses, compiled to NVIDIA kernels.
//!
//! A layer is the lens of Section 3.1: the forward map writes the output
//! together with a residual that contains only what the backward map reads.
//! Sequencing layers is optic composition, so the composite residual is the
//! tape of the parts and the backward pass runs the parts in reverse. A
//! residual connection is an add lens with a unit residual: the skip does not
//! store a second copy of the activation, it reuses the copy already saved by
//! the branch that reads it.
//!
//! [`Program::mlp`], [`Program::resnet_block`], and [`Program::transformer_block`]
//! are the three architectures. [`emit_cuda`] lowers a program to CUDA. The
//! forward kernels are the maps `X → M × Y` and the backward kernels are the
//! maps `M × dY → dX`.

/// One compiled stage. Offsets are into the flat residual tape.
#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    /// `Y = relu?(X W + b)`. The residual stores `X`, and the ReLU mask when
    /// `relu` is set. `W` and `b` stay in the parameter buffer.
    Gemm {
        batch: usize,
        din: usize,
        dout: usize,
        relu: bool,
        save_at: usize,
        mask_at: usize,
        param_at: usize,
    },
    /// Self-attention with `Q = K = V = X`. The residual stores `X` and the
    /// softmax probabilities.
    Attention {
        batch: usize,
        tokens: usize,
        dim: usize,
        x_at: usize,
        probs_at: usize,
    },
    /// `Y = Y + tape[from]`. No residual of its own.
    Add { from: usize, len: usize },
}

#[derive(Clone, Debug)]
pub struct Program {
    pub name: String,
    pub stages: Vec<Stage>,
    pub input_len: usize,
    pub output_len: usize,
    pub param_len: usize,
    pub tape_f32: usize,
    pub tape_u8: usize,
}

#[derive(Clone, Debug)]
struct Built {
    prog: Program,
}

impl Built {
    fn new(name: &str, input_len: usize) -> Self {
        Self {
            prog: Program {
                name: name.to_string(),
                stages: Vec::new(),
                input_len,
                output_len: input_len,
                param_len: 0,
                tape_f32: 0,
                tape_u8: 0,
            },
        }
    }

    fn gemm(&mut self, batch: usize, din: usize, dout: usize, relu: bool) {
        let save_at = self.prog.tape_f32;
        let mask_at = self.prog.tape_u8;
        let param_at = self.prog.param_len;
        self.prog.stages.push(Stage::Gemm {
            batch,
            din,
            dout,
            relu,
            save_at,
            mask_at,
            param_at,
        });
        self.prog.tape_f32 += batch * din;
        if relu {
            self.prog.tape_u8 += batch * dout;
        }
        self.prog.param_len += din * dout + dout;
        self.prog.output_len = batch * dout;
    }

    fn attention(&mut self, batch: usize, tokens: usize, dim: usize) -> usize {
        assert!(
            tokens <= 64,
            "the attention kernel keeps one row in registers"
        );
        let x_at = self.prog.tape_f32;
        let n = batch * tokens * dim;
        self.prog.tape_f32 += n;
        let probs_at = self.prog.tape_f32;
        self.prog.tape_f32 += batch * tokens * tokens;
        self.prog.stages.push(Stage::Attention {
            batch,
            tokens,
            dim,
            x_at,
            probs_at,
        });
        self.prog.output_len = n;
        x_at
    }

    fn add(&mut self, from: usize, len: usize) {
        self.prog.stages.push(Stage::Add { from, len });
    }
}

impl Program {
    /// Multilayer perceptron. ReLU sits between linear layers, not after the
    /// last one. Each linear lens stores its input; each ReLU stores a mask.
    pub fn mlp(batch: usize, dims: &[usize]) -> Self {
        assert!(dims.len() >= 2, "an MLP needs an input and an output width");
        let mut b = Built::new("mlp", batch * dims[0]);
        for (i, w) in dims.windows(2).enumerate() {
            let relu = i + 1 < dims.len() - 1;
            b.gemm(batch, w[0], w[1], relu);
        }
        b.prog
    }

    /// Pre-activation-free ResNet block `x + W2 relu(W1 x + b1) + b2`.
    /// The skip add stores nothing: `W1` already saved `x`.
    pub fn resnet_block(batch: usize, dim: usize, hidden: usize) -> Self {
        let mut b = Built::new("resnet_block", batch * dim);
        let x_at = {
            let at = b.prog.tape_f32;
            b.gemm(batch, dim, hidden, true);
            at
        };
        b.gemm(batch, hidden, dim, false);
        b.add(x_at, batch * dim);
        b.prog.output_len = batch * dim;
        b.prog
    }

    /// Transformer block: residual self-attention, then a residual MLP.
    /// Attention's residual is the tokens plus the softmax. The skip add
    /// reuses those tokens. The MLP's first linear saves the attention output
    /// for its own skip.
    pub fn transformer_block(batch: usize, tokens: usize, dim: usize, mlp_hidden: usize) -> Self {
        let n = batch * tokens * dim;
        let mut b = Built::new("transformer_block", n);
        let x_at = b.attention(batch, tokens, dim);
        b.add(x_at, n);
        let y_at = b.prog.tape_f32;
        b.gemm(batch * tokens, dim, mlp_hidden, true);
        b.gemm(batch * tokens, mlp_hidden, dim, false);
        b.add(y_at, n);
        b.prog.output_len = n;
        b.prog
    }
}

/// Deterministic positive inputs so ReLU stays in its linear region.
pub fn init(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| scale * ((i % 7) as f32 + 1.0) / 8.0)
        .collect()
}

/// Forward and backward for a sum loss. `dy` is identically one.
/// Returns `(output, d_input, d_params)`.
pub fn forward_backward(
    prog: &Program,
    x: &[f32],
    params: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), prog.input_len);
    assert_eq!(params.len(), prog.param_len);
    let mut tape = vec![0.0f32; prog.tape_f32];
    let mut mask = vec![0u8; prog.tape_u8];
    let mut cur = x.to_vec();
    for stage in &prog.stages {
        match *stage {
            Stage::Gemm {
                batch,
                din,
                dout,
                relu,
                save_at,
                mask_at,
                param_at,
            } => {
                tape[save_at..save_at + cur.len()].copy_from_slice(&cur);
                let w = &params[param_at..param_at + din * dout];
                let bias = &params[param_at + din * dout..param_at + din * dout + dout];
                let mut y = gemm(&cur, w, bias, batch, din, dout);
                if relu {
                    for (i, v) in y.iter_mut().enumerate() {
                        let on = *v > 0.0;
                        mask[mask_at + i] = u8::from(on);
                        if !on {
                            *v = 0.0;
                        }
                    }
                }
                cur = y;
            }
            Stage::Attention {
                batch,
                tokens,
                dim,
                x_at,
                probs_at,
            } => {
                tape[x_at..x_at + cur.len()].copy_from_slice(&cur);
                let (y, probs) = attention_forward(&cur, batch, tokens, dim);
                let plen = batch * tokens * tokens;
                tape[probs_at..probs_at + plen].copy_from_slice(&probs);
                cur = y;
            }
            Stage::Add { from, len } => {
                for i in 0..len {
                    cur[i] += tape[from + i];
                }
            }
        }
    }
    let dy = vec![1.0f32; cur.len()];
    let (dx, dp) = backward(prog, params, &tape, &mask, &dy);
    (cur, dx, dp)
}

fn gemm(x: &[f32], w: &[f32], bias: &[f32], batch: usize, din: usize, dout: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * dout];
    for m in 0..batch {
        for n in 0..dout {
            let mut acc = bias[n];
            for k in 0..din {
                acc += x[m * din + k] * w[k * dout + n];
            }
            y[m * dout + n] = acc;
        }
    }
    y
}

fn attention_forward(x: &[f32], batch: usize, tokens: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut probs = vec![0.0f32; batch * tokens * tokens];
    let mut y = vec![0.0f32; batch * tokens * dim];
    for b in 0..batch {
        for i in 0..tokens {
            let mut row = vec![0.0f32; tokens];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..tokens {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += x[(b * tokens + i) * dim + d] * x[(b * tokens + j) * dim + d];
                }
                row[j] = dot * scale;
                max_s = max_s.max(row[j]);
            }
            let mut sum = 0.0;
            for j in 0..tokens {
                row[j] = (row[j] - max_s).exp();
                sum += row[j];
            }
            for j in 0..tokens {
                row[j] /= sum;
                probs[(b * tokens + i) * tokens + j] = row[j];
            }
            for d in 0..dim {
                let mut acc = 0.0;
                for j in 0..tokens {
                    acc += row[j] * x[(b * tokens + j) * dim + d];
                }
                y[(b * tokens + i) * dim + d] = acc;
            }
        }
    }
    (y, probs)
}

fn backward(
    prog: &Program,
    params: &[f32],
    tape: &[f32],
    mask: &[u8],
    dy0: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut dy = dy0.to_vec();
    let mut dparams = vec![0.0f32; prog.param_len];
    let mut side: Vec<(usize, Vec<f32>)> = Vec::new();
    for stage in prog.stages.iter().rev() {
        match *stage {
            Stage::Add { from, len } => {
                side.push((from, dy[..len].to_vec()));
            }
            Stage::Gemm {
                batch,
                din,
                dout,
                relu,
                save_at,
                mask_at,
                param_at,
            } => {
                if relu {
                    for i in 0..batch * dout {
                        if mask[mask_at + i] == 0 {
                            dy[i] = 0.0;
                        }
                    }
                }
                let x = &tape[save_at..save_at + batch * din];
                let w = &params[param_at..param_at + din * dout];
                let mut dx = vec![0.0f32; batch * din];
                for m in 0..batch {
                    for k in 0..din {
                        let mut acc = 0.0;
                        for n in 0..dout {
                            acc += dy[m * dout + n] * w[k * dout + n];
                        }
                        dx[m * din + k] = acc;
                    }
                }
                for k in 0..din {
                    for n in 0..dout {
                        let mut acc = 0.0;
                        for m in 0..batch {
                            acc += x[m * din + k] * dy[m * dout + n];
                        }
                        dparams[param_at + k * dout + n] = acc;
                    }
                }
                let b_at = param_at + din * dout;
                for n in 0..dout {
                    let mut acc = 0.0;
                    for m in 0..batch {
                        acc += dy[m * dout + n];
                    }
                    dparams[b_at + n] = acc;
                }
                add_side(&mut side, save_at, &mut dx);
                dy = dx;
            }
            Stage::Attention {
                batch,
                tokens,
                dim,
                x_at,
                probs_at,
            } => {
                let n = batch * tokens * dim;
                let x = &tape[x_at..x_at + n];
                let probs = &tape[probs_at..probs_at + batch * tokens * tokens];
                let mut dx = attention_backward(x, probs, &dy, batch, tokens, dim);
                add_side(&mut side, x_at, &mut dx);
                dy = dx;
            }
        }
    }
    assert!(side.is_empty(), "skip cotangent had no saved activation");
    (dy, dparams)
}

fn add_side(side: &mut Vec<(usize, Vec<f32>)>, at: usize, dx: &mut [f32]) {
    let mut i = 0;
    while i < side.len() {
        if side[i].0 == at {
            let extra = side.remove(i);
            for (dst, src) in dx.iter_mut().zip(extra.1) {
                *dst += src;
            }
        } else {
            i += 1;
        }
    }
}

fn attention_backward(
    x: &[f32],
    probs: &[f32],
    dy: &[f32],
    batch: usize,
    tokens: usize,
    dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut d_v = vec![0.0f32; x.len()];
    let mut d_a = vec![0.0f32; batch * tokens * tokens];
    for b in 0..batch {
        for i in 0..tokens {
            for j in 0..tokens {
                let p = probs[(b * tokens + i) * tokens + j];
                let mut dot = 0.0;
                for d in 0..dim {
                    let y = dy[(b * tokens + i) * dim + d];
                    d_v[(b * tokens + j) * dim + d] += p * y;
                    dot += y * x[(b * tokens + j) * dim + d];
                }
                d_a[(b * tokens + i) * tokens + j] = dot;
            }
        }
    }
    let mut d_s = vec![0.0f32; d_a.len()];
    for b in 0..batch {
        for i in 0..tokens {
            let mut dot = 0.0;
            for j in 0..tokens {
                let p = probs[(b * tokens + i) * tokens + j];
                dot += p * d_a[(b * tokens + i) * tokens + j];
            }
            for j in 0..tokens {
                let p = probs[(b * tokens + i) * tokens + j];
                d_s[(b * tokens + i) * tokens + j] = p * (d_a[(b * tokens + i) * tokens + j] - dot);
            }
        }
    }
    let mut d_q = vec![0.0f32; x.len()];
    let mut d_k = vec![0.0f32; x.len()];
    for b in 0..batch {
        for i in 0..tokens {
            for j in 0..tokens {
                let ds = d_s[(b * tokens + i) * tokens + j] * scale;
                for d in 0..dim {
                    d_q[(b * tokens + i) * dim + d] += ds * x[(b * tokens + j) * dim + d];
                    d_k[(b * tokens + j) * dim + d] += ds * x[(b * tokens + i) * dim + d];
                }
            }
        }
    }
    let mut dx = d_v;
    for i in 0..dx.len() {
        dx[i] += d_q[i] + d_k[i];
    }
    dx
}

fn finite_difference(
    prog: &Program,
    x: &[f32],
    params: &[f32],
    bump_input: Option<usize>,
    bump_param: Option<usize>,
) -> f32 {
    let eps = 1.0e-3;
    let mut x1 = x.to_vec();
    let mut x0 = x.to_vec();
    let mut p1 = params.to_vec();
    let mut p0 = params.to_vec();
    if let Some(i) = bump_input {
        x1[i] += eps;
        x0[i] -= eps;
    }
    if let Some(i) = bump_param {
        p1[i] += eps;
        p0[i] -= eps;
    }
    let (y1, _, _) = forward_backward(prog, &x1, &p1);
    let (y0, _, _) = forward_backward(prog, &x0, &p0);
    let s1: f32 = y1.iter().sum();
    let s0: f32 = y0.iter().sum();
    (s1 - s0) / (2.0 * eps)
}

/// CUDA source whose `main` runs this program and compares against `expect`.
pub fn emit_cuda(
    prog: &Program,
    x: &[f32],
    params: &[f32],
    expect_y: &[f32],
    expect_dx: &[f32],
    expect_dp: &[f32],
) -> String {
    let mut kernels = String::new();
    let mut fwd = String::new();
    let mut bwd_steps: Vec<String> = Vec::new();
    fwd.push_str("    cudaMemcpy(buf0, h_x, in_bytes, cudaMemcpyHostToDevice);\n");
    fwd.push_str("    cudaMemset(dtape, 0, tape_f32_bytes);\n");
    fwd.push_str("    cudaMemset(dmask, 0, tape_u8_bytes);\n");
    let mut bwd = String::from(
        "    cudaMemset(ddy, 0, max_activ * sizeof(float));\n    std::vector<float> h_dy(out_len, 1.f);\n    cudaMemcpy(ddy, h_dy.data(), out_bytes, cudaMemcpyHostToDevice);\n    cudaMemset(ddp, 0, param_bytes);\n    cudaMemset(dside, 0, tape_f32_bytes);\n",
    );
    // Ping-pong: after forward, the output lives in `cur_is_buf1`.
    let mut cur_is_buf1 = false;
    for (idx, stage) in prog.stages.iter().enumerate() {
        match *stage {
            Stage::Gemm {
                batch,
                din,
                dout,
                relu,
                save_at,
                mask_at,
                param_at,
            } => {
                let src = if cur_is_buf1 { "buf1" } else { "buf0" };
                let dst = if cur_is_buf1 { "buf0" } else { "buf1" };
                kernels.push_str(&gemm_kernels(idx));
                let mut step = String::new();
                fwd.push_str(&format!(
                    "    cudaMemcpy(dtape + {save_at}, {src}, {n} * sizeof(float), cudaMemcpyDeviceToDevice);\n",
                    n = batch * din,
                ));
                fwd.push_str(&format!(
                    "    fwd_gemm_{idx}<<<grid({out}), 128>>>({src}, dparams + {param_at}, dparams + {bias}, {dst}, dmask + {mask_at}, {batch}, {din}, {dout}, {relu});\n",
                    out = batch * dout,
                    bias = param_at + din * dout,
                    relu = relu as i32,
                ));
                step.push_str(&format!(
                    "    if ({relu}) bwd_relu_{idx}<<<grid({out}), 128>>>(ddy, dmask + {mask_at}, {out});\n",
                    out = batch * dout,
                    relu = relu as i32,
                ));
                step.push_str(&format!(
                    "    bwd_dx_{idx}<<<grid({xin}), 128>>>(ddy, dparams + {param_at}, ddx_tmp, {batch}, {din}, {dout});\n",
                    xin = batch * din,
                ));
                step.push_str(&format!(
                    "    bwd_dw_{idx}<<<grid({wn}), 128>>>(dtape + {save_at}, ddy, ddp + {param_at}, {batch}, {din}, {dout});\n",
                    wn = din * dout,
                ));
                step.push_str(&format!(
                    "    bwd_db_{idx}<<<grid({dout}), 128>>>(ddy, ddp + {bias}, {batch}, {dout});\n",
                    bias = param_at + din * dout,
                ));
                step.push_str(&format!(
                    "    add_skip_{idx}<<<grid({xin}), 128>>>(ddx_tmp, dside + {save_at}, {xin});\n",
                    xin = batch * din,
                ));
                step.push_str(&format!(
                    "    cudaMemcpy(ddy, ddx_tmp, {xin} * sizeof(float), cudaMemcpyDeviceToDevice);\n",
                    xin = batch * din,
                ));
                bwd_steps.push(step);
                cur_is_buf1 = !cur_is_buf1;
            }
            Stage::Attention {
                batch,
                tokens,
                dim,
                x_at,
                probs_at,
            } => {
                let src = if cur_is_buf1 { "buf1" } else { "buf0" };
                let dst = if cur_is_buf1 { "buf0" } else { "buf1" };
                let n = batch * tokens * dim;
                kernels.push_str(&attention_kernels(idx));
                let mut step = String::new();
                fwd.push_str(&format!(
                    "    cudaMemcpy(dtape + {x_at}, {src}, {n} * sizeof(float), cudaMemcpyDeviceToDevice);\n",
                ));
                fwd.push_str(&format!(
                    "    fwd_attn_{idx}<<<grid({rows}), 128>>>({src}, dtape + {probs_at}, {dst}, {batch}, {tokens}, {dim});\n",
                    rows = batch * tokens,
                ));
                step.push_str(&format!(
                    "    bwd_attn_{idx}<<<grid({n}), 128>>>(dtape + {x_at}, dtape + {probs_at}, ddy, ddx_tmp, {batch}, {tokens}, {dim});\n",
                ));
                step.push_str(&format!(
                    "    add_skip_{idx}<<<grid({n}), 128>>>(ddx_tmp, dside + {x_at}, {n});\n",
                ));
                step.push_str(&format!(
                    "    cudaMemcpy(ddy, ddx_tmp, {n} * sizeof(float), cudaMemcpyDeviceToDevice);\n",
                ));
                bwd_steps.push(step);
                cur_is_buf1 = !cur_is_buf1;
            }
            Stage::Add { from, len } => {
                let cur = if cur_is_buf1 { "buf1" } else { "buf0" };
                kernels.push_str(&add_kernel(idx));
                fwd.push_str(&format!(
                    "    add_fwd_{idx}<<<grid({len}), 128>>>({cur}, dtape + {from}, {len});\n",
                ));
                bwd_steps.push(format!(
                    "    add_fwd_{idx}<<<grid({len}), 128>>>(dside + {from}, ddy, {len});\n",
                ));
            }
        }
    }
    bwd.push_str(&bwd_steps.into_iter().rev().collect::<String>());
    let y_buf = if cur_is_buf1 { "buf1" } else { "buf0" };
    // After the reverse walk, ddy holds dX but may have been allocated to the
    // maximum width. The input cotangent is the last copy into ddy.
    let src = source_template(
        prog, &kernels, &fwd, &bwd, y_buf, x, params, expect_y, expect_dx, expect_dp,
    );
    src
}

fn grid_helper() -> &'static str {
    "int grid(int n) { return (n + 127) / 128; }\n"
}

fn gemm_kernels(idx: usize) -> String {
    format!(
        r#"
__global__ void fwd_gemm_{idx}(const float* X, const float* W, const float* bias, float* Y, unsigned char* mask,
                                int B, int Din, int Dout, int relu) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Dout) return;
    int m = idx / Dout;
    int n = idx - m * Dout;
    float acc = bias[n];
    for (int k = 0; k < Din; ++k) acc += X[m * Din + k] * W[k * Dout + n];
    if (relu) {{
        mask[idx] = acc > 0.f ? 1 : 0;
        Y[idx] = acc > 0.f ? acc : 0.f;
    }} else {{
        Y[idx] = acc;
    }}
}}
__global__ void bwd_relu_{idx}(float* dY, const unsigned char* mask, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i] == 0) dY[i] = 0.f;
}}
__global__ void bwd_dx_{idx}(const float* dY, const float* W, float* dX, int B, int Din, int Dout) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Din) return;
    int m = idx / Din;
    int k = idx - m * Din;
    float acc = 0.f;
    for (int n = 0; n < Dout; ++n) acc += dY[m * Dout + n] * W[k * Dout + n];
    dX[idx] = acc;
}}
__global__ void bwd_dw_{idx}(const float* X, const float* dY, float* dW, int B, int Din, int Dout) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Din * Dout) return;
    int k = idx / Dout;
    int n = idx - k * Dout;
    float acc = 0.f;
    for (int m = 0; m < B; ++m) acc += X[m * Din + k] * dY[m * Dout + n];
    dW[idx] = acc;
}}
__global__ void bwd_db_{idx}(const float* dY, float* db, int B, int Dout) {{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Dout) return;
    float acc = 0.f;
    for (int m = 0; m < B; ++m) acc += dY[m * Dout + n];
    db[n] = acc;
}}
__global__ void add_skip_{idx}(float* dx, const float* extra, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += extra[i];
}}
"#
    )
}

fn add_kernel(idx: usize) -> String {
    format!(
        r#"
__global__ void add_fwd_{idx}(float* y, const float* extra, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += extra[i];
}}
"#
    )
}

fn attention_kernels(idx: usize) -> String {
    format!(
        r#"
__global__ void fwd_attn_{idx}(const float* X, float* probs, float* Y, int B, int T, int D) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * T) return;
    int b = row / T;
    int i = row - b * T;
    float scale = rsqrtf((float)D);
    float row_s[64];
    float max_s = -1e30f;
    for (int j = 0; j < T; ++j) {{
        float dot = 0.f;
        for (int d = 0; d < D; ++d)
            dot += X[(b * T + i) * D + d] * X[(b * T + j) * D + d];
        row_s[j] = dot * scale;
        max_s = fmaxf(max_s, row_s[j]);
    }}
    float sum = 0.f;
    for (int j = 0; j < T; ++j) {{
        row_s[j] = expf(row_s[j] - max_s);
        sum += row_s[j];
    }}
    for (int j = 0; j < T; ++j) {{
        row_s[j] /= sum;
        probs[(b * T + i) * T + j] = row_s[j];
    }}
    for (int d = 0; d < D; ++d) {{
        float acc = 0.f;
        for (int j = 0; j < T; ++j) acc += row_s[j] * X[(b * T + j) * D + d];
        Y[(b * T + i) * D + d] = acc;
    }}
}}
__global__ void bwd_attn_{idx}(const float* X, const float* probs, const float* dY, float* dX,
                                int B, int T, int D) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B * T * D;
    if (idx >= n) return;
    int tmp = idx;
    int d = tmp % D; tmp /= D;
    int t = tmp % T; tmp /= T;
    int b = tmp;
    float scale = rsqrtf((float)D);
    float dv = 0.f, dq = 0.f, dk = 0.f;
    for (int i = 0; i < T; ++i) {{
        float p = probs[(b * T + i) * T + t];
        dv += p * dY[(b * T + i) * D + d];
    }}
    for (int j = 0; j < T; ++j) {{
        float dot_da = 0.f;
        float mean = 0.f;
        for (int k = 0; k < T; ++k) {{
            float dak = 0.f;
            for (int e = 0; e < D; ++e)
                dak += dY[(b * T + t) * D + e] * X[(b * T + k) * D + e];
            float pk = probs[(b * T + t) * T + k];
            mean += pk * dak;
            if (k == j) dot_da = dak;
        }}
        float p = probs[(b * T + t) * T + j];
        float ds = p * (dot_da - mean) * scale;
        dq += ds * X[(b * T + j) * D + d];
    }}
    for (int i = 0; i < T; ++i) {{
        float dot_da = 0.f;
        float mean = 0.f;
        for (int k = 0; k < T; ++k) {{
            float dak = 0.f;
            for (int e = 0; e < D; ++e)
                dak += dY[(b * T + i) * D + e] * X[(b * T + k) * D + e];
            float pk = probs[(b * T + i) * T + k];
            mean += pk * dak;
            if (k == t) dot_da = dak;
        }}
        float p = probs[(b * T + i) * T + t];
        float ds = p * (dot_da - mean) * scale;
        dk += ds * X[(b * T + i) * D + d];
    }}
    dX[(b * T + t) * D + d] = dv + dq + dk;
}}
__global__ void add_skip_{idx}(float* dx, const float* extra, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += extra[i];
}}
"#
    )
}

fn f32s(xs: &[f32]) -> String {
    xs.iter()
        .map(|v| format!("{v:.8e}f"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn source_template(
    prog: &Program,
    kernels: &str,
    fwd: &str,
    bwd: &str,
    y_buf: &str,
    x: &[f32],
    params: &[f32],
    expect_y: &[f32],
    expect_dx: &[f32],
    expect_dp: &[f32],
) -> String {
    let max_activ = prog
        .stages
        .iter()
        .map(|s| match *s {
            Stage::Gemm {
                batch, din, dout, ..
            } => batch * din.max(dout),
            Stage::Attention {
                batch, tokens, dim, ..
            } => batch * tokens * dim,
            Stage::Add { len, .. } => len,
        })
        .max()
        .unwrap_or(prog.input_len)
        .max(prog.input_len)
        .max(prog.output_len);
    format!(
        r#"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
{grid}
{kernels}
static int fail(const char* msg) {{ printf("FAIL %s\n", msg); return 1; }}
int main() {{
    const int in_len = {in_len};
    const int out_len = {out_len};
    const int param_len = {param_len};
    const int tape_f32 = {tape_f32};
    const int tape_u8 = {tape_u8};
    const int max_activ = {max_activ};
    const size_t in_bytes = in_len * sizeof(float);
    const size_t out_bytes = out_len * sizeof(float);
    const size_t param_bytes = param_len * sizeof(float);
    const size_t tape_f32_bytes = (tape_f32 > 0 ? tape_f32 : 1) * sizeof(float);
    const size_t tape_u8_bytes = (tape_u8 > 0 ? tape_u8 : 1) * sizeof(unsigned char);
    const float h_x[] = {{ {x} }};
    const float h_p[] = {{ {p} }};
    const float h_ey[] = {{ {ey} }};
    const float h_edx[] = {{ {edx} }};
    const float h_edp[] = {{ {edp} }};
    float *buf0, *buf1, *dparams, *dtape, *ddy, *ddx, *ddx_tmp, *ddp, *dside;
    unsigned char* dmask;
    cudaMalloc(&buf0, max_activ * sizeof(float));
    cudaMalloc(&buf1, max_activ * sizeof(float));
    cudaMalloc(&dparams, param_bytes == 0 ? sizeof(float) : param_bytes);
    cudaMalloc(&dtape, tape_f32_bytes);
    cudaMalloc(&dmask, tape_u8_bytes);
    cudaMalloc(&ddy, max_activ * sizeof(float));
    cudaMalloc(&ddx, max_activ * sizeof(float));
    cudaMalloc(&ddx_tmp, max_activ * sizeof(float));
    cudaMalloc(&ddp, param_bytes == 0 ? sizeof(float) : param_bytes);
    cudaMalloc(&dside, tape_f32_bytes);
    cudaMemset(dside, 0, tape_f32_bytes);
    cudaMemcpy(dparams, h_p, param_bytes, cudaMemcpyHostToDevice);
{fwd}
    std::vector<float> y(out_len), dx(in_len), dp(param_len);
    cudaMemcpy(y.data(), {y_buf}, out_bytes, cudaMemcpyDeviceToHost);
{bwd}
    cudaMemcpy(dx.data(), ddy, in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(dp.data(), ddp, param_bytes, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaDeviceSynchronize();
    if (err) return fail(cudaGetErrorString(err));
    auto close = [](float a, float b) {{ return fabsf(a - b) <= 1e-3f + 1e-3f * fabsf(b); }};
    for (int i = 0; i < out_len; ++i) if (!close(y[i], h_ey[i])) {{
        printf("FAIL y[%d] gpu %g cpu %g\n", i, y[i], h_ey[i]); return 1;
    }}
    for (int i = 0; i < in_len; ++i) if (!close(dx[i], h_edx[i])) {{
        printf("FAIL dx[%d] gpu %g cpu %g\n", i, dx[i], h_edx[i]); return 1;
    }}
    for (int i = 0; i < param_len; ++i) if (!close(dp[i], h_edp[i])) {{
        printf("FAIL dp[%d] gpu %g cpu %g\n", i, dp[i], h_edp[i]); return 1;
    }}
    printf("OK {name}\n");
    return 0;
}}
"#,
        grid = grid_helper(),
        kernels = kernels,
        in_len = prog.input_len,
        out_len = prog.output_len,
        param_len = prog.param_len,
        tape_f32 = prog.tape_f32,
        tape_u8 = prog.tape_u8,
        max_activ = max_activ,
        x = f32s(x),
        p = if params.is_empty() {
            "0.f".into()
        } else {
            f32s(params)
        },
        ey = f32s(expect_y),
        edx = f32s(expect_dx),
        edp = if expect_dp.is_empty() {
            "0.f".into()
        } else {
            f32s(expect_dp)
        },
        fwd = fwd,
        bwd = bwd,
        y_buf = y_buf,
        name = prog.name,
    )
}

/// Compile `src` with nvcc and run it. Returns the compiler/runtime output.
pub fn nvcc_run(src: &str, name: &str) -> Result<String, String> {
    let dir = std::env::temp_dir();
    let cu = dir.join(format!("{name}.cu"));
    let bin = dir.join(name);
    std::fs::write(&cu, src).map_err(|e| e.to_string())?;
    let compile = std::process::Command::new("/usr/local/cuda/bin/nvcc")
        .args([
            "-arch=sm_75",
            "-ccbin",
            "g++-11",
            "-O2",
            cu.to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !compile.status.success() {
        return Err(format!(
            "nvcc failed\n{}{}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr)
        ));
    }
    let run = std::process::Command::new(&bin)
        .output()
        .map_err(|e| e.to_string())?;
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    if !run.status.success() {
        return Err(out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_grad(prog: &Program) {
        let x = init(prog.input_len, 0.5);
        let params = init(prog.param_len, 0.25);
        let (_y, dx, dp) = forward_backward(prog, &x, &params);
        for i in 0..prog.input_len {
            let num = finite_difference(prog, &x, &params, Some(i), None);
            let err = (num - dx[i]).abs();
            assert!(
                err < 2e-2,
                "{} dX[{i}] analytic {} fd {num}",
                prog.name,
                dx[i]
            );
        }
        for i in 0..prog.param_len {
            let num = finite_difference(prog, &x, &params, None, Some(i));
            let err = (num - dp[i]).abs();
            assert!(
                err < 2e-2,
                "{} dP[{i}] analytic {} fd {num}",
                prog.name,
                dp[i]
            );
        }
    }

    #[test]
    fn residuals_store_only_backward_inputs() {
        let mlp = Program::mlp(2, &[3, 4, 2]);
        // First linear saves the 2×3 input and a 2×4 ReLU mask. Second linear
        // saves the 2×4 hidden state and no mask. The outputs are not stored.
        assert_eq!(mlp.tape_f32, 2 * 3 + 2 * 4);
        assert_eq!(mlp.tape_u8, 2 * 4);
        assert_eq!(mlp.param_len, 3 * 4 + 4 + 4 * 2 + 2);

        let block = Program::resnet_block(2, 4, 6);
        // W1 saved x (2×4). The skip add does not store another copy.
        // W2 saved the hidden state (2×6).
        assert_eq!(block.tape_f32, 2 * 4 + 2 * 6);
        assert_eq!(block.tape_u8, 2 * 6);
        assert!(matches!(
            block.stages.last(),
            Some(Stage::Add { from: 0, .. })
        ));

        let tr = Program::transformer_block(2, 3, 4, 8);
        let n = 2 * 3 * 4;
        // Attention stores tokens and softmax, the MLP stores its input and hidden.
        assert_eq!(tr.tape_f32, n + 2 * 3 * 3 + n + 2 * 3 * 8);
        assert_eq!(tr.tape_u8, 2 * 3 * 8);
    }

    #[test]
    fn mlp_resnet_and_transformer_match_finite_differences() {
        check_grad(&Program::mlp(2, &[3, 4, 2]));
        check_grad(&Program::resnet_block(2, 3, 4));
        check_grad(&Program::transformer_block(1, 3, 4, 5));
    }

    #[test]
    fn nvidia_kernels_match_the_lens() {
        let prog = Program::mlp(2, &[3, 4, 2]);
        let x = init(prog.input_len, 0.5);
        let params = init(prog.param_len, 0.25);
        let (y, dx, dp) = forward_backward(&prog, &x, &params);
        let src = emit_cuda(&prog, &x, &params, &y, &dx, &dp);
        let out = nvcc_run(&src, "depopt_mlp").expect("nvcc");
        assert!(out.contains("OK mlp"), "{out}");

        let prog = Program::resnet_block(2, 3, 4);
        let x = init(prog.input_len, 0.5);
        let params = init(prog.param_len, 0.25);
        let (y, dx, dp) = forward_backward(&prog, &x, &params);
        let src = emit_cuda(&prog, &x, &params, &y, &dx, &dp);
        let out = nvcc_run(&src, "depopt_resnet").expect("nvcc");
        assert!(out.contains("OK resnet_block"), "{out}");

        let prog = Program::transformer_block(1, 3, 4, 5);
        let x = init(prog.input_len, 0.5);
        let params = init(prog.param_len, 0.25);
        let (y, dx, dp) = forward_backward(&prog, &x, &params);
        let src = emit_cuda(&prog, &x, &params, &y, &dx, &dp);
        let out = nvcc_run(&src, "depopt_transformer").expect("nvcc");
        assert!(out.contains("OK transformer_block"), "{out}");
    }
}
