//! Mixtral 8x7B as a dependent lens, with the adjoint required for SGD.
//!
//! Mixtral (Apache-2.0, Mistral AI, arXiv:2401.04088) is a decoder-only
//! transformer with Mistral's attention stack and a sparse feed-forward block:
//! RMSNorm, grouped-query attention, rotary positions, a sliding causal window,
//! then eight SwiGLU experts of which the router keeps two. The reference
//! formulas are the ones in Hugging Face `modeling_mixtral`:
//!
//! - RMSNorm is `x * rsqrt(mean(x²) + ε) * weight`
//! - RoPE pairs the two halves of each head (`rotate_half`)
//! - the router softmaxes over every expert, keeps the top two probabilities,
//!   and renormalizes those two
//! - each expert is `down(silu(x W_gate) * x W_up)`
//! - residuals are added around attention and around the mixture
//! - the language-model head is untied
//!
//! Each of those maps is a lens. The forward pass writes the tensors the
//! backward pass reads (the pre-norm activation and its `rsqrt`, the pre-RoPE
//! queries and keys, the values, the attention probabilities, the mixture
//! weights, and the selected experts' gate and up projections). Parameters
//! that the backward pass still has in hand are not copied into the tape.
//! Cross-entropy against the target tokens is the loss whose adjoint is the
//! stochastic gradient.
//!
//! [`MixtralConfig::mixtral_8x7b`] is the published shape (about 46.7 billion
//! parameters). [`MixtralConfig::demo`] is the same block at a width that fits
//! on one GPU. [`emit_cuda`] lowers a cluster stage plan to a CUDA program and
//! [`crate::kernel::nvcc_run`] compiles and runs it. Each window runs on the
//! CPU or GPU the plan assigned. One SGD step on those gradients has to
//! decrease the loss.

/// Hyperparameters of one Mixtral model. `sliding_window` is the number of
/// keys a query may attend to, including itself.
#[derive(Clone, Debug)]
pub struct MixtralConfig {
    pub vocab: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
}

impl MixtralConfig {
    /// Published Mixtral-8x7B-v0.1 configuration.
    pub fn mixtral_8x7b() -> Self {
        Self {
            vocab: 32_000,
            dim: 4096,
            layers: 32,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 14_336,
            experts: 8,
            top_k: 2,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window: 4096,
        }
    }

    /// Same block, small enough to differentiate and launch.
    pub fn demo() -> Self {
        Self {
            vocab: 32,
            dim: 32,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            intermediate: 48,
            experts: 4,
            top_k: 2,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window: 4,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn group(&self) -> usize {
        assert!(self.heads % self.kv_heads == 0);
        self.heads / self.kv_heads
    }
}

/// Byte-free parameter layout. Every matrix is row-major with the contraction
/// index first: `W[k * dout + n]`.
#[derive(Clone, Debug)]
pub struct Layout {
    pub embed: usize,
    pub layers: Vec<LayerLayout>,
    pub final_norm: usize,
    pub lm_head: usize,
    pub total: usize,
}

#[derive(Clone, Debug)]
pub struct LayerLayout {
    pub attn_norm: usize,
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub o: usize,
    pub ffn_norm: usize,
    pub router: usize,
    pub gate: Vec<usize>,
    pub up: Vec<usize>,
    pub down: Vec<usize>,
}

pub fn layout(cfg: &MixtralConfig) -> Layout {
    let mut at = 0usize;
    let embed = at;
    at += cfg.vocab * cfg.dim;
    let mut layers = Vec::with_capacity(cfg.layers);
    for _ in 0..cfg.layers {
        let attn_norm = at;
        at += cfg.dim;
        let q = at;
        at += cfg.dim * cfg.q_dim();
        let k = at;
        at += cfg.dim * cfg.kv_dim();
        let v = at;
        at += cfg.dim * cfg.kv_dim();
        let o = at;
        at += cfg.q_dim() * cfg.dim;
        let ffn_norm = at;
        at += cfg.dim;
        let router = at;
        at += cfg.dim * cfg.experts;
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for _e in 0..cfg.experts {
            gate.push(at);
            at += cfg.dim * cfg.intermediate;
            up.push(at);
            at += cfg.dim * cfg.intermediate;
            down.push(at);
            at += cfg.intermediate * cfg.dim;
        }
        layers.push(LayerLayout {
            attn_norm,
            q,
            k,
            v,
            o,
            ffn_norm,
            router,
            gate,
            up,
            down,
        });
    }
    let final_norm = at;
    at += cfg.dim;
    let lm_head = at;
    at += cfg.dim * cfg.vocab;
    Layout {
        embed,
        layers,
        final_norm,
        lm_head,
        total: at,
    }
}

pub fn parameter_count(cfg: &MixtralConfig) -> u64 {
    layout(cfg).total as u64
}

/// Positive, expert-separated weights. The router columns increase with the
/// expert index so top-2 routing has a margin and does not flip under a small
/// finite-difference step.
pub fn init_params(cfg: &MixtralConfig) -> Vec<f32> {
    let n = layout(cfg).total;
    (0..n)
        .map(|i| {
            let base = 0.02 * ((i % 5) as f32 + 1.0);
            base
        })
        .collect()
}

pub fn init_router(cfg: &MixtralConfig, params: &mut [f32]) {
    let lay = layout(cfg);
    for layer in &lay.layers {
        for d in 0..cfg.dim {
            for e in 0..cfg.experts {
                params[layer.router + d * cfg.experts + e] =
                    0.05 * (e as f32 + 1.0) + 0.001 * (d as f32);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct LayerTape {
    attn_in: Vec<f32>,
    attn_inv: Vec<f32>,
    q_pre: Vec<f32>,
    k_pre: Vec<f32>,
    v: Vec<f32>,
    probs: Vec<f32>,
    mix: Vec<f32>,
    ffn_in: Vec<f32>,
    ffn_inv: Vec<f32>,
    router_probs: Vec<f32>,
    top_i: Vec<usize>,
    top_p: Vec<f32>,
    top_w: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Tape {
    layers: Vec<LayerTape>,
    final_in: Vec<f32>,
    final_inv: Vec<f32>,
}

/// Mean next-token cross-entropy and the parameter adjoint.
pub struct LossGrad {
    pub loss: f32,
    pub grad: Vec<f32>,
    pub logits: Vec<f32>,
}

pub fn forward_backward(
    cfg: &MixtralConfig,
    params: &[f32],
    tokens: &[usize],
    targets: &[usize],
    batch: usize,
    seq: usize,
) -> LossGrad {
    assert_eq!(tokens.len(), batch * seq);
    assert_eq!(targets.len(), batch * seq);
    let lay = layout(cfg);
    assert_eq!(params.len(), lay.total);
    let (logits, tape) = forward(cfg, &lay, params, tokens, batch, seq);
    let (loss, dlogits) = cross_entropy(&logits, targets, batch, seq, cfg.vocab);
    let grad = backward(cfg, &lay, params, &tape, tokens, &dlogits, batch, seq);
    LossGrad { loss, grad, logits }
}

fn forward(
    cfg: &MixtralConfig,
    lay: &Layout,
    params: &[f32],
    tokens: &[usize],
    batch: usize,
    seq: usize,
) -> (Vec<f32>, Tape) {
    let n = batch * seq;
    let mut x = embed(cfg, &params[lay.embed..], tokens);
    let mut layers = Vec::new();
    for layer in &lay.layers {
        let (y, tape) = forward_layer(cfg, layer, params, &x, batch, seq);
        x = y;
        layers.push(tape);
    }
    let (final_h, final_inv) = rmsnorm(&x, &params[lay.final_norm..], n, cfg.dim, cfg.rms_eps);
    let logits = linear(&final_h, &params[lay.lm_head..], n, cfg.dim, cfg.vocab);
    (
        logits,
        Tape {
            layers,
            final_in: x,
            final_inv,
        },
    )
}

fn forward_layer(
    cfg: &MixtralConfig,
    layer: &LayerLayout,
    params: &[f32],
    x: &[f32],
    batch: usize,
    seq: usize,
) -> (Vec<f32>, LayerTape) {
    let n = batch * seq;
    let (xn, attn_inv) = rmsnorm(x, &params[layer.attn_norm..], n, cfg.dim, cfg.rms_eps);
    let q_pre = linear(&xn, &params[layer.q..], n, cfg.dim, cfg.q_dim());
    let k_pre = linear(&xn, &params[layer.k..], n, cfg.dim, cfg.kv_dim());
    let v = linear(&xn, &params[layer.v..], n, cfg.dim, cfg.kv_dim());
    let q = rope_qkv(cfg, &q_pre, batch, seq, cfg.heads);
    let k = rope_qkv(cfg, &k_pre, batch, seq, cfg.kv_heads);
    let (probs, mix) = gqa_forward(cfg, &q, &k, &v, batch, seq);
    let attn = linear(&mix, &params[layer.o..], n, cfg.q_dim(), cfg.dim);
    let mut mid = x.to_vec();
    add_assign(&mut mid, &attn);

    let (xn2, ffn_inv) = rmsnorm(&mid, &params[layer.ffn_norm..], n, cfg.dim, cfg.rms_eps);
    let router_logits = linear(&xn2, &params[layer.router..], n, cfg.dim, cfg.experts);
    let mut router_probs = vec![0.0; n * cfg.experts];
    let mut top_i = vec![0usize; n * cfg.top_k];
    let mut top_p = vec![0.0; n * cfg.top_k];
    let mut top_w = vec![0.0; n * cfg.top_k];
    for t in 0..n {
        let row = softmax(&router_logits[t * cfg.experts..(t + 1) * cfg.experts]);
        router_probs[t * cfg.experts..(t + 1) * cfg.experts].copy_from_slice(&row);
        let (idx, p, w) = topk_renorm(&row, cfg.top_k);
        for k in 0..cfg.top_k {
            top_i[t * cfg.top_k + k] = idx[k];
            top_p[t * cfg.top_k + k] = p[k];
            top_w[t * cfg.top_k + k] = w[k];
        }
    }
    let mut gate = vec![0.0; n * cfg.top_k * cfg.intermediate];
    let mut up = vec![0.0; n * cfg.top_k * cfg.intermediate];
    let mut mixed = vec![0.0; n * cfg.dim];
    for t in 0..n {
        for k in 0..cfg.top_k {
            let e = top_i[t * cfg.top_k + k];
            let g = linear(
                &xn2[t * cfg.dim..(t + 1) * cfg.dim],
                &params[layer.gate[e]..],
                1,
                cfg.dim,
                cfg.intermediate,
            );
            let u = linear(
                &xn2[t * cfg.dim..(t + 1) * cfg.dim],
                &params[layer.up[e]..],
                1,
                cfg.dim,
                cfg.intermediate,
            );
            let hid = silu_mul(&g, &u);
            let out = linear(&hid, &params[layer.down[e]..], 1, cfg.intermediate, cfg.dim);
            let w = top_w[t * cfg.top_k + k];
            for d in 0..cfg.dim {
                mixed[t * cfg.dim + d] += w * out[d];
            }
            let base = (t * cfg.top_k + k) * cfg.intermediate;
            gate[base..base + cfg.intermediate].copy_from_slice(&g);
            up[base..base + cfg.intermediate].copy_from_slice(&u);
        }
    }
    let mut y = mid.clone();
    add_assign(&mut y, &mixed);
    (
        y,
        LayerTape {
            attn_in: x.to_vec(),
            attn_inv,
            q_pre,
            k_pre,
            v,
            probs,
            mix,
            ffn_in: mid,
            ffn_inv,
            router_probs,
            top_i,
            top_p,
            top_w,
            gate,
            up,
        },
    )
}

fn backward(
    cfg: &MixtralConfig,
    lay: &Layout,
    params: &[f32],
    tape: &Tape,
    tokens: &[usize],
    dlogits: &[f32],
    batch: usize,
    seq: usize,
) -> Vec<f32> {
    let n = batch * seq;
    let _ = n;
    let mut grad = vec![0.0f32; lay.total];
    let (mut dx, dhead) = linear_bwd(
        &rmsnorm_out(
            &tape.final_in,
            &tape.final_inv,
            &params[lay.final_norm..],
            n,
            cfg.dim,
        ),
        dlogits,
        &params[lay.lm_head..],
        n,
        cfg.dim,
        cfg.vocab,
    );
    add_assign(
        &mut grad[lay.lm_head..lay.lm_head + cfg.dim * cfg.vocab],
        &dhead,
    );
    let (dx_norm, dweight) = rmsnorm_bwd(
        &dx,
        &tape.final_in,
        &tape.final_inv,
        &params[lay.final_norm..],
        n,
        cfg.dim,
    );
    dx = dx_norm;
    add_assign(
        &mut grad[lay.final_norm..lay.final_norm + cfg.dim],
        &dweight,
    );

    for (layer, lt) in lay.layers.iter().zip(tape.layers.iter()).rev() {
        dx = backward_layer(cfg, layer, params, lt, &dx, &mut grad, batch, seq);
    }
    embed_bwd(cfg, &mut grad[lay.embed..], tokens, &dx);
    grad
}

fn backward_layer(
    cfg: &MixtralConfig,
    layer: &LayerLayout,
    params: &[f32],
    lt: &LayerTape,
    dy: &[f32],
    grad: &mut [f32],
    batch: usize,
    seq: usize,
) -> Vec<f32> {
    let n = batch * seq;
    // Second residual: both the mixture and the skip receive `dy`.
    let d_mixed = dy;

    let xn2 = rmsnorm_out(
        &lt.ffn_in,
        &lt.ffn_inv,
        &params[layer.ffn_norm..],
        n,
        cfg.dim,
    );
    let mut d_xn2 = vec![0.0f32; n * cfg.dim];
    for t in 0..n {
        let mut dvalues = vec![0.0f32; cfg.top_k];
        let mut weight_dot = 0.0f32;
        let s: f32 = (0..cfg.top_k).map(|k| lt.top_p[t * cfg.top_k + k]).sum();
        for k in 0..cfg.top_k {
            let e = lt.top_i[t * cfg.top_k + k];
            let base = (t * cfg.top_k + k) * cfg.intermediate;
            let g = &lt.gate[base..base + cfg.intermediate];
            let u = &lt.up[base..base + cfg.intermediate];
            let hid = silu_mul(g, u);
            let out = linear(&hid, &params[layer.down[e]..], 1, cfg.intermediate, cfg.dim);
            let mut dw = 0.0f32;
            for d in 0..cfg.dim {
                dw += d_mixed[t * cfg.dim + d] * out[d];
            }
            weight_dot += dw * lt.top_w[t * cfg.top_k + k];
            dvalues[k] = dw;
            let w = lt.top_w[t * cfg.top_k + k];
            let mut d_out = vec![0.0f32; cfg.dim];
            for d in 0..cfg.dim {
                d_out[d] = w * d_mixed[t * cfg.dim + d];
            }
            let (d_hid, d_down) = linear_bwd(
                &hid,
                &d_out,
                &params[layer.down[e]..],
                1,
                cfg.intermediate,
                cfg.dim,
            );
            add_assign(
                &mut grad[layer.down[e]..layer.down[e] + cfg.intermediate * cfg.dim],
                &d_down,
            );
            let (d_gate, d_up) = silu_mul_bwd(g, u, &d_hid);
            let xrow = &xn2[t * cfg.dim..(t + 1) * cfg.dim];
            let (dx_g, dw_g) = linear_bwd(
                xrow,
                &d_gate,
                &params[layer.gate[e]..],
                1,
                cfg.dim,
                cfg.intermediate,
            );
            let (dx_u, dw_u) = linear_bwd(
                xrow,
                &d_up,
                &params[layer.up[e]..],
                1,
                cfg.dim,
                cfg.intermediate,
            );
            add_assign(
                &mut grad[layer.gate[e]..layer.gate[e] + cfg.dim * cfg.intermediate],
                &dw_g,
            );
            add_assign(
                &mut grad[layer.up[e]..layer.up[e] + cfg.dim * cfg.intermediate],
                &dw_u,
            );
            for d in 0..cfg.dim {
                d_xn2[t * cfg.dim + d] += dx_g[d] + dx_u[d];
            }
        }
        for k in 0..cfg.top_k {
            dvalues[k] = dvalues[k] / s - weight_dot / s;
        }
        let mut dp = vec![0.0f32; cfg.experts];
        for k in 0..cfg.top_k {
            dp[lt.top_i[t * cfg.top_k + k]] = dvalues[k];
        }
        let p = &lt.router_probs[t * cfg.experts..(t + 1) * cfg.experts];
        let dot: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
        let mut dz = vec![0.0f32; cfg.experts];
        for e in 0..cfg.experts {
            dz[e] = p[e] * (dp[e] - dot);
        }
        let xrow = &xn2[t * cfg.dim..(t + 1) * cfg.dim];
        let (dx_r, dw_r) = linear_bwd(xrow, &dz, &params[layer.router..], 1, cfg.dim, cfg.experts);
        add_assign(
            &mut grad[layer.router..layer.router + cfg.dim * cfg.experts],
            &dw_r,
        );
        for d in 0..cfg.dim {
            d_xn2[t * cfg.dim + d] += dx_r[d];
        }
    }
    let (d_mid, d_ffn_w) = rmsnorm_bwd(
        &d_xn2,
        &lt.ffn_in,
        &lt.ffn_inv,
        &params[layer.ffn_norm..],
        n,
        cfg.dim,
    );
    add_assign(
        &mut grad[layer.ffn_norm..layer.ffn_norm + cfg.dim],
        &d_ffn_w,
    );
    let mut d_mid = d_mid;
    // Skip around the mixture.
    add_assign(&mut d_mid, dy);

    // Attention residual. `d_mid` is the cotangent of the attention block output.
    let (d_mix, d_o) = linear_bwd(&lt.mix, &d_mid, &params[layer.o..], n, cfg.q_dim(), cfg.dim);
    add_assign(&mut grad[layer.o..layer.o + cfg.q_dim() * cfg.dim], &d_o);
    let q = rope_qkv(cfg, &lt.q_pre, batch, seq, cfg.heads);
    let k = rope_qkv(cfg, &lt.k_pre, batch, seq, cfg.kv_heads);
    let (dq, dk, dv) = gqa_backward(cfg, &q, &k, &lt.v, &lt.probs, &d_mix, batch, seq);
    let dq = rope_qkv_transpose(cfg, &dq, batch, seq, cfg.heads);
    let dk = rope_qkv_transpose(cfg, &dk, batch, seq, cfg.kv_heads);
    let xn = rmsnorm_out(
        &lt.attn_in,
        &lt.attn_inv,
        &params[layer.attn_norm..],
        n,
        cfg.dim,
    );
    let (dx_q, dw_q) = linear_bwd(&xn, &dq, &params[layer.q..], n, cfg.dim, cfg.q_dim());
    let (dx_k, dw_k) = linear_bwd(&xn, &dk, &params[layer.k..], n, cfg.dim, cfg.kv_dim());
    let (dx_v, dw_v) = linear_bwd(&xn, &dv, &params[layer.v..], n, cfg.dim, cfg.kv_dim());
    add_assign(&mut grad[layer.q..layer.q + cfg.dim * cfg.q_dim()], &dw_q);
    add_assign(&mut grad[layer.k..layer.k + cfg.dim * cfg.kv_dim()], &dw_k);
    add_assign(&mut grad[layer.v..layer.v + cfg.dim * cfg.kv_dim()], &dw_v);
    let mut d_xn = dx_q;
    add_assign(&mut d_xn, &dx_k);
    add_assign(&mut d_xn, &dx_v);
    let (mut dx, d_attn_w) = rmsnorm_bwd(
        &d_xn,
        &lt.attn_in,
        &lt.attn_inv,
        &params[layer.attn_norm..],
        n,
        cfg.dim,
    );
    add_assign(
        &mut grad[layer.attn_norm..layer.attn_norm + cfg.dim],
        &d_attn_w,
    );
    // `x_in + attn_out = mid`, so the skip carries the same cotangent as the branch.
    add_assign(&mut dx, &d_mid);
    dx
}

fn embed(cfg: &MixtralConfig, table: &[f32], tokens: &[usize]) -> Vec<f32> {
    let mut y = vec![0.0; tokens.len() * cfg.dim];
    for (t, &id) in tokens.iter().enumerate() {
        y[t * cfg.dim..(t + 1) * cfg.dim].copy_from_slice(&table[id * cfg.dim..(id + 1) * cfg.dim]);
    }
    y
}

fn embed_bwd(cfg: &MixtralConfig, dtable: &mut [f32], tokens: &[usize], dx: &[f32]) {
    for (t, &id) in tokens.iter().enumerate() {
        for d in 0..cfg.dim {
            dtable[id * cfg.dim + d] += dx[t * cfg.dim + d];
        }
    }
}

fn linear(x: &[f32], w: &[f32], n: usize, din: usize, dout: usize) -> Vec<f32> {
    let mut y = vec![0.0; n * dout];
    for m in 0..n {
        for o in 0..dout {
            let mut acc = 0.0;
            for k in 0..din {
                acc += x[m * din + k] * w[k * dout + o];
            }
            y[m * dout + o] = acc;
        }
    }
    y
}

fn linear_bwd(
    x: &[f32],
    dy: &[f32],
    w: &[f32],
    n: usize,
    din: usize,
    dout: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; n * din];
    let mut dw = vec![0.0; din * dout];
    for m in 0..n {
        for k in 0..din {
            let mut acc = 0.0;
            for o in 0..dout {
                acc += dy[m * dout + o] * w[k * dout + o];
            }
            dx[m * din + k] = acc;
        }
    }
    for k in 0..din {
        for o in 0..dout {
            let mut acc = 0.0;
            for m in 0..n {
                acc += x[m * din + k] * dy[m * dout + o];
            }
            dw[k * dout + o] = acc;
        }
    }
    (dx, dw)
}

fn add_assign(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

fn rmsnorm(x: &[f32], w: &[f32], n: usize, dim: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0; n * dim];
    let mut inv = vec![0.0; n];
    for t in 0..n {
        let mut mean = 0.0;
        for i in 0..dim {
            let v = x[t * dim + i];
            mean += v * v;
        }
        mean = mean / dim as f32 + eps;
        let r = mean.sqrt().recip();
        inv[t] = r;
        for i in 0..dim {
            y[t * dim + i] = x[t * dim + i] * r * w[i];
        }
    }
    (y, inv)
}

fn rmsnorm_out(x: &[f32], inv: &[f32], w: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0.0; n * dim];
    for t in 0..n {
        for i in 0..dim {
            y[t * dim + i] = x[t * dim + i] * inv[t] * w[i];
        }
    }
    y
}

fn rmsnorm_bwd(
    dy: &[f32],
    x: &[f32],
    inv: &[f32],
    w: &[f32],
    n: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; n * dim];
    let mut dw = vec![0.0; dim];
    for t in 0..n {
        let r = inv[t];
        let mut dot = 0.0;
        for i in 0..dim {
            let xhat = x[t * dim + i] * r;
            let dxhat = dy[t * dim + i] * w[i];
            dw[i] += dy[t * dim + i] * xhat;
            dot += dxhat * xhat;
        }
        let mean = dot / dim as f32;
        for i in 0..dim {
            let xhat = x[t * dim + i] * r;
            let dxhat = dy[t * dim + i] * w[i];
            dx[t * dim + i] = r * (dxhat - xhat * mean);
        }
    }
    (dx, dw)
}

fn rope_angles(cfg: &MixtralConfig, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = cfg.head_dim / 2;
    let mut cos = vec![0.0; cfg.head_dim];
    let mut sin = vec![0.0; cfg.head_dim];
    for i in 0..half {
        let freq = cfg.rope_theta.powf(-(2.0 * i as f32) / cfg.head_dim as f32);
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        cos[i] = c;
        cos[half + i] = c;
        sin[i] = s;
        sin[half + i] = s;
    }
    (cos, sin)
}

fn rope_one(q: &[f32], cos: &[f32], sin: &[f32], sign: f32) -> Vec<f32> {
    let hd = q.len();
    let half = hd / 2;
    let mut y = vec![0.0; hd];
    for i in 0..half {
        let q1 = q[i];
        let q2 = q[half + i];
        let s = sign * sin[i];
        y[i] = q1 * cos[i] - q2 * s;
        y[half + i] = q2 * cos[half + i] + q1 * s;
    }
    y
}

fn rope_qkv(
    cfg: &MixtralConfig,
    src: &[f32],
    batch: usize,
    seq: usize,
    n_heads: usize,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let mut y = vec![0.0; src.len()];
    for b in 0..batch {
        for t in 0..seq {
            let (cos, sin) = rope_angles(cfg, t);
            for h in 0..n_heads {
                let base = ((b * seq + t) * n_heads + h) * hd;
                let out = rope_one(&src[base..base + hd], &cos, &sin, 1.0);
                y[base..base + hd].copy_from_slice(&out);
            }
        }
    }
    y
}

fn rope_qkv_transpose(
    cfg: &MixtralConfig,
    src: &[f32],
    batch: usize,
    seq: usize,
    n_heads: usize,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let mut y = vec![0.0; src.len()];
    for b in 0..batch {
        for t in 0..seq {
            let (cos, sin) = rope_angles(cfg, t);
            for h in 0..n_heads {
                let base = ((b * seq + t) * n_heads + h) * hd;
                let out = rope_one(&src[base..base + hd], &cos, &sin, -1.0);
                y[base..base + hd].copy_from_slice(&out);
            }
        }
    }
    y
}

fn visible(cfg: &MixtralConfig, query: usize, key: usize) -> bool {
    key <= query && query - key < cfg.sliding_window
}

fn gqa_forward(
    cfg: &MixtralConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    let hd = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let group = cfg.group();
    let mut probs = vec![0.0; batch * cfg.heads * seq * seq];
    let mut mix = vec![0.0; batch * seq * cfg.q_dim()];
    for b in 0..batch {
        for h in 0..cfg.heads {
            let kv = h / group;
            for i in 0..seq {
                let mut row = vec![f32::NEG_INFINITY; seq];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..seq {
                    if !visible(cfg, i, j) {
                        continue;
                    }
                    let mut dot = 0.0;
                    for d in 0..hd {
                        let qv = q[((b * seq + i) * cfg.heads + h) * hd + d];
                        let kvv = k[((b * seq + j) * cfg.kv_heads + kv) * hd + d];
                        dot += qv * kvv;
                    }
                    row[j] = dot * scale;
                    max_s = max_s.max(row[j]);
                }
                let mut sum = 0.0;
                for j in 0..seq {
                    if row[j].is_finite() {
                        row[j] = (row[j] - max_s).exp();
                        sum += row[j];
                    } else {
                        row[j] = 0.0;
                    }
                }
                for j in 0..seq {
                    row[j] /= sum;
                    probs[((b * cfg.heads + h) * seq + i) * seq + j] = row[j];
                }
                for d in 0..hd {
                    let mut acc = 0.0;
                    for j in 0..seq {
                        acc += row[j] * v[((b * seq + j) * cfg.kv_heads + kv) * hd + d];
                    }
                    mix[((b * seq + i) * cfg.heads + h) * hd + d] = acc;
                }
            }
        }
    }
    (probs, mix)
}

fn gqa_backward(
    cfg: &MixtralConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    dmix: &[f32],
    batch: usize,
    seq: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = cfg.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let group = cfg.group();
    let mut dq = vec![0.0; q.len()];
    let mut dk = vec![0.0; k.len()];
    let mut dv = vec![0.0; v.len()];
    for b in 0..batch {
        for h in 0..cfg.heads {
            let kvh = h / group;
            for i in 0..seq {
                let mut da = vec![0.0; seq];
                for j in 0..seq {
                    if probs[((b * cfg.heads + h) * seq + i) * seq + j] == 0.0
                        && !visible(cfg, i, j)
                    {
                        continue;
                    }
                    let mut dot = 0.0;
                    for d in 0..hd {
                        dot += dmix[((b * seq + i) * cfg.heads + h) * hd + d]
                            * v[((b * seq + j) * cfg.kv_heads + kvh) * hd + d];
                    }
                    da[j] = dot;
                    let p = probs[((b * cfg.heads + h) * seq + i) * seq + j];
                    for d in 0..hd {
                        dv[((b * seq + j) * cfg.kv_heads + kvh) * hd + d] +=
                            p * dmix[((b * seq + i) * cfg.heads + h) * hd + d];
                    }
                }
                let mut dot_pa = 0.0;
                for j in 0..seq {
                    let p = probs[((b * cfg.heads + h) * seq + i) * seq + j];
                    dot_pa += p * da[j];
                }
                for j in 0..seq {
                    if !visible(cfg, i, j) {
                        continue;
                    }
                    let p = probs[((b * cfg.heads + h) * seq + i) * seq + j];
                    let ds = p * (da[j] - dot_pa) * scale;
                    for d in 0..hd {
                        dq[((b * seq + i) * cfg.heads + h) * hd + d] +=
                            ds * k[((b * seq + j) * cfg.kv_heads + kvh) * hd + d];
                        dk[((b * seq + j) * cfg.kv_heads + kvh) * hd + d] +=
                            ds * q[((b * seq + i) * cfg.heads + h) * hd + d];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / s).collect()
}

fn topk_renorm(probs: &[f32], k: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    let p: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
    let s: f32 = p.iter().sum();
    let w: Vec<f32> = p.iter().map(|v| v / s).collect();
    (idx, p, w)
}

fn silu(z: f32) -> f32 {
    z * sigmoid(z)
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

fn silu_mul_bwd(gate: &[f32], up: &[f32], dhid: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut dg = vec![0.0; gate.len()];
    let mut du = vec![0.0; up.len()];
    for i in 0..gate.len() {
        let z = gate[i];
        let s = sigmoid(z);
        let dsilu = s * (1.0 + z * (1.0 - s));
        dg[i] = dhid[i] * up[i] * dsilu;
        du[i] = dhid[i] * silu(z);
    }
    (dg, du)
}

fn cross_entropy(
    logits: &[f32],
    targets: &[usize],
    batch: usize,
    seq: usize,
    vocab: usize,
) -> (f32, Vec<f32>) {
    let n = batch * seq;
    let mut d = vec![0.0; n * vocab];
    let mut loss = 0.0;
    for t in 0..n {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row {
            sum += (v - m).exp();
        }
        let lse = m + sum.ln();
        loss += lse - row[targets[t]];
        for v in 0..vocab {
            d[t * vocab + v] = ((row[v] - m).exp() / sum) / n as f32;
        }
        d[t * vocab + targets[t]] -= 1.0 / n as f32;
    }
    (loss / n as f32, d)
}

/// One SGD step `p ← p − lr · ∇p`. Returns the Euclidean norm of `grad`.
pub fn sgd_step(params: &mut [f32], grad: &[f32], lr: f32) -> f32 {
    let mut norm = 0.0f32;
    for (p, g) in params.iter_mut().zip(grad) {
        norm += g * g;
        *p -= lr * g;
    }
    norm.sqrt()
}

pub fn demo_batch(cfg: &MixtralConfig) -> (usize, usize, Vec<usize>, Vec<usize>) {
    let batch = 2;
    let seq = 8;
    let tokens: Vec<usize> = (0..batch * seq).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let targets: Vec<usize> = tokens.iter().map(|t| (t + 3) % cfg.vocab).collect();
    (batch, seq, tokens, targets)
}

/// Step size of the demo's one-step descent check.
pub const DEMO_LR: f32 = 0.05;

/// Parameter indices printed by the CUDA program and compared with the Rust adjoint.
pub fn demo_probes(cfg: &MixtralConfig) -> Vec<usize> {
    let lay = layout(cfg);
    assert!(cfg.layers >= 2, "demo probes use two layers");
    assert!(cfg.experts >= 4, "demo probes use four experts");
    let a = &lay.layers[0];
    let b = &lay.layers[1];
    let last = cfg.experts - 1;
    let probes = vec![
        lay.embed + 3,
        a.attn_norm + 1,
        a.q + 10,
        a.k + 4,
        a.v + 6,
        a.o + 4,
        a.ffn_norm + 2,
        a.router + 5,
        a.gate[0] + 1,
        a.up[last] + 3,
        a.down[2] + 5,
        b.gate[last] + 7,
        lay.final_norm + 2,
        lay.lm_head + 9,
    ];
    assert!(a.q + 10 < a.k);
    assert!(a.k + 4 < a.v);
    assert!(a.v + 6 < a.o);
    assert!(a.o + 4 < a.ffn_norm);
    assert!(a.ffn_norm + 2 < a.router);
    assert!(a.router + 5 < a.gate[0]);
    assert!(a.gate[0] + 1 < a.up[0]);
    assert!(a.up[last] + 3 < a.down[last]);
    assert!(a.down[2] + 5 < a.down[2] + cfg.intermediate * cfg.dim);
    assert!(b.gate[last] + 7 < b.up[last]);
    assert!(lay.final_norm + 2 < lay.lm_head);
    assert!(lay.lm_head + 9 < lay.total);
    probes
}

fn window_step(window: &crate::cluster::Window) -> String {
    format!(
        "{{{dev}, {home}, {pass}, {exec}, {off}ull, {b0}ull, {b1}ull, {r0}ull, {r1}ull, {c0}ull, {c1}ull, {din}ull, {dout}ull, {layer}, {expert}}}",
        dev = window.device,
        home = window.home,
        pass = if window.pass == crate::cluster::Pass::Fwd { 0 } else { 1 },
        exec = window.exec as u8,
        off = window.param_offset,
        b0 = window.batch_begin,
        b1 = window.batch_end,
        r0 = window.row_begin,
        r1 = window.row_end,
        c0 = window.col_begin,
        c1 = window.col_end,
        din = window.din,
        dout = window.dout,
        layer = window.layer,
        expert = window.expert,
    )
}

fn hop_step(hop: &crate::cluster::Hop) -> String {
    format!(
        "{{{to}, {from}, 2, 14, {off}ull, {elems}ull, {bytes}ull, {r0}ull, {r1}ull, {c0}ull, {c1}ull, 0ull, {stride}ull, {kind}, 0}}",
        to = hop.to,
        from = hop.from,
        off = hop.offset,
        elems = hop.elems,
        bytes = hop.bytes,
        r0 = hop.row_begin,
        r1 = hop.row_end,
        c0 = hop.col_begin,
        c1 = hop.col_end,
        stride = hop.stride,
        kind = hop.kind as u8,
    )
}

fn csv(values: impl IntoIterator<Item = usize>) -> String {
    values
        .into_iter()
        .map(|v| format!("{v}ull"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn ull(v: usize) -> String {
    format!("{v}ull")
}

fn c_f32(v: f32) -> String {
    if v.to_bits() == 1e-5f32.to_bits() {
        "1e-5f".to_string()
    } else if v.to_bits() == 1_000_000f32.to_bits() {
        "1000000.f".to_string()
    } else if v.to_bits() == DEMO_LR.to_bits() {
        "0.05f".to_string()
    } else {
        format!("{v:.9e}f")
    }
}

/// CUDA source that executes the staged schedule for `cluster`. Parameter
/// homes stay on the device the plan chose. Each window runs there, and a
/// GPU window is a real kernel launch plus the transfers of its tile.
pub fn emit_cuda(
    cfg: &MixtralConfig,
    batch: usize,
    seq: usize,
    probes: &[usize],
    cluster: &crate::cluster::Cluster,
) -> String {
    assert!(cfg.layers >= 1 && cfg.experts >= 1 && cfg.top_k >= 1);
    assert!(batch >= 1 && seq >= 1);
    assert!(cfg.head_dim >= 2 && cfg.head_dim % 2 == 0);
    assert!(cfg.heads >= 1 && cfg.kv_heads >= 1 && cfg.heads % cfg.kv_heads == 0);
    assert!(cfg.vocab >= 1 && cfg.dim >= 1 && cfg.intermediate >= 1);
    assert!(!probes.is_empty());
    assert!(!cluster.devices.is_empty());
    let lay = layout(cfg);
    for &p in probes {
        assert!(p < lay.total, "probe {p} outside {}", lay.total);
    }
    let compiled = crate::cluster::compile_mixtral(cfg, batch as u64, seq as u64, cluster)
        .unwrap_or_else(|err| panic!("cluster staging failed: {err:?}"));
    let mut lines = Vec::new();
    for window in &compiled.plan.windows {
        for hop in &window.inbound {
            lines.push(hop_step(hop));
        }
        lines.push(window_step(window));
        for hop in &window.outbound {
            lines.push(hop_step(hop));
        }
    }
    let steps = lines.join(",\n");
    let kinds = cluster
        .devices
        .iter()
        .map(|d| {
            if d.kind == crate::cluster::DeviceKind::Gpu {
                "1"
            } else {
                "0"
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let scratch = compiled
        .plan
        .windows
        .iter()
        .map(|w| w.resident_bytes)
        .max()
        .unwrap_or(1)
        .max(1);
    let gate: Vec<Vec<usize>> = lay.layers.iter().map(|l| l.gate.clone()).collect();
    let up: Vec<Vec<usize>> = lay.layers.iter().map(|l| l.up.clone()).collect();
    let down: Vec<Vec<usize>> = lay.layers.iter().map(|l| l.down.clone()).collect();
    let rows = |groups: &[Vec<usize>]| {
        groups
            .iter()
            .map(|g| format!("{{ {} }}", csv(g.iter().copied())))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut src = include_str!("staged_mixtral.cu.tpl").to_string();
    let pairs = [
        ("@@PROBES@@", csv(probes.iter().copied())),
        ("@@STEPS@@", steps),
        ("@@KINDS@@", kinds),
        ("@@ROUTER@@", csv(lay.layers.iter().map(|l| l.router))),
        ("@@GATE@@", rows(&gate)),
        ("@@DOWN@@", rows(&down)),
        ("@@VOFF@@", csv(lay.layers.iter().map(|l| l.v))),
        ("@@TOPK@@", ull(cfg.top_k)),
        ("@@SCRATCH@@", ull(scratch as usize)),
        ("@@WINDOW@@", ull(cfg.sliding_window)),
        ("@@THETA@@", c_f32(cfg.rope_theta)),
        ("@@NDEV@@", cluster.devices.len().to_string()),
        ("@@SEQ@@", ull(seq)),
        ("@@EPS@@", c_f32(cfg.rms_eps)),
        ("@@KV@@", ull(cfg.kv_heads)),
        ("@@HD@@", ull(cfg.head_dim)),
        ("@@UP@@", rows(&up)),
        ("@@LM@@", ull(lay.lm_head)),
        ("@@LR@@", c_f32(DEMO_LR)),
        ("@@Q@@", csv(lay.layers.iter().map(|l| l.q))),
        ("@@K@@", csv(lay.layers.iter().map(|l| l.k))),
        ("@@O@@", csv(lay.layers.iter().map(|l| l.o))),
        ("@@L@@", ull(cfg.layers)),
        ("@@H@@", ull(cfg.heads)),
        ("@@E@@", ull(cfg.experts)),
        ("@@I@@", ull(cfg.intermediate)),
        ("@@D@@", ull(cfg.dim)),
        ("@@V@@", ull(cfg.vocab)),
        ("@@B@@", ull(batch)),
        ("@@P@@", ull(lay.total)),
    ];
    for (key, value) in pairs {
        assert!(src.contains(key), "template is missing {key}");
        src = src.replace(key, &value);
    }
    if let Some(i) = src.find("@@") {
        let end = (i + 24).min(src.len());
        panic!("unreplaced placeholder near {}", &src[i..end]);
    }
    src
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MixtralConfig {
        MixtralConfig::demo()
    }

    fn params() -> Vec<f32> {
        let c = cfg();
        let mut p = init_params(&c);
        init_router(&c, &mut p);
        p
    }

    #[test]
    fn published_mixtral_8x7b_parameter_count() {
        let n = parameter_count(&MixtralConfig::mixtral_8x7b());
        assert!(
            (46_500_000_000..47_000_000_000).contains(&n),
            "published count {n}"
        );
    }

    #[test]
    fn adjoint_matches_finite_differences_and_unselected_experts_get_no_gradient() {
        let c = cfg();
        let lay = layout(&c);
        let mut p = params();
        let (batch, seq, tokens, targets) = demo_batch(&c);
        let before = forward_backward(&c, &p, &tokens, &targets, batch, seq);
        let eps = 1e-3;
        let probes = [
            lay.embed + 3,
            lay.layers[0].attn_norm + 1,
            lay.layers[0].q + 10,
            lay.layers[0].o + 4,
            lay.layers[1].gate[3] + 7,
            lay.layers[0].down[2] + 5,
            lay.final_norm + 2,
            lay.lm_head + 9,
        ];
        for &i in &probes {
            let mut plus = p.clone();
            let mut minus = p.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let lp = forward_backward(&c, &plus, &tokens, &targets, batch, seq).loss;
            let lm = forward_backward(&c, &minus, &tokens, &targets, batch, seq).loss;
            let num = (lp - lm) / (2.0 * eps);
            let err = (num - before.grad[i]).abs();
            assert!(
                err < 2e-2,
                "param {i}: analytic {} fd {num}",
                before.grad[i]
            );
        }

        // An expert that no token selected has a zero parameter adjoint.
        let (logits_unused, tape) = {
            let result = forward_backward(&c, &p, &tokens, &targets, batch, seq);
            // Recompute the tape by a second forward through the public loss.
            let _ = result;
            let (_b, _s, tokens, targets) = demo_batch(&c);
            let lay = layout(&c);
            let (logits, tape) = super::forward(&c, &lay, &p, &tokens, batch, seq);
            let _ = targets;
            (logits, tape)
        };
        let _ = logits_unused;
        let n = batch * seq;
        let mut used = vec![false; c.experts];
        for lt in &tape.layers {
            for t in 0..n {
                for k in 0..c.top_k {
                    used[lt.top_i[t * c.top_k + k]] = true;
                }
            }
        }
        assert!(
            used.iter().any(|u| *u) && used.iter().any(|u| !*u),
            "demo routing should be sparse: {used:?}"
        );
        for (e, &on) in used.iter().enumerate() {
            if on {
                continue;
            }
            let start = lay.layers[0].gate[e];
            let stop = lay.layers[0].down[e] + c.intermediate * c.dim;
            let mass: f32 = before.grad[start..stop].iter().map(|g| g.abs()).sum();
            assert!(mass < 1e-6, "unselected expert {e} grad mass {mass}");
        }

        let loss0 = before.loss;
        let mut step = p.clone();
        let gnorm = sgd_step(&mut step, &before.grad, DEMO_LR);
        assert!(gnorm.is_finite() && gnorm > 0.0, "gradient norm {gnorm}");
        let loss1 = forward_backward(&c, &step, &tokens, &targets, batch, seq).loss;
        assert!(loss1 < loss0, "SGD should descend: {loss0} -> {loss1}");
        // Keep `p` so the assertion above used the original point.
        p.copy_from_slice(&step);
        let _ = p;
    }

    fn field(out: &str, key: &str) -> f64 {
        out.lines()
            .find_map(|line| {
                let mut parts = line.split_whitespace();
                (parts.next() == Some(key)).then(|| parts.next().unwrap().parse::<f64>().unwrap())
            })
            .unwrap_or_else(|| panic!("missing {key}\n{out}"))
    }

    fn close(name: &str, got: f64, expect: f64) {
        let tol = 1e-3 + 1e-3 * expect.abs();
        let err = (got - expect).abs();
        assert!(
            err < tol,
            "{name}: gpu {got:.8e} cpu {expect:.8e} err {err:.8e} tol {tol:.8e}"
        );
    }

    fn run_cluster() -> crate::cluster::Cluster {
        use crate::cluster::{Cluster, Device, DeviceKind, CPU_RUNTIME_BYTES};
        Cluster::devices(vec![
            Device {
                name: "cpu0".into(),
                kind: DeviceKind::Cpu,
                buffer_bytes: 64 << 20,
                code_bytes: CPU_RUNTIME_BYTES,
            },
            Device {
                name: "gpu0".into(),
                kind: DeviceKind::Gpu,
                buffer_bytes: 16 << 10,
                code_bytes: 1 << 20,
            },
        ])
    }

    #[test]
    fn cuda_source_is_specialized_to_the_demo() {
        let c = cfg();
        let (batch, seq, _, _) = demo_batch(&c);
        let src = emit_cuda(&c, batch, seq, &demo_probes(&c), &run_cluster());
        assert!(!src.contains("@@"), "unreplaced placeholder");
        assert!(src.contains("constexpr ull D = 32ull;"));
        assert!(src.contains("constexpr ull E = 4ull;"));
        assert!(src.contains("constexpr ull L = 2ull;"));
        assert!(src.contains("constexpr ull SEQ = 8ull;"));
        assert!(src.contains("constexpr float LR = 0.05f;"));
        assert!(src.contains(&format!("constexpr ull P = {}ull;", layout(&c).total)));
        assert!(src.contains("const Step STEPS[]"));
        assert!(src.contains("k_gqa_fwd"));
        assert!(src.contains("k_gqa_bwd"));
        assert!(src.contains("k_rmsnorm_bwd"));
        assert!(src.contains("k_sgd"));
        assert!(src.contains("k_router"));
        assert!(src.contains("k_silu_bwd"));
    }

    #[test]
    fn published_mixtral_compiles_at_full_parameter_count() {
        use crate::cluster::{Cluster, Device, DeviceKind, CPU_RUNTIME_BYTES, KERNEL_CODE_BYTES};
        let c = MixtralConfig::mixtral_8x7b();
        let n = parameter_count(&c);
        assert!(n > i32::MAX as u64, "published count {n}");
        let lay = layout(&c);
        let cluster = Cluster::devices(vec![
            Device {
                name: "cpu0".into(),
                kind: DeviceKind::Cpu,
                buffer_bytes: 256 << 30,
                code_bytes: CPU_RUNTIME_BYTES,
            },
            Device {
                name: "gpu0".into(),
                kind: DeviceKind::Gpu,
                buffer_bytes: 6 << 30,
                code_bytes: KERNEL_CODE_BYTES,
            },
        ]);
        let src = emit_cuda(&c, 1, 1, &[0, lay.lm_head, lay.total - 1], &cluster);
        assert!(src.contains(&format!("constexpr ull P = {n}ull;")), "published length missing");
        crate::kernel::nvcc_compile(&src, "mixtral_8x7b").unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn nvidia_mixtral_matches_the_adjoint() {
        let c = cfg();
        let lay = layout(&c);
        let p = params();
        let (batch, seq, tokens, targets) = demo_batch(&c);
        let before = forward_backward(&c, &p, &tokens, &targets, batch, seq);
        let mut stepped = p.clone();
        let gnorm = sgd_step(&mut stepped, &before.grad, DEMO_LR);
        let loss1 = forward_backward(&c, &stepped, &tokens, &targets, batch, seq).loss;
        let (logits, tape) = super::forward(&c, &lay, &p, &tokens, batch, seq);
        assert!((logits.iter().sum::<f32>() - before.logits.iter().sum::<f32>()).abs() < 1e-5);

        let probes = demo_probes(&c);
        let cluster = run_cluster();
        let src = emit_cuda(&c, batch, seq, &probes, &cluster);
        let out =
            crate::kernel::nvcc_run(&src, "mixtral_staged").unwrap_or_else(|err| panic!("{err}"));
        assert!(!out.lines().any(|line| line.starts_with("FAIL")), "{out}");
        assert!(out.contains("OK mixtral"), "{out}");
        assert!(field(&out, "GPUOPS") > 0.0, "schedule did not launch on the GPU\n{out}");
        assert!(field(&out, "XFERS") > 0.0, "schedule did not move a tile\n{out}");

        assert_eq!(field(&out, "PCOUNT") as usize, lay.total);
        close("loss0", field(&out, "LOSS0"), before.loss as f64);
        close("loss1", field(&out, "LOSS1"), loss1 as f64);
        assert!(field(&out, "LOSS1") < field(&out, "LOSS0"), "{out}");

        let gsum: f64 = before.grad.iter().map(|v| *v as f64).sum();
        let gabs: f64 = before.grad.iter().map(|v| (*v as f64).abs()).sum();
        close("gsum", field(&out, "GSUM"), gsum);
        close("gabs", field(&out, "GABS"), gabs);
        close("gnorm", field(&out, "GNORM"), gnorm as f64);
        let _ = tape;

        let mut gpu_grad = std::collections::BTreeMap::new();
        for line in out.lines() {
            let mut parts = line.split_whitespace();
            if parts.next() != Some("GRAD") {
                continue;
            }
            let index: usize = parts.next().unwrap().parse().unwrap();
            let value: f64 = parts.next().unwrap().parse().unwrap();
            gpu_grad.insert(index, value);
        }
        for &index in &probes {
            close(
                &format!("grad {index}"),
                gpu_grad[&index],
                before.grad[index] as f64,
            );
        }
    }
}
