// Mixtral demo lens. Constants are substituted from mixtral::layout.
// The forward pass writes the residual the backward pass reads. The backward
// pass returns the parameter cotangent, and main takes one SGD step.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int grid(int n) { return n <= 0 ? 1 : (n + 127) / 128; }

void must(const char* stage) {
    cudaError_t launch = cudaGetLastError();
    if (launch) {
        printf("FAIL %s %s\n", stage, cudaGetErrorString(launch));
        exit(1);
    }
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync) {
        printf("FAIL %s %s\n", stage, cudaGetErrorString(sync));
        exit(1);
    }
}

void d2d(void* dst, const void* src, size_t bytes) {
    cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (e) {
        printf("FAIL memcpy %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

float* fbuf(size_t n) {
    float* p = nullptr;
    size_t bytes = (n ? n : 1) * sizeof(float);
    if (cudaMalloc(&p, bytes)) {
        printf("FAIL malloc\n");
        exit(1);
    }
    cudaMemset(p, 0, bytes);
    return p;
}

int* ibuf(size_t n) {
    int* p = nullptr;
    size_t bytes = (n ? n : 1) * sizeof(int);
    if (cudaMalloc(&p, bytes)) {
        printf("FAIL malloc\n");
        exit(1);
    }
    cudaMemset(p, 0, bytes);
    return p;
}

constexpr int B = @@B@@;
constexpr int SEQ = @@SEQ@@;
constexpr int DIM = @@DIM@@;
constexpr int NLAYER = @@NLAYER@@;
constexpr int NHEAD = @@NHEAD@@;
constexpr int NKV = @@NKV@@;
constexpr int HD = @@HD@@;
constexpr int INTER = @@INTER@@;
constexpr int NEXP = @@NEXP@@;
constexpr int TOPK = @@TOPK@@;
constexpr int VOCAB = @@VOCAB@@;
constexpr int WINDOW = @@WINDOW@@;
constexpr float EPS = @@EPS@@;
constexpr float THETA = @@THETA@@;
constexpr float LR = @@LR@@;
constexpr int N = B * SEQ;
constexpr int QDIM = NHEAD * HD;
constexpr int KVDIM = NKV * HD;
constexpr int P = @@P@@;

static_assert(SEQ <= 64, "one attention row stays in a fixed array");
static_assert(HD <= 128 && HD % 2 == 0, "rotate_half");
static_assert(NEXP <= 32 && TOPK <= 8 && TOPK >= 1, "router scratch");
static_assert(NHEAD % NKV == 0, "grouped-query attention");
static_assert(NLAYER >= 1 && N >= 1 && DIM >= 1, "empty model");

const int ATTN_NORM[NLAYER] = { @@ATTN_NORM@@ };
const int QOFF[NLAYER] = { @@QOFF@@ };
const int KOFF[NLAYER] = { @@KOFF@@ };
const int VOFF[NLAYER] = { @@VOFF@@ };
const int OOFF[NLAYER] = { @@OOFF@@ };
const int FFN_NORM[NLAYER] = { @@FFN_NORM@@ };
const int ROUTER[NLAYER] = { @@ROUTER@@ };
const int GATE[NLAYER][NEXP] = { @@GATE@@ };
const int UP[NLAYER][NEXP] = { @@UP@@ };
const int DOWN[NLAYER][NEXP] = { @@DOWN@@ };
constexpr int EMBED = @@EMBED@@;
constexpr int FINAL_NORM = @@FINAL@@;
constexpr int LM_HEAD = @@LM@@;
const int PROBES[] = { @@PROBES@@ };
const int NPROBES = sizeof(PROBES) / sizeof(PROBES[0]);

__global__ void k_embed(const float* table, const int* tokens, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM) return;
    int t = i / DIM;
    int d = i - t * DIM;
    y[i] = table[tokens[t] * DIM + d];
}

__global__ void k_embed_bwd(float* table, const int* tokens, const float* dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM) return;
    int t = i / DIM;
    int d = i - t * DIM;
    atomicAdd(table + tokens[t] * DIM + d, dx[i]);
}

__global__ void k_rmsnorm(const float* x, const float* w, float* y, float* inv) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    float mean = 0.f;
    for (int i = 0; i < DIM; ++i) {
        float v = x[t * DIM + i];
        mean += v * v;
    }
    mean = mean / DIM + EPS;
    float r = 1.f / sqrtf(mean);
    inv[t] = r;
    for (int i = 0; i < DIM; ++i) y[t * DIM + i] = x[t * DIM + i] * r * w[i];
}

__global__ void k_rmsnorm_apply(const float* x, const float* inv, const float* w, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM) return;
    int t = i / DIM;
    int d = i - t * DIM;
    y[i] = x[i] * inv[t] * w[d];
}

__global__ void k_rmsnorm_bwd(const float* dy, const float* x, const float* inv, const float* w, float* dx, float* dw) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    float r = inv[t];
    float dot = 0.f;
    for (int i = 0; i < DIM; ++i) {
        float xhat = x[t * DIM + i] * r;
        float dxhat = dy[t * DIM + i] * w[i];
        atomicAdd(dw + i, dy[t * DIM + i] * xhat);
        dot += dxhat * xhat;
    }
    float mean = dot / DIM;
    for (int i = 0; i < DIM; ++i) {
        float xhat = x[t * DIM + i] * r;
        float dxhat = dy[t * DIM + i] * w[i];
        dx[t * DIM + i] = r * (dxhat - xhat * mean);
    }
}

__global__ void k_linear(const float* X, const float* W, float* Y, int Din, int Dout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Dout) return;
    int m = idx / Dout;
    int o = idx - m * Dout;
    float acc = 0.f;
    for (int k = 0; k < Din; ++k) acc += X[m * Din + k] * W[k * Dout + o];
    Y[idx] = acc;
}

__global__ void k_linear_dx(const float* dY, const float* W, float* dX, int Din, int Dout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * Din) return;
    int m = idx / Din;
    int k = idx - m * Din;
    float acc = 0.f;
    for (int o = 0; o < Dout; ++o) acc += dY[m * Dout + o] * W[k * Dout + o];
    dX[idx] = acc;
}

__global__ void k_linear_dw(const float* X, const float* dY, float* dW, int Din, int Dout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Din * Dout) return;
    int k = idx / Dout;
    int o = idx - k * Dout;
    float acc = 0.f;
    for (int m = 0; m < N; ++m) acc += X[m * Din + k] * dY[m * Dout + o];
    dW[idx] = acc;
}

__global__ void k_add(float* y, const float* extra, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += extra[i];
}

__global__ void k_rope(float* q, int heads, int sign_neg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * heads) return;
    int h = idx % heads;
    int tmp = idx / heads;
    int t = tmp % SEQ;
    int b = tmp / SEQ;
    int half = HD / 2;
    float sign = sign_neg ? -1.f : 1.f;
    float* row = q + ((b * SEQ + t) * heads + h) * HD;
    float out[HD];
    for (int i = 0; i < half; ++i) {
        float freq = powf(THETA, -(2.f * i) / (float)HD);
        float ang = t * freq;
        float c = cosf(ang);
        float s = sign * sinf(ang);
        float q1 = row[i];
        float q2 = row[half + i];
        out[i] = q1 * c - q2 * s;
        out[half + i] = q2 * c + q1 * s;
    }
    for (int i = 0; i < HD; ++i) row[i] = out[i];
}

__global__ void k_gqa_fwd(const float* q, const float* k, const float* v, float* probs, float* mix) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * NHEAD * SEQ) return;
    int i = row % SEQ;
    int tmp = row / SEQ;
    int h = tmp % NHEAD;
    int b = tmp / NHEAD;
    int kv = h / (NHEAD / NKV);
    float scale = 1.f / sqrtf((float)HD);
    float scores[SEQ];
    float maxs = -INFINITY;
    for (int j = 0; j < SEQ; ++j) {
        bool vis = j <= i && (i - j) < WINDOW;
        if (!vis) {
            scores[j] = -INFINITY;
            continue;
        }
        float dot = 0.f;
        for (int d = 0; d < HD; ++d) {
            float qv = q[((b * SEQ + i) * NHEAD + h) * HD + d];
            float kvv = k[((b * SEQ + j) * NKV + kv) * HD + d];
            dot += qv * kvv;
        }
        scores[j] = dot * scale;
        maxs = fmaxf(maxs, scores[j]);
    }
    float sum = 0.f;
    for (int j = 0; j < SEQ; ++j) {
        if (j <= i && (i - j) < WINDOW) {
            scores[j] = expf(scores[j] - maxs);
            sum += scores[j];
        } else {
            scores[j] = 0.f;
        }
    }
    for (int j = 0; j < SEQ; ++j) {
        scores[j] /= sum;
        probs[((b * NHEAD + h) * SEQ + i) * SEQ + j] = scores[j];
    }
    for (int d = 0; d < HD; ++d) {
        float acc = 0.f;
        for (int j = 0; j < SEQ; ++j) acc += scores[j] * v[((b * SEQ + j) * NKV + kv) * HD + d];
        mix[((b * SEQ + i) * NHEAD + h) * HD + d] = acc;
    }
}

__global__ void k_gqa_bwd(const float* q, const float* k, const float* v, const float* probs,
                          const float* dmix, float* dq, float* dk, float* dv) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * NHEAD * SEQ) return;
    int i = row % SEQ;
    int tmp = row / SEQ;
    int h = tmp % NHEAD;
    int b = tmp / NHEAD;
    int kv = h / (NHEAD / NKV);
    float scale = 1.f / sqrtf((float)HD);
    float da[SEQ];
    for (int j = 0; j < SEQ; ++j) {
        float dot = 0.f;
        const float* dm = dmix + ((b * SEQ + i) * NHEAD + h) * HD;
        const float* vv = v + ((b * SEQ + j) * NKV + kv) * HD;
        for (int d = 0; d < HD; ++d) dot += dm[d] * vv[d];
        da[j] = dot;
        float p = probs[((b * NHEAD + h) * SEQ + i) * SEQ + j];
        for (int d = 0; d < HD; ++d) atomicAdd(dv + ((b * SEQ + j) * NKV + kv) * HD + d, p * dm[d]);
    }
    float dot_pa = 0.f;
    for (int j = 0; j < SEQ; ++j) dot_pa += probs[((b * NHEAD + h) * SEQ + i) * SEQ + j] * da[j];
    for (int j = 0; j < SEQ; ++j) {
        if (!(j <= i && (i - j) < WINDOW)) continue;
        float p = probs[((b * NHEAD + h) * SEQ + i) * SEQ + j];
        float ds = p * (da[j] - dot_pa) * scale;
        for (int d = 0; d < HD; ++d) {
            dq[((b * SEQ + i) * NHEAD + h) * HD + d] += ds * k[((b * SEQ + j) * NKV + kv) * HD + d];
            atomicAdd(dk + ((b * SEQ + j) * NKV + kv) * HD + d, ds * q[((b * SEQ + i) * NHEAD + h) * HD + d]);
        }
    }
}

__global__ void k_router(const float* logits, float* probs, int* idx, float* top_p, float* top_w) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    float m = -INFINITY;
    for (int e = 0; e < NEXP; ++e) m = fmaxf(m, logits[t * NEXP + e]);
    float sum = 0.f;
    for (int e = 0; e < NEXP; ++e) {
        float p = expf(logits[t * NEXP + e] - m);
        probs[t * NEXP + e] = p;
        sum += p;
    }
    for (int e = 0; e < NEXP; ++e) probs[t * NEXP + e] /= sum;
    for (int k = 0; k < TOPK; ++k) {
        int best = -1;
        float bv = -1.f;
        for (int e = 0; e < NEXP; ++e) {
            bool taken = false;
            for (int j = 0; j < k; ++j) if (idx[t * TOPK + j] == e) taken = true;
            if (taken) continue;
            float p = probs[t * NEXP + e];
            if (best < 0 || p > bv || (p == bv && e < best)) {
                best = e;
                bv = p;
            }
        }
        idx[t * TOPK + k] = best;
        top_p[t * TOPK + k] = bv;
    }
    float s = 0.f;
    for (int k = 0; k < TOPK; ++k) s += top_p[t * TOPK + k];
    for (int k = 0; k < TOPK; ++k) top_w[t * TOPK + k] = top_p[t * TOPK + k] / s;
}

__global__ void k_mix_experts(const float* outs, const int* idx, const float* top_w, float* mixed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM) return;
    int t = i / DIM;
    int d = i - t * DIM;
    float acc = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        int e = idx[t * TOPK + k];
        acc += top_w[t * TOPK + k] * outs[(e * N + t) * DIM + d];
    }
    mixed[i] = acc;
}

__global__ void k_silu_mul(const float* gate, const float* up, float* hid, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = gate[i];
    float s = 1.f / (1.f + expf(-z));
    hid[i] = z * s * up[i];
}

__global__ void k_expert_mask(const float* dy, const int* idx, const float* top_w, float* dout, int expert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * DIM) return;
    int t = i / DIM;
    float scale = 0.f;
    for (int k = 0; k < TOPK; ++k) if (idx[t * TOPK + k] == expert) scale += top_w[t * TOPK + k];
    dout[i] = scale * dy[i];
}

__global__ void k_router_bwd(const float* dy, const float* xn, const float* probs, const int* idx,
                             const float* top_p, const float* top_w, const float* expert_out,
                             const float* wrouter, float* d_xn, float* drouter) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    float dw[TOPK];
    float s = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        s += top_p[t * TOPK + k];
        int e = idx[t * TOPK + k];
        float dot = 0.f;
        for (int d = 0; d < DIM; ++d) dot += dy[t * DIM + d] * expert_out[(e * N + t) * DIM + d];
        dw[k] = dot;
    }
    float weight_dot = 0.f;
    for (int k = 0; k < TOPK; ++k) weight_dot += dw[k] * top_w[t * TOPK + k];
    float dp[NEXP];
    for (int e = 0; e < NEXP; ++e) dp[e] = 0.f;
    for (int k = 0; k < TOPK; ++k) dp[idx[t * TOPK + k]] = dw[k] / s - weight_dot / s;
    float dotp = 0.f;
    for (int e = 0; e < NEXP; ++e) dotp += probs[t * NEXP + e] * dp[e];
    for (int e = 0; e < NEXP; ++e) {
        float dz = probs[t * NEXP + e] * (dp[e] - dotp);
        for (int d = 0; d < DIM; ++d) {
            atomicAdd(drouter + d * NEXP + e, xn[t * DIM + d] * dz);
            atomicAdd(d_xn + t * DIM + d, wrouter[d * NEXP + e] * dz);
        }
    }
}

__global__ void k_silu_bwd(const float* gate, const float* up, const float* dhid, float* dgate, float* dup, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = gate[i];
    float s = 1.f / (1.f + expf(-z));
    float dsilu = s * (1.f + z * (1.f - s));
    dgate[i] = dhid[i] * up[i] * dsilu;
    dup[i] = dhid[i] * (z * s);
}

__global__ void k_ce(const float* logits, const int* targets, float* dlogits, float* loss) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    const float* row = logits + t * VOCAB;
    float m = -INFINITY;
    for (int v = 0; v < VOCAB; ++v) m = fmaxf(m, row[v]);
    float sum = 0.f;
    for (int v = 0; v < VOCAB; ++v) sum += expf(row[v] - m);
    atomicAdd(loss, (m + logf(sum) - row[targets[t]]) / N);
    for (int v = 0; v < VOCAB; ++v) dlogits[t * VOCAB + v] = expf(row[v] - m) / sum / N;
    dlogits[t * VOCAB + targets[t]] -= 1.f / N;
}

__global__ void k_sgd(float* p, const float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] -= LR * g[i];
}

struct Ws {
    float *x, *xn, *q, *k, *v, *probs, *mix, *attn, *mixed;
    float *gate, *up, *hid, *eout, *logits, *rlogits, *rprobs, *top_p, *top_w;
    int *idx, *tokens, *targets;
    float *inv_attn, *inv_ffn, *saved_x, *saved_mid, *q_pre, *k_pre, *v_pre;
    float *final_in, *final_inv, *dlogits, *dx, *d_xn, *tmp, *dmix, *dq, *dk, *dv;
    float *dgate, *dup, *dhid, *dout, *loss;
};

void alloc(Ws& w) {
    w.x = fbuf(N * DIM);
    w.xn = fbuf(N * DIM);
    w.q = fbuf(N * QDIM);
    w.k = fbuf(N * KVDIM);
    w.v = fbuf(N * KVDIM);
    w.probs = fbuf((size_t)NLAYER * B * NHEAD * SEQ * SEQ);
    w.mix = fbuf((size_t)NLAYER * N * QDIM);
    w.attn = fbuf(N * DIM);
    w.mixed = fbuf(N * DIM);
    w.gate = fbuf((size_t)NLAYER * NEXP * N * INTER);
    w.up = fbuf((size_t)NLAYER * NEXP * N * INTER);
    w.hid = fbuf((size_t)NEXP * N * INTER);
    w.eout = fbuf((size_t)NEXP * N * DIM);
    w.logits = fbuf(N * VOCAB);
    w.rlogits = fbuf(N * NEXP);
    w.rprobs = fbuf((size_t)NLAYER * N * NEXP);
    w.top_p = fbuf((size_t)NLAYER * N * TOPK);
    w.top_w = fbuf((size_t)NLAYER * N * TOPK);
    w.idx = ibuf((size_t)NLAYER * N * TOPK);
    w.tokens = ibuf(N);
    w.targets = ibuf(N);
    w.inv_attn = fbuf((size_t)NLAYER * N);
    w.inv_ffn = fbuf((size_t)NLAYER * N);
    w.saved_x = fbuf((size_t)NLAYER * N * DIM);
    w.saved_mid = fbuf((size_t)NLAYER * N * DIM);
    w.q_pre = fbuf((size_t)NLAYER * N * QDIM);
    w.k_pre = fbuf((size_t)NLAYER * N * KVDIM);
    w.v_pre = fbuf((size_t)NLAYER * N * KVDIM);
    w.final_in = fbuf(N * DIM);
    w.final_inv = fbuf(N);
    w.dlogits = fbuf(N * VOCAB);
    w.dx = fbuf(N * DIM);
    w.d_xn = fbuf(N * DIM);
    w.tmp = fbuf(N * DIM);
    w.dmix = fbuf(N * QDIM);
    w.dq = fbuf(N * QDIM);
    w.dk = fbuf(N * KVDIM);
    w.dv = fbuf(N * KVDIM);
    w.dgate = fbuf(N * INTER);
    w.dup = fbuf(N * INTER);
    w.dhid = fbuf(N * INTER);
    w.dout = fbuf(N * DIM);
    w.loss = fbuf(1);
}

void forward_layers(float* params, Ws& w) {
    k_embed<<<grid(N * DIM), 128>>>(params + EMBED, w.tokens, w.x);
    for (int li = 0; li < NLAYER; ++li) {
        d2d(w.saved_x + li * N * DIM, w.x, N * DIM * sizeof(float));
        k_rmsnorm<<<grid(N), 128>>>(w.x, params + ATTN_NORM[li], w.xn, w.inv_attn + li * N);
        k_linear<<<grid(N * QDIM), 128>>>(w.xn, params + QOFF[li], w.q, DIM, QDIM);
        k_linear<<<grid(N * KVDIM), 128>>>(w.xn, params + KOFF[li], w.k, DIM, KVDIM);
        k_linear<<<grid(N * KVDIM), 128>>>(w.xn, params + VOFF[li], w.v, DIM, KVDIM);
        d2d(w.q_pre + (size_t)li * N * QDIM, w.q, (size_t)N * QDIM * sizeof(float));
        d2d(w.k_pre + (size_t)li * N * KVDIM, w.k, (size_t)N * KVDIM * sizeof(float));
        d2d(w.v_pre + (size_t)li * N * KVDIM, w.v, (size_t)N * KVDIM * sizeof(float));
        k_rope<<<grid(N * NHEAD), 128>>>(w.q, NHEAD, 0);
        k_rope<<<grid(N * NKV), 128>>>(w.k, NKV, 0);
        k_gqa_fwd<<<grid(B * NHEAD * SEQ), 128>>>(w.q, w.k, w.v, w.probs + (size_t)li * B * NHEAD * SEQ * SEQ, w.mix + (size_t)li * N * QDIM);
        k_linear<<<grid(N * DIM), 128>>>(w.mix + (size_t)li * N * QDIM, params + OOFF[li], w.attn, QDIM, DIM);
        k_add<<<grid(N * DIM), 128>>>(w.x, w.attn, N * DIM);
        d2d(w.saved_mid + (size_t)li * N * DIM, w.x, (size_t)N * DIM * sizeof(float));
        k_rmsnorm<<<grid(N), 128>>>(w.x, params + FFN_NORM[li], w.xn, w.inv_ffn + li * N);
        k_linear<<<grid(N * NEXP), 128>>>(w.xn, params + ROUTER[li], w.rlogits, DIM, NEXP);
        k_router<<<grid(N), 128>>>(w.rlogits, w.rprobs + li * N * NEXP, w.idx + li * N * TOPK, w.top_p + li * N * TOPK, w.top_w + li * N * TOPK);
        for (int e = 0; e < NEXP; ++e) {
            float* gate = w.gate + ((size_t)li * NEXP + e) * N * INTER;
            float* up = w.up + ((size_t)li * NEXP + e) * N * INTER;
            float* hid = w.hid + (size_t)e * N * INTER;
            float* eout = w.eout + (size_t)e * N * DIM;
            k_linear<<<grid(N * INTER), 128>>>(w.xn, params + GATE[li][e], gate, DIM, INTER);
            k_linear<<<grid(N * INTER), 128>>>(w.xn, params + UP[li][e], up, DIM, INTER);
            k_silu_mul<<<grid(N * INTER), 128>>>(gate, up, hid, N * INTER);
            k_linear<<<grid(N * DIM), 128>>>(hid, params + DOWN[li][e], eout, INTER, DIM);
        }
        k_mix_experts<<<grid(N * DIM), 128>>>(w.eout, w.idx + li * N * TOPK, w.top_w + li * N * TOPK, w.mixed);
        k_add<<<grid(N * DIM), 128>>>(w.x, w.mixed, N * DIM);
    }
}

float loss_of(float* params, Ws& w) {
    forward_layers(params, w);
    must("forward");
    d2d(w.final_in, w.x, N * DIM * sizeof(float));
    k_rmsnorm<<<grid(N), 128>>>(w.x, params + FINAL_NORM, w.xn, w.final_inv);
    k_linear<<<grid(N * VOCAB), 128>>>(w.xn, params + LM_HEAD, w.logits, DIM, VOCAB);
    cudaMemset(w.loss, 0, sizeof(float));
    k_ce<<<grid(N), 128>>>(w.logits, w.targets, w.dlogits, w.loss);
    must("loss");
    float loss = 0.f;
    cudaMemcpy(&loss, w.loss, sizeof(float), cudaMemcpyDeviceToHost);
    return loss;
}

void backward_layer(int li, float* params, float* dparams, Ws& w) {
    const float* x_in = w.saved_x + (size_t)li * N * DIM;
    const float* mid = w.saved_mid + (size_t)li * N * DIM;
    // Weighted feed-forward norm: the vector the router and the experts read.
    k_rmsnorm_apply<<<grid(N * DIM), 128>>>(mid, w.inv_ffn + li * N, params + FFN_NORM[li], w.xn);
    for (int e = 0; e < NEXP; ++e) {
        float* gate = w.gate + ((size_t)li * NEXP + e) * N * INTER;
        float* up = w.up + ((size_t)li * NEXP + e) * N * INTER;
        float* hid = w.hid + (size_t)e * N * INTER;
        float* eout = w.eout + (size_t)e * N * DIM;
        k_silu_mul<<<grid(N * INTER), 128>>>(gate, up, hid, N * INTER);
        k_linear<<<grid(N * DIM), 128>>>(hid, params + DOWN[li][e], eout, INTER, DIM);
    }
    cudaMemset(w.d_xn, 0, N * DIM * sizeof(float));
    for (int e = 0; e < NEXP; ++e) {
        float* gate = w.gate + ((size_t)li * NEXP + e) * N * INTER;
        float* up = w.up + ((size_t)li * NEXP + e) * N * INTER;
        float* hid = w.hid + (size_t)e * N * INTER;
        k_expert_mask<<<grid(N * DIM), 128>>>(w.x, w.idx + li * N * TOPK, w.top_w + li * N * TOPK, w.dout, e);
        k_linear_dx<<<grid(N * INTER), 128>>>(w.dout, params + DOWN[li][e], w.dhid, INTER, DIM);
        k_linear_dw<<<grid(INTER * DIM), 128>>>(hid, w.dout, dparams + DOWN[li][e], INTER, DIM);
        k_silu_bwd<<<grid(N * INTER), 128>>>(gate, up, w.dhid, w.dgate, w.dup, N * INTER);
        k_linear_dx<<<grid(N * DIM), 128>>>(w.dgate, params + GATE[li][e], w.tmp, DIM, INTER);
        k_linear_dw<<<grid(DIM * INTER), 128>>>(w.xn, w.dgate, dparams + GATE[li][e], DIM, INTER);
        k_add<<<grid(N * DIM), 128>>>(w.d_xn, w.tmp, N * DIM);
        k_linear_dx<<<grid(N * DIM), 128>>>(w.dup, params + UP[li][e], w.tmp, DIM, INTER);
        k_linear_dw<<<grid(DIM * INTER), 128>>>(w.xn, w.dup, dparams + UP[li][e], DIM, INTER);
        k_add<<<grid(N * DIM), 128>>>(w.d_xn, w.tmp, N * DIM);
    }
    k_router_bwd<<<grid(N), 128>>>(w.x, w.xn, w.rprobs + li * N * NEXP, w.idx + li * N * TOPK,
        w.top_p + li * N * TOPK, w.top_w + li * N * TOPK, w.eout, params + ROUTER[li], w.d_xn, dparams + ROUTER[li]);
    // Feed-forward norm, then the residual skip carries the same cotangent.
    k_rmsnorm_bwd<<<grid(N), 128>>>(w.d_xn, mid, w.inv_ffn + li * N, params + FFN_NORM[li], w.attn, dparams + FFN_NORM[li]);
    k_add<<<grid(N * DIM), 128>>>(w.attn, w.x, N * DIM);
    k_linear_dx<<<grid(N * QDIM), 128>>>(w.attn, params + OOFF[li], w.dmix, QDIM, DIM);
    k_linear_dw<<<grid(QDIM * DIM), 128>>>(w.mix + (size_t)li * N * QDIM, w.attn, dparams + OOFF[li], QDIM, DIM);
    d2d(w.q, w.q_pre + (size_t)li * N * QDIM, (size_t)N * QDIM * sizeof(float));
    d2d(w.k, w.k_pre + (size_t)li * N * KVDIM, (size_t)N * KVDIM * sizeof(float));
    k_rope<<<grid(N * NHEAD), 128>>>(w.q, NHEAD, 0);
    k_rope<<<grid(N * NKV), 128>>>(w.k, NKV, 0);
    cudaMemset(w.dq, 0, (size_t)N * QDIM * sizeof(float));
    cudaMemset(w.dk, 0, (size_t)N * KVDIM * sizeof(float));
    cudaMemset(w.dv, 0, (size_t)N * KVDIM * sizeof(float));
    k_gqa_bwd<<<grid(B * NHEAD * SEQ), 128>>>(w.q, w.k, w.v_pre + (size_t)li * N * KVDIM,
        w.probs + (size_t)li * B * NHEAD * SEQ * SEQ, w.dmix, w.dq, w.dk, w.dv);
    k_rope<<<grid(N * NHEAD), 128>>>(w.dq, NHEAD, 1);
    k_rope<<<grid(N * NKV), 128>>>(w.dk, NKV, 1);
    k_rmsnorm_apply<<<grid(N * DIM), 128>>>(x_in, w.inv_attn + li * N, params + ATTN_NORM[li], w.xn);
    k_linear_dx<<<grid(N * DIM), 128>>>(w.dq, params + QOFF[li], w.tmp, DIM, QDIM);
    k_linear_dw<<<grid(DIM * QDIM), 128>>>(w.xn, w.dq, dparams + QOFF[li], DIM, QDIM);
    d2d(w.d_xn, w.tmp, N * DIM * sizeof(float));
    k_linear_dx<<<grid(N * DIM), 128>>>(w.dk, params + KOFF[li], w.tmp, DIM, KVDIM);
    k_linear_dw<<<grid(DIM * KVDIM), 128>>>(w.xn, w.dk, dparams + KOFF[li], DIM, KVDIM);
    k_add<<<grid(N * DIM), 128>>>(w.d_xn, w.tmp, N * DIM);
    k_linear_dx<<<grid(N * DIM), 128>>>(w.dv, params + VOFF[li], w.tmp, DIM, KVDIM);
    k_linear_dw<<<grid(DIM * KVDIM), 128>>>(w.xn, w.dv, dparams + VOFF[li], DIM, KVDIM);
    k_add<<<grid(N * DIM), 128>>>(w.d_xn, w.tmp, N * DIM);
    k_rmsnorm_bwd<<<grid(N), 128>>>(w.d_xn, x_in, w.inv_attn + li * N, params + ATTN_NORM[li], w.x, dparams + ATTN_NORM[li]);
    k_add<<<grid(N * DIM), 128>>>(w.x, w.attn, N * DIM);
    char stage[32];
    snprintf(stage, sizeof(stage), "layer %d", li);
    must(stage);
}

void backward(float* params, float* dparams, Ws& w) {
    cudaMemset(dparams, 0, (size_t)P * sizeof(float));
    k_linear_dx<<<grid(N * DIM), 128>>>(w.dlogits, params + LM_HEAD, w.dx, DIM, VOCAB);
    k_linear_dw<<<grid(DIM * VOCAB), 128>>>(w.xn, w.dlogits, dparams + LM_HEAD, DIM, VOCAB);
    k_rmsnorm_bwd<<<grid(N), 128>>>(w.dx, w.final_in, w.final_inv, params + FINAL_NORM, w.x, dparams + FINAL_NORM);
    must("head");
    for (int li = NLAYER - 1; li >= 0; --li) backward_layer(li, params, dparams, w);
    k_embed_bwd<<<grid(N * DIM), 128>>>(dparams + EMBED, w.tokens, w.x);
    must("embed");
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Ws w;
    alloc(w);
    std::vector<float> h_params(P);
    for (int i = 0; i < P; ++i) h_params[i] = 0.02f * ((i % 5) + 1.0f);
    for (int li = 0; li < NLAYER; ++li) {
        for (int d = 0; d < DIM; ++d)
            for (int e = 0; e < NEXP; ++e)
                h_params[ROUTER[li] + d * NEXP + e] = 0.05f * (e + 1.0f) + 0.001f * (float)d;
    }
    std::vector<int> h_tok(N), h_tgt(N);
    for (int i = 0; i < N; ++i) {
        h_tok[i] = (i * 5 + 1) % VOCAB;
        h_tgt[i] = (h_tok[i] + 3) % VOCAB;
    }
    float* params = fbuf(P);
    float* dparams = fbuf(P);
    cudaMemcpy(params, h_params.data(), P * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w.tokens, h_tok.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(w.targets, h_tgt.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    float loss0 = loss_of(params, w);
    backward(params, dparams, w);

    std::vector<float> h_grad(P), h_logits(N * VOCAB), h_w((size_t)NLAYER * N * TOPK);
    std::vector<int> h_idx((size_t)NLAYER * N * TOPK);
    cudaMemcpy(h_grad.data(), dparams, P * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_logits.data(), w.logits, N * VOCAB * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w.data(), w.top_w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), w.idx, h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost);
    must("copy");

    double lsum = 0.0, wsum = 0.0, gsum = 0.0, gabs = 0.0, g2 = 0.0;
    for (float v : h_logits) lsum += v;
    for (float v : h_w) wsum += v;
    for (float v : h_grad) {
        gsum += v;
        gabs += fabs(v);
        g2 += (double)v * v;
    }
    printf("PCOUNT %d\n", P);
    printf("LOSS0 %.8e\n", loss0);
    printf("LSUM %.8e\n", lsum);
    printf("WSUM %.8e\n", wsum);
    printf("GSUM %.8e\n", gsum);
    printf("GABS %.8e\n", gabs);
    printf("GNORM %.8e\n", sqrt(g2));
    printf("IDX");
    int count[32] = {};
    for (int id : h_idx) {
        if (id < 0 || id >= NEXP) {
            printf("\nFAIL idx %d\n", id);
            return 1;
        }
        count[id] += 1;
        printf(" %d", id);
    }
    printf("\n");
    for (int i = 0; i < NPROBES; ++i) {
        if (PROBES[i] < 0 || PROBES[i] >= P) {
            printf("FAIL probe %d\n", PROBES[i]);
            return 1;
        }
        printf("GRAD %d %.8e\n", PROBES[i], h_grad[PROBES[i]]);
    }
    for (int e = 0; e < NEXP; ++e) {
        double mass = 0.0;
        for (int li = 0; li < NLAYER; ++li) {
            int begin = GATE[li][e];
            int end = DOWN[li][e] + INTER * DIM;
            for (int i = begin; i < end; ++i) mass += fabs(h_grad[i]);
        }
        printf("EMASS %d %d %.8e\n", e, count[e], mass);
    }

    k_sgd<<<grid(P), 128>>>(params, dparams, P);
    must("sgd");
    float loss1 = loss_of(params, w);
    printf("LOSS1 %.8e\n", loss1);
    if (!std::isfinite(loss0) || !std::isfinite(loss1) || !std::isfinite(gsum)) {
        printf("FAIL nonfinite\n");
        return 1;
    }
    printf("OK mixtral\n");
    return 0;
}
