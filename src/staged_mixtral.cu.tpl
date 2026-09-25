// Runs a staged Mixtral schedule as a cluster. The main thread is the
// coordinator. Each device is a member. Members pass parameter, adjoint, and
// activation tiles to each other over TCP, then the member that owns a window
// runs that operation.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using ull = unsigned long long;

constexpr ull B = @@B@@;
constexpr ull SEQ = @@SEQ@@;
constexpr ull D = @@D@@;
constexpr ull L = @@L@@;
constexpr ull H = @@H@@;
constexpr ull KV = @@KV@@;
constexpr ull HD = @@HD@@;
constexpr ull I = @@I@@;
constexpr ull E = @@E@@;
constexpr ull TOPK = @@TOPK@@;
constexpr ull V = @@V@@;
constexpr ull WINDOW = @@WINDOW@@;
constexpr float EPS = @@EPS@@;
constexpr float THETA = @@THETA@@;
constexpr float LR = @@LR@@;
constexpr ull N = B * SEQ;
constexpr ull QDIM = H * HD;
constexpr ull KVDIM = KV * HD;
constexpr ull P = @@P@@;
constexpr ull LM = @@LM@@;
constexpr int NDEV = @@NDEV@@;
constexpr ull SCRATCH = @@SCRATCH@@;

const int DEV_KIND[NDEV] = { @@KINDS@@ };
const ull QOFF[L] = { @@Q@@ };
const ull KOFF[L] = { @@K@@ };
const ull VOFF[L] = { @@VOFF@@ };
const ull OOFF[L] = { @@O@@ };
const ull ROUTER[L] = { @@ROUTER@@ };
const ull GATE[L][E] = { @@GATE@@ };
const ull UP[L][E] = { @@UP@@ };
const ull DOWN[L][E] = { @@DOWN@@ };

enum Exec {
    Embed = 1, RmsNorm = 2, Linear = 3, RopeQ = 4, RopeK = 5, Gqa = 6,
    AddAttn = 7, AddMoe = 8, Router = 9, Silu = 10, Mix = 11, Loss = 12, Sgd = 13, Send = 14
};

struct Step {
    int dev, home, pass, exec;
    ull p_off, b0, b1, r0, r1, c0, c1, din, dout;
    int layer, expert;
};

const Step STEPS[] = { @@STEPS@@ };
const int NSTEPS = sizeof(STEPS) / sizeof(STEPS[0]);

int gpu_ops = 0, cpu_ops = 0, xfers = 0;
int phys[NDEV];
std::vector<int> g_peer;

void must(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e) { printf("FAIL %s %s\n", what, cudaGetErrorString(e)); exit(1); }
    e = cudaDeviceSynchronize();
    if (e) { printf("FAIL %s %s\n", what, cudaGetErrorString(e)); exit(1); }
}

int blocks_for(ull n) {
    if (n == 0) return 1;
    ull b = (n + 127ull) / 128ull;
    if (b > 2147483647ull) b = 2147483647ull;
    return (int)b;
}

#define LAUNCH(dev, n, ...) do { \
    cudaSetDevice(phys[(dev)]); \
    const ull _n = (ull)(n); \
    const ull _chunk = 2147483647ull * 128ull; \
    for (ull base = 0; base < _n; base += _chunk) { \
        ull count = _n - base; \
        if (count > _chunk) count = _chunk; \
        ull limit = base + count; \
        __VA_ARGS__; \
        must("launch"); \
    } \
    gpu_ops++; \
} while (0)

float* dev_f(ull n) {
    float* p = nullptr;
    cudaMalloc(&p, (size_t)(n == 0 ? 1 : n) * sizeof(float));
    must("malloc");
    return p;
}
ull* dev_u(ull n) {
    ull* p = nullptr;
    cudaMalloc(&p, (size_t)(n == 0 ? 1 : n) * sizeof(ull));
    must("malloc");
    return p;
}
void h2d(void* d, const void* h, ull bytes) {
    if (bytes) cudaMemcpy(d, h, (size_t)bytes, cudaMemcpyHostToDevice);
    must("h2d");
    xfers++;
}
void d2h(void* h, void* d, ull bytes) {
    if (bytes) cudaMemcpy(h, d, (size_t)bytes, cudaMemcpyDeviceToHost);
    must("d2h");
    xfers++;
}

__host__ __device__ ull ix(ull layer, ull t, ull e, ull inner) {
    return ((layer * E + e) * N + t) * inner;
}
__host__ __device__ bool visible(ull query, ull key) {
    return key <= query && query - key < WINDOW;
}
__host__ __device__ ull q_at(ull b, ull t, ull h, ull d) {
    return ((b * SEQ + t) * H + h) * HD + d;
}
__host__ __device__ ull k_at(ull b, ull t, ull h, ull d) {
    return ((b * SEQ + t) * KV + h) * HD + d;
}
__host__ __device__ ull p_at(ull layer, ull b, ull h, ull i, ull j) {
    return ((((layer * B + b) * H + h) * SEQ + i) * SEQ) + j;
}

__host__ __device__ void rope_head(float* row, ull t, int sign_neg) {
    ull half = HD / 2;
    for (ull i = 0; i < half; ++i) {
        float freq = powf(THETA, -(2.f * (float)i) / (float)HD);
        float ang = (float)t * freq;
        float c = cosf(ang);
        float s = (sign_neg ? -1.f : 1.f) * sinf(ang);
        float q1 = row[i], q2 = row[half + i];
        row[i] = q1 * c - q2 * s;
        row[half + i] = q2 * c + q1 * s;
    }
}

__global__ void k_lin_y(const float* X, const float* W, float* Y, ull k, ull c, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull m = idx / c, o = idx - m * c;
    float acc = 0.f;
    for (ull i = 0; i < k; ++i) acc += X[m * k + i] * W[i * c + o];
    Y[idx] = acc;
}
__global__ void k_lin_dx(const float* dY, const float* W, float* dX, ull k, ull c, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull m = idx / k, i = idx - m * k;
    float acc = 0.f;
    for (ull o = 0; o < c; ++o) acc += dY[m * c + o] * W[i * c + o];
    dX[idx] = acc;
}
__global__ void k_lin_dw(const float* X, const float* dY, float* dW, ull brows, ull k, ull c, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull i = idx / c, o = idx - i * c;
    float acc = 0.f;
    for (ull m = 0; m < brows; ++m) acc += X[m * k + i] * dY[m * c + o];
    dW[idx] = acc;
}
__global__ void k_rmsnorm(const float* x, const float* w, float* xn, float* inv, ull ntok, float eps, ull limit, ull base) {
    ull t = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= limit || t >= ntok) return;
    float ss = 0.f;
    for (ull i = 0; i < D; ++i) { float v = x[t * D + i]; ss += v * v; }
    float r = 1.f / sqrtf(ss / (float)D + eps);
    inv[t] = r;
    for (ull i = 0; i < D; ++i) xn[t * D + i] = x[t * D + i] * r * w[i];
}
__global__ void k_rmsnorm_bwd(const float* x, const float* dy, const float* w, const float* inv, float* dx, ull ntok, ull limit, ull base) {
    ull t = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= limit || t >= ntok) return;
    float r = inv[t];
    float dot = 0.f;
    for (ull i = 0; i < D; ++i) {
        float xhat = x[t * D + i] * r;
        float dxhat = dy[t * D + i] * w[i];
        dot += dxhat * xhat;
    }
    float mean = dot / (float)D;
    for (ull i = 0; i < D; ++i) {
        float xhat = x[t * D + i] * r;
        float dxhat = dy[t * D + i] * w[i];
        dx[t * D + i] = r * (dxhat - xhat * mean);
    }
}
__global__ void k_rmsnorm_dw(const float* x, const float* dy, const float* inv, float* dw, ull ntok, ull f0, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull i = f0 + idx;
    if (i >= D) return;
    float acc = 0.f;
    for (ull t = 0; t < ntok; ++t) acc += dy[t * D + i] * (x[t * D + i] * inv[t]);
    dw[i] = acc;
}
__global__ void k_rope(float* q, int heads, int sign_neg, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull h = idx % (ull)heads;
    ull tmp = idx / (ull)heads;
    ull t = tmp % SEQ;
    ull b = tmp / SEQ;
    rope_head(q + ((b * SEQ + t) * (ull)heads + h) * HD, t, sign_neg);
}
__global__ void k_gqa_fwd(const float* q, const float* k, const float* v, float* probs, float* mix, int layer, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull i = idx % SEQ;
    ull tmp = idx / SEQ;
    ull h = tmp % H;
    ull b = tmp / H;
    ull kvh = h / (H / KV);
    float scale = 1.f / sqrtf((float)HD);
    float mx = -INFINITY;
    bool any = false;
    for (ull j = 0; j < SEQ; ++j) {
        if (!visible(i, j)) continue;
        float dot = 0.f;
        for (ull d = 0; d < HD; ++d) dot += q[q_at(b, i, h, d)] * k[k_at(b, j, kvh, d)];
        mx = fmaxf(mx, dot * scale);
        any = true;
    }
    float sum = 0.f;
    for (ull j = 0; j < SEQ; ++j) {
        if (!visible(i, j)) continue;
        float dot = 0.f;
        for (ull d = 0; d < HD; ++d) dot += q[q_at(b, i, h, d)] * k[k_at(b, j, kvh, d)];
        sum += expf(dot * scale - mx);
    }
    for (ull j = 0; j < SEQ; ++j) {
        float p = 0.f;
        if (visible(i, j) && any && sum != 0.f) {
            float dot = 0.f;
            for (ull d = 0; d < HD; ++d) dot += q[q_at(b, i, h, d)] * k[k_at(b, j, kvh, d)];
            p = expf(dot * scale - mx) / sum;
        }
        probs[p_at((ull)layer, b, h, i, j)] = p;
    }
    for (ull d = 0; d < HD; ++d) {
        float acc = 0.f;
        for (ull j = 0; j < SEQ; ++j) {
            float p = probs[p_at((ull)layer, b, h, i, j)];
            acc += p * v[k_at(b, j, kvh, d)];
        }
        mix[q_at(b, i, h, d)] = acc;
    }
}
__global__ void k_gqa_bwd(const float* q, const float* k, const float* v, const float* probs, const float* dmix,
                          float* dq, float* dk, float* dv, int layer, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull i = idx % SEQ;
    ull tmp = idx / SEQ;
    ull h = tmp % H;
    ull b = tmp / H;
    ull kvh = h / (H / KV);
    float scale = 1.f / sqrtf((float)HD);
    float dot_pa = 0.f;
    for (ull j = 0; j < SEQ; ++j) {
        float da = 0.f;
        for (ull d = 0; d < HD; ++d) da += dmix[q_at(b, i, h, d)] * v[k_at(b, j, kvh, d)];
        float p = probs[p_at((ull)layer, b, h, i, j)];
        dot_pa += p * da;
        for (ull d = 0; d < HD; ++d) atomicAdd(dv + k_at(b, j, kvh, d), p * dmix[q_at(b, i, h, d)]);
    }
    for (ull j = 0; j < SEQ; ++j) {
        if (!visible(i, j)) continue;
        float da = 0.f;
        for (ull d = 0; d < HD; ++d) da += dmix[q_at(b, i, h, d)] * v[k_at(b, j, kvh, d)];
        float p = probs[p_at((ull)layer, b, h, i, j)];
        float ds = p * (da - dot_pa) * scale;
        for (ull d = 0; d < HD; ++d) {
            dq[q_at(b, i, h, d)] += ds * k[k_at(b, j, kvh, d)];
            atomicAdd(dk + k_at(b, j, kvh, d), ds * q[q_at(b, i, h, d)]);
        }
    }
}
__global__ void k_router(const float* rlog, float* rprob, ull* topi, float* topp, float* topw, int layer, ull limit, ull base) {
    ull t = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= limit) return;
    float m = -INFINITY, sum = 0.f;
    for (ull e = 0; e < E; ++e) m = fmaxf(m, rlog[t * E + e]);
    for (ull e = 0; e < E; ++e) {
        float p = expf(rlog[t * E + e] - m);
        rprob[(ull)layer * N * E + t * E + e] = p;
        sum += p;
    }
    for (ull e = 0; e < E; ++e) rprob[(ull)layer * N * E + t * E + e] /= sum;
    for (ull k = 0; k < TOPK; ++k) {
        ull best = 0;
        bool any = false;
        float bv = -1.f;
        for (ull e = 0; e < E; ++e) {
            bool taken = false;
            for (ull j = 0; j < k; ++j)
                if (topi[((ull)layer * N + t) * TOPK + j] == e) taken = true;
            if (taken) continue;
            float p = rprob[(ull)layer * N * E + t * E + e];
            if (!any || p > bv || (p == bv && e < best)) { best = e; bv = p; any = true; }
        }
        topi[((ull)layer * N + t) * TOPK + k] = best;
        topp[((ull)layer * N + t) * TOPK + k] = bv;
    }
    float s = 0.f;
    for (ull k = 0; k < TOPK; ++k) s += topp[((ull)layer * N + t) * TOPK + k];
    for (ull k = 0; k < TOPK; ++k)
        topw[((ull)layer * N + t) * TOPK + k] = topp[((ull)layer * N + t) * TOPK + k] / s;
}
__global__ void k_silu_fwd(const float* gate, const float* up, float* hid, ull limit, ull base) {
    ull i = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= limit) return;
    float z = gate[i];
    float s = 1.f / (1.f + expf(-z));
    hid[i] = z * s * up[i];
}
__global__ void k_silu_bwd(const float* gate, const float* up, const float* dhid, float* dgate, float* dup, ull limit, ull base) {
    ull i = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= limit) return;
    float z = gate[i];
    float sg = 1.f / (1.f + expf(-z));
    float dsilu = sg * (1.f + z * (1.f - sg));
    dgate[i] = dhid[i] * up[i] * dsilu;
    dup[i] = dhid[i] * (z * sg);
}
__global__ void k_mix_fwd(const float* eout, const ull* topi, const float* topw, float* mixed, int layer, ull limit, ull base) {
    ull t = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= limit) return;
    for (ull d = 0; d < D; ++d) mixed[t * D + d] = 0.f;
    for (ull k = 0; k < TOPK; ++k) {
        ull e = topi[((ull)layer * N + t) * TOPK + k];
        float w = topw[((ull)layer * N + t) * TOPK + k];
        for (ull d = 0; d < D; ++d) mixed[t * D + d] += w * eout[ix((ull)layer, t, e, D) + d];
    }
}
__global__ void k_mix_bwd(const float* dx, const float* eout, const float* rprob, const ull* topi, const float* topp, const float* topw,
                          float* deout, float* dhid, float* dz, float* scratch, int layer, ull limit, ull base) {
    ull t = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= limit) return;
    for (ull e = 0; e < E; ++e) {
        for (ull d = 0; d < D; ++d) deout[ix((ull)layer, t, e, D) + d] = 0.f;
        for (ull i = 0; i < I; ++i) dhid[ix((ull)layer, t, e, I) + i] = 0.f;
    }
    float* dw = scratch + t * TOPK;
    float* dp = scratch + N * TOPK + t * E;
    for (ull e = 0; e < E; ++e) dp[e] = 0.f;
    float s = 0.f, weight_dot = 0.f;
    for (ull k = 0; k < TOPK; ++k) {
        s += topp[((ull)layer * N + t) * TOPK + k];
        ull e = topi[((ull)layer * N + t) * TOPK + k];
        float dot = 0.f;
        for (ull d = 0; d < D; ++d) dot += dx[t * D + d] * eout[ix((ull)layer, t, e, D) + d];
        dw[k] = dot;
        float w = topw[((ull)layer * N + t) * TOPK + k];
        weight_dot += dot * w;
        for (ull d = 0; d < D; ++d) deout[ix((ull)layer, t, e, D) + d] += w * dx[t * D + d];
    }
    for (ull k = 0; k < TOPK; ++k) {
        float dvalues = dw[k] / s - weight_dot / s;
        dp[topi[((ull)layer * N + t) * TOPK + k]] = dvalues;
    }
    float dotp = 0.f;
    for (ull e = 0; e < E; ++e) dotp += rprob[(ull)layer * N * E + t * E + e] * dp[e];
    for (ull e = 0; e < E; ++e)
        dz[t * E + e] = rprob[(ull)layer * N * E + t * E + e] * (dp[e] - dotp);
}
__global__ void k_embed(const float* table, const ull* tok, float* x, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull t = idx / D, d = idx - t * D;
    x[idx] = table[tok[t] * D + d];
}
__global__ void k_embed_bwd(float* dtable, const ull* tok, const float* dx, ull limit, ull base) {
    ull idx = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit) return;
    ull t = idx / D, d = idx - t * D;
    atomicAdd(dtable + tok[t] * D + d, dx[idx]);
}
__global__ void k_add(float* x, const float* y, ull limit, ull base) {
    ull i = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= limit) return;
    x[i] += y[i];
}
__global__ void k_zero(float* x, ull limit, ull base) {
    ull i = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= limit) return;
    x[i] = 0.f;
}
__global__ void k_loss(const float* logits, const ull* tgt, float* dlogits, float* tok_loss, ull limit, ull base) {
    ull t = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= limit) return;
    float m = -INFINITY, sum = 0.f;
    for (ull v = 0; v < V; ++v) m = fmaxf(m, logits[t * V + v]);
    for (ull v = 0; v < V; ++v) sum += expf(logits[t * V + v] - m);
    tok_loss[t] = (m + logf(sum) - logits[t * V + tgt[t]]) / (float)N;
    for (ull v = 0; v < V; ++v) dlogits[t * V + v] = expf(logits[t * V + v] - m) / sum / (float)N;
    dlogits[t * V + tgt[t]] -= 1.f / (float)N;
}
__global__ void k_sgd(float* tile, const float* grad, float lr, ull limit, ull base) {
    ull i = base + (ull)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= limit) return;
    tile[i] -= lr * grad[i];
}

void gpu_gemm(const float* X, const float* W, const float* dYin, float* Y, float* dXout, float* dWout,
              ull brows, ull k, ull c, int kind, int dev) {
    cudaSetDevice(phys[dev]);
    ull nx = brows * k, nw = k * c, ny = brows * c;
    float* xbuf = dev_f(nx);
    float* wbuf = dev_f(nw);
    float* ybuf = dev_f(ny);
    ull nout = kind == 2 ? nw : (kind == 1 ? nx : ny);
    float* obuf = dev_f(nout);
    h2d(xbuf, X, nx * 4);
    h2d(wbuf, W, nw * 4);
    if (kind == 0) {
        LAUNCH(dev, ny, k_lin_y<<<blocks_for(count), 128>>>(xbuf, wbuf, obuf, k, c, limit, base));
        d2h(Y, obuf, ny * 4);
    } else if (kind == 1) {
        h2d(ybuf, dYin, ny * 4);
        LAUNCH(dev, nx, k_lin_dx<<<blocks_for(count), 128>>>(ybuf, wbuf, obuf, k, c, limit, base));
        d2h(dXout, obuf, nx * 4);
    } else {
        h2d(ybuf, dYin, ny * 4);
        LAUNCH(dev, nw, k_lin_dw<<<blocks_for(count), 128>>>(xbuf, ybuf, obuf, brows, k, c, limit, base));
        d2h(dWout, obuf, nw * 4);
    }
    cudaFree(xbuf); cudaFree(wbuf); cudaFree(ybuf); cudaFree(obuf);
}

struct State {
    std::vector<float> P, dP, x, dx, xn, q, k, v, mix, attn, logits, dlogits;
    std::vector<float> qpre, kpre, vp, probs, ain, ainv, mid, finv, gate, up, hid, eout;
    std::vector<float> rprob, topp, topw, rlog, dz, dgate, dup, deout, dhid;
    std::vector<float> fin, finv_s, mixed, dmid, dxn;
    std::vector<float> xn_attn, xn_ffn, xn_final, mix_save;
    std::vector<ull> topi, tok, tgt;
    float loss = 0;
};

void alloc_state(State& st) {
    st.P.assign((size_t)P, 0.f);
    st.dP.assign((size_t)P, 0.f);
    st.x.assign((size_t)(N * D), 0.f);
    st.dx.assign((size_t)(N * D), 0.f);
    st.xn.assign((size_t)(N * D), 0.f);
    st.q.assign((size_t)(N * QDIM), 0.f);
    st.k.assign((size_t)(N * KVDIM), 0.f);
    st.v.assign((size_t)(N * KVDIM), 0.f);
    st.mix.assign((size_t)(N * QDIM), 0.f);
    st.attn.assign((size_t)(N * D), 0.f);
    st.logits.assign((size_t)(N * V), 0.f);
    st.dlogits.assign((size_t)(N * V), 0.f);
    st.qpre.assign((size_t)(L * N * QDIM), 0.f);
    st.kpre.assign((size_t)(L * N * KVDIM), 0.f);
    st.vp.assign((size_t)(L * N * KVDIM), 0.f);
    st.probs.assign((size_t)(L * B * H * SEQ * SEQ), 0.f);
    st.ain.assign((size_t)(L * N * D), 0.f);
    st.ainv.assign((size_t)(L * N), 0.f);
    st.mid.assign((size_t)(L * N * D), 0.f);
    st.finv.assign((size_t)(L * N), 0.f);
    st.gate.assign((size_t)(L * E * N * I), 0.f);
    st.up.assign((size_t)(L * E * N * I), 0.f);
    st.hid.assign((size_t)(L * E * N * I), 0.f);
    st.eout.assign((size_t)(L * E * N * D), 0.f);
    st.rprob.assign((size_t)(L * N * E), 0.f);
    st.topp.assign((size_t)(L * N * TOPK), 0.f);
    st.topw.assign((size_t)(L * N * TOPK), 0.f);
    st.topi.assign((size_t)(L * N * TOPK), 0);
    st.rlog.assign((size_t)(N * E), 0.f);
    st.dz.assign((size_t)(N * E), 0.f);
    st.dgate.assign((size_t)(L * E * N * I), 0.f);
    st.dup.assign((size_t)(L * E * N * I), 0.f);
    st.deout.assign((size_t)(L * E * N * D), 0.f);
    st.dhid.assign((size_t)(L * E * N * I), 0.f);
    st.fin.assign((size_t)(N * D), 0.f);
    st.finv_s.assign((size_t)N, 0.f);
    st.mixed.assign((size_t)(N * D), 0.f);
    st.dmid.assign((size_t)(N * D), 0.f);
    st.dxn.assign((size_t)(N * D), 0.f);
    st.xn_attn.assign((size_t)(L * N * D), 0.f);
    st.xn_ffn.assign((size_t)(L * N * D), 0.f);
    st.xn_final.assign((size_t)(N * D), 0.f);
    st.mix_save.assign((size_t)(L * N * QDIM), 0.f);
    st.tok.resize((size_t)N);
    st.tgt.resize((size_t)N);
    (void)SCRATCH;
}

void add_rect(std::vector<float>& dst, ull stride, ull b0, ull b1, ull c0, ull c1, const std::vector<float>& src, bool replace) {
    ull brows = b1 - b0, c = c1 - c0;
    for (ull m = 0; m < brows; ++m)
        for (ull o = 0; o < c; ++o) {
            float& slot = dst[(size_t)((b0 + m) * stride + c0 + o)];
            if (replace) slot = src[(size_t)(m * c + o)];
            else slot += src[(size_t)(m * c + o)];
        }
}

int linear_role(const Step& s, int& expert) {
    expert = 0;
    if ((ull)s.layer >= L) return 0;
    if (s.p_off == QOFF[s.layer]) return 1;
    if (s.p_off == KOFF[s.layer]) return 2;
    if (s.p_off == VOFF[s.layer]) return 3;
    if (s.p_off == OOFF[s.layer]) return 4;
    if (s.p_off == ROUTER[s.layer]) return 5;
    if (s.p_off == LM) return 0;
    for (ull e = 0; e < E; ++e) {
        if (s.p_off == GATE[s.layer][e]) { expert = (int)e; return 6; }
        if (s.p_off == UP[s.layer][e]) { expert = (int)e; return 7; }
        if (s.p_off == DOWN[s.layer][e]) { expert = (int)e; return 8; }
    }
    return 0;
}

float linear_x(const State& st, const Step& s, int role, int expert, ull row, ull feat) {
    if (role == 4) {
        if (s.pass == 0) return st.mix[(size_t)(row * QDIM + feat)];
        return st.mix_save[(size_t)(((ull)s.layer * N + row) * QDIM + feat)];
    }
    if (role == 8) return st.hid[(size_t)ix((ull)s.layer, row, (ull)expert, I) + (size_t)feat];
    if (s.pass == 0) return st.xn[(size_t)(row * D + feat)];
    if (role == 0) return st.xn_final[(size_t)(row * D + feat)];
    if (role == 5 || role == 6 || role == 7)
        return st.xn_ffn[(size_t)(((ull)s.layer * N + row) * D + feat)];
    return st.xn_attn[(size_t)(((ull)s.layer * N + row) * D + feat)];
}

float linear_dy(const State& st, const Step& s, int role, int expert, ull row, ull col) {
    if (role == 1) return st.q[(size_t)(row * s.dout + col)];
    if (role == 2) return st.k[(size_t)(row * s.dout + col)];
    if (role == 3) return st.v[(size_t)(row * s.dout + col)];
    if (role == 4) return st.dmid[(size_t)(row * s.dout + col)];
    if (role == 5) return st.dz[(size_t)(row * s.dout + col)];
    if (role == 6) return st.dgate[(size_t)ix((ull)s.layer, row, (ull)expert, I) + (size_t)col];
    if (role == 7) return st.dup[(size_t)ix((ull)s.layer, row, (ull)expert, I) + (size_t)col];
    if (role == 8) return st.deout[(size_t)ix((ull)s.layer, row, (ull)expert, D) + (size_t)col];
    return st.dlogits[(size_t)(row * s.dout + col)];
}

void scatter_y(State& st, const Step& s, int role, int expert, const std::vector<float>& Y) {
    ull brows = s.b1 - s.b0, c = s.c1 - s.c0;
    bool replace = s.r0 == 0;
    if (role == 1) add_rect(st.q, s.dout, s.b0, s.b1, s.c0, s.c1, Y, replace);
    else if (role == 2) add_rect(st.k, s.dout, s.b0, s.b1, s.c0, s.c1, Y, replace);
    else if (role == 3) add_rect(st.v, s.dout, s.b0, s.b1, s.c0, s.c1, Y, replace);
    else if (role == 4) add_rect(st.attn, s.dout, s.b0, s.b1, s.c0, s.c1, Y, replace);
    else if (role == 5) add_rect(st.rlog, s.dout, s.b0, s.b1, s.c0, s.c1, Y, replace);
    else if (role == 6 || role == 7 || role == 8) {
        ull inner = role == 8 ? D : I;
        std::vector<float>& dst = role == 6 ? st.gate : (role == 7 ? st.up : st.eout);
        for (ull m = 0; m < brows; ++m)
            for (ull o = 0; o < c; ++o) {
                float& slot = dst[(size_t)ix((ull)s.layer, s.b0 + m, (ull)expert, inner) + (size_t)(s.c0 + o)];
                slot = replace ? Y[(size_t)(m * c + o)] : slot + Y[(size_t)(m * c + o)];
            }
    } else add_rect(st.logits, s.dout, s.b0, s.b1, s.c0, s.c1, Y, replace);
}

void scatter_dx(State& st, const Step& s, int role, int expert, const std::vector<float>& dX) {
    ull brows = s.b1 - s.b0, k = s.r1 - s.r0;
    if (role == 4) {
        add_rect(st.mix, QDIM, s.b0, s.b1, s.r0, s.r1, dX, false);
        return;
    }
    if (role == 8) {
        for (ull m = 0; m < brows; ++m)
            for (ull i = 0; i < k; ++i)
                st.dhid[(size_t)ix((ull)s.layer, s.b0 + m, (ull)expert, I) + (size_t)(s.r0 + i)] += dX[(size_t)(m * k + i)];
        return;
    }
    add_rect(st.dxn, s.din, s.b0, s.b1, s.r0, s.r1, dX, false);
}

void run_linear(State& st, const Step& s, bool gpu) {
    int expert = 0;
    int role = linear_role(s, expert);
    ull brows = s.b1 - s.b0, k = s.r1 - s.r0, c = s.c1 - s.c0;
    std::vector<float> X(brows * k), W(k * c), Y(brows * c), dY(brows * c), dX(brows * k), dW(k * c);
    for (ull m = 0; m < brows; ++m)
        for (ull i = 0; i < k; ++i)
            X[(size_t)(m * k + i)] = linear_x(st, s, role, expert, s.b0 + m, s.r0 + i);
    for (ull i = 0; i < k; ++i)
        for (ull o = 0; o < c; ++o)
            W[(size_t)(i * c + o)] = st.P[(size_t)(s.p_off + (s.r0 + i) * s.dout + s.c0 + o)];
    if (s.pass == 0) {
        if (gpu) gpu_gemm(X.data(), W.data(), nullptr, Y.data(), nullptr, nullptr, brows, k, c, 0, s.dev);
        else {
            cpu_ops++;
            for (ull m = 0; m < brows; ++m)
                for (ull o = 0; o < c; ++o) {
                    float acc = 0.f;
                    for (ull i = 0; i < k; ++i) acc += X[(size_t)(m * k + i)] * W[(size_t)(i * c + o)];
                    Y[(size_t)(m * c + o)] = acc;
                }
        }
        scatter_y(st, s, role, expert, Y);
        if (s.r1 == s.din && s.c1 == s.dout && s.b1 == N) {
            if (role == 1) std::copy(st.q.begin(), st.q.end(), st.qpre.begin() + (size_t)((ull)s.layer * N * QDIM));
            if (role == 2) std::copy(st.k.begin(), st.k.end(), st.kpre.begin() + (size_t)((ull)s.layer * N * KVDIM));
            if (role == 3) std::copy(st.v.begin(), st.v.end(), st.vp.begin() + (size_t)((ull)s.layer * N * KVDIM));
        }
        return;
    }
    for (ull m = 0; m < brows; ++m)
        for (ull o = 0; o < c; ++o)
            dY[(size_t)(m * c + o)] = linear_dy(st, s, role, expert, s.b0 + m, s.c0 + o);
    if (gpu) {
        gpu_gemm(X.data(), W.data(), dY.data(), nullptr, dX.data(), nullptr, brows, k, c, 1, s.dev);
        gpu_gemm(X.data(), W.data(), dY.data(), nullptr, nullptr, dW.data(), brows, k, c, 2, s.dev);
    } else {
        cpu_ops++;
        for (ull m = 0; m < brows; ++m)
            for (ull i = 0; i < k; ++i) {
                float acc = 0.f;
                for (ull o = 0; o < c; ++o) acc += dY[(size_t)(m * c + o)] * W[(size_t)(i * c + o)];
                dX[(size_t)(m * k + i)] = acc;
            }
        for (ull i = 0; i < k; ++i)
            for (ull o = 0; o < c; ++o) {
                float acc = 0.f;
                for (ull m = 0; m < brows; ++m) acc += X[(size_t)(m * k + i)] * dY[(size_t)(m * c + o)];
                dW[(size_t)(i * c + o)] = acc;
            }
    }
    scatter_dx(st, s, role, expert, dX);
    for (ull i = 0; i < k; ++i)
        for (ull o = 0; o < c; ++o)
            st.dP[(size_t)(s.p_off + (s.r0 + i) * s.dout + s.c0 + o)] += dW[(size_t)(i * c + o)];
}

void save_rows(std::vector<float>& dst, ull dst_base, const std::vector<float>& src, ull b0, ull b1) {
    for (ull t = b0; t < b1; ++t)
        for (ull i = 0; i < D; ++i)
            dst[(size_t)(dst_base + t * D + i)] = src[(size_t)(t * D + i)];
}

void rms_compute(const float* x, const float* w, const float* dy, float* xn, float* inv, float* dx, float* dw, ull ntok, bool backward) {
    for (ull t = 0; t < ntok; ++t) {
        if (!backward) {
            float ss = 0.f;
            for (ull i = 0; i < D; ++i) { float v = x[t * D + i]; ss += v * v; }
            float r = 1.f / sqrtf(ss / (float)D + EPS);
            inv[t] = r;
            for (ull i = 0; i < D; ++i) xn[t * D + i] = x[t * D + i] * r * w[i];
        }
    }
    if (!backward) return;
    for (ull i = 0; i < D; ++i) dw[i] = 0.f;
    for (ull t = 0; t < ntok; ++t) {
        float r = inv[t];
        float dot = 0.f;
        for (ull i = 0; i < D; ++i) {
            float xhat = x[t * D + i] * r;
            float dxhat = dy[t * D + i] * w[i];
            dw[i] += dy[t * D + i] * xhat;
            dot += dxhat * xhat;
        }
        float mean = dot / (float)D;
        for (ull i = 0; i < D; ++i) {
            float xhat = x[t * D + i] * r;
            float dxhat = dy[t * D + i] * w[i];
            dx[t * D + i] = r * (dxhat - xhat * mean);
        }
    }
}

void rms_fwd(State& st, const Step& s, bool gpu) {
    if (s.r0 != 0) return;
    ull ntok = s.b1 - s.b0;
    int layer = s.layer;
    ull save_base = (ull)layer * N * D;
    if (s.expert == 0) save_rows(st.ain, save_base, st.x, s.b0, s.b1);
    else if (s.expert == 1) save_rows(st.mid, save_base, st.x, s.b0, s.b1);
    else save_rows(st.fin, 0, st.x, s.b0, s.b1);
    std::vector<float> x((size_t)(ntok * D)), w((size_t)D), xn((size_t)(ntok * D)), inv((size_t)ntok);
    for (ull t = 0; t < ntok; ++t)
        for (ull i = 0; i < D; ++i) x[(size_t)(t * D + i)] = st.x[(size_t)((s.b0 + t) * D + i)];
    for (ull i = 0; i < D; ++i) w[(size_t)i] = st.P[(size_t)(s.p_off + i)];
    if (gpu) {
        float* dxbuf = dev_f(ntok * D);
        float* dwbuf = dev_f(D);
        float* xbuf = dev_f(ntok * D);
        float* xnbuf = dev_f(ntok * D);
        float* ibuf = dev_f(ntok);
        h2d(xbuf, x.data(), ntok * D * 4);
        h2d(dwbuf, w.data(), D * 4);
        LAUNCH(s.dev, ntok, k_rmsnorm<<<blocks_for(count), 128>>>(xbuf, dwbuf, xnbuf, ibuf, ntok, EPS, limit, base));
        d2h(xn.data(), xnbuf, ntok * D * 4);
        d2h(inv.data(), ibuf, ntok * 4);
        cudaFree(dxbuf); cudaFree(dwbuf); cudaFree(xbuf); cudaFree(xnbuf); cudaFree(ibuf);
    } else {
        cpu_ops++;
        rms_compute(x.data(), w.data(), nullptr, xn.data(), inv.data(), nullptr, nullptr, ntok, false);
    }
    for (ull t = 0; t < ntok; ++t) {
        if (s.expert == 0) st.ainv[(size_t)((ull)layer * N + s.b0 + t)] = inv[(size_t)t];
        else if (s.expert == 1) st.finv[(size_t)((ull)layer * N + s.b0 + t)] = inv[(size_t)t];
        else st.finv_s[(size_t)(s.b0 + t)] = inv[(size_t)t];
        for (ull i = 0; i < D; ++i) {
            st.xn[(size_t)((s.b0 + t) * D + i)] = xn[(size_t)(t * D + i)];
            if (s.expert == 0) st.xn_attn[(size_t)(save_base + (s.b0 + t) * D + i)] = xn[(size_t)(t * D + i)];
            else if (s.expert == 1) st.xn_ffn[(size_t)(save_base + (s.b0 + t) * D + i)] = xn[(size_t)(t * D + i)];
            else st.xn_final[(size_t)((s.b0 + t) * D + i)] = xn[(size_t)(t * D + i)];
        }
    }
}

void rms_bwd(State& st, const Step& s, bool gpu) {
    if (s.r0 != 0) return;
    ull ntok = s.b1 - s.b0;
    int layer = s.layer;
    const std::vector<float>* src = &st.fin;
    const std::vector<float>* invs = &st.finv_s;
    ull src_base = 0, inv_base = 0;
    if (s.expert == 0) { src = &st.ain; invs = &st.ainv; src_base = (ull)layer * N * D; inv_base = (ull)layer * N; }
    else if (s.expert == 1) { src = &st.mid; invs = &st.finv; src_base = (ull)layer * N * D; inv_base = (ull)layer * N; }
    std::vector<float> x((size_t)(ntok * D)), dy((size_t)(ntok * D)), w((size_t)D), inv((size_t)ntok);
    std::vector<float> dx((size_t)(ntok * D)), dw((size_t)D);
    for (ull t = 0; t < ntok; ++t) {
        inv[(size_t)t] = (*invs)[(size_t)(inv_base + s.b0 + t)];
        for (ull i = 0; i < D; ++i) {
            x[(size_t)(t * D + i)] = (*src)[(size_t)(src_base + (s.b0 + t) * D + i)];
            dy[(size_t)(t * D + i)] = st.dxn[(size_t)((s.b0 + t) * D + i)];
        }
    }
    for (ull i = 0; i < D; ++i) w[(size_t)i] = st.P[(size_t)(s.p_off + i)];
    if (gpu) {
        float* xbuf = dev_f(ntok * D);
        float* dybuf = dev_f(ntok * D);
        float* wbuf = dev_f(D);
        float* ibuf = dev_f(ntok);
        float* dxbuf = dev_f(ntok * D);
        float* dwbuf = dev_f(D);
        h2d(xbuf, x.data(), ntok * D * 4);
        h2d(dybuf, dy.data(), ntok * D * 4);
        h2d(wbuf, w.data(), D * 4);
        h2d(ibuf, inv.data(), ntok * 4);
        LAUNCH(s.dev, ntok, k_rmsnorm_bwd<<<blocks_for(count), 128>>>(xbuf, dybuf, wbuf, ibuf, dxbuf, ntok, limit, base));
        LAUNCH(s.dev, D, k_rmsnorm_dw<<<blocks_for(count), 128>>>(xbuf, dybuf, ibuf, dwbuf, ntok, 0, limit, base));
        d2h(dx.data(), dxbuf, ntok * D * 4);
        d2h(dw.data(), dwbuf, D * 4);
        cudaFree(xbuf); cudaFree(dybuf); cudaFree(wbuf); cudaFree(ibuf); cudaFree(dxbuf); cudaFree(dwbuf);
    } else {
        cpu_ops++;
        rms_compute(x.data(), w.data(), dy.data(), nullptr, inv.data(), dx.data(), dw.data(), ntok, true);
    }
    for (ull t = 0; t < ntok; ++t)
        for (ull i = 0; i < D; ++i) {
            float v = dx[(size_t)(t * D + i)];
            ull row = s.b0 + t;
            if (s.expert == 1) st.dmid[(size_t)(row * D + i)] = v + st.dx[(size_t)(row * D + i)];
            else if (s.expert == 0) st.dx[(size_t)(row * D + i)] = v + st.dmid[(size_t)(row * D + i)];
            else st.dx[(size_t)(row * D + i)] = v;
        }
    for (ull i = 0; i < D; ++i) st.dP[(size_t)(s.p_off + i)] += dw[(size_t)i];
}

void rope_apply(std::vector<float>& q, int heads, int sign_neg, bool gpu, int dev) {
    ull npos = B * SEQ * (ull)heads;
    if (!gpu) {
        cpu_ops++;
        for (ull b = 0; b < B; ++b)
            for (ull t = 0; t < SEQ; ++t)
                for (int h = 0; h < heads; ++h)
                    rope_head(q.data() + (size_t)(((b * SEQ + t) * (ull)heads + (ull)h) * HD), t, sign_neg);
        return;
    }
    float* buf = dev_f(q.size());
    h2d(buf, q.data(), q.size() * 4);
    LAUNCH(dev, npos, k_rope<<<blocks_for(count), 128>>>(buf, heads, sign_neg, limit, base));
    d2h(q.data(), buf, q.size() * 4);
    cudaFree(buf);
}

void gqa_fwd(State& st, const Step& s, bool gpu) {
    if (!(s.b0 == 0 && s.r0 == 0 && s.c0 == 0)) return;
    int layer = s.layer;
    ull npos = B * H * SEQ;
    if (gpu) {
        float* qbuf = dev_f(st.q.size());
        float* kbuf = dev_f(st.k.size());
        float* vbuf = dev_f(st.v.size());
        float* pbuf = dev_f(st.probs.size());
        float* mbuf = dev_f(st.mix.size());
        h2d(qbuf, st.q.data(), st.q.size() * 4);
        h2d(kbuf, st.k.data(), st.k.size() * 4);
        h2d(vbuf, st.v.data(), st.v.size() * 4);
        h2d(pbuf, st.probs.data(), st.probs.size() * 4);
        LAUNCH(s.dev, npos, k_gqa_fwd<<<blocks_for(count), 128>>>(qbuf, kbuf, vbuf, pbuf, mbuf, layer, limit, base));
        d2h(st.probs.data(), pbuf, st.probs.size() * 4);
        d2h(st.mix.data(), mbuf, st.mix.size() * 4);
        cudaFree(qbuf); cudaFree(kbuf); cudaFree(vbuf); cudaFree(pbuf); cudaFree(mbuf);
    } else {
        cpu_ops++;
        float scale = 1.f / sqrtf((float)HD);
        for (ull b = 0; b < B; ++b)
            for (ull h = 0; h < H; ++h) {
                ull kvh = h / (H / KV);
                for (ull i = 0; i < SEQ; ++i) {
                    float mx = -INFINITY;
                    bool any = false;
                    for (ull j = 0; j < SEQ; ++j) {
                        if (!visible(i, j)) continue;
                        float dot = 0.f;
                        for (ull d = 0; d < HD; ++d) dot += st.q[(size_t)q_at(b, i, h, d)] * st.k[(size_t)k_at(b, j, kvh, d)];
                        mx = fmaxf(mx, dot * scale);
                        any = true;
                    }
                    float sum = 0.f;
                    for (ull j = 0; j < SEQ; ++j) {
                        if (!visible(i, j)) continue;
                        float dot = 0.f;
                        for (ull d = 0; d < HD; ++d) dot += st.q[(size_t)q_at(b, i, h, d)] * st.k[(size_t)k_at(b, j, kvh, d)];
                        sum += expf(dot * scale - mx);
                    }
                    for (ull j = 0; j < SEQ; ++j) {
                        float p = 0.f;
                        if (visible(i, j) && any && sum != 0.f) {
                            float dot = 0.f;
                            for (ull d = 0; d < HD; ++d) dot += st.q[(size_t)q_at(b, i, h, d)] * st.k[(size_t)k_at(b, j, kvh, d)];
                            p = expf(dot * scale - mx) / sum;
                        }
                        st.probs[(size_t)p_at((ull)layer, b, h, i, j)] = p;
                    }
                    for (ull d = 0; d < HD; ++d) {
                        float acc = 0.f;
                        for (ull j = 0; j < SEQ; ++j) acc += st.probs[(size_t)p_at((ull)layer, b, h, i, j)] * st.v[(size_t)k_at(b, j, kvh, d)];
                        st.mix[(size_t)q_at(b, i, h, d)] = acc;
                    }
                }
            }
    }
    std::copy(st.mix.begin(), st.mix.end(), st.mix_save.begin() + (size_t)((ull)layer * N * QDIM));
}

void gqa_bwd(State& st, const Step& s, bool gpu) {
    if (!(s.b0 == 0 && s.r0 == 0 && s.c0 == 0)) return;
    int layer = s.layer;
    std::vector<float> post_q(st.qpre.begin() + (size_t)((ull)layer * N * QDIM), st.qpre.begin() + (size_t)(((ull)layer + 1) * N * QDIM));
    std::vector<float> post_k(st.kpre.begin() + (size_t)((ull)layer * N * KVDIM), st.kpre.begin() + (size_t)(((ull)layer + 1) * N * KVDIM));
    std::vector<float> cur_v(st.vp.begin() + (size_t)((ull)layer * N * KVDIM), st.vp.begin() + (size_t)(((ull)layer + 1) * N * KVDIM));
    rope_apply(post_q, (int)H, 0, gpu, s.dev);
    rope_apply(post_k, (int)KV, 0, gpu, s.dev);
    std::fill(st.q.begin(), st.q.end(), 0.f);
    std::fill(st.k.begin(), st.k.end(), 0.f);
    std::fill(st.v.begin(), st.v.end(), 0.f);
    ull npos = B * H * SEQ;
    if (gpu) {
        float* qbuf = dev_f(post_q.size());
        float* kbuf = dev_f(post_k.size());
        float* vbuf = dev_f(cur_v.size());
        float* pbuf = dev_f(st.probs.size());
        float* mbuf = dev_f(st.mix.size());
        float* dq = dev_f(st.q.size());
        float* dk = dev_f(st.k.size());
        float* dv = dev_f(st.v.size());
        h2d(qbuf, post_q.data(), post_q.size() * 4);
        h2d(kbuf, post_k.data(), post_k.size() * 4);
        h2d(vbuf, cur_v.data(), cur_v.size() * 4);
        h2d(pbuf, st.probs.data(), st.probs.size() * 4);
        h2d(mbuf, st.mix.data(), st.mix.size() * 4);
        cudaMemset(dq, 0, st.q.size() * 4);
        cudaMemset(dk, 0, st.k.size() * 4);
        cudaMemset(dv, 0, st.v.size() * 4);
        must("zero gqa");
        LAUNCH(s.dev, npos, k_gqa_bwd<<<blocks_for(count), 128>>>(qbuf, kbuf, vbuf, pbuf, mbuf, dq, dk, dv, layer, limit, base));
        d2h(st.q.data(), dq, st.q.size() * 4);
        d2h(st.k.data(), dk, st.k.size() * 4);
        d2h(st.v.data(), dv, st.v.size() * 4);
        cudaFree(qbuf); cudaFree(kbuf); cudaFree(vbuf); cudaFree(pbuf); cudaFree(mbuf); cudaFree(dq); cudaFree(dk); cudaFree(dv);
        return;
    }
    cpu_ops++;
    float scale = 1.f / sqrtf((float)HD);
    for (ull b = 0; b < B; ++b)
        for (ull h = 0; h < H; ++h) {
            ull kvh = h / (H / KV);
            for (ull i = 0; i < SEQ; ++i) {
                float dot_pa = 0.f;
                for (ull j = 0; j < SEQ; ++j) {
                    float da = 0.f;
                    for (ull d = 0; d < HD; ++d) da += st.mix[(size_t)q_at(b, i, h, d)] * cur_v[(size_t)k_at(b, j, kvh, d)];
                    float p = st.probs[(size_t)p_at((ull)layer, b, h, i, j)];
                    dot_pa += p * da;
                    for (ull d = 0; d < HD; ++d) st.v[(size_t)k_at(b, j, kvh, d)] += p * st.mix[(size_t)q_at(b, i, h, d)];
                }
                for (ull j = 0; j < SEQ; ++j) {
                    if (!visible(i, j)) continue;
                    float da = 0.f;
                    for (ull d = 0; d < HD; ++d) da += st.mix[(size_t)q_at(b, i, h, d)] * cur_v[(size_t)k_at(b, j, kvh, d)];
                    float p = st.probs[(size_t)p_at((ull)layer, b, h, i, j)];
                    float ds = p * (da - dot_pa) * scale;
                    for (ull d = 0; d < HD; ++d) {
                        st.q[(size_t)q_at(b, i, h, d)] += ds * post_k[(size_t)k_at(b, j, kvh, d)];
                        st.k[(size_t)k_at(b, j, kvh, d)] += ds * post_q[(size_t)q_at(b, i, h, d)];
                    }
                }
            }
        }
}

void router_fwd(State& st, const Step& s, bool gpu) {
    int layer = s.layer;
    if (gpu) {
        float* logb = dev_f(st.rlog.size());
        float* pb = dev_f(st.rprob.size());
        ull* tb = dev_u(st.topi.size());
        float* pp = dev_f(st.topp.size());
        float* pw = dev_f(st.topw.size());
        h2d(logb, st.rlog.data(), st.rlog.size() * 4);
        h2d(pb, st.rprob.data(), st.rprob.size() * 4);
        h2d(tb, st.topi.data(), st.topi.size() * sizeof(ull));
        h2d(pp, st.topp.data(), st.topp.size() * 4);
        h2d(pw, st.topw.data(), st.topw.size() * 4);
        LAUNCH(s.dev, N, k_router<<<blocks_for(count), 128>>>(logb, pb, tb, pp, pw, layer, limit, base));
        d2h(st.rprob.data(), pb, st.rprob.size() * 4);
        d2h(st.topi.data(), tb, st.topi.size() * sizeof(ull));
        d2h(st.topp.data(), pp, st.topp.size() * 4);
        d2h(st.topw.data(), pw, st.topw.size() * 4);
        cudaFree(logb); cudaFree(pb); cudaFree(tb); cudaFree(pp); cudaFree(pw);
        return;
    }
    cpu_ops++;
    for (ull t = 0; t < N; ++t) {
        float m = -INFINITY, sum = 0.f;
        for (ull e = 0; e < E; ++e) m = fmaxf(m, st.rlog[(size_t)(t * E + e)]);
        for (ull e = 0; e < E; ++e) {
            float p = expf(st.rlog[(size_t)(t * E + e)] - m);
            st.rprob[(size_t)((ull)layer * N * E + t * E + e)] = p;
            sum += p;
        }
        for (ull e = 0; e < E; ++e) st.rprob[(size_t)((ull)layer * N * E + t * E + e)] /= sum;
        for (ull k = 0; k < TOPK; ++k) {
            ull best = 0;
            bool any = false;
            float bv = -1.f;
            for (ull e = 0; e < E; ++e) {
                bool taken = false;
                for (ull j = 0; j < k; ++j)
                    if (st.topi[(size_t)(((ull)layer * N + t) * TOPK + j)] == e) taken = true;
                if (taken) continue;
                float p = st.rprob[(size_t)((ull)layer * N * E + t * E + e)];
                if (!any || p > bv || (p == bv && e < best)) { best = e; bv = p; any = true; }
            }
            st.topi[(size_t)(((ull)layer * N + t) * TOPK + k)] = best;
            st.topp[(size_t)(((ull)layer * N + t) * TOPK + k)] = bv;
        }
        float sm = 0.f;
        for (ull k = 0; k < TOPK; ++k) sm += st.topp[(size_t)(((ull)layer * N + t) * TOPK + k)];
        for (ull k = 0; k < TOPK; ++k)
            st.topw[(size_t)(((ull)layer * N + t) * TOPK + k)] = st.topp[(size_t)(((ull)layer * N + t) * TOPK + k)] / sm;
    }
}

void silu_fwd(State& st, const Step& s, bool gpu) {
    ull n = N * I;
    ull off = ix((ull)s.layer, 0, (ull)s.expert, I);
    if (gpu) {
        float* g = dev_f(n);
        float* u = dev_f(n);
        float* h = dev_f(n);
        h2d(g, st.gate.data() + off, n * 4);
        h2d(u, st.up.data() + off, n * 4);
        LAUNCH(s.dev, n, k_silu_fwd<<<blocks_for(count), 128>>>(g, u, h, limit, base));
        d2h(st.hid.data() + off, h, n * 4);
        cudaFree(g); cudaFree(u); cudaFree(h);
        return;
    }
    cpu_ops++;
    for (ull t = 0; t < N; ++t)
        for (ull i = 0; i < I; ++i) {
            float z = st.gate[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i];
            float u = st.up[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i];
            float sg = 1.f / (1.f + expf(-z));
            st.hid[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i] = z * sg * u;
        }
}
void silu_bwd(State& st, const Step& s, bool gpu) {
    ull n = N * I;
    ull off = ix((ull)s.layer, 0, (ull)s.expert, I);
    if (gpu) {
        float* g = dev_f(n);
        float* u = dev_f(n);
        float* dh = dev_f(n);
        float* dg = dev_f(n);
        float* du = dev_f(n);
        h2d(g, st.gate.data() + off, n * 4);
        h2d(u, st.up.data() + off, n * 4);
        h2d(dh, st.dhid.data() + off, n * 4);
        LAUNCH(s.dev, n, k_silu_bwd<<<blocks_for(count), 128>>>(g, u, dh, dg, du, limit, base));
        d2h(st.dgate.data() + off, dg, n * 4);
        d2h(st.dup.data() + off, du, n * 4);
        cudaFree(g); cudaFree(u); cudaFree(dh); cudaFree(dg); cudaFree(du);
        return;
    }
    cpu_ops++;
    for (ull t = 0; t < N; ++t)
        for (ull i = 0; i < I; ++i) {
            float z = st.gate[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i];
            float u = st.up[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i];
            float dh = st.dhid[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i];
            float sg = 1.f / (1.f + expf(-z));
            float dsilu = sg * (1.f + z * (1.f - sg));
            st.dgate[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i] = dh * u * dsilu;
            st.dup[(size_t)ix((ull)s.layer, t, (ull)s.expert, I) + (size_t)i] = dh * (z * sg);
        }
}

void mix_fwd(State& st, const Step& s, bool gpu) {
    int layer = s.layer;
    if (gpu) {
        float* eo = dev_f(st.eout.size());
        ull* ti = dev_u(st.topi.size());
        float* tw = dev_f(st.topw.size());
        float* mx = dev_f(st.mixed.size());
        h2d(eo, st.eout.data(), st.eout.size() * 4);
        h2d(ti, st.topi.data(), st.topi.size() * sizeof(ull));
        h2d(tw, st.topw.data(), st.topw.size() * 4);
        LAUNCH(s.dev, N, k_mix_fwd<<<blocks_for(count), 128>>>(eo, ti, tw, mx, layer, limit, base));
        d2h(st.mixed.data(), mx, st.mixed.size() * 4);
        cudaFree(eo); cudaFree(ti); cudaFree(tw); cudaFree(mx);
        return;
    }
    cpu_ops++;
    std::fill(st.mixed.begin(), st.mixed.end(), 0.f);
    for (ull t = 0; t < N; ++t)
        for (ull k = 0; k < TOPK; ++k) {
            ull e = st.topi[(size_t)(((ull)layer * N + t) * TOPK + k)];
            float w = st.topw[(size_t)(((ull)layer * N + t) * TOPK + k)];
            for (ull d = 0; d < D; ++d)
                st.mixed[(size_t)(t * D + d)] += w * st.eout[(size_t)ix((ull)layer, t, e, D) + (size_t)d];
        }
}

void mix_bwd(State& st, const Step& s, bool gpu) {
    int layer = s.layer;
    if (gpu) {
        ull scratch_n = N * TOPK + N * E;
        float* dx = dev_f(st.dx.size());
        float* eo = dev_f(st.eout.size());
        float* rp = dev_f(st.rprob.size());
        ull* ti = dev_u(st.topi.size());
        float* tp = dev_f(st.topp.size());
        float* tw = dev_f(st.topw.size());
        float* de = dev_f(st.deout.size());
        float* dh = dev_f(st.dhid.size());
        float* dz = dev_f(st.dz.size());
        float* sc = dev_f(scratch_n);
        h2d(dx, st.dx.data(), st.dx.size() * 4);
        h2d(eo, st.eout.data(), st.eout.size() * 4);
        h2d(rp, st.rprob.data(), st.rprob.size() * 4);
        h2d(ti, st.topi.data(), st.topi.size() * sizeof(ull));
        h2d(tp, st.topp.data(), st.topp.size() * 4);
        h2d(tw, st.topw.data(), st.topw.size() * 4);
        h2d(de, st.deout.data(), st.deout.size() * 4);
        h2d(dh, st.dhid.data(), st.dhid.size() * 4);
        LAUNCH(s.dev, N, k_mix_bwd<<<blocks_for(count), 128>>>(dx, eo, rp, ti, tp, tw, de, dh, dz, sc, layer, limit, base));
        d2h(st.deout.data(), de, st.deout.size() * 4);
        d2h(st.dhid.data(), dh, st.dhid.size() * 4);
        d2h(st.dz.data(), dz, st.dz.size() * 4);
        cudaFree(dx); cudaFree(eo); cudaFree(rp); cudaFree(ti); cudaFree(tp); cudaFree(tw);
        cudaFree(de); cudaFree(dh); cudaFree(dz); cudaFree(sc);
        return;
    }
    cpu_ops++;
    for (ull t = 0; t < N; ++t)
        for (ull e = 0; e < E; ++e) {
            for (ull d = 0; d < D; ++d) st.deout[(size_t)ix((ull)layer, t, e, D) + (size_t)d] = 0.f;
            for (ull i = 0; i < I; ++i) st.dhid[(size_t)ix((ull)layer, t, e, I) + (size_t)i] = 0.f;
        }
    std::fill(st.dz.begin(), st.dz.end(), 0.f);
    for (ull t = 0; t < N; ++t) {
        std::vector<float> dw((size_t)TOPK), dp((size_t)E, 0.f);
        float ssum = 0.f, weight_dot = 0.f;
        for (ull k = 0; k < TOPK; ++k) {
            ssum += st.topp[(size_t)(((ull)layer * N + t) * TOPK + k)];
            ull e = st.topi[(size_t)(((ull)layer * N + t) * TOPK + k)];
            float dot = 0.f;
            for (ull d = 0; d < D; ++d) dot += st.dx[(size_t)(t * D + d)] * st.eout[(size_t)ix((ull)layer, t, e, D) + (size_t)d];
            dw[(size_t)k] = dot;
            float w = st.topw[(size_t)(((ull)layer * N + t) * TOPK + k)];
            weight_dot += dot * w;
            for (ull d = 0; d < D; ++d) st.deout[(size_t)ix((ull)layer, t, e, D) + (size_t)d] += w * st.dx[(size_t)(t * D + d)];
        }
        for (ull k = 0; k < TOPK; ++k) {
            float dvalues = dw[(size_t)k] / ssum - weight_dot / ssum;
            dp[(size_t)st.topi[(size_t)(((ull)layer * N + t) * TOPK + k)]] = dvalues;
        }
        float dotp = 0.f;
        for (ull e = 0; e < E; ++e) dotp += st.rprob[(size_t)((ull)layer * N * E + t * E + e)] * dp[(size_t)e];
        for (ull e = 0; e < E; ++e)
            st.dz[(size_t)(t * E + e)] = st.rprob[(size_t)((ull)layer * N * E + t * E + e)] * (dp[(size_t)e] - dotp);
    }
}

void loss_fwd(State& st, const Step& s, bool gpu) {
    if (!(s.b0 == 0 && s.c0 == 0)) return;
    if (s.pass == 1) std::fill(st.dxn.begin(), st.dxn.end(), 0.f);
    if (gpu) {
        float* lg = dev_f(st.logits.size());
        ull* tg = dev_u(st.tgt.size());
        float* dl = dev_f(st.dlogits.size());
        float* tl = dev_f(N);
        std::vector<float> tok((size_t)N);
        h2d(lg, st.logits.data(), st.logits.size() * 4);
        h2d(tg, st.tgt.data(), st.tgt.size() * sizeof(ull));
        LAUNCH(s.dev, N, k_loss<<<blocks_for(count), 128>>>(lg, tg, dl, tl, limit, base));
        d2h(st.dlogits.data(), dl, st.dlogits.size() * 4);
        d2h(tok.data(), tl, N * 4);
        st.loss = 0.f;
        for (float v : tok) st.loss += v;
        cudaFree(lg); cudaFree(tg); cudaFree(dl); cudaFree(tl);
        return;
    }
    cpu_ops++;
    st.loss = 0.f;
    for (ull t = 0; t < N; ++t) {
        float m = -INFINITY, sum = 0.f;
        for (ull v = 0; v < V; ++v) m = fmaxf(m, st.logits[(size_t)(t * V + v)]);
        for (ull v = 0; v < V; ++v) sum += expf(st.logits[(size_t)(t * V + v)] - m);
        st.loss += (m + logf(sum) - st.logits[(size_t)(t * V + st.tgt[(size_t)t])]) / (float)N;
        for (ull v = 0; v < V; ++v) st.dlogits[(size_t)(t * V + v)] = expf(st.logits[(size_t)(t * V + v)] - m) / sum / (float)N;
        st.dlogits[(size_t)(t * V + st.tgt[(size_t)t])] -= 1.f / (float)N;
    }
}

void embed_fwd(State& st, bool gpu, int dev) {
    if (gpu) {
        float* table = dev_f(V * D);
        ull* tok = dev_u(N);
        float* x = dev_f(N * D);
        h2d(table, st.P.data(), V * D * 4);
        h2d(tok, st.tok.data(), N * sizeof(ull));
        LAUNCH(dev, N * D, k_embed<<<blocks_for(count), 128>>>(table, tok, x, limit, base));
        d2h(st.x.data(), x, N * D * 4);
        cudaFree(table); cudaFree(tok); cudaFree(x);
        return;
    }
    cpu_ops++;
    for (ull t = 0; t < N; ++t)
        for (ull d = 0; d < D; ++d) st.x[(size_t)(t * D + d)] = st.P[(size_t)(st.tok[(size_t)t] * D + d)];
}
void embed_bwd(State& st, bool gpu, int dev) {
    if (gpu) {
        float* dtable = dev_f(V * D);
        ull* tok = dev_u(N);
        float* dx = dev_f(N * D);
        h2d(dtable, st.dP.data(), V * D * 4);
        h2d(tok, st.tok.data(), N * sizeof(ull));
        h2d(dx, st.dx.data(), N * D * 4);
        LAUNCH(dev, N * D, k_embed_bwd<<<blocks_for(count), 128>>>(dtable, tok, dx, limit, base));
        d2h(st.dP.data(), dtable, V * D * 4);
        cudaFree(dtable); cudaFree(tok); cudaFree(dx);
        return;
    }
    cpu_ops++;
    for (ull t = 0; t < N; ++t)
        for (ull d = 0; d < D; ++d) st.dP[(size_t)(st.tok[(size_t)t] * D + d)] += st.dx[(size_t)(t * D + d)];
}

void add_buf(std::vector<float>& x, const std::vector<float>& y, bool gpu, int dev) {
    if (gpu) {
        float* xb = dev_f(x.size());
        float* yb = dev_f(y.size());
        h2d(xb, x.data(), x.size() * 4);
        h2d(yb, y.data(), y.size() * 4);
        LAUNCH(dev, x.size(), k_add<<<blocks_for(count), 128>>>(xb, yb, limit, base));
        d2h(x.data(), xb, x.size() * 4);
        cudaFree(xb); cudaFree(yb);
        return;
    }
    cpu_ops++;
    for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}
void zero_buf(std::vector<float>& x, bool gpu, int dev) {
    if (gpu) {
        float* xb = dev_f(x.size());
        LAUNCH(dev, x.size(), k_zero<<<blocks_for(count), 128>>>(xb, limit, base));
        d2h(x.data(), xb, x.size() * 4);
        cudaFree(xb);
        return;
    }
    cpu_ops++;
    std::fill(x.begin(), x.end(), 0.f);
}

void sgd_home(State& st, const Step& s, bool gpu) {
    ull rows = s.r1 - s.r0, cols = s.c1 - s.c0;
    ull n = rows * cols;
    std::vector<float> tile((size_t)n), grad((size_t)n);
    for (ull r = 0; r < rows; ++r)
        for (ull c = 0; c < cols; ++c) {
            ull i = s.p_off + (s.r0 + r) * s.dout + s.c0 + c;
            tile[(size_t)(r * cols + c)] = st.P[(size_t)i];
            grad[(size_t)(r * cols + c)] = st.dP[(size_t)i];
        }
    if (gpu) {
        float* tb = dev_f(n);
        float* gb = dev_f(n);
        h2d(tb, tile.data(), n * 4);
        h2d(gb, grad.data(), n * 4);
        LAUNCH(s.dev, n, k_sgd<<<blocks_for(count), 128>>>(tb, gb, LR, limit, base));
        d2h(tile.data(), tb, n * 4);
        cudaFree(tb); cudaFree(gb);
    } else {
        cpu_ops++;
        for (ull i = 0; i < n; ++i) tile[(size_t)i] -= LR * grad[(size_t)i];
    }
    for (ull r = 0; r < rows; ++r)
        for (ull c = 0; c < cols; ++c)
            st.P[(size_t)(s.p_off + (s.r0 + r) * s.dout + s.c0 + c)] = tile[(size_t)(r * cols + c)];
}

std::mutex state_mu;
State g_state;

void io_full(int fd, void* buf, size_t n, bool writing) {
    char* p = static_cast<char*>(buf);
    while (n) {
        ssize_t k = writing ? ::send(fd, p, n, MSG_NOSIGNAL) : ::recv(fd, p, n, 0);
        if (k <= 0) { printf("FAIL socket\n"); exit(1); }
        p += k;
        n -= (size_t)k;
    }
}

std::vector<float> pack_tile(State& st, const Step& s) {
    std::vector<float> tile;
    if (s.layer == 2) {
        ull n = s.b0;
        if (n > st.x.size()) n = (ull)st.x.size();
        tile.assign(st.x.begin(), st.x.begin() + (size_t)n);
        return tile;
    }
    ull rows = s.r1 > s.r0 ? s.r1 - s.r0 : 0;
    ull cols = s.c1 > s.c0 ? s.c1 - s.c0 : 0;
    tile.resize((size_t)(rows * cols));
    const std::vector<float>& buf = s.layer == 1 ? st.dP : st.P;
    for (ull r = 0; r < rows; ++r)
        for (ull c = 0; c < cols; ++c) {
            ull i = s.p_off + (s.r0 + r) * s.dout + s.c0 + c;
            tile[(size_t)(r * cols + c)] = i < buf.size() ? buf[(size_t)i] : 0.f;
        }
    return tile;
}

void apply_tile(State& st, const Step& s, const std::vector<float>& tile) {
    if (s.layer == 2) {
        for (size_t i = 0; i < tile.size() && i < st.x.size(); ++i) st.x[i] = tile[i];
        return;
    }
    ull rows = s.r1 > s.r0 ? s.r1 - s.r0 : 0;
    ull cols = s.c1 > s.c0 ? s.c1 - s.c0 : 0;
    std::vector<float>& buf = s.layer == 1 ? st.dP : st.P;
    for (ull r = 0; r < rows && (size_t)(r * cols) < tile.size(); ++r)
        for (ull c = 0; c < cols; ++c) {
            ull i = s.p_off + (s.r0 + r) * s.dout + s.c0 + c;
            size_t at = (size_t)(r * cols + c);
            if (i < buf.size() && at < tile.size()) buf[(size_t)i] = tile[at];
        }
}

void write_tile(int fd, const std::vector<float>& tile) {
    uint32_t n = (uint32_t)tile.size();
    io_full(fd, &n, 4, true);
    if (n) io_full(fd, (void*)tile.data(), (size_t)n * 4, true);
}

std::vector<float> read_tile(int fd) {
    uint32_t n = 0;
    io_full(fd, &n, 4, false);
    std::vector<float> tile(n);
    if (n) io_full(fd, tile.data(), (size_t)n * 4, false);
    return tile;
}

int peer_fd(int rank, int other) {
    return g_peer[(size_t)rank * (size_t)NDEV + (size_t)other];
}

void set_peer_fd(int rank, int other, int fd) {
    g_peer[(size_t)rank * (size_t)NDEV + (size_t)other] = fd;
}

int listen_ephemeral(int& port, int backlog) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { printf("FAIL socket\n"); exit(1); }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (backlog < 1) backlog = 1;
    if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0 || listen(fd, backlog) < 0) {
        printf("FAIL bind\n"); exit(1);
    }
    socklen_t len = sizeof(addr);
    getsockname(fd, (sockaddr*)&addr, &len);
    port = ntohs(addr.sin_port);
    return fd;
}

void run_step(State& st, const Step& s) {
    if (s.dev < 0 || s.dev >= NDEV) { printf("FAIL device %d\n", s.dev); exit(1); }
    if (s.exec == Send) { printf("FAIL send is a member message\n"); exit(1); }
    bool gpu = DEV_KIND[s.dev] == 1;
    switch (s.exec) {
    case Embed:
        if (s.pass == 0) embed_fwd(st, gpu, s.dev);
        else embed_bwd(st, gpu, s.dev);
        break;
    case RmsNorm:
        if (s.pass == 0) rms_fwd(st, s, gpu);
        else rms_bwd(st, s, gpu);
        break;
    case Linear:
        run_linear(st, s, gpu);
        break;
    case RopeQ:
        if (s.pass == 1) zero_buf(st.dxn, gpu, s.dev);
        rope_apply(st.q, (int)H, s.pass == 1, gpu, s.dev);
        break;
    case RopeK:
        rope_apply(st.k, (int)KV, s.pass == 1, gpu, s.dev);
        break;
    case Gqa:
        if (s.pass == 0) gqa_fwd(st, s, gpu);
        else gqa_bwd(st, s, gpu);
        break;
    case AddAttn:
        if (s.pass == 0) add_buf(st.x, st.attn, gpu, s.dev);
        else zero_buf(st.mix, gpu, s.dev);
        break;
    case AddMoe:
        if (s.pass == 0) add_buf(st.x, st.mixed, gpu, s.dev);
        else zero_buf(st.dxn, gpu, s.dev);
        break;
    case Router:
        if (s.pass == 0) router_fwd(st, s, gpu);
        break;
    case Silu:
        if (s.pass == 0) silu_fwd(st, s, gpu);
        else silu_bwd(st, s, gpu);
        break;
    case Mix:
        if (s.pass == 0) mix_fwd(st, s, gpu);
        else mix_bwd(st, s, gpu);
        break;
    case Loss:
        loss_fwd(st, s, gpu);
        break;
    case Sgd:
        sgd_home(st, s, gpu);
        break;
    default:
        break;
    }
}

enum CmdOp { Finish = 0, Compute = 1, SendTile = 2, RecvTile = 3, Ready = 4, Done = 5 };

struct Cmd {
    int op, step, other;
};

struct Boot {
    int rank;
    int coord_port;
    int peer_port;
    int peer_fd;
};

void member_loop(Boot boot) {
    int ctrl = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)boot.coord_port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(ctrl, (sockaddr*)&addr, sizeof(addr)) < 0) { printf("FAIL connect\n"); exit(1); }
    int hello[2] = { boot.rank, boot.peer_port };
    io_full(ctrl, hello, sizeof(hello), true);
    std::vector<int> ports(NDEV);
    io_full(ctrl, ports.data(), ports.size() * sizeof(int), false);
    for (int i = 0; i < boot.rank; ++i) {
        sockaddr_in peer{};
        socklen_t len = sizeof(peer);
        int fd = accept(boot.peer_fd, (sockaddr*)&peer, &len);
        if (fd < 0) { printf("FAIL accept\n"); exit(1); }
        int other = 0;
        io_full(fd, &other, sizeof(other), false);
        set_peer_fd(boot.rank, other, fd);
    }
    for (int other = boot.rank + 1; other < NDEV; ++other) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in peer{};
        peer.sin_family = AF_INET;
        peer.sin_port = htons((uint16_t)ports[other]);
        peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (sockaddr*)&peer, sizeof(peer)) < 0) { printf("FAIL peer\n"); exit(1); }
        int me = boot.rank;
        io_full(fd, &me, sizeof(me), true);
        set_peer_fd(boot.rank, other, fd);
    }
    int up = 1;
    io_full(ctrl, &up, sizeof(up), true);
    for (;;) {
        Cmd cmd{};
        io_full(ctrl, &cmd, sizeof(cmd), false);
        if (cmd.op == Finish) break;
        if (cmd.op == SendTile) {
            std::vector<float> tile;
            {
                std::lock_guard<std::mutex> guard(state_mu);
                tile = pack_tile(g_state, STEPS[cmd.step]);
            }
            write_tile(peer_fd(boot.rank, cmd.other), tile);
        } else if (cmd.op == RecvTile) {
            Cmd ready{ Ready, cmd.step, boot.rank };
            io_full(ctrl, &ready, sizeof(ready), true);
            std::vector<float> tile = read_tile(peer_fd(boot.rank, cmd.other));
            std::lock_guard<std::mutex> guard(state_mu);
            apply_tile(g_state, STEPS[cmd.step], tile);
        } else if (cmd.op == Compute) {
            std::lock_guard<std::mutex> guard(state_mu);
            run_step(g_state, STEPS[cmd.step]);
        } else {
            printf("FAIL command %d\n", cmd.op);
            exit(1);
        }
        if (cmd.op != RecvTile) {
            Cmd done{ Done, cmd.step, boot.rank };
            io_full(ctrl, &done, sizeof(done), true);
        } else {
            Cmd done{ Done, cmd.step, boot.rank };
            io_full(ctrl, &done, sizeof(done), true);
        }
    }
    ::close(ctrl);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const ull PROBES[] = { @@PROBES@@ };
    if (sizeof(PROBES) == 0) { printf("FAIL device table\n"); return 1; }
    g_peer.assign((size_t)NDEV * (size_t)NDEV, -1);
    int nd = 0;
    cudaGetDeviceCount(&nd);
    int seen = 0;
    for (int i = 0; i < NDEV; ++i) {
        if (DEV_KIND[i] == 1) {
            if (nd <= 0) { printf("FAIL no cuda device for %d\n", i); return 1; }
            phys[i] = seen % nd;
            seen++;
        } else phys[i] = -1;
    }
    int coord_port = 0;
    int coord_fd = listen_ephemeral(coord_port, NDEV);
    std::vector<int> peer_ports(NDEV);
    std::vector<int> peer_fds(NDEV);
    std::vector<std::thread> members;
    for (int rank = 0; rank < NDEV; ++rank) {
        peer_fds[rank] = listen_ephemeral(peer_ports[rank], rank < 1 ? 1 : rank);
        Boot boot{ rank, coord_port, peer_ports[rank], peer_fds[rank] };
        members.emplace_back(member_loop, boot);
    }
    std::vector<int> ctrl(NDEV, -1);
    for (int n = 0; n < NDEV; ++n) {
        int fd = accept(coord_fd, nullptr, nullptr);
        if (fd < 0) { printf("FAIL accept\n"); return 1; }
        int hello[2] = { 0, 0 };
        io_full(fd, hello, sizeof(hello), false);
        if (hello[0] < 0 || hello[0] >= NDEV || ctrl[hello[0]] >= 0) { printf("FAIL join\n"); return 1; }
        ctrl[hello[0]] = fd;
    }
    printf("MEMBERS %d\n", NDEV);
    for (int rank = 0; rank < NDEV; ++rank) printf("JOIN %d\n", rank);
    for (int rank = 0; rank < NDEV; ++rank)
        io_full(ctrl[rank], peer_ports.data(), peer_ports.size() * sizeof(int), true);
    for (int rank = 0; rank < NDEV; ++rank) {
        int up = 0;
        io_full(ctrl[rank], &up, sizeof(up), false);
        if (up != 1) { printf("FAIL peers\n"); return 1; }
    }
    {
        std::lock_guard<std::mutex> guard(state_mu);
        alloc_state(g_state);
    }
    for (ull i = 0; i < P; ++i) g_state.P[(size_t)i] = 0.02f * ((i % 5ull) + 1.f);
    for (ull li = 0; li < L; ++li)
        for (ull d = 0; d < D; ++d)
            for (ull e = 0; e < E; ++e)
                g_state.P[(size_t)(ROUTER[li] + d * E + e)] = 0.05f * (e + 1.f) + 0.001f * (float)d;
    for (ull i = 0; i < N; ++i) {
        g_state.tok[(size_t)i] = (i * 5ull + 1ull) % V;
        g_state.tgt[(size_t)i] = (g_state.tok[(size_t)i] + 3ull) % V;
    }
    for (int i = 0; i < NSTEPS; ++i) {
        const Step& s = STEPS[i];
        if (s.exec == Send) {
            printf("HOP %d %d %d %llu\n", s.home, s.dev, s.layer, s.b1);
            Cmd recv{ RecvTile, i, s.home };
            io_full(ctrl[s.dev], &recv, sizeof(recv), true);
            Cmd ready{};
            io_full(ctrl[s.dev], &ready, sizeof(ready), false);
            if (ready.op != Ready) { printf("FAIL ready\n"); return 1; }
            Cmd send{ SendTile, i, s.dev };
            io_full(ctrl[s.home], &send, sizeof(send), true);
            Cmd done_s{};
            io_full(ctrl[s.home], &done_s, sizeof(done_s), false);
            Cmd done_r{};
            io_full(ctrl[s.dev], &done_r, sizeof(done_r), false);
            if (done_s.op != Done || done_r.op != Done) { printf("FAIL hop ack\n"); return 1; }
        } else {
            printf("COMPUTE %d %d %d\n", s.dev, s.exec, s.pass);
            Cmd cmd{ Compute, i, 0 };
            io_full(ctrl[s.dev], &cmd, sizeof(cmd), true);
            Cmd done{};
            io_full(ctrl[s.dev], &done, sizeof(done), false);
            if (done.op != Done) { printf("FAIL compute ack\n"); return 1; }
        }
    }
    for (int rank = 0; rank < NDEV; ++rank) {
        Cmd cmd{ Finish, 0, 0 };
        io_full(ctrl[rank], &cmd, sizeof(cmd), true);
        ::close(ctrl[rank]);
    }
    for (auto& member : members) member.join();
    ::close(coord_fd);
    printf("DONE\n");
    return 0;
}
