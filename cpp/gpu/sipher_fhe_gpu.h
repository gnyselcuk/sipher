// Sipher FHE GPU Inference Engine — FIDESlib CKKS
// GPU-accelerated FHE-native Turkish Finance LM
// Bootstrap destekli 20 katman tam inference
#pragma once

#include <fideslib.hpp>

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <cassert>
#include <set>

using namespace fideslib;

// ── Config ──
struct SipherGPUConfig {
    int vocab_size = 16000;
    int d_model = 1024;
    int seq_len = 256;
    int n_layers = 20;
    int n_heads = 16;
    int d_k = 64;
    int emb_dim = 128;
    int window = 32;
    int ffn_h = 4096;
    int feat_degree = 1;   // feature-map derecesi: 1 = kapalı (linear global), 2 = [x; x²]

    // FHE parameters
    int ring_dim = 16384;   // RTX 4060 8GB: 16384 (32768 OOM)
    int scale_bits = 59;
    int first_mod_bits = 60;
    int mult_depth = 15;
    int bootstrap_interval = 5;
    std::vector<uint32_t> level_budget = {3, 3};
    std::vector<int> gpu_devices = {0};

    static SipherGPUConfig load(const std::string& path);
};

// ── Tensor (plaintext, row-major) ──
struct GPUTensor {
    std::vector<double> data;
    std::vector<int> shape;

    GPUTensor() = default;
    GPUTensor(const std::vector<double>& d, const std::vector<int>& s) : data(d), shape(s) {}

    static GPUTensor load_bin(const std::string& path);
    double& at(int i) { return data[i]; }
    double at(int i) const { return data[i]; }
    int size() const { return data.size(); }
    int rows() const { return shape[0]; }
    int cols() const { return shape.size() > 1 ? shape[1] : 1; }
    double& at(int r, int c) { return data[r * cols() + c]; }
    double at(int r, int c) const { return data[r * cols() + c]; }
};

// ── Debug: >=0 ise bu katmanın FHE ara değerlerini plaintext referansla karşılaştır ──
extern int g_dump_layer;

// ── FIDESlib CKKS Wrapper (GPU) ──
class FIDESContext {
public:
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
    uint32_t n_slots;
    uint32_t ring_dim;
    uint32_t depth;
    bool gpu_loaded = false;

    void init(const SipherGPUConfig& cfg);

    Ciphertext<DCRTPoly> encrypt(const std::vector<double>& plain, uint32_t level = 0);
    std::vector<double> decrypt(Ciphertext<DCRTPoly>& ct, int n = -1);

    // GPU-accelerated operations
    Ciphertext<DCRTPoly> add(Ciphertext<DCRTPoly>& a, Ciphertext<DCRTPoly>& b);
    Ciphertext<DCRTPoly> add_plain(Ciphertext<DCRTPoly>& ct, const std::vector<double>& plain);
    Ciphertext<DCRTPoly> mult(Ciphertext<DCRTPoly>& a, Ciphertext<DCRTPoly>& b);
    Ciphertext<DCRTPoly> mult_plain(Ciphertext<DCRTPoly>& ct, const std::vector<double>& plain);
    Ciphertext<DCRTPoly> mult_scalar(Ciphertext<DCRTPoly>& ct, double scalar);
    Ciphertext<DCRTPoly> rotate(Ciphertext<DCRTPoly>& ct, int steps);
    Ciphertext<DCRTPoly> negate(Ciphertext<DCRTPoly>& ct);

    // BSGS matvec: y = W @ x (encrypted), GPU-accelerated
    Ciphertext<DCRTPoly> matvec(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                 int in_dim, int out_dim);

    // Hoisted BSGS matvec: 1 decompose + N cheap rotations (baby-step hoisting)
    Ciphertext<DCRTPoly> matvec_hoisted(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                         int in_dim, int out_dim);

    // ConvolutionTransform matvec: fused GPU kernel + pre-encoded plaintext
    Ciphertext<DCRTPoly> matvec_conv(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                      int in_dim, int out_dim);

    // Pre-encoded matvec: diagonals already encoded, pure GPU
    std::vector<Plaintext> encode_diagonals(const GPUTensor& W, int in_dim, int out_dim);
    Ciphertext<DCRTPoly> matvec_pre(Ciphertext<DCRTPoly>& enc_x,
                                     std::vector<Plaintext>& diag_pts,
                                     int in_dim, int out_dim);

    // Tek çağrıda encode + pre-eval (benchmark: naive'e göre ~13.5× hızlı, d=1024)
    Ciphertext<DCRTPoly> matvec_fast(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                     int in_dim, int out_dim);

    // Replicated BSGS (mamba3 "input-replicated true BSGS"):
    // R köşegeni TEK plaintext'te → encode + ct-pt çarpım ~R× azalır.
    // out_dim ≤ n_slots/4 - r + 1 olan matvec'lerde r=4; büyük out'ta matvec_fast'e düşer.
    Ciphertext<DCRTPoly> matvec_rep(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                    int in_dim, int out_dim);

    // Bootstrap: refresh noise budget
    Ciphertext<DCRTPoly> bootstrap(Ciphertext<DCRTPoly>& ct);

    // Precompute rotation keys for matvec
    std::vector<int32_t> compute_rotation_indices(int max_dim);
};

// ── Sipher Layer (GPU) ──
class SipherGPULayer {
public:
    GPUTensor W_Q, W_K, W_V, W_O;
    GPUTensor ffn_up, ffn_down;
    double gate_global, gate_local;
    double poly_c0, poly_c1, poly_c2;
    GPUTensor norm1_w, norm1_b, norm2_w, norm2_b;
    int feat_degree = 1;

    void load(const std::string& dir, int layer_idx);

    Ciphertext<DCRTPoly> forward(
        FIDESContext& fides,
        Ciphertext<DCRTPoly>& enc_x,
        const GPUTensor& K_all,
        const GPUTensor& V_all,
        const std::vector<bool>& window_mask,
        int pos,
        int n_heads,
        int d_k,
        int seq_len
    );

    // Plaintext LayerNorm: (x - mean)/std * gamma + beta (eps=1e-5, eğitilmiş modelle aynı)
    static void norm_plain(const std::vector<double>& in, const GPUTensor& w,
                           const GPUTensor& b, int D, std::vector<double>& out);

    // Plaintext mirror forward: eğitilmiş modelin pre-norm bloğunu TÜM pozisyonlar için hesaplar.
    // K/V protokolü (her katmanın K/V'si bir önceki katman çıktısından) + parity referansı.
    //   x → norm1 → Q,K,V → attn → gate → W_O → +residual
    //       → norm2 → FFN(polyact) → +residual
    // K_all/V_all: norm1(hs)'ten hesaplanmış olarak verilir (infer'da üretilir).
    void forward_plaintext(
        const std::vector<std::vector<double>>& hs,
        const GPUTensor& K_all, const GPUTensor& V_all,
        int window_size, int n_heads, int d_k, int seq_len,
        std::vector<std::vector<double>>& out);
};

// ── Sipher Model (GPU) ──
class SipherGPUModel {
public:
    SipherGPUConfig config;
    GPUTensor token_emb, emb_proj, pos_emb, head_proj;
    std::vector<SipherGPULayer> layers;

    void load(const std::string& weights_dir);

    Ciphertext<DCRTPoly> embed(FIDESContext& fides, const std::vector<int>& tokens, int pos);

    std::vector<double> infer(FIDESContext& fides, const std::vector<int>& tokens);

    int generate_next(FIDESContext& fides, const std::vector<int>& tokens,
                      double temp = 0.7, int top_k = 40);
};

// ── Timing ──
struct GPUTimer {
    std::chrono::high_resolution_clock::time_point start;
    GPUTimer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
};
