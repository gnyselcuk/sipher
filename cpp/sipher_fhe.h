// Sipher FHE Inference Engine — OpenFHE CKKS
// FHE-native Turkish Finance LM inference
#pragma once

#include <openfhe.h>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>

using namespace lbcrypto;

// ── Config ──
struct SipherConfig {
    int vocab_size = 16000;
    int d_model = 1024;
    int seq_len = 256;
    int n_layers = 20;
    int n_heads = 16;
    int d_k = 64;
    int emb_dim = 128;
    int window = 32;
    int ffn_h = 4096;  // d_model * 4

    static SipherConfig load(const std::string& path);
};

// ── Tensor (plaintext, row-major) ──
struct Tensor {
    std::vector<double> data;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(const std::vector<double>& d, const std::vector<int>& s) : data(d), shape(s) {}

    static Tensor load_bin(const std::string& path);
    double& at(int i) { return data[i]; }
    double at(int i) const { return data[i]; }
    int size() const { return data.size(); }

    // Matrix ops (2D only)
    int rows() const { return shape[0]; }
    int cols() const { return shape.size() > 1 ? shape[1] : 1; }
    double& at(int r, int c) { return data[r * cols() + c]; }
    double at(int r, int c) const { return data[r * cols() + c]; }
};

// ── CKKS Wrapper ──
class CKKSContext {
public:
    CryptoContext<DCRTPoly> cc;
    PublicKey<DCRTPoly> pub;
    PrivateKey<DCRTPoly> sec;
    int n_slots;
    double scale;

    void init(int poly_mod = 16384, int scale_bits = 40, int levels = 9);

    Ciphertext<DCRTPoly> encrypt(const std::vector<double>& plain);
    std::vector<double> decrypt(Ciphertext<DCRTPoly> ct, int n = -1);

    Ciphertext<DCRTPoly> add(Ciphertext<DCRTPoly> a, Ciphertext<DCRTPoly> b);
    Ciphertext<DCRTPoly> add_plain(Ciphertext<DCRTPoly> ct, const std::vector<double>& plain);
    Ciphertext<DCRTPoly> mult(Ciphertext<DCRTPoly> a, Ciphertext<DCRTPoly> b);
    Ciphertext<DCRTPoly> mult_plain(Ciphertext<DCRTPoly> ct, const std::vector<double>& plain);
    Ciphertext<DCRTPoly> mult_scalar(Ciphertext<DCRTPoly> ct, double scalar);
    Ciphertext<DCRTPoly> rotate(Ciphertext<DCRTPoly> ct, int steps);
    Ciphertext<DCRTPoly> negate(Ciphertext<DCRTPoly> ct);

    // BSGS matvec: y = W @ x (encrypted)
    Ciphertext<DCRTPoly> matvec(Ciphertext<DCRTPoly> enc_x, const Tensor& W,
                                 int in_dim, int out_dim);
};

// ── Sipher Layer ──
class SipherLayer {
public:
    Tensor W_Q, W_K, W_V, W_O;
    Tensor ffn_up, ffn_down;
    double gate_global, gate_local;
    double poly_c0, poly_c1, poly_c2;
    Tensor norm1_w, norm1_b, norm2_w, norm2_b;

    void load(const std::string& dir, int layer_idx);

    // Full layer inference (1 position)
    // enc_x: encrypted input [d_model]
    // K_all, V_all: plaintext K,V for all positions [seq, n_heads*d_k]
    // Returns: encrypted output [d_model]
    Ciphertext<DCRTPoly> forward(
        CKKSContext& ckks,
        Ciphertext<DCRTPoly> enc_x,
        const Tensor& K_all,
        const Tensor& V_all,
        const std::vector<bool>& window_mask,
        int pos
    );
};

// ── Sipher Model ──
class SipherModel {
public:
    SipherConfig config;
    Tensor token_emb, emb_proj, pos_emb, head_proj;
    std::vector<SipherLayer> layers;

    void load(const std::string& weights_dir);

    // Embedding lookup (plaintext) + encrypt
    Ciphertext<DCRTPoly> embed(CKKSContext& ckks, const std::vector<int>& tokens, int pos);

    // Full inference: 1 token, all layers
    // Returns: logits [vocab_size]
    std::vector<double> infer(CKKSContext& ckks, const std::vector<int>& tokens);

    // Generate next token
    int generate_next(CKKSContext& ckks, const std::vector<int>& tokens,
                      double temp = 0.7, int top_k = 40);
};

// ── Timing ──
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start).count();
    }
};
