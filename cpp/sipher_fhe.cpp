// Sipher FHE Inference Engine — Implementation
#include "sipher_fhe.h"
#include <algorithm>
#include <random>
#include <cassert>

// ═══════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════

SipherConfig SipherConfig::load(const std::string& path) {
    SipherConfig cfg;
    std::ifstream f(path);
    std::string key;
    while (f >> key) {
        if (key == "vocab_size") f >> cfg.vocab_size;
        else if (key == "d_model") f >> cfg.d_model;
        else if (key == "seq_len") f >> cfg.seq_len;
        else if (key == "n_layers") f >> cfg.n_layers;
        else if (key == "n_heads") f >> cfg.n_heads;
        else if (key == "d_k") f >> cfg.d_k;
        else if (key == "emb_dim") f >> cfg.emb_dim;
        else if (key == "window") f >> cfg.window;
        else if (key == "ffn_h") f >> cfg.ffn_h;
    }
    return cfg;
}

// ═══════════════════════════════════════════════════════════
// Tensor
// ═══════════════════════════════════════════════════════════

Tensor Tensor::load_bin(const std::string& path) {
    // Read metadata
    std::string meta_path = path + ".meta";
    // Try .meta file first, then .bin.meta
    std::ifstream mf(meta_path);
    if (!mf.is_open()) {
        // Try without .bin extension
        std::string base = path.substr(0, path.find(".bin"));
        mf.open(base + ".meta");
    }

    Tensor t;
    if (mf.is_open()) {
        int ndim;
        mf >> ndim;
        t.shape.resize(ndim);
        for (int i = 0; i < ndim; i++) mf >> t.shape[i];
    }

    // Read binary data
    std::ifstream bf(path, std::ios::binary);
    if (!bf.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return t;
    }
    bf.seekg(0, std::ios::end);
    size_t bytes = bf.tellg();
    bf.seekg(0, std::ios::beg);
    size_t n = bytes / sizeof(double);
    t.data.resize(n);
    bf.read(reinterpret_cast<char*>(t.data.data()), bytes);

    if (t.shape.empty()) {
        t.shape = {(int)n};
    }
    return t;
}

// ═══════════════════════════════════════════════════════════
// CKKS Context
// ═══════════════════════════════════════════════════════════

void CKKSContext::init(int poly_mod, int scale_bits, int levels) {
    // OpenFHE 128-bit security: ring dim >= 32768 for deep circuits
    if (poly_mod < 32768) poly_mod = 32768;
    n_slots = poly_mod / 2;
    scale = std::pow(2.0, scale_bits);

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(levels);
    params.SetScalingModSize(scale_bits);
    params.SetBatchSize(n_slots);
    params.SetSecurityLevel(HEStd_128_classic);
    params.SetRingDim(poly_mod);

    cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keys = cc->KeyGen();
    pub = keys.publicKey;
    sec = keys.secretKey;
    cc->EvalMultKeyGen(sec);
    // BSGS matvec için rotasyon anahtarları (bellek optimize)
    // Baby steps: 1..63, Giant steps: 8'in katları 64..4096
    std::vector<int32_t> rot_indices;
    for (int i = 1; i <= 63; i++) rot_indices.push_back(i);
    for (int i = 64; i <= 4096; i += 8) rot_indices.push_back(i);
    std::cout << "  Generating " << rot_indices.size() << " rotation keys..." << std::endl;
    cc->EvalRotateKeyGen(sec, rot_indices);

    std::cout << "  CKKS: poly=" << poly_mod << " slots=" << n_slots
              << " scale=2^" << scale_bits << " levels=" << levels << std::endl;
}

Ciphertext<DCRTPoly> CKKSContext::encrypt(const std::vector<double>& plain) {
    std::vector<double> padded(n_slots, 0.0);
    for (size_t i = 0; i < plain.size() && i < (size_t)n_slots; i++)
        padded[i] = plain[i];
    auto pt = cc->MakeCKKSPackedPlaintext(padded);
    return cc->Encrypt(pub, pt);
}

std::vector<double> CKKSContext::decrypt(Ciphertext<DCRTPoly> ct, int n) {
    Plaintext pt;
    cc->Decrypt(sec, ct, &pt);
    auto vals = pt->GetCKKSPackedValue();
    if (n < 0) n = n_slots;
    std::vector<double> result(n);
    for (int i = 0; i < n && i < (int)vals.size(); i++)
        result[i] = vals[i].real();
    return result;
}

Ciphertext<DCRTPoly> CKKSContext::add(Ciphertext<DCRTPoly> a, Ciphertext<DCRTPoly> b) {
    return cc->EvalAdd(a, b);
}

Ciphertext<DCRTPoly> CKKSContext::add_plain(Ciphertext<DCRTPoly> ct, const std::vector<double>& plain) {
    std::vector<double> padded(n_slots, 0.0);
    for (size_t i = 0; i < plain.size() && i < (size_t)n_slots; i++)
        padded[i] = plain[i];
    auto pt = cc->MakeCKKSPackedPlaintext(padded);
    return cc->EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CKKSContext::mult(Ciphertext<DCRTPoly> a, Ciphertext<DCRTPoly> b) {
    auto result = cc->EvalMult(a, b);
    cc->Rescale(result);
    return result;
}

Ciphertext<DCRTPoly> CKKSContext::mult_plain(Ciphertext<DCRTPoly> ct, const std::vector<double>& plain) {
    std::vector<double> padded(n_slots, 0.0);
    for (size_t i = 0; i < plain.size() && i < (size_t)n_slots; i++)
        padded[i] = plain[i];
    auto pt = cc->MakeCKKSPackedPlaintext(padded);
    return cc->EvalMult(ct, pt);
}

Ciphertext<DCRTPoly> CKKSContext::mult_scalar(Ciphertext<DCRTPoly> ct, double scalar) {
    std::vector<double> s(n_slots, scalar);
    auto pt = cc->MakeCKKSPackedPlaintext(s);
    return cc->EvalMult(ct, pt);
}

Ciphertext<DCRTPoly> CKKSContext::rotate(Ciphertext<DCRTPoly> ct, int steps) {
    return cc->EvalRotate(ct, steps);
}

Ciphertext<DCRTPoly> CKKSContext::negate(Ciphertext<DCRTPoly> ct) {
    return cc->EvalNegate(ct);
}

// ═══════════════════════════════════════════════════════════
// BSGS Matvec: y = W @ x (encrypted) — OPTİMİZE
// EvalFastRotationPrecompute ile hoisted rotations
// ═══════════════════════════════════════════════════════════

Ciphertext<DCRTPoly> CKKSContext::matvec(Ciphertext<DCRTPoly> enc_x, const Tensor& W,
                                          int in_dim, int out_dim) {
    int baby = (int)std::ceil(std::sqrt((double)in_dim));
    int giant = (int)std::ceil((double)in_dim / baby);

    // Input replikasyonu: periyodik N slot (diagonal wrapping için)
    Ciphertext<DCRTPoly> enc_rep = enc_x;
    for (int shift = in_dim; shift < n_slots; shift *= 2) {
        auto shifted = rotate(enc_rep, shift);
        enc_rep = add(enc_rep, shifted);
    }

    // Baby-step rotations
    std::vector<Ciphertext<DCRTPoly>> baby_rots(baby);
    baby_rots[0] = enc_rep;
    for (int b = 1; b < baby; b++) {
        baby_rots[b] = rotate(enc_rep, b);
    }

    // Giant steps (Halevi-Shoup BSGS: slot shift + col wrapping)
    Ciphertext<DCRTPoly> acc;
    bool first = true;

    for (int g = 0; g < giant; g++) {
        Ciphertext<DCRTPoly> z_g;
        bool first_inner = false;

        for (int b = 0; b < baby; b++) {
            int k = g * baby + b;
            if (k >= in_dim) break;

            // BSGS diagonal: slot g*baby+row, col wrapping
            std::vector<double> diag(n_slots, 0.0);
            bool has_nonzero = false;
            for (int i = g * baby; i < g * baby + out_dim && i < n_slots; i++) {
                int row = i - g * baby;
                int col = (i + b) % in_dim;
                diag[i] = W.at(row, col);
                if (std::abs(diag[i]) > 1e-15) has_nonzero = true;
            }
            if (!has_nonzero) continue;

            auto prod = mult_plain(baby_rots[b], diag);
            if (!first_inner) {
                z_g = prod;
                first_inner = true;
            } else {
                z_g = add(z_g, prod);
            }
        }

        if (z_g) {
            if (g > 0) {
                z_g = rotate(z_g, g * baby);
            }
            if (first) {
                acc = z_g;
                first = false;
            } else {
                acc = add(acc, z_g);
            }
        }
    }

    return acc;
}

// ═══════════════════════════════════════════════════════════
// Sipher Layer
// ═══════════════════════════════════════════════════════════

void SipherLayer::load(const std::string& dir, int idx) {
    std::string p = dir + "/layers." + std::to_string(idx);
    W_Q = Tensor::load_bin(p + ".W_Q.bin");
    W_K = Tensor::load_bin(p + ".W_K.bin");
    W_V = Tensor::load_bin(p + ".W_V.bin");
    W_O = Tensor::load_bin(p + ".W_O.bin");
    ffn_up = Tensor::load_bin(p + ".ffn_up.bin");
    ffn_down = Tensor::load_bin(p + ".ffn_down.bin");
    norm1_w = Tensor::load_bin(p + ".norm1_w.bin");
    norm1_b = Tensor::load_bin(p + ".norm1_b.bin");
    norm2_w = Tensor::load_bin(p + ".norm2_w.bin");
    norm2_b = Tensor::load_bin(p + ".norm2_b.bin");

    auto gg = Tensor::load_bin(p + ".gate_global.bin");
    auto gl = Tensor::load_bin(p + ".gate_local.bin");
    auto pc = Tensor::load_bin(p + ".poly_coeffs.bin");
    gate_global = gg.data[0];
    gate_local = gl.data[0];
    poly_c0 = pc.data[0];
    poly_c1 = pc.data[1];
    poly_c2 = pc.data[2];
}

Ciphertext<DCRTPoly> SipherLayer::forward(
    CKKSContext& ckks,
    Ciphertext<DCRTPoly> enc_x,
    const Tensor& K_all,
    const Tensor& V_all,
    const std::vector<bool>& window_mask,
    int pos)
{
    int D = W_Q.rows();
    int NH = W_Q.rows() / (W_Q.cols() > 1 ? W_Q.cols() : 1);
    // Infer NH and DK from weight shapes
    int nh_dk = W_Q.rows();  // W_Q: [nh*dk, d]
    int DK = W_Q.cols() > 0 ? nh_dk / (nh_dk / (W_Q.cols() > 1 ? W_Q.cols() : 1)) : 64;
    // Simplified: use config values
    int n_heads = 16;
    int d_k = 64;
    int seq = K_all.rows();

    // Q matvec (FHE)
    auto enc_q = ckks.matvec(enc_x, W_Q, D, nh_dk);
    auto q_dec = ckks.decrypt(enc_q, nh_dk);
    // Scale by 1/sqrt(dk)
    for (auto& v : q_dec) v /= std::sqrt((double)d_k);

    // Global causal linear attention (client-side KV_cum + FHE Q@KV)
    std::vector<double> global_out(nh_dk, 0.0);
    for (int h = 0; h < n_heads; h++) {
        // KV_cum for head h
        std::vector<std::vector<double>> KV_cum(d_k, std::vector<double>(d_k, 0.0));
        for (int t = 0; t <= pos; t++) {
            for (int i = 0; i < d_k; i++) {
                for (int j = 0; j < d_k; j++) {
                    KV_cum[i][j] += K_all.at(t, h * d_k + i) * V_all.at(t, h * d_k + j);
                }
            }
        }
        // Q_h @ KV_cum (FHE matvec)
        std::vector<double> q_h(q_dec.begin() + h * d_k, q_dec.begin() + (h + 1) * d_k);
        auto enc_q_h = ckks.encrypt(q_h);

        // KV_cum as matrix [dk, dk]
        Tensor KV_tensor;
        KV_tensor.shape = {d_k, d_k};
        KV_tensor.data.resize(d_k * d_k);
        for (int i = 0; i < d_k; i++)
            for (int j = 0; j < d_k; j++)
                KV_tensor.at(i, j) = KV_cum[i][j];

        auto enc_g = ckks.matvec(enc_q_h, KV_tensor, d_k, d_k);
        auto g_dec = ckks.decrypt(enc_g, d_k);
        for (int i = 0; i < d_k; i++)
            global_out[h * d_k + i] = g_dec[i];
    }

    // Local window attention (FHE Q@K^T → client softmax → FHE attn@V)
    std::vector<double> local_out(nh_dk, 0.0);
    for (int h = 0; h < n_heads; h++) {
        std::vector<double> q_h(q_dec.begin() + h * d_k, q_dec.begin() + (h + 1) * d_k);
        auto enc_q_h = ckks.encrypt(q_h);

        // K_h: [seq, dk]
        Tensor K_h;
        K_h.shape = {seq, d_k};
        K_h.data.resize(seq * d_k);
        for (int t = 0; t < seq; t++)
            for (int i = 0; i < d_k; i++)
                K_h.at(t, i) = K_all.at(t, h * d_k + i);

        // scores = Q_h @ K_h^T (FHE matvec: [1,dk] @ [dk,seq] → [1,seq])
        Tensor K_h_T;
        K_h_T.shape = {d_k, seq};
        K_h_T.data.resize(d_k * seq);
        for (int i = 0; i < d_k; i++)
            for (int t = 0; t < seq; t++)
                K_h_T.at(i, t) = K_h.at(t, i);

        auto enc_scores = ckks.matvec(enc_q_h, K_h_T, d_k, seq);
        auto scores = ckks.decrypt(enc_scores, seq);

        // Client-side: mask + softmax
        for (int t = 0; t < seq; t++) {
            if (!window_mask[t]) scores[t] = -1e9;
        }
        double max_s = *std::max_element(scores.begin(), scores.end());
        double sum_exp = 0.0;
        for (int t = 0; t < seq; t++) {
            scores[t] = std::exp(scores[t] - max_s);
            sum_exp += scores[t];
        }
        for (int t = 0; t < seq; t++) scores[t] /= sum_exp;

        // attn @ V_h (FHE matvec)
        auto enc_attn = ckks.encrypt(scores);
        Tensor V_h;
        V_h.shape = {seq, d_k};
        V_h.data.resize(seq * d_k);
        for (int t = 0; t < seq; t++)
            for (int i = 0; i < d_k; i++)
                V_h.at(t, i) = V_all.at(t, h * d_k + i);

        auto enc_l = ckks.matvec(enc_attn, V_h, seq, d_k);
        auto l_dec = ckks.decrypt(enc_l, d_k);
        for (int i = 0; i < d_k; i++)
            local_out[h * d_k + i] = l_dec[i];
    }

    // Gate combine + W_O + residual (FHE)
    std::vector<double> combined(nh_dk);
    for (int i = 0; i < nh_dk; i++)
        combined[i] = gate_global * global_out[i] + gate_local * local_out[i];

    auto enc_comb = ckks.encrypt(combined);
    auto enc_wo = ckks.matvec(enc_comb, W_O, nh_dk, D);
    auto enc_post = ckks.add(enc_x, enc_wo);  // residual

    // PolyFFN (FHE: matvec → ct×ct → matvec)
    auto enc_up = ckks.matvec(enc_post, ffn_up, D, ffn_up.rows());

    // PolyAct: c1*x + c2*x²
    auto enc_x2 = ckks.mult(enc_up, enc_up);  // ct×ct!
    auto enc_act = ckks.mult_scalar(enc_up, poly_c1);
    auto enc_x2_scaled = ckks.mult_scalar(enc_x2, poly_c2);
    enc_act = ckks.add(enc_act, enc_x2_scaled);
    if (std::abs(poly_c0) > 1e-10) {
        std::vector<double> c0_vec(ffn_up.rows(), poly_c0);
        enc_act = ckks.add_plain(enc_act, c0_vec);
    }

    auto enc_down = ckks.matvec(enc_act, ffn_down, ffn_up.rows(), D);
    auto enc_out = ckks.add(enc_post, enc_down);  // residual

    return enc_out;
}

// ═══════════════════════════════════════════════════════════
// Sipher Model
// ═══════════════════════════════════════════════════════════

void SipherModel::load(const std::string& dir) {
    config = SipherConfig::load(dir + "/config.txt");
    token_emb = Tensor::load_bin(dir + "/token_emb.bin");
    emb_proj = Tensor::load_bin(dir + "/emb_proj.bin");
    pos_emb = Tensor::load_bin(dir + "/pos_emb.bin");
    head_proj = Tensor::load_bin(dir + "/head_proj.bin");

    layers.resize(config.n_layers);
    for (int i = 0; i < config.n_layers; i++) {
        layers[i].load(dir, i);
        std::cout << "  Layer " << i << " loaded" << std::endl;
    }
    std::cout << "  Model: " << config.n_layers << " layers, d=" << config.d_model << std::endl;
}

Ciphertext<DCRTPoly> SipherModel::embed(CKKSContext& ckks, const std::vector<int>& tokens, int pos) {
    int D = config.d_model;
    int E = config.emb_dim;

    // Embedding lookup (plaintext)
    std::vector<double> emb(D, 0.0);
    int tok = tokens[pos];
    for (int i = 0; i < E; i++) {
        double e = token_emb.at(tok, i);
        for (int j = 0; j < D; j++) {
            emb[j] += e * emb_proj.at(j, i);
        }
    }
    // Add positional embedding
    for (int j = 0; j < D; j++) {
        emb[j] += pos_emb.at(pos, j);
    }

    return ckks.encrypt(emb);
}

std::vector<double> SipherModel::infer(CKKSContext& ckks, const std::vector<int>& tokens) {
    int D = config.d_model;
    int SEQ = config.seq_len;
    int NH = config.n_heads;
    int DK = config.d_k;
    int pos = SEQ - 1;

    // Compute K, V for all positions (plaintext, layer 0 weights)
    Tensor K_all, V_all;
    K_all.shape = {SEQ, NH * DK};
    V_all.shape = {SEQ, NH * DK};
    K_all.data.resize(SEQ * NH * DK, 0.0);
    V_all.data.resize(SEQ * NH * DK, 0.0);

    for (int t = 0; t < SEQ; t++) {
        int tok = (t < (int)tokens.size()) ? tokens[t] : 0;
        std::vector<double> emb_t(D, 0.0);
        for (int i = 0; i < config.emb_dim; i++) {
            double e = token_emb.at(tok, i);
            for (int j = 0; j < D; j++)
                emb_t[j] += e * emb_proj.at(j, i);
        }
        for (int j = 0; j < D; j++)
            emb_t[j] += pos_emb.at(t, j);

        auto& W_K = layers[0].W_K;
        auto& W_V = layers[0].W_V;
        for (int i = 0; i < NH * DK; i++) {
            double k_val = 0.0, v_val = 0.0;
            for (int j = 0; j < D; j++) {
                k_val += emb_t[j] * W_K.at(i, j);
                v_val += emb_t[j] * W_V.at(i, j);
            }
            K_all.at(t, i) = k_val;
            V_all.at(t, i) = v_val;
        }
    }

    // Window mask
    std::vector<bool> window_mask(SEQ, false);
    for (int t = 0; t < SEQ; t++) {
        int diff = pos - t;
        if (diff >= 0 && diff < config.window)
            window_mask[t] = true;
    }

    // Embed + encrypt
    auto enc_x = embed(ckks, tokens, pos);

    // Multi-layer inference with bootstrapping
    int bootstrap_interval = 5;  // Bootstrap every 5 layers (safety margin)
    double total_layer_time = 0;

    for (int l = 0; l < config.n_layers; l++) {
        Timer timer;
        enc_x = layers[l].forward(ckks, enc_x, K_all, V_all, window_mask, pos);
        double layer_time = timer.elapsed();
        total_layer_time += layer_time;
        std::cout << "  Layer " << l << " FHE: " << layer_time << "s" << std::endl;

        // Bootstrap to refresh noise budget
        if ((l + 1) % bootstrap_interval == 0 && l + 1 < config.n_layers) {
            std::cout << "  Bootstrap after layer " << l << "..." << std::flush;
            Timer bt;
            try {
                enc_x = ckks.cc->EvalBootstrap(enc_x);
                std::cout << " done (" << bt.elapsed() << "s)" << std::endl;
            } catch (const std::exception& e) {
                std::cout << " FAILED: " << e.what() << std::endl;
                std::cout << "  Continuing without bootstrap (noise will accumulate)" << std::endl;
            }
        }
    }

    std::cout << "  Total layer time: " << total_layer_time << "s" << std::endl;

    // Head (FHE matvec)
    auto fhe_hidden = ckks.decrypt(enc_x, D);
    auto enc_head_in = ckks.encrypt(fhe_hidden);
    auto enc_head = ckks.matvec(enc_head_in, head_proj, D, config.emb_dim);
    auto fhe_emb_out = ckks.decrypt(enc_head, config.emb_dim);

    // Logits (plaintext)
    std::vector<double> logits(config.vocab_size, 0.0);
    for (int v = 0; v < config.vocab_size; v++) {
        for (int i = 0; i < config.emb_dim; i++) {
            logits[v] += fhe_emb_out[i] * token_emb.at(v, i);
        }
    }

    return logits;
}

int SipherModel::generate_next(CKKSContext& ckks, const std::vector<int>& tokens,
                                double temp, int top_k) {
    auto logits = infer(ckks, tokens);

    // Repetition penalty
    for (int t : tokens) {
        if (t < (int)logits.size()) {
            if (logits[t] > 0) logits[t] /= 2.0;
            else logits[t] *= 2.0;
        }
    }

    // Top-k
    if (top_k > 0) {
        std::vector<std::pair<double, int>> indexed;
        for (int i = 0; i < (int)logits.size(); i++)
            indexed.push_back({logits[i], i});
        std::partial_sort(indexed.begin(), indexed.begin() + top_k, indexed.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        double threshold = indexed[top_k - 1].first;
        for (auto& l : logits)
            if (l < threshold) l = -1e9;
    }

    // Softmax + sample
    double max_l = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (auto& l : logits) {
        l = std::exp((l - max_l) / temp);
        sum += l;
    }
    for (auto& l : logits) l /= sum;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(logits.begin(), logits.end());
    return dist(gen);
}
