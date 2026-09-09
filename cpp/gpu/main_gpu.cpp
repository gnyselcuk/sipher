// Sipher FHE GPU Inference — CLI
// Usage:
//   ./sipher_fhe_gpu --benchmark          GPU FHE benchmark
//   ./sipher_fhe_gpu --infer <weights>    Full model inference
//   ./sipher_fhe_gpu --layers N <weights> Test N layers
//   ./sipher_fhe_gpu --parity N <weights> [--dump <file>]  FHE vs plaintext mirror parity
#include "sipher_fhe_gpu.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <cstdlib>

void print_usage(const char* prog) {
    std::cout << "Sipher FHE GPU Engine (FIDESlib)\n"
              << "Usage:\n"
              << "  " << prog << " --benchmark              GPU FHE primitive benchmark\n"
              << "  " << prog << " --infer <weights_dir> [--ring 16384|32768] [--bootstrap]\n"
              << "                              Full 20-layer inference\n"
              << "  " << prog << " --layers N <weights_dir> [--ring N] [--bootstrap]\n"
              << "                              Test first N layers\n"
              << "  " << prog << " --parity N <weights_dir> [--dump <file>] [--ring N] [--bootstrap]\n"
              << "                              FHE vs plaintext mirror parity (cos_sim/max_err)\n"
              << "  " << prog << " --bootstrap-test         Bootstrap timing test\n"
              << std::endl;
}

void run_benchmark(int ring_dim) {
    std::cout << "═══ Sipher FIDESlib GPU Benchmark (ring=" << ring_dim << ") ═══\n" << std::endl;

    SipherGPUConfig cfg;
    cfg.ring_dim = ring_dim;
    cfg.mult_depth = 9;
    cfg.scale_bits = 40;
    cfg.first_mod_bits = 40;
    cfg.gpu_devices = {0};

    FIDESContext fides;
    std::cout << "Initializing FIDESlib GPU context..." << std::endl;
    GPUTimer init_timer;
    fides.init(cfg);
    std::cout << "Init time: " << init_timer.elapsed() << "s\n" << std::endl;

    int dims[] = {32, 64, 256, 1024};

    std::cout << "── Matvec (BSGS, GPU) ──" << std::endl;
    std::cout << "  dim     time(s)   err" << std::endl;

    for (int d : dims) {
        // Random input
        std::vector<double> x(d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& v : x) v = dis(gen);

        // Random weight matrix
        GPUTensor W;
        W.shape = {d, d};
        W.data.resize(d * d);
        for (auto& v : W.data) v = dis(gen) * 0.01;

        // Plaintext reference
        std::vector<double> ref(d, 0.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                ref[i] += W.at(i, j) * x[j];

        // FHE matvec
        auto enc_x = fides.encrypt(x);
        GPUTimer t;
        auto enc_y = fides.matvec(enc_x, W, d, d);
        double elapsed = t.elapsed();

        auto y = fides.decrypt(enc_y, d);

        // Error
        double max_err = 0;
        for (int i = 0; i < d; i++)
            max_err = std::max(max_err, std::abs(y[i] - ref[i]));

        std::cout << "  d=" << d << ":\t" << elapsed << "s\t" << max_err << std::endl;
    }

    std::cout << "\n── Matvec (ConvolutionTransform, GPU) ──" << std::endl;
    std::cout << "  dim     time(s)   err" << std::endl;

    for (int d : dims) {
        std::vector<double> x(d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& v : x) v = dis(gen);

        GPUTensor W;
        W.shape = {d, d};
        W.data.resize(d * d);
        for (auto& v : W.data) v = dis(gen) * 0.01;

        std::vector<double> ref(d, 0.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                ref[i] += W.at(i, j) * x[j];

        auto enc_x = fides.encrypt(x);
        GPUTimer t;
        Ciphertext<DCRTPoly> enc_y;
        try {
            enc_y = fides.matvec_conv(enc_x, W, d, d);
        } catch (const std::exception& e) {
            std::cout << "  d=" << d << ":\tFAILED: " << e.what() << std::endl;
            continue;
        }
        double elapsed = t.elapsed();

        auto y = fides.decrypt(enc_y, d);
        double max_err = 0;
        for (int i = 0; i < d; i++)
            max_err = std::max(max_err, std::abs(y[i] - ref[i]));

        std::cout << "  d=" << d << ":\t" << elapsed << "s\t" << max_err << std::endl;
    }

    std::cout << "\n── Matvec (Pre-encoded BSGS, GPU) ──" << std::endl;
    std::cout << "  dim     time(s)   err" << std::endl;

    for (int d : dims) {
        std::vector<double> x(d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& v : x) v = dis(gen);

        GPUTensor W;
        W.shape = {d, d};
        W.data.resize(d * d);
        for (auto& v : W.data) v = dis(gen) * 0.01;

        std::vector<double> ref(d, 0.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                ref[i] += W.at(i, j) * x[j];

        auto enc_x = fides.encrypt(x);
        auto diag_pts = fides.encode_diagonals(W, d, d);
        GPUTimer t;
        auto enc_y = fides.matvec_pre(enc_x, diag_pts, d, d);
        double elapsed = t.elapsed();

        auto y = fides.decrypt(enc_y, d);
        double max_err = 0;
        for (int i = 0; i < d; i++)
            max_err = std::max(max_err, std::abs(y[i] - ref[i]));

        std::cout << "  d=" << d << ":\t" << elapsed << "s\t" << max_err << std::endl;
    }

    std::cout << "\n── Matvec (Replicated BSGS, GPU) ──" << std::endl;
    std::cout << "  dim     time(s)   err" << std::endl;

    for (int d : dims) {
        std::vector<double> x(d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& v : x) v = dis(gen);

        GPUTensor W;
        W.shape = {d, d};
        W.data.resize(d * d);
        for (auto& v : W.data) v = dis(gen) * 0.01;

        std::vector<double> ref(d, 0.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                ref[i] += W.at(i, j) * x[j];

        auto enc_x = fides.encrypt(x);
        GPUTimer t;
        auto enc_y = fides.matvec_rep(enc_x, W, d, d);
        double elapsed = t.elapsed();

        auto y = fides.decrypt(enc_y, d);
        double max_err = 0;
        for (int i = 0; i < d; i++)
            max_err = std::max(max_err, std::abs(y[i] - ref[i]));

        std::cout << "  d=" << d << ":\t" << elapsed << "s\t" << max_err << std::endl;
    }

    std::cout << "\n── ct×ct (PolyAct, GPU) ──" << std::endl;
    for (int d : dims) {
        std::vector<double> x(d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-0.5, 0.5);
        for (auto& v : x) v = dis(gen);

        auto enc_x = fides.encrypt(x);
        GPUTimer t;
        auto enc_x2 = fides.mult(enc_x, enc_x);
        double elapsed = t.elapsed();

        auto x2 = fides.decrypt(enc_x2, d);
        double max_err = 0;
        for (int i = 0; i < d; i++)
            max_err = std::max(max_err, std::abs(x2[i] - x[i] * x[i]));

        std::cout << "  d=" << d << ":\t" << elapsed << "s\t" << max_err << std::endl;
    }

    std::cout << "\n── Bootstrap (GPU) ──" << std::endl;
    {
        uint32_t depth = cfg.mult_depth;
        std::vector<double> x(1024, 0.5);
        auto enc_x = fides.encrypt(x, depth - 1);

        for (int i = 0; i < 2; i++)
            enc_x = fides.mult(enc_x, enc_x);

        GPUTimer t;
        auto enc_refreshed = fides.bootstrap(enc_x);
        double elapsed = t.elapsed();
        std::cout << "  Bootstrap time: " << elapsed << "s" << std::endl;

        try {
            auto refreshed = fides.decrypt(enc_refreshed, 8);
            std::cout << "  Values: ";
            for (int i = 0; i < 4; i++) std::cout << refreshed[i] << " ";
            std::cout << "... ✓" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  Decrypt: " << e.what() << std::endl;
        }
    }
}

void run_bootstrap_test() {
    std::cout << "═══ Bootstrap Correctness Test ═══\n" << std::endl;

    SipherGPUConfig cfg;
    cfg.ring_dim = 16384;   // gerçek pipeline ile aynı
    cfg.mult_depth = 25;    // bootstrap 15 tüketir → 10 kalan
    // scale 59/60 = FIDESlib default'ları — GPU bootstrap'in çalıştığı tek config
    cfg.gpu_devices = {0};

    FIDESContext fides;
    std::cout << "Initializing..." << std::endl;
    GPUTimer init_timer;
    fides.init(cfg);
    std::cout << "Init: " << init_timer.elapsed() << "s\n" << std::endl;

    // Gerçek pipeline senaryosu: level 0 (full chain) encrypt → 3 EvalMult (3 katman simülasyonu)
    // → çok-limb girdiyle bootstrap → decrypt doğrulama
    uint32_t depth = cfg.mult_depth;
    std::vector<double> x(fides.n_slots, 0.0);
    for (int i = 0; i < 1024; i++) x[i] = 1.0 + 0.01 * (i % 7);  // O(1) değerler — x^30 hâlâ makul

    auto enc_x = fides.encrypt(x, 0);
    std::cout << "Encrypted (level 0): levels=" << depth - enc_x->GetLevel()
              << " (level=" << enc_x->GetLevel() << ")" << std::endl;

    // 3 katman simülasyonu: her katman birkaç mult → seviye yakılır
    for (uint32_t i = 0; i < 3; i++) {
        enc_x = fides.mult(enc_x, enc_x);
        enc_x = fides.mult_plain(enc_x, x);
    }
    std::cout << "After 3 layers: level=" << enc_x->GetLevel()
              << " (kalan seviye=" << depth - enc_x->GetLevel() << ")" << std::endl;

    // Bootstrap (çok-limb girdi — SetLevel hack'i OLMADAN)
    std::cout << "Bootstrapping (multi-limb input)..." << std::flush;
    GPUTimer bt;
    enc_x = fides.bootstrap(enc_x);
    double bts_time = bt.elapsed();
    std::cout << " done (" << bts_time << "s)" << std::endl;
    std::cout << "After bootstrap: level=" << enc_x->GetLevel()
              << " (kalan seviye=" << depth - enc_x->GetLevel() << ")" << std::endl;

    // Bootstrap sonrası taze seviyelerle 1 mult daha
    enc_x = fides.mult(enc_x, enc_x);
    std::cout << "After post-bootstrap mult: level=" << enc_x->GetLevel() << std::endl;

    // Doğrulama: 3 iterasyon × (x² sonra ·x) → x^15; bootstrap; 1 mult → x^30
    std::vector<double> expected(8);
    for (int i = 0; i < 8; i++) {
        double v = 1.0 + 0.01 * (i % 7);
        expected[i] = std::pow(v, 30);
    }

    try {
        auto result = fides.decrypt(enc_x, 8);
        std::cout << "Expected: ";
        for (int i = 0; i < 4; i++) std::cout << expected[i] << " ";
        std::cout << "..." << std::endl;
        std::cout << "Got:      ";
        for (int i = 0; i < 4; i++) std::cout << result[i] << " ";
        std::cout << "..." << std::endl;
        double max_err = 0.0, max_abs = 0.0;
        for (int i = 0; i < 8; i++) {
            max_err = std::max(max_err, std::abs(result[i] - expected[i]));
            max_abs = std::max(max_abs, std::abs(expected[i]));
        }
        std::cout << "Max error: " << max_err << " (rel " << max_err / max_abs << ")" << std::endl;
        if (max_err < 0.05 * max_abs)
            std::cout << "BOOTSTRAP OK ✓ (multi-limb input, " << bts_time << "s)" << std::endl;
        else
            std::cout << "BOOTSTRAP FAILED ✗ (max_err=" << max_err << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Decrypt: " << e.what() << std::endl;
    }
}

void run_inference(const std::string& weights_dir, int max_layers = -1,
                   int ring_dim = 32768, bool use_bootstrap = false) {
    std::cout << "═══ Sipher GPU FHE Inference ═══" << std::endl;
    std::cout << "Weights: " << weights_dir << std::endl;

    // Load model
    SipherGPUModel model;
    std::cout << "\nLoading model..." << std::endl;
    GPUTimer load_timer;
    model.load(weights_dir);
    std::cout << "Model loaded in " << load_timer.elapsed() << "s" << std::endl;

    if (max_layers > 0 && max_layers < model.config.n_layers) {
        model.config.n_layers = max_layers;
        std::cout << "Testing first " << max_layers << " layers only" << std::endl;
    }

    // ring=16384: replikasyon ile VRAM'e sığar, matvec doğrulanmış
    model.config.ring_dim = ring_dim;
    if (use_bootstrap) {
        // FIDESlib GPU bootstrap scale 2^59/60 İSTER (scale 2^40'ta çıktı çöp — test edildi)
        model.config.scale_bits = 59;
        model.config.first_mod_bits = 60;
        model.config.mult_depth = 25;   // bootstrap 15 seviye tüketir → 10 kalan (depth 15'te 0 kalan → ölür)
        model.config.bootstrap_interval = 3;   // 3 katman ~5 seviye yakar → 10 yeterli
        std::cout << "Bootstrap ENABLED (depth=25, interval=3, scale=2^59)" << std::endl;
    } else {
        model.config.scale_bits = 40;
        model.config.first_mod_bits = 40;
        model.config.mult_depth = 9;
        model.config.bootstrap_interval = 999;
    }

    // Initialize FIDESlib GPU
    FIDESContext fides;
    std::cout << "\nInitializing FIDESlib GPU..." << std::endl;
    GPUTimer init_timer;
    fides.init(model.config);
    std::cout << "FIDESlib init: " << init_timer.elapsed() << "s" << std::endl;

    // Test tokens: "Merkez Bankası faiz"
    std::vector<int> tokens = {4412, 7314, 6006}; // "Merkez Bankası faiz" (Sipher 16K)
    std::cout << "\nInput tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]" << std::endl;

    // Pad to seq_len
    while ((int)tokens.size() < model.config.seq_len)
        tokens.push_back(0);

    // Run inference
    std::cout << "\nRunning " << model.config.n_layers << "-layer GPU FHE inference..." << std::endl;
    GPUTimer total_timer;
    int next_token = model.generate_next(fides, tokens, 0.7, 40);
    double total_time = total_timer.elapsed();

    std::cout << "\n═══ Results ═══" << std::endl;
    std::cout << "Next token ID: " << next_token << std::endl;
    std::cout << "Total time:    " << total_time << "s" << std::endl;
    std::cout << "Per layer:     " << total_time / model.config.n_layers << "s" << std::endl;
}

// ═══ FHE'de ardışık token üretimi (cümle) ═══
void run_generate(const std::string& weights_dir, int n_tokens, int ring_dim, bool use_bootstrap,
                  const std::vector<int>& seed_ids) {
    std::cout << "═══ Sipher GPU FHE Generation ═══" << std::endl;
    std::cout << "Weights: " << weights_dir << " | " << n_tokens << " token üretilecek" << std::endl;

    SipherGPUModel model;
    model.load(weights_dir);

    model.config.ring_dim = ring_dim;
    if (use_bootstrap) {
        model.config.scale_bits = 59;
        model.config.first_mod_bits = 60;
        model.config.mult_depth = 25;
        model.config.bootstrap_interval = 3;
        std::cout << "Bootstrap ENABLED (depth=25, interval=3, scale=2^59)" << std::endl;
    } else {
        model.config.scale_bits = 40;
        model.config.first_mod_bits = 40;
        model.config.mult_depth = 9;
        model.config.bootstrap_interval = 999;
    }

    FIDESContext fides;
    std::cout << "\nInitializing FIDESlib GPU..." << std::endl;
    GPUTimer init_timer;
    fides.init(model.config);
    std::cout << "FIDESlib init: " << init_timer.elapsed() << "s" << std::endl;

    int SEQ = model.config.seq_len;
    // Seed → pencerenin SONUNA (kausal: son slot tahmin slotu)
    std::vector<int> seed = seed_ids.empty() ? std::vector<int>{4412, 7314, 6006} : seed_ids;
    std::vector<int> window(SEQ, 0);
    for (size_t i = 0; i < seed.size(); i++)
        window[SEQ - 1 - (int)seed.size() + (int)i] = seed[i];

    std::cout << "\nSeed: [";
    for (size_t i = 0; i < seed.size(); i++) { if (i) std::cout << ", "; std::cout << seed[i]; }
    std::cout << "]" << std::endl;

    std::vector<int> generated;
    for (int step = 0; step < n_tokens; step++) {
        std::cout << "\n── Step " << step << " (20L FHE infer, ~13 dk) ──" << std::endl;
        GPUTimer step_timer;
        int tok = model.generate_next(fides, window, 0.7, 40);
        std::cout << "→ Token " << tok << "  (" << step_timer.elapsed() << "s)" << std::endl;
        generated.push_back(tok);
        // Pencereyi kaydır: en eskiyi at, yeni token'ı sona ekle
        for (int i = 0; i < SEQ - 1; i++) window[i] = window[i + 1];
        window[SEQ - 1] = tok;
    }

    std::cout << "\n═══ Generated sequence (token IDs) ═══" << std::endl;
    std::cout << "Seed:    ";
    for (int t : seed) std::cout << t << " ";
    std::cout << "\nGenerated: ";
    for (int t : generated) std::cout << t << " ";
    std::cout << std::endl;
}

void run_parity(const std::string& weights_dir, int n_layers, const std::string& dump_path,
                int ring_dim, bool use_bootstrap, const std::vector<int>& seed_ids = {}) {
    std::cout << "═══ Sipher GPU FHE vs Plaintext Mirror Parity ═══" << std::endl;
    std::cout << "Weights: " << weights_dir << std::endl;

    SipherGPUModel model;
    std::cout << "\nLoading model..." << std::endl;
    GPUTimer load_timer;
    model.load(weights_dir);
    std::cout << "Model loaded in " << load_timer.elapsed() << "s" << std::endl;

    if (n_layers <= 0 || n_layers > model.config.n_layers)
        n_layers = model.config.n_layers;
    std::cout << "Parity layers: " << n_layers << " / " << model.config.n_layers << std::endl;

    // FHE konfigürasyonu (--infer ile aynı)
    model.config.ring_dim = ring_dim;
    if (use_bootstrap) {
        // FIDESlib GPU bootstrap scale 2^59/60 ister (2^40'ta çöp)
        model.config.scale_bits = 59;
        model.config.first_mod_bits = 60;
        model.config.mult_depth = 25;   // bootstrap 15 seviye tüketir → 10 kalan
        model.config.bootstrap_interval = 3;
        std::cout << "Bootstrap ENABLED (depth=25, interval=3, scale=2^59)" << std::endl;
    } else {
        model.config.scale_bits = 40;
        model.config.first_mod_bits = 40;
        model.config.mult_depth = 9;
        model.config.bootstrap_interval = 999;
    }

    FIDESContext fides;
    std::cout << "\nInitializing FIDESlib GPU..." << std::endl;
    GPUTimer init_timer;
    fides.init(model.config);
    std::cout << "FIDESlib init: " << init_timer.elapsed() << "s" << std::endl;

    int D = model.config.d_model;
    int SEQ = model.config.seq_len;
    int NH = model.config.n_heads;
    int DK = model.config.d_k;
    int pos = SEQ - 1;

    // Test tokenları (--infer ile aynı; --seed-tokens ile değiştirilebilir)
    std::vector<int> tokens = {4412, 7314, 6006}; // "Merkez Bankası faiz" (Sipher 16K)
    if (!seed_ids.empty()) {
        tokens = seed_ids;
        std::cout << "Seed tokens: ";
        for (int t : tokens) std::cout << t << " ";
        std::cout << std::endl;
    }
    while ((int)tokens.size() < SEQ) tokens.push_back(0);

    // Plaintext embeddings (tüm pozisyonlar — K/V kaynağı)
    std::vector<std::vector<double>> hidden_states(SEQ, std::vector<double>(D, 0.0));
    for (int t = 0; t < SEQ; t++) {
        int tok = tokens[t];
        for (int i = 0; i < model.config.emb_dim; i++) {
            double e = model.token_emb.at(tok, i);
            for (int j = 0; j < D; j++)
                hidden_states[t][j] += e * model.emb_proj.at(j, i);
        }
        for (int j = 0; j < D; j++)
            hidden_states[t][j] += model.pos_emb.at(t, j);
    }

    // Encrypted target pozisyon
    auto enc_x = model.embed(fides, tokens, pos);

    std::vector<bool> window_mask(SEQ, false);
    for (int t = 0; t < SEQ; t++) {
        int diff = pos - t;
        if (diff >= 0 && diff < model.config.window) window_mask[t] = true;
    }

    std::vector<std::vector<double>> dump;
    double total_fhe = 0.0;
    bool all_ok = true;

    for (int l = 0; l < n_layers; l++) {
        // K/V from norm1(hidden_states)
        GPUTensor K_all, V_all;
        K_all.shape = {SEQ, NH * DK};
        V_all.shape = {SEQ, NH * DK};
        K_all.data.resize(SEQ * NH * DK, 0.0);
        V_all.data.resize(SEQ * NH * DK, 0.0);
        for (int t = 0; t < SEQ; t++) {
            std::vector<double> hnorm;
            SipherGPULayer::norm_plain(hidden_states[t], model.layers[l].norm1_w,
                                       model.layers[l].norm1_b, D, hnorm);
            for (int i = 0; i < NH * DK; i++) {
                double k_val = 0.0, v_val = 0.0;
                for (int j = 0; j < D; j++) {
                    k_val += hnorm[j] * model.layers[l].W_K.at(i, j);
                    v_val += hnorm[j] * model.layers[l].W_V.at(i, j);
                }
                K_all.at(t, i) = k_val;
                V_all.at(t, i) = v_val;
            }
        }

        // Bootstrap: residual noise'ını tazele (katman başlangıcı)
        if (model.config.bootstrap_interval > 0 && l > 0 && l % model.config.bootstrap_interval == 0) {
            GPUTimer bt;
            enc_x = fides.bootstrap(enc_x);
            std::cout << "         bootstrap: " << bt.elapsed() << "s" << std::endl;
        }

        // FHE forward (pozisyon pos)
        GPUTimer ft;
        enc_x = model.layers[l].forward(fides, enc_x, K_all, V_all, window_mask,
                                        pos, NH, DK, SEQ);
        double fhe_time = ft.elapsed();
        total_fhe += fhe_time;
        auto fhe_out = fides.decrypt(enc_x, D);

        // Plaintext mirror (tüm pozisyonlar)
        GPUTimer pt;
        std::vector<std::vector<double>> hs_next;
        model.layers[l].forward_plaintext(hidden_states, K_all, V_all, model.config.window,
                                          NH, DK, SEQ, hs_next);
        double mirror_time = pt.elapsed();
        hidden_states = hs_next;
        dump.push_back(hidden_states[pos]);

        // cos_sim + max_err (göreli: max_err / max|referans|)
        double dot = 0, na = 0, nb = 0, max_err = 0, max_ref = 0;
        for (int i = 0; i < D; i++) {
            dot += fhe_out[i] * hidden_states[pos][i];
            na += fhe_out[i] * fhe_out[i];
            nb += hidden_states[pos][i] * hidden_states[pos][i];
            max_err = std::max(max_err, std::abs(fhe_out[i] - hidden_states[pos][i]));
            max_ref = std::max(max_ref, std::abs(hidden_states[pos][i]));
        }
        double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        double rel_err = max_err / (max_ref + 1e-30);
        // d=1024 CKKS: Python referansı da err~0.21/cos 0.9989 veriyor (HANDOFF) —
        // mutlak eşik yerine cos_sim + göreli hata kullanılır
        bool ok = cos_sim > 0.999 && rel_err < 0.10;
        all_ok = all_ok && ok;

        std::cout << "  Layer " << l << ": FHE " << fhe_time << "s | mirror " << mirror_time
                  << "s | cos_sim=" << cos_sim << " | max_err=" << max_err
                  << " (rel " << 100.0 * rel_err << "%)"
                  << (ok ? "  ✓" : "  ✗") << std::endl;
    }

    // Head + logits (FHE vs mirror)
    auto fhe_hidden = fides.decrypt(enc_x, D);
    auto enc_head_in = fides.encrypt(fhe_hidden);
    auto enc_head = fides.matvec_rep(enc_head_in, model.head_proj, D, model.config.emb_dim);
    auto fhe_emb = fides.decrypt(enc_head, model.config.emb_dim);

    std::vector<double> logits_fhe(model.config.vocab_size, 0.0);
    std::vector<double> logits_pt(model.config.vocab_size, 0.0);
    for (int v = 0; v < model.config.vocab_size; v++) {
        double acc_fhe = 0, acc_pt = 0;
        for (int i = 0; i < model.config.emb_dim; i++) {
            acc_fhe += fhe_emb[i] * model.token_emb.at(v, i);
            double h_pt = 0;
            for (int j = 0; j < D; j++)
                h_pt += model.head_proj.at(i, j) * hidden_states[pos][j];
            acc_pt += h_pt * model.token_emb.at(v, i);
        }
        logits_fhe[v] = acc_fhe;
        logits_pt[v] = acc_pt;
    }

    double dot = 0, na = 0, nb = 0;
    for (int v = 0; v < model.config.vocab_size; v++) {
        dot += logits_fhe[v] * logits_pt[v];
        na += logits_fhe[v] * logits_fhe[v];
        nb += logits_pt[v] * logits_pt[v];
    }
    double cos_logits = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);

    auto top5 = [](const std::vector<double>& lg) {
        std::vector<int> idx(lg.size());
        for (size_t i = 0; i < lg.size(); i++) idx[i] = (int)i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        return std::vector<int>(idx.begin(), idx.begin() + 5);
    };
    auto t5_fhe = top5(logits_fhe);
    auto t5_pt = top5(logits_pt);
    int hit = 0;
    for (int a : t5_fhe)
        for (int b : t5_pt)
            if (a == b) hit++;

    std::cout << "\n  ── Head / Logits ──" << std::endl;
    std::cout << "  cos_sim(logits): " << cos_logits << std::endl;
    std::cout << "  Top-5 uyumu: " << hit << "/5" << std::endl;
    std::cout << "  Top-5 FHE:  ";
    for (int v : t5_fhe) std::cout << v << " ";
    std::cout << "\n  Top-5 PT:   ";
    for (int v : t5_pt) std::cout << v << " ";
    std::cout << std::endl;

    // Genel sonuç: katman cos_sim + logits top-5 uyumu (≥4/5)
    all_ok = all_ok && (hit >= 4);

    std::cout << "\n═══ PARITY " << (all_ok ? "OK ✓" : "FAIL ✗")
              << " ═══ (FHE toplam: " << total_fhe << "s)" << std::endl;

    if (!dump_path.empty()) {
        std::ofstream of(dump_path, std::ios::binary);
        int32_t nl = (int32_t)dump.size(), dim = D;
        of.write((const char*)&nl, sizeof(nl));
        of.write((const char*)&dim, sizeof(dim));
        for (auto& v : dump)
            of.write((const char*)v.data(), D * sizeof(double));
        of.close();
        std::cout << "  Mirror hidden states → " << dump_path << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════
// Encode overhead teşhisi: mask kurulumu / MakeCKKSPackedPlaintext /
// EvalMult / taze encode+mult ayrı ayrı zamanlanır
// ═══════════════════════════════════════════════════════════
void run_encode_bench() {
    std::cout << "═══ Encode Overhead Diagnostic ═══\n" << std::endl;

    SipherGPUConfig cfg;
    cfg.ring_dim = 32768;
    cfg.mult_depth = 9;
    cfg.scale_bits = 40;
    cfg.first_mod_bits = 40;
    cfg.gpu_devices = {0};

    FIDESContext fides;
    fides.init(cfg);

    const int N = 200;
    const int SLOTS = (int)fides.n_slots;   // 16384
    GPUTensor W;
    W.shape = {1024, 1024};
    W.data.resize(1024 * 1024, 0.01);

    // 1) Mask vektör kurulumu (saf CPU, FHE yok) — matvec_rep'teki gibi
    {
        GPUTimer t;
        volatile double sink = 0;
        for (int k = 0; k < N; k++) {
            std::vector<double> mask(SLOTS, 0.0);
            for (int j = 0; j < 4; j++) {
                int d = j + k * 4;
                if (d >= 1024) continue;
                for (int i = 0; i < 1024; i++) {
                    int slot = j * 2048 + i + j;
                    int col = (i + d) % 1024;
                    mask[slot] = W.at(i, col);
                }
            }
            sink += mask[k];
        }
        double t1 = t.elapsed();
        std::cout << "1) Mask kurulumu:         " << t1 << "s → "
                  << t1 / N * 1000 << " ms/adet" << std::endl;
    }

    // 2) MakeCKKSPackedPlaintext (encode, pt tutulmaz)
    double t2 = 0, t3 = 0;
    {
        std::vector<double> mask(SLOTS, 0.0);
        for (int i = 0; i < 1024; i++) mask[i] = 0.01;
        GPUTimer t;
        for (int k = 0; k < N; k++) {
            auto pt = fides.cc->MakeCKKSPackedPlaintext(mask);
        }
        t2 = t.elapsed();
        std::cout << "2) MakeCKKSPackedPlaintext: " << t2 << "s → "
                  << t2 / N * 1000 << " ms/adet" << std::endl;
    }

    // 3) EvalMult — önceden encode edilmiş SABİT pt ile (encode yok)
    {
        std::vector<double> x(1024, 0.5);
        auto enc_x = fides.encrypt(x);
        std::vector<double> mask(SLOTS, 0.0);
        for (int i = 0; i < 1024; i++) mask[i] = 0.01;
        auto pt = fides.cc->MakeCKKSPackedPlaintext(mask);
        GPUTimer t;
        Ciphertext<DCRTPoly> acc;
        bool first = true;
        for (int k = 0; k < N; k++) {
            auto prod = fides.cc->EvalMult(enc_x, pt);
            if (first) { acc = prod; first = false; }
            else fides.cc->EvalAddInPlace(acc, prod);
        }
        t3 = t.elapsed();
        std::cout << "3) EvalMult (aynı pt):    " << t3 << "s → "
                  << t3 / N * 1000 << " ms/adet" << std::endl;
    }

    // 4) Taze encode + mult (matvec_rep'teki gerçek yol)
    {
        std::vector<double> x(1024, 0.5);
        auto enc_x = fides.encrypt(x);
        GPUTimer t;
        for (int k = 0; k < N; k++) {
            std::vector<double> mask(SLOTS, 0.0);
            for (int i = 0; i < 1024; i++) mask[i] = 0.01;
            auto pt = fides.cc->MakeCKKSPackedPlaintext(mask);
            auto prod = fides.cc->EvalMult(enc_x, pt);
        }
        double t4 = t.elapsed();
        std::cout << "4) Taze encode+mult:      " << t4 << "s → "
                  << t4 / N * 1000 << " ms/adet" << std::endl;
        std::cout << "   (2)+(3)=" << (t2 + t3) / N * 1000 << " ms/adet → fark (overhead): "
                  << (t4 - t2 - t3) / N * 1000 << " ms/adet" << std::endl;
    }

    // 5) Paralel encode (8 thread) — mask'lar bağımsız, thread-safe mi?
    {
        std::vector<double> mask(SLOTS, 0.0);
        for (int i = 0; i < 1024; i++) mask[i] = 0.01;
        const int NT = 8;
        int per_thread = N / NT;
        GPUTimer t;
        std::vector<std::thread> threads;
        for (int ti = 0; ti < NT; ti++) {
            threads.emplace_back([&, ti]() {
                for (int k = 0; k < per_thread; k++)
                    auto pt = fides.cc->MakeCKKSPackedPlaintext(mask);
            });
        }
        for (auto& th : threads) th.join();
        double t5 = t.elapsed();
        std::cout << "5) Paralel encode (" << NT << "×" << per_thread << "): " << t5 << "s → "
                  << t5 / N * 1000 << " ms/adet efektif (tekil 5.24 → hızlanma "
                  << (t2 / N) / (t5 / N) << "×)" << std::endl;
    }

    std::cout << "\nOMP_NUM_THREADS=" << (getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(set değil)")
              << " | nproc=" << std::thread::hardware_concurrency() << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "--benchmark") {
        int ring = 16384;
        for (int i = 2; i < argc; i++)
            if (std::string(argv[i]) == "--ring" && i + 1 < argc) ring = std::atoi(argv[++i]);
        run_benchmark(ring);
    } else if (cmd == "--bootstrap-test") {
        run_bootstrap_test();
    } else if (cmd == "--infer" && argc >= 3) {
        int ring = 32768; bool boot = false;
        for (int i = 3; i < argc; i++) {
            if (std::string(argv[i]) == "--ring" && i + 1 < argc) ring = std::atoi(argv[++i]);
            if (std::string(argv[i]) == "--bootstrap") boot = true;
        }
        run_inference(argv[2], -1, ring, boot);
    } else if (cmd == "--layers" && argc >= 4) {
        int n = std::atoi(argv[2]);
        int ring = 32768; bool boot = false;
        for (int i = 4; i < argc; i++) {
            if (std::string(argv[i]) == "--ring" && i + 1 < argc) ring = std::atoi(argv[++i]);
            if (std::string(argv[i]) == "--bootstrap") boot = true;
        }
        run_inference(argv[3], n, ring, boot);
    } else if (cmd == "--generate" && argc >= 4) {
        int n = std::atoi(argv[2]);
        int ring = 32768; bool boot = false;
        std::vector<int> seed_ids;
        for (int i = 4; i < argc; i++) {
            if (std::string(argv[i]) == "--ring" && i + 1 < argc) ring = std::atoi(argv[++i]);
            if (std::string(argv[i]) == "--bootstrap") boot = true;
            if (std::string(argv[i]) == "--seed-tokens" && i + 1 < argc) {
                std::string s = argv[++i];
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t c = s.find(',', pos);
                    seed_ids.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
        }
        run_generate(argv[3], n, ring, boot, seed_ids);
    } else if (cmd == "--parity" && argc >= 4) {
        int n = std::atoi(argv[2]);
        std::string dump = "";
        int ring = 32768; bool boot = false;
        std::vector<int> seed_ids;
        for (int i = 4; i < argc; i++) {
            if (std::string(argv[i]) == "--dump" && i + 1 < argc) dump = argv[i + 1];
            if (std::string(argv[i]) == "--ring" && i + 1 < argc) ring = std::atoi(argv[++i]);
            if (std::string(argv[i]) == "--bootstrap") boot = true;
            if (std::string(argv[i]) == "--dump-layer" && i + 1 < argc) g_dump_layer = std::atoi(argv[++i]);
            if (std::string(argv[i]) == "--seed-tokens" && i + 1 < argc) {
                std::string s = argv[++i];
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t c = s.find(',', pos);
                    seed_ids.push_back(std::atoi(s.substr(pos, c - pos).c_str()));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
        }
        run_parity(argv[3], n, dump, ring, boot, seed_ids);
    } else if (cmd == "--encode-bench") {
        run_encode_bench();
    } else {
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
