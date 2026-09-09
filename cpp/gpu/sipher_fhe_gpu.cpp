// Sipher FHE GPU Inference Engine — FIDESlib Implementation
// GPU-accelerated CKKS with bootstrapping for 20-layer inference
#include "sipher_fhe_gpu.h"
#include <thread>
#include <atomic>

int g_dump_layer = -1;

// ═══════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════

SipherGPUConfig SipherGPUConfig::load(const std::string& path) {
    SipherGPUConfig cfg;
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
        else if (key == "feat_degree") f >> cfg.feat_degree;
    }
    return cfg;
}

// ═══════════════════════════════════════════════════════════
// Tensor
// ═══════════════════════════════════════════════════════════

GPUTensor GPUTensor::load_bin(const std::string& path) {
    std::string meta_path = path + ".meta";
    std::ifstream mf(meta_path);
    if (!mf.is_open()) {
        std::string base = path.substr(0, path.find(".bin"));
        mf.open(base + ".meta");
    }

    GPUTensor t;
    if (mf.is_open()) {
        int ndim;
        mf >> ndim;
        t.shape.resize(ndim);
        for (int i = 0; i < ndim; i++) mf >> t.shape[i];
    }

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
// FIDESlib Context (GPU)
// ═══════════════════════════════════════════════════════════

std::vector<int32_t> FIDESContext::compute_rotation_indices(int max_dim) {
    std::set<int32_t> indices;

    // Matvec rotasyonları hep POZİTİF yönde (+b baby, +gB giant, +shift replikasyon).
    // A1 (head-grouped V) sonrası negatif rotasyon kullanılmıyor → key seti ~%40 küçülür
    // (init hızı + VRAM: ring=16384'te ~206 → ~123 anahtar).
    // Baby steps: 1..63
    for (int i = 1; i <= 63; i++) {
        indices.insert(i);
    }
    // Giant steps
    int dims[] = {64, 256, 1024};
    for (int d : dims) {
        int baby = (int)std::ceil(std::sqrt((double)d));
        for (int g = baby; g <= d; g += baby) {
            indices.insert(g);
        }
    }
    // Replikasyon rotasyonları (input periodic extension için)
    for (int r = 2048; r <= 8192; r *= 2) {
        indices.insert(r);
    }

    // ── GENEL: max_dim'e (d_model) göre — d=512 (İngilizce model) desteği ──
    {
        // matvec_rep giant rotasyonları (r=4 ve r=8 replica düzenleri)
        for (int rr : {4, 8}) {
            int per = (max_dim + rr - 1) / rr;
            int baby = (int)std::ceil(std::sqrt((double)per));
            int giant = (int)std::ceil((double)per / baby);
            for (int g = 1; g <= giant; g++)
                indices.insert(g * baby * rr);
            // Baby rotasyonları (bi*rr; bkz. matvec_rep: EvalRotate(enc_rep, bi*r)).
            // Yukarıdaki 1..63 loop'u bi*rr≤63'ü kapsar, ama d=1024/rr=8/baby=12 için
            // bi=10,11 → 80,88 >63 EKSIK → global matvec "Rotation index 88 not found"
            // ile çöküyordu. Block-diagonal KVg bu baby adımlarını kullanıyor (dense Q
            // matvec zero-diag skip sayesinde atlayabiliyordu, o yüzden local-only çalıştı).
            for (int i = 1; i < baby; i++)
                indices.insert(i * rr);
        }
        // replikasyon (in_dim adımları: d, 2d, 4d, ...)
        for (int s = max_dim; s <= n_slots; s *= 2)
            indices.insert(s);
        // down/up chunk taşıma: c*max_dim (c=1..4; d=512: 512/1024/1536/2048, d=1024: 1024/2048/3072/4096)
        for (int c = 1; c <= 4; c++)
            indices.insert(c * max_dim);
    }
    // Replicated BSGS (matvec_rep):
    //   - V grubu (in=128, per_replica=32, baby=6): giant roll g*24 → 72, 120 eksik
    //   - fold: roll by (window+1) doubling → window=1024 (r=8, küçük in_dim): 1025, 2050, 4100
    //   - fold: window=2048 (r=4/8): 2049, 4098
    //   - FFN up chunk birleştirme: slot taşıma k*1024 → 3072 (k=3)
    indices.insert(72);
    indices.insert(120);
    indices.insert(1025);
    indices.insert(2050);
    indices.insert(4100);
    indices.insert(2049);
    indices.insert(4098);
    indices.insert(3072);

    // FFN-down chunk taşıma: chunk [c*1024,(c+1)*1024) → [0,1024) NEGATİF rotasyon ister.
    // OpenFHE negatif index'i n-index'e normalize eder → pozitif-tümleyen key'ler:
    // 8192-1024=7168, 8192-2048=6144, 8192-3072=5120 (EvalRotate(-1024/-2048/-3072))
    indices.insert(7168);
    indices.insert(6144);
    indices.insert(5120);

    // Ring=32768 (n_slots=16384): r=8 square + r=3 FFN-up (out=4096)
    if (n_slots >= 16384) {
        // r=8 square (per_replica=128, baby=12): baby roll bi*8 → 80, 88 eksik; fold 4*2049=8196
        indices.insert(80);
        indices.insert(88);
        indices.insert(8196);
        // r=3 FFN-up (per_replica=342, baby=19, giant=18): giant roll g*57 (57..969, step 57)
        for (int g = 57; g <= 969; g += 57) indices.insert(g);
        // r=3 fold (window=5120): 5121, 2*5121=10242
        indices.insert(5121);
        indices.insert(10242);
        // FFN-down mask+rotate: chunk c'yi [c*1024, (c+1)*1024) → [0, 1024) (c=3 → 3072)
        indices.insert(3072);
    }

    return std::vector<int32_t>(indices.begin(), indices.end());
}

void FIDESContext::init(const SipherGPUConfig& cfg) {
    ring_dim = cfg.ring_dim;
    n_slots = ring_dim / 2;
    depth = cfg.mult_depth;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(cfg.mult_depth);
    params.SetScalingModSize(cfg.scale_bits);
    params.SetFirstModSize(cfg.first_mod_bits);
    params.SetScalingTechnique(FIXEDAUTO);
    params.SetKeySwitchTechnique(HYBRID);
    params.SetSecretKeyDist(SPARSE_TERNARY);
    // Bootstrap (depth>=15) → 660+ bit modulus ring=32768'de 128-bit kontrolüne sığmaz → geliştirme modu
    params.SetSecurityLevel(cfg.mult_depth >= 15 ? HEStd_NotSet
                           : (ring_dim >= 32768 ? HEStd_128_classic : HEStd_NotSet));
    params.SetRingDim(ring_dim);
    params.SetBatchSize(n_slots);

    // GPU configuration
    params.SetDevices(std::vector<int>(cfg.gpu_devices));
    params.SetPlaintextAutoload(false);
    params.SetCiphertextAutoload(true);

    cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);  // Required for bootstrapping

    keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Rotation keys for BSGS matvec
    auto rot_indices = compute_rotation_indices(cfg.d_model);
    std::cout << "  Generating " << rot_indices.size() << " rotation keys..." << std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, rot_indices);

    // Bootstrap setup (BEFORE LoadContext!)
    if (cfg.mult_depth >= 15) {
        std::cout << "  Setting up bootstrapping (levelBudget={"
                  << cfg.level_budget[0] << "," << cfg.level_budget[1]
                  << "})..." << std::endl;
        cc->EvalBootstrapSetup(cfg.level_budget, {0, 0}, n_slots, 0);
        cc->EvalBootstrapKeyGen(keys.secretKey, n_slots);
    } else {
        std::cout << "  Bootstrap skipped (depth=" << cfg.mult_depth << " < 20)" << std::endl;
    }

    // Load context to GPU (after all key generation)
    std::cout << "  Loading context to GPU..." << std::endl;
    cc->LoadContext(keys.publicKey);
    gpu_loaded = true;

    std::cout << "  FIDESlib GPU: ring=" << ring_dim
              << " slots=" << n_slots
              << " scale=2^" << cfg.scale_bits
              << " depth=" << cfg.mult_depth
              << " devices={" << cfg.gpu_devices[0] << "}"
              << std::endl;
}

Ciphertext<DCRTPoly> FIDESContext::encrypt(const std::vector<double>& plain, uint32_t level) {
    // FIXEDAUTO: level=0 = tüm moduli zinciri = maksimum noise budget + rescale alanı
    // (FLEXIBLEAUTO'da depth-1 kullanılır, FIXEDAUTO'da 0 doğru)
    std::vector<double> padded(n_slots, 0.0);
    for (size_t i = 0; i < plain.size() && i < n_slots; i++)
        padded[i] = plain[i];
    auto pt = cc->MakeCKKSPackedPlaintext(padded, 1, level);
    return cc->Encrypt(keys.publicKey, pt);
}

std::vector<double> FIDESContext::decrypt(Ciphertext<DCRTPoly>& ct, int n) {
    Plaintext pt;
    cc->Decrypt(keys.secretKey, ct, &pt);
    auto vals = pt->GetCKKSPackedValue();
    if (n < 0) n = n_slots;
    std::vector<double> result(n);
    for (int i = 0; i < n && i < (int)vals.size(); i++)
        result[i] = vals[i].real();
    return result;
}

Ciphertext<DCRTPoly> FIDESContext::add(Ciphertext<DCRTPoly>& a, Ciphertext<DCRTPoly>& b) {
    return cc->EvalAdd(a, b);
}

Ciphertext<DCRTPoly> FIDESContext::add_plain(Ciphertext<DCRTPoly>& ct, const std::vector<double>& plain) {
    std::vector<double> padded(n_slots, 0.0);
    for (size_t i = 0; i < plain.size() && i < n_slots; i++)
        padded[i] = plain[i];
    // KRİTİK: plaintext ciphertext'in LEVEL'ında encode edilmeli — level 0'da kalırsa
    // FIDESlib GPU EvalAdd uyumsuz level'ı handle etmiyor → terim sessizce düşüyor
    // (poly_c0 bias'ı: act = c0+c1·u+c2·u² → c0 hiç eklenmiyordu!)
    auto pt = cc->MakeCKKSPackedPlaintext(padded, 1, ct->GetLevel());
    return cc->EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> FIDESContext::mult(Ciphertext<DCRTPoly>& a, Ciphertext<DCRTPoly>& b) {
    return cc->EvalMult(a, b);  // FIXEDAUTO: otomatik rescale
}

Ciphertext<DCRTPoly> FIDESContext::mult_plain(Ciphertext<DCRTPoly>& ct, const std::vector<double>& plain) {
    std::vector<double> padded(n_slots, 0.0);
    for (size_t i = 0; i < plain.size() && i < n_slots; i++)
        padded[i] = plain[i];
    auto pt = cc->MakeCKKSPackedPlaintext(padded);
    return cc->EvalMult(ct, pt);
}

Ciphertext<DCRTPoly> FIDESContext::mult_scalar(Ciphertext<DCRTPoly>& ct, double scalar) {
    return cc->EvalMult(ct, scalar);
}

Ciphertext<DCRTPoly> FIDESContext::rotate(Ciphertext<DCRTPoly>& ct, int steps) {
    return cc->EvalRotate(ct, steps);
}

Ciphertext<DCRTPoly> FIDESContext::negate(Ciphertext<DCRTPoly>& ct) {
    return cc->EvalNegate(ct);
}

Ciphertext<DCRTPoly> FIDESContext::bootstrap(Ciphertext<DCRTPoly>& ct) {
    // FIDESlib GPU Bootstrap, ModRaise içinde multScalar+rescale+dropToLevel(0) ile
    // girdiyi fiziksel olarak dibe indirir — OpenFHE CPU EvalBootstrap'taki
    // ModReduceInternalInPlace'in GPU karşılığı. SetLevel(ct,0) METADATA-ONLY idi:
    // level takibini 0'da dondurup rescale/dropToLevel'i etkisiz bırakıyordu → çöp.
    return cc->EvalBootstrap(ct);
}

// ═══════════════════════════════════════════════════════════
// BSGS Matvec (GPU-accelerated)
// y = W @ x, x encrypted, W plaintext
// ═══════════════════════════════════════════════════════════

Ciphertext<DCRTPoly> FIDESContext::matvec(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                           int in_dim, int out_dim) {
    int baby = (int)std::ceil(std::sqrt((double)in_dim));
    int giant = (int)std::ceil((double)in_dim / baby);

    // Input replikasyonu: x[0..in_dim-1] → periyodik N slot
    // (diagonal wrapping mod in_dim çalışsın diye)
    Ciphertext<DCRTPoly> enc_rep = enc_x;
    for (int shift = in_dim; shift < (int)n_slots; shift *= 2) {
        auto shifted = cc->EvalRotate(enc_rep, shift);
        cc->EvalAddInPlace(enc_rep, shifted);
    }

    // Baby-step rotations (GPU-accelerated)
    std::vector<Ciphertext<DCRTPoly>> baby_rots(baby);
    baby_rots[0] = enc_rep;
    for (int b = 1; b < baby; b++) {
        baby_rots[b] = cc->EvalRotate(enc_rep, b);
    }

    // Giant steps with diagonal encoding
    Ciphertext<DCRTPoly> acc;
    bool first = true;

    for (int g = 0; g < giant; g++) {
        Ciphertext<DCRTPoly> z_g;
        bool first_inner = true;

        for (int b = 0; b < baby; b++) {
            int k = g * baby + b;
            if (k >= in_dim) break;

            // BSGS diagonal (Halevi-Shoup): slot g*baby+row, col wrapping
            std::vector<double> diag(n_slots, 0.0);
            bool has_nonzero = false;
            for (int i = g * baby; i < g * baby + out_dim && i < (int)n_slots; i++) {
                int row = i - g * baby;
                int col = (i + b) % in_dim;
                diag[i] = W.at(row, col);
                if (std::abs(diag[i]) > 1e-15) has_nonzero = true;
            }
            if (!has_nonzero) continue;

            auto pt = cc->MakeCKKSPackedPlaintext(diag);
            auto prod = cc->EvalMult(baby_rots[b], pt);

            if (first_inner) {
                z_g = prod;
                first_inner = false;
            } else {
                cc->EvalAddInPlace(z_g, prod);
            }
        }

        if (z_g) {
            if (g > 0) {
                z_g = cc->EvalRotate(z_g, g * baby);
            }
            if (first) {
                acc = z_g;
                first = false;
            } else {
                cc->EvalAddInPlace(acc, z_g);
            }
        }
    }

    return acc;
}

// ═══════════════════════════════════════════════════════════
// Hoisted BSGS Matvec — 1 decompose + N cheap baby rotations
// EvalFastRotationPrecompute + batch EvalFastRotation
// ═══════════════════════════════════════════════════════════

Ciphertext<DCRTPoly> FIDESContext::matvec_hoisted(Ciphertext<DCRTPoly>& enc_x,
                                                    const GPUTensor& W,
                                                    int in_dim, int out_dim) {
    int baby = (int)std::ceil(std::sqrt((double)in_dim));
    int giant = (int)std::ceil((double)in_dim / baby);
    uint32_t M = 2 * ring_dim;

    // Input replikasyonu: periyodik N slot (diagonal wrapping için)
    Ciphertext<DCRTPoly> enc_rep = enc_x;
    for (int shift = in_dim; shift < (int)n_slots; shift *= 2) {
        auto shifted = cc->EvalRotate(enc_rep, shift);
        cc->EvalAddInPlace(enc_rep, shifted);
    }

    // HOISTED baby-step: 1 decompose, N cheap automorphisms
    auto precomp = cc->EvalFastRotationPrecompute(enc_rep);

    std::vector<int32_t> baby_indices;
    for (int b = 1; b < baby; b++) baby_indices.push_back(b);

    // Tüm baby rotasyonları TEK çağrıda
    std::vector<Ciphertext<DCRTPoly>> baby_rots;
    if (!baby_indices.empty())
        baby_rots = cc->EvalFastRotation(enc_rep, baby_indices, M, precomp);

    // Giant steps
    Ciphertext<DCRTPoly> acc;
    bool first = true;

    for (int g = 0; g < giant; g++) {
        Ciphertext<DCRTPoly> z_g;
        bool first_inner = true;

        for (int b = 0; b < baby; b++) {
            int k = g * baby + b;
            if (k >= in_dim) break;

            // BSGS diagonal (Halevi-Shoup): slot g*baby+row, col wrapping
            std::vector<double> diag(n_slots, 0.0);
            bool has_nonzero = false;
            for (int i = g * baby; i < g * baby + out_dim && i < (int)n_slots; i++) {
                int row = i - g * baby;
                int col = (i + b) % in_dim;
                diag[i] = W.at(row, col);
                if (std::abs(diag[i]) > 1e-15) has_nonzero = true;
            }
            if (!has_nonzero) continue;

            auto pt = cc->MakeCKKSPackedPlaintext(diag);
            auto& ct = (b == 0) ? enc_rep : baby_rots[b - 1];
            auto prod = cc->EvalMult(ct, pt);

            if (first_inner) { z_g = prod; first_inner = false; }
            else cc->EvalAddInPlace(z_g, prod);
        }

        if (z_g) {
            if (g > 0) z_g = cc->EvalRotate(z_g, g * baby);
            if (first) { acc = z_g; first = false; }
            else cc->EvalAddInPlace(acc, z_g);
        }
    }

    return acc;
}

// ═══════════════════════════════════════════════════════════
// Pre-encoded Matvec — encode once, GPU-only inference
// ═══════════════════════════════════════════════════════════

std::vector<Plaintext> FIDESContext::encode_diagonals(const GPUTensor& W,
                                                       int in_dim, int out_dim) {
    int baby = (int)std::ceil(std::sqrt((double)in_dim));
    std::vector<Plaintext> pts;
    pts.reserve(in_dim);
    for (int k = 0; k < in_dim; k++) {
        int g = k / baby;
        int b = k % baby;
        // BSGS shifted diagonal: slot g*baby+row, col wrapping
        std::vector<double> diag(n_slots, 0.0);
        bool has_nonzero = false;
        for (int i = g * baby; i < g * baby + out_dim && i < (int)n_slots; i++) {
            int row = i - g * baby;
            int col = (i + b) % in_dim;
            diag[i] = W.at(row, col);
            if (std::abs(diag[i]) > 1e-15) has_nonzero = true;
        }
        if (has_nonzero)
            pts.push_back(cc->MakeCKKSPackedPlaintext(diag));
        else
            pts.push_back(Plaintext());
    }
    return pts;
}

Ciphertext<DCRTPoly> FIDESContext::matvec_pre(Ciphertext<DCRTPoly>& enc_x,
                                               std::vector<Plaintext>& diag_pts,
                                               int in_dim, int out_dim) {
    int baby = (int)std::ceil(std::sqrt((double)in_dim));
    int giant = (int)std::ceil((double)in_dim / baby);
    uint32_t M = 2 * ring_dim;

    // Input replikasyonu: x[0..in_dim-1] → periyodik N slot (cyclic col wrapping için)
    Ciphertext<DCRTPoly> enc_rep = enc_x;
    for (int shift = in_dim; shift < (int)n_slots; shift *= 2) {
        auto shifted = cc->EvalRotate(enc_rep, shift);
        cc->EvalAddInPlace(enc_rep, shifted);
    }

    // Hoisted baby-step rotations
    auto precomp = cc->EvalFastRotationPrecompute(enc_rep);
    std::vector<int32_t> baby_indices;
    for (int b = 1; b < baby; b++) baby_indices.push_back(b);
    std::vector<Ciphertext<DCRTPoly>> baby_rots;
    if (!baby_indices.empty())
        baby_rots = cc->EvalFastRotation(enc_rep, baby_indices, M, precomp);

    // Giant steps — pure GPU, no encoding
    Ciphertext<DCRTPoly> acc;
    bool first = true;

    for (int g = 0; g < giant; g++) {
        Ciphertext<DCRTPoly> z_g;
        bool first_inner = true;

        for (int b = 0; b < baby; b++) {
            int k = g * baby + b;
            if (k >= in_dim) break;
            if (!diag_pts[k]) continue;  // skip zero diagonals

            auto& ct = (b == 0) ? enc_rep : baby_rots[b - 1];
            auto prod = cc->EvalMult(ct, diag_pts[k]);

            if (first_inner) { z_g = prod; first_inner = false; }
            else cc->EvalAddInPlace(z_g, prod);
        }

        if (z_g) {
            if (g > 0) z_g = cc->EvalRotate(z_g, g * baby);
            if (first) { acc = z_g; first = false; }
            else cc->EvalAddInPlace(acc, z_g);
        }
    }

    return acc;
}

// ═══════════════════════════════════════════════════════════
// ConvolutionTransform Matvec — fused GPU kernel
// ═══════════════════════════════════════════════════════════

Ciphertext<DCRTPoly> FIDESContext::matvec_conv(Ciphertext<DCRTPoly>& enc_x,
                                                const GPUTensor& W,
                                                int in_dim, int out_dim) {
    int bStep = (int)std::ceil(std::sqrt((double)in_dim));
    int gStep = (int)std::ceil((double)in_dim / bStep);

    // Pre-encode diagonal plaintexts
    std::vector<Plaintext> pts;
    pts.reserve(in_dim);
    for (int k = 0; k < in_dim; k++) {
        std::vector<double> diag(n_slots, 0.0);
        for (int i = 0; i < out_dim && i < (int)n_slots; i++) {
            if (i + k < in_dim) diag[i] = W.at(i, i + k);
        }
        pts.push_back(cc->MakeCKKSPackedPlaintext(diag));
    }

    // Baby step rotation indices
    std::vector<int> indexes(bStep);
    for (int b = 0; b < bStep; b++) indexes[b] = b;

    // ConvolutionTransform: fused hoisted rotation + batched ct-pt mult
    auto result = enc_x;
    cc->ConvolutionTransformInPlace(result, gStep, bStep, pts, indexes, bStep, in_dim);

    return result;
}

Ciphertext<DCRTPoly> FIDESContext::matvec_fast(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                                int in_dim, int out_dim) {
    auto pts = encode_diagonals(W, in_dim, out_dim);
    return matvec_pre(enc_x, pts, in_dim, out_dim);
}

// ═══════════════════════════════════════════════════════════
// Replicated BSGS (mamba3 "input-replicated true BSGS")
// y = W @ x; W [out_dim, in_dim]. R köşegeni TEK plaintext'te:
//   mask slot (j*window + i + j) = W[i, (i + d) mod in_dim], d = j + k*r
//   giriş: x periyodik replikasyon (slot s = x[s mod in_dim])
//   acc = Σ_k roll(input, k*r) ⊙ mask_k  — BSGS: baby roll (bi*r) + mask pre-rotation + giant roll
//   fold: roll(window+1) doubling → replica'ları topla → y[i] slot i'de
// Encode + ct-pt çarpım sayısı: in_dim → in_dim/r (~4×).
// ═══════════════════════════════════════════════════════════
Ciphertext<DCRTPoly> FIDESContext::matvec_rep(Ciphertext<DCRTPoly>& enc_x, const GPUTensor& W,
                                               int in_dim, int out_dim) {
    // r/window: window = in_dim'in katı, r·window ≤ n_slots, window ≥ out_dim + r - 1
    // Öncelik: r=8 (ring=32768, square), r=4 (ring=16384), r=3 (FFN-up out=4096), yoksa fallback
    int r = 0, window = 0;
    for (int rr : {8, 4, 3}) {
        int w = ((int)n_slots / rr / in_dim) * in_dim;   // in_dim katı, rr·w ≤ n_slots
        if (w >= out_dim + rr - 1) { r = rr; window = w; break; }
    }
    if (r < 2)
        return matvec_fast(enc_x, W, in_dim, out_dim);   // out çok büyük — paketleme yok

    // Input replikasyonu: slot s = x[s mod in_dim]
    Ciphertext<DCRTPoly> enc_rep = enc_x;
    for (int shift = in_dim; shift < (int)n_slots; shift *= 2) {
        auto shifted = cc->EvalRotate(enc_rep, shift);
        cc->EvalAddInPlace(enc_rep, shifted);
    }

    int per_replica = (in_dim + r - 1) / r;   // grup sayısı
    int baby = (int)std::ceil(std::sqrt((double)per_replica));
    int giant = (int)std::ceil((double)per_replica / baby);

    // Baby rotations: roll(input, bi*r)
    std::vector<Ciphertext<DCRTPoly>> baby_rots(baby);
    baby_rots[0] = enc_rep;
    for (int bi = 1; bi < baby; bi++)
        baby_rots[bi] = cc->EvalRotate(enc_rep, bi * r);

    // ── PIPELINED FAZ 1+2: mask encode (16 thread, CPU) ∥ BSGS eval (ana thread, GPU) ──
    // Encode producer mask_pts[k]'yi üretir; eval consumer her g bloğu için ready flag bekler.
    std::vector<Plaintext> mask_pts(per_replica);
    std::vector<std::atomic<bool>> ready(per_replica);
    for (auto& rflag : ready) rflag.store(false);

    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    auto encode_worker = [&]() {
        while (true) {
            size_t k = next.fetch_add(1);
            if (k >= (size_t)per_replica) break;
            int g = (int)k / baby;
            std::vector<double> mask(n_slots, 0.0);
            bool has_nonzero = false;
            for (int j = 0; j < r; j++) {
                int d = j + (int)k * r;
                if (d >= in_dim) continue;
                for (int i = 0; i < out_dim; i++) {
                    int slot = j * window + i + j;
                    if (slot >= (int)n_slots) continue;
                    int col = (i + d) % in_dim;
                    mask[slot] = W.at(i, col);
                    if (std::abs(mask[slot]) > 1e-15) has_nonzero = true;
                }
            }
            if (has_nonzero) {
                int giant_off = g * baby * r;
                if (giant_off != 0) {
                    std::vector<double> rot(n_slots, 0.0);
                    for (int s = 0; s < (int)n_slots; s++)
                        rot[s] = mask[(s - giant_off + (int)n_slots) % n_slots];
                    mask.swap(rot);
                }
                mask_pts[k] = cc->MakeCKKSPackedPlaintext(mask);
            }
            ready[k].store(true, std::memory_order_release);
        }
    };
    const unsigned int NT = std::min(16u, std::thread::hardware_concurrency());
    for (unsigned int ti = 0; ti < NT; ti++)
        pool.emplace_back(encode_worker);

    auto wait_block = [&](int g) {
        int from = g * baby, to = std::min(from + baby, per_replica);
        for (int k = from; k < to; k++)
            while (!ready[k].load(std::memory_order_acquire))
                std::this_thread::yield();
    };

    // Eval (ana thread) — BSGS, her g bloğu için encode'u bekler
    Ciphertext<DCRTPoly> acc;
    bool first = true;

    for (int g = 0; g < giant; g++) {
        wait_block(g);
        Ciphertext<DCRTPoly> z_g;
        bool first_inner = true;

        for (int bi = 0; bi < baby; bi++) {
            int k = g * baby + bi;
            if (k >= per_replica) break;
            if (!mask_pts[k]) continue;   // sıfır köşegen — atla

            auto prod = cc->EvalMult(baby_rots[bi], mask_pts[k]);

            if (first_inner) { z_g = prod; first_inner = false; }
            else cc->EvalAddInPlace(z_g, prod);
        }

        if (z_g) {
            if (g > 0) z_g = cc->EvalRotate(z_g, g * baby * r);
            if (first) { acc = z_g; first = false; }
            else cc->EvalAddInPlace(acc, z_g);
        }
    }
    for (auto& th : pool) th.join();

    // Fold: replica'ları topla — r=3: direkt (2 roll); güçlü 2 (r=4/8): doubling
    if (r == 3) {
        auto f1 = cc->EvalRotate(acc, window + 1);
        auto f2 = cc->EvalRotate(acc, 2 * (window + 1));
        cc->EvalAddInPlace(acc, f1);
        cc->EvalAddInPlace(acc, f2);
    } else {
        for (int s = 1; s < r; s *= 2) {
            auto folded = cc->EvalRotate(acc, s * (window + 1));
            cc->EvalAddInPlace(acc, folded);
        }
    }

    return acc;
}

// ═══════════════════════════════════════════════════════════
// Sipher Layer (GPU)
// ═══════════════════════════════════════════════════════════

void SipherGPULayer::load(const std::string& dir, int idx) {
    std::string p = dir + "/layers." + std::to_string(idx);
    W_Q = GPUTensor::load_bin(p + ".W_Q.bin");
    W_K = GPUTensor::load_bin(p + ".W_K.bin");
    W_V = GPUTensor::load_bin(p + ".W_V.bin");
    W_O = GPUTensor::load_bin(p + ".W_O.bin");
    ffn_up = GPUTensor::load_bin(p + ".ffn_up.bin");
    ffn_down = GPUTensor::load_bin(p + ".ffn_down.bin");
    norm1_w = GPUTensor::load_bin(p + ".norm1_w.bin");
    norm1_b = GPUTensor::load_bin(p + ".norm1_b.bin");
    norm2_w = GPUTensor::load_bin(p + ".norm2_w.bin");
    norm2_b = GPUTensor::load_bin(p + ".norm2_b.bin");

    auto gg = GPUTensor::load_bin(p + ".gate_global.bin");
    auto gl = GPUTensor::load_bin(p + ".gate_local.bin");
    auto pc = GPUTensor::load_bin(p + ".poly_coeffs.bin");
    gate_global = gg.data[0];
    gate_local = gl.data[0];
    poly_c0 = pc.data[0];
    poly_c1 = pc.data[1];
    poly_c2 = pc.data[2];
}

Ciphertext<DCRTPoly> SipherGPULayer::forward(
    FIDESContext& fides,
    Ciphertext<DCRTPoly>& enc_x,
    const GPUTensor& K_all,
    const GPUTensor& V_all,
    const std::vector<bool>& window_mask,
    int pos,
    int n_heads,
    int d_k,
    int seq_len)
{
    int D = W_Q.rows();
    int nh_dk = n_heads * d_k;
    uint32_t N = fides.n_slots;
    GPUTimer sec_timer;
    double t_encode, t_q, t_kv_build, t_global, t_local, t_gate, t_wo, t_ffn_up, t_polyact, t_ffn_down;
    static int s_fwd_cnt = 0;
    int my_layer = s_fwd_cnt++;

    // ── Pre-norm1: norm1(x) → attention'a girer ──
    sec_timer = GPUTimer();
    auto x_dec = fides.decrypt(enc_x, D);
    std::vector<double> x_orig = x_dec;   // residual için orijinal x (norm1 x_dec'i IN-PLACE değiştirir)
    {
        double mean = 0;
        for (int i = 0; i < D; i++) mean += x_dec[i];
        mean /= D;
        double var = 0;
        for (int i = 0; i < D; i++) { x_dec[i] -= mean; var += x_dec[i] * x_dec[i]; }
        var /= D;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < D; i++)
            x_dec[i] = norm1_w.at(i) * x_dec[i] * inv_std + norm1_b.at(i);
    }
    auto enc_normed = fides.encrypt(x_dec);

    // /√d_k'yı W_Q'ya GÖM (mult_scalar yerine — daha güvenli).
    GPUTensor WQ_s = W_Q;
    double qscale = 1.0 / std::sqrt((double)d_k);
    for (auto& v : WQ_s.data) v *= qscale;
    auto enc_q = fides.matvec_rep(enc_normed, WQ_s, D, nh_dk);
    // GERÇEK q'yı mutasyondan ÖNCE yakala (matvec'ler girdiyi REPLİKE edip bozuyor —
    // mutate edilmiş enc_q decrypt edilince 8× çıkıyor!). Global ve local için TAZE kopyalar.
    auto q_true = fides.decrypt(enc_q, nh_dk);
    if (g_dump_layer == my_layer) {
        std::cerr << "[DBG q-early]";
        for (int i = 0; i < 8; i++) std::cerr << " " << q_true[i];
        std::cerr << std::endl;
    }
    t_q = sec_timer.elapsed();

    // ── 2. Global attention: FEATURE-MAP (derece 2) linear attention ──
    // φ(x) = [x; x²] → global = (Σ V·K)@Q + (Σ V·K²)@Q²  (iki ayrı sparse matvec, concat yok)
    // feat_degree=1 ise sadece linear kısım (eski davranış — Kaggle checkpoint'leri)
    sec_timer = GPUTimer();
    GPUTensor KV_block, KV_sq_block;
    KV_block.shape = {nh_dk, nh_dk};
    KV_block.data.resize(nh_dk * nh_dk, 0.0);
    if (feat_degree >= 2) {
        KV_sq_block.shape = {nh_dk, nh_dk};
        KV_sq_block.data.resize(nh_dk * nh_dk, 0.0);
    }
    for (int h = 0; h < n_heads; h++)
        for (int t = 0; t <= pos; t++)
            for (int i = 0; i < d_k; i++)
                for (int j = 0; j < d_k; j++) {
                    double k = K_all.at(t, h*d_k+j);
                    KV_block.at(h*d_k+i, h*d_k+j) += V_all.at(t, h*d_k+i) * k;
                    if (feat_degree >= 2)
                        KV_sq_block.at(h*d_k+i, h*d_k+j) += V_all.at(t, h*d_k+i) * k * k;
                }
    t_kv_build = sec_timer.elapsed();
    sec_timer = GPUTimer();
    // gate_global'i KV_block'a GÖM (mult_scalar scale bozuk — c0/gate bug'ı).
    // gate_global=0 ise global katkı yok (sıfır matvec = boş ciphertext → CRASH! o yüzden atla).
    Ciphertext<DCRTPoly> enc_global;
    if (std::abs(gate_global) > 1e-12) {
        GPUTensor KVg = KV_block, KVg_sq = KV_sq_block;
        for (auto& v : KVg.data) v *= gate_global;
        for (auto& v : KVg_sq.data) v *= gate_global;
        auto enc_q_g = fides.encrypt(q_true);   // TAZE kopya (matvec girdiyi mutate eder)
        enc_global = fides.matvec_rep(enc_q_g, KVg, nh_dk, nh_dk);
        if (feat_degree >= 2) {
            // enc_q_g yukarıdaki matvec_rep tarafından mutasyona uğradı (replica kalıntısı
            // [0,nh_dk) dışında sıfırlanmaz). q²'yi bu bozuk ciphertext'ten hesaplamak global'i
            // ~2x saptırır; ölçüldü: feat_degree=1 (q² yok) cos=1, feat_degree=2 global 2x yanlış.
            // Bu yol gate_global=0'da hep atlandığı için bug hiç yakalanmamıştı. Taze encrypt şart.
            auto enc_q_sq = fides.encrypt(q_true);
            auto enc_q2 = fides.mult(enc_q_sq, enc_q_sq);   // Q² (ct×ct), gerçekten taze q'dan
            auto enc_global_sq = fides.matvec_rep(enc_q2, KVg_sq, nh_dk, nh_dk);
            fides.cc->EvalAddInPlace(enc_global, enc_global_sq);
        }
    } else {
        enc_global = fides.encrypt(std::vector<double>(nh_dk, 0.0));
    }
    t_global = sec_timer.elapsed();

    // ── 3. Local attention: pencere-only (32 pozisyon) — scores 8×, V 32× küçülür ──
    sec_timer = GPUTimer();

    // Pencere boyutunu window_mask'tan çıkar (true sayısı)
    int WINDOW = 0;
    for (int t = 0; t < seq_len; t++) if (window_mask[t]) WINDOW++;
    int w_start = pos - WINDOW + 1;   // 255 - 32 + 1 = 224

    // K_block: [n_heads*WINDOW, nh_dk] — sadece pencere pozisyonları
    int sc_dim = n_heads * WINDOW;
    GPUTensor K_block;
    K_block.shape = {sc_dim, nh_dk};
    K_block.data.resize(sc_dim * nh_dk, 0.0);
    for (int h = 0; h < n_heads; h++)
        for (int w = 0; w < WINDOW; w++)
            for (int i = 0; i < d_k; i++)
                K_block.at(h*WINDOW + w, h*d_k + i) = K_all.at(w_start + w, h*d_k + i);
    // Local skores için TAZE q (global matvec enc_q_g'yi REPLİKE edip mutate etti — tekrar
    // replike edilirse q 2× olur → softmax(2s) çarpılır → attention bozulur!)
    auto enc_q_fresh = fides.encrypt(q_true);
    auto enc_scores_all = fides.matvec_rep(enc_q_fresh, K_block, nh_dk, sc_dim);
    auto all_scores = fides.decrypt(enc_scores_all, sc_dim);

    // Softmax (sadece pencere — mask gerekmez)
    std::vector<double> attn_all(n_heads * WINDOW, 0.0);
    for (int h = 0; h < n_heads; h++) {
        double* sc = &attn_all[h * WINDOW];
        for (int w = 0; w < WINDOW; w++) sc[w] = all_scores[h * WINDOW + w];
        double max_s = *std::max_element(sc, sc + WINDOW);
        double sum_exp = 0.0;
        for (int w = 0; w < WINDOW; w++) { sc[w] = std::exp(sc[w] - max_s); sum_exp += sc[w]; }
        for (int w = 0; w < WINDOW; w++) sc[w] /= sum_exp;
    }

    // V matvec: 4 kafa × 32 pencere = 128 slot — block-diagonal, çıktı direkt kafa pozisyonlarında
    const int GROUP = 4;
    const int PACK = GROUP * WINDOW;   // 128
    Ciphertext<DCRTPoly> enc_local;
    bool first_local = true;
    for (int g = 0; g < n_heads / GROUP; g++) {
        std::vector<double> packed(PACK, 0.0);
        for (int hl = 0; hl < GROUP; hl++) {
            int h = g * GROUP + hl;
            for (int w = 0; w < WINDOW; w++)
                packed[hl * WINDOW + w] = attn_all[h * WINDOW + w] * gate_local;  // gate'i GÖM (mult_scalar scale bozuk!)
        }
        auto enc_packed = fides.encrypt(packed);

        GPUTensor Vg;
        Vg.shape = {nh_dk, PACK};
        Vg.data.resize(nh_dk * PACK, 0.0);
        for (int hl = 0; hl < GROUP; hl++) {
            int h = g * GROUP + hl;
            for (int w = 0; w < WINDOW; w++)
                for (int i = 0; i < d_k; i++)
                    Vg.at(h * d_k + i, hl * WINDOW + w) = V_all.at(w_start + w, h * d_k + i);
        }
        auto enc_lg = fides.matvec_rep(enc_packed, Vg, PACK, nh_dk);

        if (first_local) { enc_local = enc_lg; first_local = false; }
        else fides.cc->EvalAddInPlace(enc_local, enc_lg);
    }

    t_local = sec_timer.elapsed();

    // ── 4. Gate combine — gate'ler zaten gömülü (global: KV'de, local: attn'de) ──
    // gate_global=0 (local-only): add'e gerek yok — farklı level'daki ciphertext'leri
    // EvalAdd etmek FIDESlib GPU'da bozuk (sessiz düşme/scale hatası).
    sec_timer = GPUTimer();
    Ciphertext<DCRTPoly> enc_gated;
    if (std::abs(gate_global) > 1e-12)
        enc_gated = fides.add(enc_global, enc_local);
    else
        enc_gated = enc_local;
    t_gate = sec_timer.elapsed();

    // ── 5. W_O + residual (pre-norm: residual ORİJİNAL x'e) ──
    sec_timer = GPUTimer();
    // matvec_rep çıktıları replica-blokları içerir (fold toplar, sıfırlamaz!) → sonraki
    // matvec_rep'in replikasyonu onları [0,1024)'e taşıyıp girdiyi BOZAR. W_O öncesi temizle.
    auto gated_dec = fides.decrypt(enc_gated, nh_dk);
    auto enc_gated_clean = fides.encrypt(gated_dec);
    auto enc_wo = fides.matvec_rep(enc_gated_clean, W_O, nh_dk, D);
    auto enc_post = fides.add(enc_x, enc_wo);  // x + attn(norm1(x))
    t_wo = sec_timer.elapsed();

    // ── Pre-norm2: norm2(post) → FFN'e girer ──
    auto post_dec = fides.decrypt(enc_post, D);
    {
        double mean = 0;
        for (int i = 0; i < D; i++) mean += post_dec[i];
        mean /= D;
        double var = 0;
        for (int i = 0; i < D; i++) { post_dec[i] -= mean; var += post_dec[i] * post_dec[i]; }
        var /= D;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < D; i++)
            post_dec[i] = norm2_w.at(i) * post_dec[i] * inv_std + norm2_b.at(i);
    }
    auto enc_ffn_in = fides.encrypt(post_dec);

    // ── 6. PolyFFN (norm2(post) → FFN) ──
    // FFN up: 1024→4096 — 4 CHUNK matvec_rep (her biri 1024→1024, ~1.2s) —
    // naive matvec_fast (4096 çıktı, ~32s) yerine → ~6× hız.
    // Ders: matvec_rep çıktısı replica-blokları içerir → chunk arası re-encrypt şart;
    // girdi de mutate olur → her chunk'a TAZE encrypt.
    sec_timer = GPUTimer();
    int ffn_h = ffn_up.rows();
    int UP_CHUNK = 1024;
    auto ffn_in_dec = fides.decrypt(enc_ffn_in, D);
    Ciphertext<DCRTPoly> enc_up_acc;
    bool up_fc = true;
    for (int c = 0; c < ffn_h / UP_CHUNK; c++) {
        GPUTensor blk;
        blk.shape = {UP_CHUNK, D};   // [çıktı bloğu, girdi boyutu] — d'ye göre (d=512: [1024,512])
        blk.data.resize(UP_CHUNK * D);
        for (int i = 0; i < UP_CHUNK; i++)
            for (int j = 0; j < D; j++)
                blk.at(i, j) = ffn_up.at(c * UP_CHUNK + i, j);
        auto enc_in_c = fides.encrypt(ffn_in_dec);   // TAZE (matvec girdiyi mutate eder)
        auto enc_c = fides.matvec_rep(enc_in_c, blk, D, UP_CHUNK);
        auto cd = fides.decrypt(enc_c, UP_CHUNK);
        auto enc_clean = fides.encrypt(cd);          // replica bloklarını temizle
        if (c > 0) enc_clean = fides.rotate(enc_clean, -c * UP_CHUNK);  // [0,1024) → bloğa YUKARI taşı (negatif → 7168/6144/5120 key'leri)
        if (up_fc) { enc_up_acc = enc_clean; up_fc = false; }
        else fides.cc->EvalAddInPlace(enc_up_acc, enc_clean);
    }
    auto enc_up = enc_up_acc;
    t_ffn_up = sec_timer.elapsed();

    // PolyAct: c1*x + c2*x²
    sec_timer = GPUTimer();
    auto enc_x2 = fides.mult(enc_up, enc_up);
    auto enc_act = fides.mult_scalar(enc_up, poly_c1);
    auto enc_x2_s = fides.mult_scalar(enc_x2, poly_c2);
    fides.cc->EvalAddInPlace(enc_act, enc_x2_s);
    if (std::abs(poly_c0) > 1e-10) {
        // Plaintext-add (EvalAdd(ct, pt)) GPU'da SESSİZCE düşüyor: addPt içindeki
        // adjustPlaintextToCiphertext başarısız olursa assert(false) Release'te no-op → return.
        // Sağlam yol: c0'ı ciphertext olarak şifrele (aynı level) → ct-ct EvalAdd (kanıtlı yol).
        std::vector<double> c0_all(fides.n_slots, 0.0);
        for (int i = 0; i < ffn_up.rows(); i++) c0_all[i] = poly_c0;
        auto enc_c0 = fides.encrypt(c0_all, enc_act->GetLevel());
        enc_act = fides.add(enc_act, enc_c0);
    }
    t_polyact = sec_timer.elapsed();

    // FFN down: mask + rotate tabanlı (decrypt YOK — şifreli kalır)
    // enc_act (4096 slot) → mask ile chunk'ı ayır → rotate ile [0, 1024)'e taşı → matvec
    sec_timer = GPUTimer();
    Ciphertext<DCRTPoly> enc_down_result;
    {
        int chunk = D;
        Ciphertext<DCRTPoly> enc_down_acc;
        bool fc = true;
        for (int c = 0; c < ffn_h / chunk; c++) {
            // 1) mask: act'in [c*1024, (c+1)*1024) bloğunu ayır (ct×pt, ucuz)
            std::vector<double> cmask(fides.n_slots, 0.0);
            for (int s = c * chunk; s < (c + 1) * chunk; s++) cmask[s] = 1.0;
            auto enc_masked = fides.mult_plain(enc_act, cmask);
            // 2) bloğu [0, 1024)'e taşı: EvalRotate(+k) değeri slot s → s+k'ya taşır.
            //    chunk [c*1024,(c+1)*1024) → [0,1024): k = n - c*1024 (pozitif tümleyen —
            //    FIDESlib GPU negatif index key'i aramaz, pozitif key'ler üretildi)
            if (c > 0) enc_masked = fides.rotate(enc_masked, c * chunk);  // TEST: orijinal +1024 yönü
            if (g_dump_layer == my_layer) {
                // Chunk verisinin rotasyondan sonra hangi slot'lara gittiğini bul (tam tarama)
                auto mfull = fides.decrypt(enc_masked, (int)fides.n_slots);
                double peak = 0.0; int peak_s = -1;
                for (int s = 0; s < (int)fides.n_slots; s++) {
                    if (std::abs(mfull[s]) > peak) { peak = std::abs(mfull[s]); peak_s = s; }
                }
                std::cerr << "[DBG chunk" << c << "] peak=" << peak << " at slot " << peak_s;
                if (peak_s >= 0) {
                    std::cerr << " | vals[0..8]:";
                    for (int s = 0; s < 8; s++) std::cerr << " " << mfull[s];
                }
                std::cerr << std::endl;
            }
            // 3) blok matvec [1024, 1024]
            GPUTensor blk;
            blk.shape = {D, chunk};
            blk.data.resize(D * chunk);
            for (int i = 0; i < D; i++)
                for (int j = 0; j < chunk; j++)
                    blk.at(i, j) = ffn_down.at(i, c * chunk + j);
            auto enc_r = fides.matvec_rep(enc_masked, blk, chunk, D);
            if (g_dump_layer == my_layer) {
                // Per-chunk matvec çıktısını referans blok katkısıyla karşılaştır
                auto rv = fides.decrypt(enc_r, D);
                auto actv = fides.decrypt(enc_act, ffn_h);
                std::vector<double> refc(D, 0.0);
                for (int i = 0; i < D; i++)
                    for (int j = 0; j < chunk; j++)
                        refc[i] += blk.at(i, j) * actv[c * chunk + j];
                double maxe = 0, rmax = 0;
                for (int i = 0; i < D; i++) maxe = std::max(maxe, std::abs(rv[i] - refc[i]));
                for (int i = 0; i < D; i++) rmax = std::max(rmax, std::abs(refc[i]));
                std::cerr << "[DBG chunk" << c << " matvec] max_err=" << maxe << " (ref_max=" << rmax << ")";
                std::cerr << " | out[0..4]:";
                for (int i = 0; i < 4; i++) std::cerr << " " << rv[i];
                std::cerr << std::endl;
            }
            if (fc) { enc_down_acc = enc_r; fc = false; }
            else fides.cc->EvalAddInPlace(enc_down_acc, enc_r);
        }
        enc_down_result = enc_down_acc;
    }
    t_ffn_down = sec_timer.elapsed();

    std::cerr << "  ── Layer Profile ──" << std::endl;
    std::cerr << "  Q matvec (d=" << D << "):        " << t_q << "s" << std::endl;
    std::cerr << "  KV_cum build (CPU):      " << t_kv_build << "s" << std::endl;
    std::cerr << "  Global attn matvec:      " << t_global << "s" << std::endl;
    std::cerr << "  Local attn (block-diag): " << t_local << "s" << std::endl;
    std::cerr << "  Gate combine:            " << t_gate << "s" << std::endl;
    std::cerr << "  W_O matvec + residual:   " << t_wo << "s" << std::endl;
    std::cerr << "  FFN up matvec:           " << t_ffn_up << "s" << std::endl;
    std::cerr << "  PolyAct ct×ct:           " << t_polyact << "s" << std::endl;
    std::cerr << "  FFN down (4×chunk):      " << t_ffn_down << "s" << std::endl;
    std::cerr << "  TOTAL:                   " << t_q+t_kv_build+t_global+t_local+t_gate+t_wo+t_ffn_up+t_polyact+t_ffn_down << "s" << std::endl;

    // ── 7. FFN residual: post + ffn(norm2(post)) ──
    // Pre-norm: norm2 zaten FFN öncesi uygulandı, çıkışa norm yok
    auto enc_final = fides.add(enc_post, enc_down_result);

    // ══ DEBUG: ara değerleri plaintext referansla karşılaştır (g_dump_layer) ══
    if (g_dump_layer == my_layer) {
            auto cmp = [&](const char* name, Ciphertext<DCRTPoly>& ct, const std::vector<double>& ref, int n) {
                auto v = fides.decrypt(ct, n);
                double max_e = 0.0;
                for (int i = 0; i < n; i++)
                    max_e = std::max(max_e, std::abs(v[i] - ref[i]));
                std::cerr << "[DBG " << name << "] max_err=" << max_e << " | FHE:";
                for (int i = 0; i < std::min(n, 8); i++) std::cerr << " " << v[i];
                std::cerr << " | REF:";
                for (int i = 0; i < std::min(n, 8); i++) std::cerr << " " << ref[i];
                std::cerr << std::endl;
            };
            // Plaintext referans (x_dec girdisinden, float64)
            auto norm1_ref = [&](const std::vector<double>& in) {
                std::vector<double> o(D, 0.0);
                double mean = 0; for (int i = 0; i < D; i++) mean += in[i];
                mean /= D;
                double var = 0; for (int i = 0; i < D; i++) { o[i] = in[i] - mean; var += o[i] * o[i]; }
                var /= D;
                double is = 1.0 / std::sqrt(var + 1e-5);
                for (int i = 0; i < D; i++) o[i] = norm1_w.at(i) * o[i] * is + norm1_b.at(i);
                return o;
            };
            auto norm2_ref = [&](const std::vector<double>& in) {
                std::vector<double> o(D, 0.0);
                double mean = 0; for (int i = 0; i < D; i++) mean += in[i];
                mean /= D;
                double var = 0; for (int i = 0; i < D; i++) { o[i] = in[i] - mean; var += o[i] * o[i]; }
                var /= D;
                double is = 1.0 / std::sqrt(var + 1e-5);
                for (int i = 0; i < D; i++) o[i] = norm2_w.at(i) * o[i] * is + norm2_b.at(i);
                return o;
            };
            auto x_norm = norm1_ref(x_orig);
            std::vector<double> q_ref(nh_dk, 0.0);
            for (int i = 0; i < nh_dk; i++) {
                for (int j = 0; j < D; j++) q_ref[i] += x_norm[j] * W_Q.at(i, j);
                q_ref[i] /= std::sqrt((double)d_k);
            }
            // q vektörlerini dosyaya yaz (offline analiz için)
            {
                auto qv = fides.decrypt(enc_q, nh_dk);
                std::ofstream qf("/tmp/q_fhe.txt");
                for (int i = 0; i < nh_dk; i++) qf << qv[i] << "\n";
                std::ofstream qr("/tmp/q_ref.txt");
                for (int i = 0; i < nh_dk; i++) qr << q_ref[i] << "\n";
                std::ofstream nf("/tmp/xnorm.txt");
                for (int i = 0; i < D; i++) nf << x_norm[i] << "\n";
                std::ofstream wf("/tmp/W_Q.bin", std::ios::binary);
                wf.write((const char*)W_Q.data.data(), W_Q.data.size() * sizeof(double));
            }
            // KV_block (global) + local referans
            std::vector<double> KVb(nh_dk * nh_dk, 0.0), KVsq(feat_degree >= 2 ? nh_dk * nh_dk : 0, 0.0);
            for (int h = 0; h < n_heads; h++)
                for (int t = 0; t <= pos; t++)
                    for (int i = 0; i < d_k; i++)
                        for (int j = 0; j < d_k; j++) {
                            double k = K_all.at(t, h*d_k+j);
                            KVb[h*d_k+i + nh_dk*(h*d_k+j)] += V_all.at(t, h*d_k+i) * k;
                            if (feat_degree >= 2) KVsq[h*d_k+i + nh_dk*(h*d_k+j)] += V_all.at(t, h*d_k+i) * k * k;
                        }
            std::vector<double> glob_ref(nh_dk, 0.0);
            for (int i = 0; i < nh_dk; i++) {
                for (int j = 0; j < nh_dk; j++)
                    glob_ref[i] += KVb[i + nh_dk*j] * q_ref[j];
                if (feat_degree >= 2)
                    for (int j = 0; j < nh_dk; j++)
                        glob_ref[i] += KVsq[i + nh_dk*j] * q_ref[j] * q_ref[j];
            }
            // local (pencere softmax + V)
            int WINDOW = 0;
            for (int t = 0; t < seq_len; t++) if (window_mask[t]) WINDOW++;
            int w_start = pos - WINDOW + 1;
            std::vector<double> local_ref(nh_dk, 0.0);
            for (int h = 0; h < n_heads; h++) {
                std::vector<double> sc(WINDOW, -1e9);
                for (int w = 0; w < WINDOW; w++) {
                    double s = 0;
                    for (int i = 0; i < d_k; i++)
                        s += q_ref[h*d_k+i] * K_all.at(w_start+w, h*d_k+i);
                    sc[w] = s;
                }
                double mx = *std::max_element(sc.begin(), sc.end());
                double se = 0;
                for (int w = 0; w < WINDOW; w++) { sc[w] = std::exp(sc[w] - mx); se += sc[w]; }
                for (int w = 0; w < WINDOW; w++) sc[w] /= se;
                for (int i = 0; i < d_k; i++) {
                    double acc = 0;
                    for (int w = 0; w < WINDOW; w++) acc += sc[w] * V_all.at(w_start+w, h*d_k+i);
                    local_ref[h*d_k+i] = acc;
                }
            }
            std::vector<double> gated_ref(nh_dk, 0.0);
            for (int i = 0; i < nh_dk; i++) gated_ref[i] = gate_global * glob_ref[i] + gate_local * local_ref[i];
            // FHE enc_global gate_global'ı KV_block'a GÖMÜYOR (enc_global = gate·global),
            // glob_ref ise gate'SİZ. Elma-elma karşılaştırma için glob_ref'i gate ile ölçekle;
            // aksi halde cmp("global") gate_global çarpanı kadar sapar (küçük gate'te noise floor).
            std::vector<double> glob_ref_gated(nh_dk, 0.0);
            for (int i = 0; i < nh_dk; i++) glob_ref_gated[i] = gate_global * glob_ref[i];
            std::vector<double> post_ref(D, 0.0);
            for (int i = 0; i < D; i++) {
                double acc = 0;
                for (int j = 0; j < nh_dk; j++) acc += gated_ref[j] * W_O.at(i, j);
                post_ref[i] = x_orig[i] + acc;
            }
            auto pn = norm2_ref(post_ref);
            int ffn_h = ffn_up.rows();
            std::vector<double> up_ref(ffn_h, 0.0);
            for (int i = 0; i < ffn_h; i++) {
                double acc = 0;
                for (int j = 0; j < D; j++) acc += pn[j] * ffn_up.at(i, j);
                up_ref[i] = acc;
            }
            std::vector<double> act_ref(ffn_h, 0.0);
            for (int i = 0; i < ffn_h; i++) {
                double a = poly_c1 * up_ref[i] + poly_c2 * up_ref[i] * up_ref[i];
                if (std::abs(poly_c0) > 1e-10) a += poly_c0;
                act_ref[i] = a;
            }
            std::vector<double> down_ref(D, 0.0);
            for (int i = 0; i < D; i++) {
                double acc = 0;
                for (int j = 0; j < ffn_h; j++) acc += act_ref[j] * ffn_down.at(i, j);
                down_ref[i] = acc;
            }
            cmp("normed", enc_normed, x_norm, D);
            cmp("q", enc_q, q_ref, nh_dk);
            cmp("global", enc_global, glob_ref_gated, nh_dk);
            cmp("local", enc_local, local_ref, nh_dk);
            cmp("gated", enc_gated, gated_ref, nh_dk);
            cmp("wo_post", enc_post, post_ref, D);
            cmp("ffn_in", enc_ffn_in, pn, D);
            cmp("up", enc_up, up_ref, ffn_h);
            cmp("x2", enc_x2, [&]{ std::vector<double> f(ffn_h); for (int i = 0; i < ffn_h; i++) f[i] = up_ref[i]*up_ref[i]; return f; }(), ffn_h);
            cmp("act", enc_act, act_ref, ffn_h);
            cmp("down", enc_down_result, down_ref, D);
            cmp("final", enc_final, [&]{ std::vector<double> f(D); for (int i = 0; i < D; i++) f[i] = post_ref[i] + down_ref[i]; return f; }(), D);
    }

    return enc_final;
}

// ═══════════════════════════════════════════════════════════
// Plaintext mirror (eğitilmiş modelin birebir yeniden uygulaması)
// ═══════════════════════════════════════════════════════════

void SipherGPULayer::norm_plain(const std::vector<double>& in, const GPUTensor& w,
                                const GPUTensor& b, int D, std::vector<double>& out) {
    double mean = 0;
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= D;
    double var = 0;
    out.resize(D);
    for (int i = 0; i < D; i++) { out[i] = in[i] - mean; var += out[i] * out[i]; }
    var /= D;
    double inv_std = 1.0 / std::sqrt(var + 1e-5);
    for (int i = 0; i < D; i++)
        out[i] = w.at(i) * out[i] * inv_std + b.at(i);
}

void SipherGPULayer::forward_plaintext(
    const std::vector<std::vector<double>>& hs,
    const GPUTensor& K_all, const GPUTensor& V_all,
    int window_size, int n_heads, int d_k, int seq_len,
    std::vector<std::vector<double>>& out)
{
    int D = W_Q.rows();
    int nh_dk = n_heads * d_k;
    out.assign(seq_len, std::vector<double>(D, 0.0));

    // ── Faz A (sıralı, ucuz): KV_cum[t] VE (feat>=2 ise) KV_sq_cum[t]'yi biriktir ──
    // KV_cum[t][h*dk*dk + i*dk + j] = Σ_{tp≤t} V[tp,h*dk+i]·K[tp,h*dk+j]
    // KV_sq_cum[t][...]                 = Σ_{tp≤t} V[tp,h*dk+i]·K[tp,h*dk+j]²  (feature-map derece 2)
    std::vector<std::vector<double>> KV_cum(seq_len, std::vector<double>(n_heads * d_k * d_k, 0.0));
    std::vector<std::vector<double>> KV_sq_cum;
    if (feat_degree >= 2)
        KV_sq_cum.assign(seq_len, std::vector<double>(n_heads * d_k * d_k, 0.0));
    {
        std::vector<double> acc(n_heads * d_k * d_k, 0.0);
        std::vector<double> acc_sq(feat_degree >= 2 ? n_heads * d_k * d_k : 0, 0.0);
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < n_heads; h++) {
                double* kvh = &acc[h * d_k * d_k];
                double* kvs = feat_degree >= 2 ? &acc_sq[h * d_k * d_k] : nullptr;
                for (int i = 0; i < d_k; i++) {
                    double V_ti = V_all.at(t, h * d_k + i);
                    if (V_ti == 0.0) continue;
                    for (int j = 0; j < d_k; j++) {
                        double k = K_all.at(t, h * d_k + j);
                        kvh[i * d_k + j] += V_ti * k;
                        if (kvs) kvs[i * d_k + j] += V_ti * k * k;
                    }
                }
            }
            KV_cum[t] = acc;
            if (feat_degree >= 2) KV_sq_cum[t] = acc_sq;
        }
    }

    // ── Faz B (paralel): pozisyonlar bağımsız — norm1/Q/global/local/W_O/FFN ──
    const unsigned int NT = std::min(16u, std::thread::hardware_concurrency());
    std::atomic<size_t> next_t{0};
    std::vector<std::thread> pool;
    auto process_pos = [&]() {
        while (true) {
            size_t t = next_t.fetch_add(1);
            if (t >= (size_t)seq_len) break;
            const auto& KV = KV_cum[t];
            const bool feat = feat_degree >= 2;
            const auto& KVs = feat ? KV_sq_cum[t] : KV_cum[t];   // feat yoksa kullanılmaz

            // norm1(x) → Q
            std::vector<double> x_norm;
            norm_plain(hs[t], norm1_w, norm1_b, D, x_norm);
            std::vector<double> q(nh_dk, 0.0);
            for (int i = 0; i < nh_dk; i++) {
                double acc = 0;
                for (int j = 0; j < D; j++) acc += x_norm[j] * W_Q.at(i, j);
                q[i] = acc / std::sqrt((double)d_k);
            }

            // Global: KV_lin @ q (+ feat>=2 ise KV_sq @ q²)
            std::vector<double> global(nh_dk, 0.0);
            for (int h = 0; h < n_heads; h++) {
                const double* kvh = &KV[h * d_k * d_k];
                const double* kvs = feat ? &KVs[h * d_k * d_k] : nullptr;
                for (int i = 0; i < d_k; i++) {
                    double acc = 0, acc2 = 0;
                    for (int j = 0; j < d_k; j++) {
                        double qj = q[h*d_k+j];
                        acc  += kvh[i * d_k + j] * qj;
                        if (kvs) acc2 += kvs[i * d_k + j] * qj * qj;
                    }
                    global[h*d_k+i] = acc + acc2;
                }
            }

            // Local: pencere softmax + V
            std::vector<double> local(nh_dk, 0.0);
            for (int h = 0; h < n_heads; h++) {
                std::vector<double> scores(seq_len, -1e9);
                for (int tp = 0; tp < seq_len; tp++) {
                    int diff = (int)t - tp;
                    if (diff < 0 || diff >= window_size) continue;
                    double s = 0;
                    for (int i = 0; i < d_k; i++)
                        s += q[h*d_k+i] * K_all.at(tp, h*d_k+i);
                    scores[tp] = s;
                }
                double max_s = *std::max_element(scores.begin(), scores.end());
                double sum_exp = 0;
                for (int tp = 0; tp < seq_len; tp++) {
                    scores[tp] = std::exp(scores[tp] - max_s);
                    sum_exp += scores[tp];
                }
                for (int tp = 0; tp < seq_len; tp++) scores[tp] /= sum_exp;
                for (int i = 0; i < d_k; i++) {
                    double acc = 0;
                    for (int tp = 0; tp < seq_len; tp++)
                        acc += scores[tp] * V_all.at(tp, h*d_k+i);
                    local[h*d_k+i] = acc;
                }
            }

            // Gate + W_O + residual (pre-norm)
            std::vector<double> post(D, 0.0);
            for (int i = 0; i < D; i++) {
                double acc = 0;
                for (int j = 0; j < nh_dk; j++)
                    acc += (gate_global * global[j] + gate_local * local[j]) * W_O.at(i, j);
                post[i] = hs[t][i] + acc;
            }

            // norm2 → PolyFFN → +residual
            std::vector<double> post_norm;
            norm_plain(post, norm2_w, norm2_b, D, post_norm);
            int ffn_h = ffn_up.rows();
            std::vector<double> up(ffn_h, 0.0);
            for (int i = 0; i < ffn_h; i++) {
                double acc = 0;
                for (int j = 0; j < D; j++) acc += post_norm[j] * ffn_up.at(i, j);
                double act = poly_c1 * acc + poly_c2 * acc * acc;
                if (std::abs(poly_c0) > 1e-10) act += poly_c0;
                up[i] = act;
            }
            for (int i = 0; i < D; i++) {
                double acc = 0;
                for (int j = 0; j < ffn_h; j++)
                    acc += up[j] * ffn_down.at(i, j);
                out[t][i] = post[i] + acc;
            }
        }
    };
    for (unsigned int ti = 0; ti < NT; ti++)
        pool.emplace_back(process_pos);
    for (auto& th : pool) th.join();
}

// ═══════════════════════════════════════════════════════════
// Sipher Model (GPU)
// ═══════════════════════════════════════════════════════════

void SipherGPUModel::load(const std::string& dir) {
    config = SipherGPUConfig::load(dir + "/config.txt");
    token_emb = GPUTensor::load_bin(dir + "/token_emb.bin");
    emb_proj = GPUTensor::load_bin(dir + "/emb_proj.bin");
    pos_emb = GPUTensor::load_bin(dir + "/pos_emb.bin");
    head_proj = GPUTensor::load_bin(dir + "/head_proj.bin");

    layers.resize(config.n_layers);
    for (int i = 0; i < config.n_layers; i++) {
        layers[i].load(dir, i);
        layers[i].feat_degree = config.feat_degree;
        std::cout << "  Layer " << i << " loaded" << std::endl;
    }
    std::cout << "  Model: " << config.n_layers << " layers, d=" << config.d_model
              << ", feat_degree=" << config.feat_degree << std::endl;
}

Ciphertext<DCRTPoly> SipherGPUModel::embed(FIDESContext& fides, const std::vector<int>& tokens, int pos) {
    int D = config.d_model;
    int E = config.emb_dim;

    std::vector<double> emb(D, 0.0);
    int tok = tokens[pos];
    for (int i = 0; i < E; i++) {
        double e = token_emb.at(tok, i);
        for (int j = 0; j < D; j++)
            emb[j] += e * emb_proj.at(j, i);
    }
    for (int j = 0; j < D; j++)
        emb[j] += pos_emb.at(pos, j);

    return fides.encrypt(emb);
}

std::vector<double> SipherGPUModel::infer(FIDESContext& fides, const std::vector<int>& tokens) {
    int D = config.d_model;
    int SEQ = config.seq_len;
    int NH = config.n_heads;
    int DK = config.d_k;
    int pos = SEQ - 1;

    // Window mask (sabit, pozisyona bağlı)
    std::vector<bool> window_mask(SEQ, false);
    for (int t = 0; t < SEQ; t++) {
        int diff = pos - t;
        if (diff >= 0 && diff < config.window)
            window_mask[t] = true;
    }

    // Tüm pozisyonların embedding'lerini hesapla (plaintext, K/V için)
    std::vector<std::vector<double>> hidden_states(SEQ, std::vector<double>(D, 0.0));
    for (int t = 0; t < SEQ; t++) {
        int tok = (t < (int)tokens.size()) ? tokens[t] : 0;
        for (int i = 0; i < config.emb_dim; i++) {
            double e = token_emb.at(tok, i);
            for (int j = 0; j < D; j++)
                hidden_states[t][j] += e * emb_proj.at(j, i);
        }
        for (int j = 0; j < D; j++)
            hidden_states[t][j] += pos_emb.at(t, j);
    }

    // Embed + encrypt (target pozisyon)
    auto enc_x = embed(fides, tokens, pos);

    // Multi-layer inference — her katman kendi K/V'sini hesaplar
    double total_layer_time = 0;

    for (int l = 0; l < config.n_layers; l++) {
        // K/V, norm1(hidden_states)'ten hesaplanır — eğitilmiş modelle aynı
        // (forward'daki Q yolu norm1(enc_x)'ten geliyor; K/V de aynı normalize girdiden gelmeli)
        GPUTensor K_all, V_all;
        K_all.shape = {SEQ, NH * DK};
        V_all.shape = {SEQ, NH * DK};
        K_all.data.resize(SEQ * NH * DK, 0.0);
        V_all.data.resize(SEQ * NH * DK, 0.0);

        auto& W_K = layers[l].W_K;
        auto& W_V = layers[l].W_V;
        for (int t = 0; t < SEQ; t++) {
            std::vector<double> hnorm;
            SipherGPULayer::norm_plain(hidden_states[t], layers[l].norm1_w,
                                       layers[l].norm1_b, D, hnorm);
            for (int i = 0; i < NH * DK; i++) {
                double k_val = 0.0, v_val = 0.0;
                for (int j = 0; j < D; j++) {
                    k_val += hnorm[j] * W_K.at(i, j);
                    v_val += hnorm[j] * W_V.at(i, j);
                }
                K_all.at(t, i) = k_val;
                V_all.at(t, i) = v_val;
            }
        }

        // Bootstrap: residual'in noise birikimini tazele (katman başlangıcı)
        // OpenFHE kuralı: EvalBootstrap GİRİŞİ level 0'da olmalı — yüksek level'de no-op/bozuk sonuç
        if (config.bootstrap_interval > 0 && l > 0 && l % config.bootstrap_interval == 0) {
            GPUTimer bt;
            enc_x = fides.bootstrap(enc_x);
            std::cout << "         bootstrap: " << bt.elapsed() << "s" << std::endl;
        }

        GPUTimer timer;
        enc_x = layers[l].forward(fides, enc_x, K_all, V_all, window_mask, pos, NH, DK, SEQ);
        double layer_time = timer.elapsed();
        total_layer_time += layer_time;
        std::cout << "  Layer " << l << " GPU FHE: " << layer_time << "s" << std::endl;

        // TÜM pozisyonların hidden state'lerini plaintext mirror ile ilerlet
        // (katman l+1'in K/V'si, katman l'in TÜM pozisyon çıktılarından hesaplanmalı)
        GPUTimer mtimer;
        std::vector<std::vector<double>> hs_next;
        layers[l].forward_plaintext(hidden_states, K_all, V_all, config.window,
                                    NH, DK, SEQ, hs_next);
        hidden_states = hs_next;
        std::cout << "         plaintext mirror: " << mtimer.elapsed() << "s" << std::endl;
    }

    std::cout << "\n  ── GPU Inference Summary ──" << std::endl;
    std::cout << "  Total layer time:     " << total_layer_time << "s" << std::endl;
    std::cout << "  Layers:               " << config.n_layers << std::endl;
    std::cout << "  Per layer avg:        " << total_layer_time / config.n_layers << "s" << std::endl;

    // Head (FHE matvec)
    auto fhe_hidden = fides.decrypt(enc_x, D);
    auto enc_head_in = fides.encrypt(fhe_hidden);
    auto enc_head = fides.matvec_rep(enc_head_in, head_proj, D, config.emb_dim);
    auto fhe_emb_out = fides.decrypt(enc_head, config.emb_dim);

    // Logits (plaintext)
    std::vector<double> logits(config.vocab_size, 0.0);
    for (int v = 0; v < config.vocab_size; v++)
        for (int i = 0; i < config.emb_dim; i++)
            logits[v] += fhe_emb_out[i] * token_emb.at(v, i);

    // ── Mirror logits karşılaştırması (FHE vs plaintext) ──
    if (!hidden_states.empty()) {
        std::vector<double> logits_pt(config.vocab_size, 0.0);
        const auto& hp = hidden_states.back();   // son pozisyon mirror çıktısı (D)
        std::vector<double> head_pt(config.emb_dim, 0.0);
        for (int i = 0; i < config.emb_dim; i++)
            for (int j = 0; j < D; j++)
                head_pt[i] += hp[j] * head_proj.at(i, j);
        for (int v = 0; v < config.vocab_size; v++)
            for (int i = 0; i < config.emb_dim; i++)
                logits_pt[v] += head_pt[i] * token_emb.at(v, i);
        double dot = 0, na = 0, nb = 0;
        for (int v = 0; v < config.vocab_size; v++) {
            dot += logits[v] * logits_pt[v];
            na += logits[v] * logits[v];
            nb += logits_pt[v] * logits_pt[v];
        }
        double cos_l = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        auto top5 = [](const std::vector<double>& lg) {
            std::vector<int> idx(lg.size());
            for (size_t i = 0; i < lg.size(); i++) idx[i] = (int)i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) { return lg[a] > lg[b]; });
            return std::vector<int>(idx.begin(), idx.begin() + 5);
        };
        auto t5f = top5(logits);
        auto t5p = top5(logits_pt);
        int match = 0;
        for (int i = 0; i < 5; i++)
            if (t5f[i] == t5p[i]) match++;
        std::cout << "  Mirror vs FHE: cos_sim(logits)=" << cos_l
                  << " | top-5 uyum=" << match << "/5"
                  << " | top-1: FHE=" << t5f[0] << " mirror=" << t5p[0] << std::endl;
        if (t5f[0] == t5p[0])
            std::cout << "  ✓ FHE top-1 = plaintext top-1 (aynı token)" << std::endl;
        else
            std::cout << "  ✗ FHE top-1 ≠ plaintext top-1 (gürültü top-1'i değiştirdi)" << std::endl;
    }

    return logits;
}

int SipherGPUModel::generate_next(FIDESContext& fides, const std::vector<int>& tokens,
                                   double temp, int top_k) {
    auto logits = infer(fides, tokens);

    // Repetition penalty
    for (int t : tokens) {
        if (t < (int)logits.size()) {
            if (logits[t] > 0) logits[t] /= 2.0;
            else logits[t] *= 2.0;
        }
    }

    // Top-k
    if (top_k > 0 && top_k < (int)logits.size()) {
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
